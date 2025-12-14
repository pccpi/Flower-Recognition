import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import img_to_array
from PIL import Image
import cv2

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Распознавание цветов", page_icon="🌸", layout="wide")

MODEL_PATH = "flower_best.keras"
IMG_SIZE = 224

# Модель обучалась на этих классах (английские имена — внутренние)
CLASS_NAMES_EN = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

# Отображение для интерфейса (русские названия + с большой буквы)
CLASS_NAMES_RU = {
    "daisy": "Ромашка",
    "dandelion": "Одуванчик",
    "rose": "Роза",
    "sunflower": "Подсолнух",
    "tulip": "Тюльпан",
}


# =========================
# MODEL LOADING (robust for Grad-CAM)
# =========================
@st.cache_resource
def load_models():
    # Загружаем сохранённый Sequential
    seq = keras.models.load_model(MODEL_PATH, compile=False)

    # "прогрев" — чтобы у Sequential появился граф/inputs/outputs
    _ = seq(tf.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32), training=False)

    # Оборачиваем в Functional-граф на тех же слоях (веса сохраняются)
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input")
    x = inp
    conv_outputs = {}  # имя Conv2D -> тензор выхода

    for layer in seq.layers:
        x = layer(x)
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_outputs[layer.name] = x

    func = tf.keras.Model(inputs=inp, outputs=x, name="functional_wrapper")
    return func, conv_outputs

model, conv_outputs_map = load_models()
CONV_LAYERS = list(conv_outputs_map.keys())


# =========================
# PREPROCESS
# =========================
def preprocess_pil(pil_img: Image.Image):
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = img_to_array(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)
    return arr


# =========================
# GRAD-CAM
# =========================
def make_gradcam_heatmap(x: tf.Tensor, conv_layer_name: str):
    conv_out = conv_outputs_map[conv_layer_name]  # тензор из functional-графа

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[conv_out, model.outputs[0]]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(x, training=False)
        tape.watch(conv_outputs)

        pred_index = tf.argmax(preds[0])
        score = preds[:, pred_index][0]

    grads = tape.gradient(score, conv_outputs)
    if grads is None:
        return None

    conv_outputs = conv_outputs[0]  # (h,w,c)
    grads = grads[0]                # (h,w,c)

    weights = tf.reduce_mean(grads, axis=(0, 1))              # (c,)
    cam = tf.reduce_sum(conv_outputs * weights, axis=-1)      # (h,w)

    cam = tf.maximum(cam, 0)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy()


def overlay_heatmap_on_pil(heatmap: np.ndarray, pil_img: Image.Image, alpha=0.40):
    # resize heatmap to original image size
    w, h = pil_img.size
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    img_rgb = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    overlay_bgr = cv2.addWeighted(heatmap_color, alpha, img_bgr, 1 - alpha, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    return overlay_rgb


def predict_with_gradcam(pil_img: Image.Image, conv_layer_preference: str | None):
    arr = preprocess_pil(pil_img)
    x = tf.convert_to_tensor(arr, dtype=tf.float32)

    probs = model(x, training=False)[0].numpy()

    class_idx = int(np.argmax(probs))
    class_name_en = CLASS_NAMES_EN[class_idx]
    class_name_ru = CLASS_NAMES_RU.get(class_name_en, class_name_en)

    if not CONV_LAYERS:
        return class_name_en, class_name_ru, probs, None, None, None

    candidates = []
    if conv_layer_preference and conv_layer_preference in CONV_LAYERS:
        candidates.append(conv_layer_preference)
    candidates += [n for n in reversed(CONV_LAYERS) if n not in candidates]

    for layer_name in candidates:
        heatmap = make_gradcam_heatmap(x, layer_name)
        if heatmap is not None:
            return class_name_en, class_name_ru, probs, heatmap, None, layer_name

    return class_name_en, class_name_ru, probs, None, None, None


# =========================
# UI
# =========================
st.title("Распознавание цветов (CNN) + Grad-CAM")
st.write(
    "Загрузите фото цветка или сделайте снимок с камеры."
    " Модель предскажет класс и покажет на какие области изображения она опиралась."
)

with st.sidebar:
    st.header("Настройки")

    st.markdown("**Классы:**")
    for k in CLASS_NAMES_EN:
        st.write(f"- {CLASS_NAMES_RU[k]} ({k})")

    st.divider()

    if CONV_LAYERS:
        st.markdown("**Слой для Grad-CAM**")
        default_idx = len(CONV_LAYERS) - 1
        selected_layer = st.selectbox(
            "Выберите слой",
            options=CONV_LAYERS,
            index=default_idx
        )
    else:
        selected_layer = None
        st.warning("Conv2D слои не найдены — Grad-CAM отключён.")

    alpha = st.slider("Наложение heatmap (alpha)", 0.0, 0.9, 0.40, 0.05)
    img_width = st.slider("Ширина изображений", 250, 900, 520, 10)

st.markdown("### Ввод изображения")
col_upload, col_cam = st.columns(2)

with col_upload:
    uploaded_file = st.file_uploader("Загрузить изображение (JPG/PNG)", type=["jpg", "jpeg", "png"])

with col_cam:
    camera_image = st.camera_input("Или сделать снимок с камеры")

pil_image = None
if camera_image is not None:
    pil_image = Image.open(camera_image).convert("RGB")
elif uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")

if pil_image is None:
    st.info("Пока нет изображения. Загрузи файл или сделай фото.")
    st.stop()

# Predict + Grad-CAM
pred_en, pred_ru, probs, heatmap, overlay, used_layer = predict_with_gradcam(pil_image, selected_layer)

# Если построили heatmap — накладываем с выбранным alpha
if heatmap is not None:
    overlay = overlay_heatmap_on_pil(heatmap, pil_image, alpha=alpha)

# Layout
c1, c2, c3 = st.columns([1.2, 1.2, 1])

with c1:
    st.markdown("### Оригинал")
    st.image(pil_image, width=img_width)

with c2:
    st.markdown("### Grad-CAM")
    if overlay is None:
        st.warning("Grad-CAM не удалось построить. Попробуй другой conv-слой в сайдбаре.")
    else:
        st.image(overlay, width=img_width)
        st.caption(f"Использованный слой: `{used_layer}`")

with c3:
    st.markdown("### Предсказание")
    st.markdown(f"**Класс:** {pred_ru}")
    st.caption(f"Англ.: {pred_en}")

    df = pd.DataFrame({
        "Класс": [CLASS_NAMES_RU[c] for c in CLASS_NAMES_EN],
        "Вероятность": probs
    }).set_index("Класс")

    st.bar_chart(df)

    st.caption(
        "Тёплые зоны на Grad-CAM показывают области, которые сильнее всего повлияли на решение модели."
    )
