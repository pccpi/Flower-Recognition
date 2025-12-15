import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Распознавание цветов (EfficientNet + Grad-CAM)", page_icon="🌸", layout="wide")

MODEL_PATH = "flower_best.keras"
IMG_SIZE = 224

CLASS_NAMES_EN = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
CLASS_NAMES_RU = {
    "daisy": "Ромашка",
    "dandelion": "Одуванчик",
    "rose": "Роза",
    "sunflower": "Подсолнух",
    "tulip": "Тюльпан",
}


# =========================
# CUSTOM LAYER (нужен для load_model)
# =========================
@tf.keras.utils.register_keras_serializable(package="Custom")
class RandomCutout(layers.Layer):
    def __init__(self, prob=0.5, ratio=(0.2, 0.4), **kwargs):
        super().__init__(**kwargs)
        self.prob = float(prob)
        self.ratio = tuple(ratio)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"prob": self.prob, "ratio": list(self.ratio)})
        return cfg

    def call(self, images, training=None):
        # В инференсе cutout не нужен
        return images


# =========================
# LOAD MODEL PARTS
# =========================
@st.cache_resource
def load_parts():
    m = keras.models.load_model(MODEL_PATH, compile=False)
    _ = m(tf.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32), training=False)

    # IMPORTANT: для инференса/Grad-CAM не используем data_augmentation и random_cutout
    backbone = m.get_layer("efficientnetb0")
    gap = m.get_layer("global_average_pooling2d_3")
    drop = m.get_layer("dropout_3")
    head = m.get_layer("dense_3")
    return m, backbone, gap, drop, head


model, backbone, gap_layer, drop_layer, head_layer = load_parts()


# =========================
# PREPROCESS (правильно для EfficientNet)
# =========================
def preprocess_pil(pil_img: Image.Image) -> tf.Tensor:
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32")  # 0..255
    arr = np.expand_dims(arr, axis=0)
    x = tf.convert_to_tensor(arr, dtype=tf.float32)
    x = keras.applications.efficientnet.preprocess_input(x)
    return x


# =========================
# HEATMAP COLORS (без matplotlib/cv2)
# =========================
def jet_like_colormap(gray_u8: np.ndarray) -> np.ndarray:
    """
    gray_u8: (H,W) uint8 0..255
    return: (H,W,3) uint8 (jet-like)
    """
    x = gray_u8.astype(np.float32)
    r = np.clip(2 * x, 0, 255)
    g = np.clip(255 - np.abs(2 * x - 255), 0, 255)
    b = np.clip(255 - 2 * x, 0, 255)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def overlay_heatmap(pil_img: Image.Image, heatmap01: np.ndarray, alpha: float) -> np.ndarray:
    w, h = pil_img.size
    hm = Image.fromarray(np.uint8(heatmap01 * 255)).resize((w, h), resample=Image.BILINEAR)
    hm_u8 = np.array(hm).astype(np.uint8)

    colored = jet_like_colormap(hm_u8).astype(np.float32) / 255.0
    img = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0

    out = (1 - alpha) * img + alpha * colored
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


# =========================
# FORWARD + GRAD-CAM (по backbone output)
# =========================
def forward_probs(x: tf.Tensor) -> np.ndarray:
    bb_out = backbone(x, training=False)
    feat = gap_layer(bb_out)
    feat = drop_layer(feat, training=False)
    probs = head_layer(feat)[0].numpy()
    return probs


def gradcam_heatmap_backbone(x: tf.Tensor, class_index: int):
    """
    Надёжный Grad-CAM: по выходу backbone (последняя feature map перед GAP)
    """
    with tf.GradientTape() as tape:
        bb_out = backbone(x, training=False)  # (1,H,W,C)
        tape.watch(bb_out)

        feat = gap_layer(bb_out)
        feat = drop_layer(feat, training=False)
        probs = head_layer(feat)[0]  # softmax

        # log(prob) даёт более живые градиенты
        score = tf.math.log(probs[class_index] + 1e-8)

    grads = tape.gradient(score, bb_out)
    if grads is None:
        return None

    bb_out = bb_out[0]   # (H,W,C)
    grads = grads[0]     # (H,W,C)

    weights = tf.reduce_mean(grads, axis=(0, 1))          # (C,)
    cam = tf.reduce_sum(bb_out * weights, axis=-1)        # (H,W)

    cam = tf.abs(cam)                                    # важный фикс против "всё синее"
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy()


# =========================
# UI
# =========================
st.title("Распознавание цветов (CNN) + Grad-CAM")
st.write("Загрузите изображение или сделайте снимок. Grad-CAM показывает зоны, влияющие на решение модели.")

with st.sidebar:
    st.header("Настройки")
    alpha = st.slider("Наложение heatmap (alpha)", 0.05, 0.90, 0.40, 0.05)
    img_width = st.slider("Ширина изображения", 300, 900, 520, 10)

col_upload, col_cam = st.columns(2)
with col_upload:
    uploaded = st.file_uploader("Загрузить изображение (JPG/PNG)", type=["jpg", "jpeg", "png"])
with col_cam:
    camera = st.camera_input("Или сделать фото")

pil_image = None
img_bytes = None

if camera is not None:
    img_bytes = camera.getvalue()
    pil_image = Image.open(camera).convert("RGB")
elif uploaded is not None:
    img_bytes = uploaded.getvalue()
    pil_image = Image.open(uploaded).convert("RGB")

if pil_image is None:
    st.info("Загрузи изображение или сделай фото.")
    st.stop()

# Cache по картинке: чтобы при любом клике не пересчитывать probs заново
if "last_hash" not in st.session_state:
    st.session_state.last_hash = None
if "x" not in st.session_state:
    st.session_state.x = None
if "probs" not in st.session_state:
    st.session_state.probs = None

h = hash(img_bytes) if img_bytes is not None else None

if h != st.session_state.last_hash:
    st.session_state.last_hash = h
    st.session_state.x = preprocess_pil(pil_image)
    with st.spinner("Считаю предсказание..."):
        st.session_state.probs = forward_probs(st.session_state.x)

x = st.session_state.x
probs = st.session_state.probs

idx = int(np.argmax(probs))
class_en = CLASS_NAMES_EN[idx]
class_ru = CLASS_NAMES_RU[class_en]

with st.spinner("Строю Grad-CAM..."):
    hm = gradcam_heatmap_backbone(x, idx)

overlay = None
if hm is not None:
    overlay = overlay_heatmap(pil_image, hm, alpha=alpha)

c1, c2, c3 = st.columns([1.2, 1.2, 1.0])

with c1:
    st.subheader("Оригинал")
    st.image(pil_image, width=img_width)

with c2:
    st.subheader("Grad-CAM")
    if overlay is None:
        st.warning("Grad-CAM не удалось построить.")
    else:
        st.image(overlay, width=img_width)
        st.caption("Grad-CAM построен по выходу backbone (последняя feature map перед GAP).")

with c3:
    st.subheader("Результат")
    st.markdown(f"### {class_ru}")
    st.caption(f"({class_en})")

    df = pd.DataFrame(
        {"Класс": [CLASS_NAMES_RU[c] for c in CLASS_NAMES_EN], "Вероятность": probs}
    ).set_index("Класс")

    st.bar_chart(df)
    st.caption("Тёплые зоны = модель сильнее опиралась на эти части изображения.")
