"""
Fawkes-style image cloaking adapted for modern TensorFlow (2.15+) / Keras 3.
============================================================================
Based on: Shan et al., "Fawkes: Protecting Privacy against Unauthorized
Deep Learning Models", USENIX Security 2020.
GitHub: https://github.com/Shawn-Shan/fawkes

This is a self-contained re-implementation that:
  - Works with TF 2.20 + Keras 3.x
  - Uses MTCNN for face detection (same as original)
  - Downloads the original Fawkes extractor models
  - Exposes a simple `cloak_image(pil_image, mode)` API for the Flask server
"""

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import hashlib
import logging
import time
from io import BytesIO

import numpy as np
import tensorflow as tf

# Limit TF GPU memory to 4GB so PyTorch SD model can coexist
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
        )
    except RuntimeError:
        pass  # Already set

import tf_keras as keras  # Legacy Keras 2 for loading old .h5 models
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_SIZE = 112
TANH_CONSTANT = 2 - 1e-6

MODE_PARAMS = {
    "low":  {"th": 0.004, "max_step": 40,  "lr": 25, "extractors": ["extractor_2"]},
    "mid":  {"th": 0.012, "max_step": 75,  "lr": 20, "extractors": ["extractor_0", "extractor_2"]},
    "high": {"th": 0.017, "max_step": 150, "lr": 15, "extractors": ["extractor_0", "extractor_2"]},
}

EXTRACTOR_HASHES = {
    "extractor_2": "ce703d481db2b83513bbdafa27434703",
    "extractor_0": "94854151fd9077997d69ceda107f9c6b",
}

MODEL_URL = "http://mirror.cs.uchicago.edu/fawkes/files/{}.h5"
MODEL_DIR = os.path.join(os.path.expanduser("~"), ".fawkes", "model")

# ---------------------------------------------------------------------------
# Face detection via MTCNN
# ---------------------------------------------------------------------------
_mtcnn_detector = None


def get_mtcnn():
    global _mtcnn_detector
    if _mtcnn_detector is None:
        from mtcnn import MTCNN
        _mtcnn_detector = MTCNN()
    return _mtcnn_detector


def detect_and_crop_face(img_array):
    """Detect the largest face, crop it, and return (cropped_face_112, bbox).
    img_array: H x W x 3, uint8 numpy array.
    Returns (face_112x112x3 float32, (x,y,w,h)) or (None, None) if no face.
    """
    detector = get_mtcnn()
    results = detector.detect_faces(img_array)
    if not results:
        return None, None
    # pick largest face by area
    best = max(results, key=lambda r: r["box"][2] * r["box"][3])
    x, y, w, h = best["box"]
    # add margin
    margin = int(max(w, h) * 0.15)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img_array.shape[1], x + w + margin)
    y2 = min(img_array.shape[0], y + h + margin)
    face_crop = img_array[y1:y2, x1:x2]
    # make it square
    side = max(face_crop.shape[0], face_crop.shape[1])
    square = np.full((side, side, 3), np.mean(face_crop, axis=(0, 1)), dtype=np.float32)
    sy = (side - face_crop.shape[0]) // 2
    sx = (side - face_crop.shape[1]) // 2
    square[sy:sy + face_crop.shape[0], sx:sx + face_crop.shape[1]] = face_crop
    # resize to 112x112
    face_pil = Image.fromarray(np.clip(square, 0, 255).astype(np.uint8))
    face_112 = np.array(face_pil.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS), dtype=np.float32)
    return face_112, (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Model loading (downloads original Fawkes .h5 weights)
# ---------------------------------------------------------------------------
_loaded_extractors = {}


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_extractor(name):
    """Load a Fawkes feature extractor .h5 model using legacy tf_keras."""
    if name in _loaded_extractors:
        return _loaded_extractors[name]

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{name}.h5")
    expected_hash = EXTRACTOR_HASHES[name]

    # Download if needed
    if not os.path.exists(model_path) or _md5(model_path) != expected_hash:
        url = MODEL_URL.format(name)
        logger.info("Downloading %s from %s ...", name, url)
        import urllib.request
        urllib.request.urlretrieve(url, model_path)
        assert _md5(model_path) == expected_hash, f"Hash mismatch for {name}"

    # Load with tf_keras (legacy Keras 2 — compatible with old .h5 format)
    model = keras.models.load_model(model_path, compile=False)
    _loaded_extractors[name] = model
    logger.info("Loaded extractor: %s (output_shape=%s)", name, model.output_shape)
    return model


# ---------------------------------------------------------------------------
# Feature extraction wrapper
# ---------------------------------------------------------------------------

def extract_features(model, imgs):
    """Extract L2-normalized feature embeddings from images (0-255 range)."""
    x = imgs / 255.0
    embeds = model(x, training=False)
    norm = tf.norm(embeds, axis=1, keepdims=True)
    return embeds / (norm + 1e-10)


# ---------------------------------------------------------------------------
# Core cloaking algorithm (adapted from FawkesMaskGeneration)
# ---------------------------------------------------------------------------

def _preprocess_arctanh(imgs):
    """Convert pixel-space images to tanh space."""
    x = imgs / 255.0
    x = x - 0.5
    x = x * TANH_CONSTANT
    return np.arctanh(x).astype(np.float32)


def _reverse_arctanh(tanh_imgs):
    """Convert tanh-space back to pixel space."""
    return (tf.tanh(tanh_imgs) / TANH_CONSTANT + 0.5) * 255.0


def generate_cloak(source_img, extractors, th=0.004, max_step=40, lr=25.0,
                   initial_const=1e7, batch_size=1, verbose=False):
    """
    Generate adversarial cloak for a single face image.

    Args:
        source_img: np.array of shape (112, 112, 3), float32, range 0-255
        extractors: list of loaded Keras models
        th: DSSIM threshold
        max_step: number of optimization iterations
        lr: learning rate for Adadelta
        initial_const: initial penalty constant
        verbose: whether to print iteration info

    Returns:
        cloaked_img: np.array same shape, cloaked version, range 0-255
    """
    source_batch = np.expand_dims(source_img, 0).astype(np.float32)
    simg_tanh = tf.Variable(_preprocess_arctanh(source_batch), dtype=tf.float32)
    simg_raw = tf.Variable(source_batch, dtype=tf.float32)

    # Initialize modifier
    modifier = tf.Variable(
        np.random.uniform(-1, 1, (1, IMG_SIZE, IMG_SIZE, 3)).astype(np.float32) * 1e-4,
        dtype=tf.float32,
    )

    optimizer = keras.optimizers.Adadelta(learning_rate=float(lr))
    const = tf.Variable(np.array([initial_const], dtype=np.float32))
    const_diff = tf.Variable(np.array([1.0], dtype=np.float32))

    best_adv = np.zeros_like(source_batch)
    best_bottlesim = np.inf
    outside = 0

    for it in range(1, max_step + 1):
        with tf.GradientTape() as tape:
            tape.watch(modifier)

            # Forward through tanh
            aimg_raw = _reverse_arctanh(simg_tanh + modifier)
            # Clip perturbation
            actual_mod = tf.clip_by_value(aimg_raw - simg_raw, -15.0, 15.0)
            aimg_raw = simg_raw + actual_mod

            current_simg_raw = _reverse_arctanh(simg_tanh)

            # DSSIM loss
            ssim_val = tf.image.ssim(aimg_raw, current_simg_raw, max_val=255.0)
            dist_raw = (1.0 - ssim_val) / 2.0
            dist = tf.maximum(dist_raw - th, 0.0)

            # Feature space loss (maximize distance from original)
            feature_loss = 0.0
            for ext in extractors:
                feat_adv = extract_features(ext, aimg_raw)
                feat_orig = extract_features(ext, current_simg_raw)
                diff = feat_adv - feat_orig
                scale = tf.sqrt(tf.reduce_sum(tf.square(feat_orig), axis=1))
                cur_loss = tf.reduce_sum(tf.square(diff), axis=1) / (scale + 1e-10)
                feature_loss += cur_loss

            # Total loss: minimize DSSIM penalty, maximize feature distance
            loss = const * tf.square(dist) - feature_loss * const_diff
            loss_sum = tf.reduce_sum(loss)

        grads = tape.gradient(loss_sum, [modifier])
        optimizer.apply_gradients(zip(grads, [modifier]))

        # On first step, add a small perturbation
        if it == 1:
            modifier.assign(modifier - tf.sign(grads[0]) * 0.01)

        # Adaptive const_diff
        d = float(dist_raw.numpy()[0])
        fd = float(feature_loss.numpy()[0])

        new_cd = float(const_diff.numpy()[0])
        if d <= th * 0.9 and new_cd <= 129:
            new_cd *= 2
            if outside == -1:
                new_cd = 1
            outside = 1
        elif d >= th * 1.1 and new_cd >= 1 / 129:
            new_cd /= 2
            if outside == 1:
                new_cd = 1
            outside = -1
        else:
            new_cd = 1.0
            outside = 0
        const_diff.assign([new_cd])

        if d <= th * 1.1 and fd < best_bottlesim:
            best_bottlesim = fd
            best_adv = aimg_raw.numpy()

        if verbose and it % 10 == 0:
            logger.info(
                "  Fawkes iter %d/%d  loss=%.2f  dssim=%.4f  feat_dist=%.4f",
                it, max_step, float(loss_sum), d, fd,
            )

    # Clamp to valid pixel range
    result = np.clip(best_adv[0], 0, 255).astype(np.float32)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_fawkes_extractors = None


def init_fawkes(mode="mid"):
    """Pre-load the Fawkes extractor models."""
    global _fawkes_extractors
    params = MODE_PARAMS.get(mode, MODE_PARAMS["mid"])
    logger.info("Loading Fawkes extractors for mode=%s ...", mode)
    _fawkes_extractors = [load_extractor(name) for name in params["extractors"]]
    logger.info("Fawkes extractors loaded.")


def cloak_image(pil_image, mode="mid"):
    """
    Cloak a PIL image using Fawkes.

    Args:
        pil_image: PIL.Image (any size, RGB)
        mode: 'low', 'mid', or 'high'

    Returns:
        cloaked PIL.Image (same size as input)
    """
    global _fawkes_extractors
    params = MODE_PARAMS[mode]

    if _fawkes_extractors is None:
        init_fawkes(mode)

    img_array = np.array(pil_image.convert("RGB")).astype(np.float32)

    # Detect face
    face_112, bbox = detect_and_crop_face(img_array.astype(np.uint8))
    if face_112 is None:
        logger.warning("No face detected — applying cloak to center crop instead")
        # Fall back: center-crop and resize
        h, w = img_array.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        crop = img_array[y0:y0 + side, x0:x0 + side]
        face_pil = Image.fromarray(crop.astype(np.uint8))
        face_112 = np.array(face_pil.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS), dtype=np.float32)
        bbox = (x0, y0, x0 + side, y0 + side)

    t0 = time.time()
    cloaked_face = generate_cloak(
        face_112,
        _fawkes_extractors,
        th=params["th"],
        max_step=params["max_step"],
        lr=params["lr"],
        verbose=True,
    )
    elapsed = time.time() - t0
    logger.info("Fawkes cloak generated in %.1fs", elapsed)

    # Merge cloak back into original image
    x1, y1, x2, y2 = bbox
    face_region = img_array[y1:y2, x1:x2]
    face_h, face_w = face_region.shape[:2]

    # Compute perturbation at 112x112 and upsample to face region size
    original_face_112 = np.array(
        Image.fromarray(face_region.astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS),
        dtype=np.float32,
    )
    perturbation_112 = cloaked_face - original_face_112

    # Upsample perturbation to original face region size
    pert_pil = Image.fromarray(
        np.clip(perturbation_112 + 128, 0, 255).astype(np.uint8)
    ).resize((face_w, face_h), Image.LANCZOS)
    perturbation_full = np.array(pert_pil, dtype=np.float32) - 128.0

    # Apply perturbation
    result = img_array.copy()
    result[y1:y2, x1:x2] = np.clip(face_region + perturbation_full, 0, 255)

    return Image.fromarray(result.astype(np.uint8))
