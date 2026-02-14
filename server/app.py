"""
DiffusionGuard + Fawkes API Server
====================================
Exposes HTTP endpoints to protect images against diffusion-based inpainting
and facial recognition.

Endpoints:
    GET  /health          - Check server status + GPU info
    POST /protect         - Protect an image with DiffusionGuard (send image + mask)
    POST /fawkes          - Cloak an image with Fawkes (send image, no mask needed)
    POST /test-inpaint    - Run inpainting on an image to demonstrate protection effectiveness

Usage:
    python app.py
"""

import io
import os
import sys
import time
import logging

import torch
import numpy as np
from PIL import Image
from flask import Flask, request, send_file, jsonify
from omegaconf import OmegaConf
from diffusers import StableDiffusionInpaintPipeline

# ---------------------------------------------------------------------------
# Ensure DiffusionGuard repo is on the path
# ---------------------------------------------------------------------------
DIFFGUARD_REPO = os.environ.get(
    "DIFFGUARD_REPO",
    os.path.join(os.path.dirname(__file__), "..", "DiffusionGuard"),
)
if os.path.isdir(DIFFGUARD_REPO):
    sys.path.insert(0, DIFFGUARD_REPO)
else:
    print(f"WARNING: DiffusionGuard repo not found at {DIFFGUARD_REPO}")
    print("Clone it:  git clone https://github.com/choi403/DiffusionGuard.git")
    sys.exit(1)

from attacks import protect_image  # noqa: E402
from utils import overlay_images, get_mask_radius_list, tensor_to_pil_image  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global model state (loaded once at startup)
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MODEL_ID = "runwayml/stable-diffusion-inpainting"
IMG_SIZE = 512

pipe = None  # loaded in load_model()

# Default DiffusionGuard config
DEFAULT_CONFIG = {
    "exp_name": "api",
    "method": "diffusionguard",
    "orig_image_name": "input.png",
    "mask_image_names": ["mask.png"],
    "model": {"inpainting": MODEL_ID},
    "training": {
        "size": IMG_SIZE,
        "iters": 500,        # Cranked up from 200 for stronger protection (~90s)
        "grad_reps": 1,
        "batch_size": 1,
        "eps": 0.12549019607843137,       # 32/255 — doubled perturbation budget
        "step_size": 0.00784313725490196,  # 2/255 — doubled step size
        "num_inference_steps": 4,
        "mask": {
            "generation_method": "contour_shrink",
            "contour_strength": 1.1,
            "contour_iters": 15,
            "contour_smoothness": 0.1,
        },
    },
}


def load_model():
    """Load the Stable Diffusion Inpainting pipeline once."""
    global pipe
    logger.info("Loading %s on %s (%s)...", MODEL_ID, DEVICE, DTYPE)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL_ID,
        variant="fp16" if DTYPE == torch.float16 else None,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    logger.info("Model loaded successfully.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def center_crop_square(img: Image.Image) -> Image.Image:
    """Center-crop an image to a square (no stretching)."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def read_image_from_request(field: str) -> Image.Image:
    """Read an image from a multipart form field (center-crop, no stretch)."""
    if field not in request.files:
        raise ValueError(f"Missing required file field: '{field}'")
    f = request.files[field]
    img = Image.open(f.stream).convert("RGB")
    img = center_crop_square(img)
    return img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> io.BytesIO:
    """Convert a PIL image to an in-memory bytes buffer."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Health check — returns GPU info."""
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    gpu_mem = ""
    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(0).total_memory
        gpu_mem = f"{mem / 1e9:.1f} GB"
    return jsonify({
        "status": "ok",
        "gpu": gpu_name,
        "gpu_memory": gpu_mem,
        "model_loaded": pipe is not None,
        "device": DEVICE,
    })


@app.route("/protect", methods=["POST"])
def protect():
    """
    Protect an image using DiffusionGuard.

    Expects multipart form data:
        image  - The source image to protect (PNG/JPEG)
        mask   - Binary mask indicating the sensitive region (white = keep, black = edit)

    Optional query params:
        iters  - Number of PGD iterations (default: 200, max: 1000)

    Returns:
        The protected (glazed) image as PNG.
    """
    try:
        src_image = read_image_from_request("image")
        mask_image = read_image_from_request("mask")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Allow callers to override iteration count
    iters = request.args.get("iters", DEFAULT_CONFIG["training"]["iters"], type=int)
    iters = min(max(iters, 10), 1000)

    logger.info("Protecting image (%d PGD iterations)...", iters)
    t0 = time.time()

    # Build config
    config = OmegaConf.create(DEFAULT_CONFIG)
    config.training.iters = iters

    mask_image_list = [mask_image]
    mask_image_combined = overlay_images(mask_image_list)
    mask_radius_list = get_mask_radius_list(mask_image_list)

    # Run DiffusionGuard
    adv_tensor = protect_image(
        config.method,
        pipe,
        src_image,
        mask_image_list,
        mask_image_combined,
        mask_radius_list,
        config,
    )

    adv_image = tensor_to_pil_image(adv_tensor.detach().cpu())
    elapsed = time.time() - t0
    logger.info("Protection complete in %.1fs", elapsed)

    return send_file(
        pil_to_bytes(adv_image),
        mimetype="image/png",
        download_name="protected.png",
    )


@app.route("/test-inpaint", methods=["POST"])
def test_inpaint():
    """
    Run Stable Diffusion Inpainting on an image to test whether protection worked.

    Expects multipart form data:
        image   - The image to inpaint (protected or unprotected)
        mask    - Binary mask (white = region to KEEP, black = region to regenerate)

    Optional query params:
        prompt  - Text prompt for inpainting (default: "a person in a hospital")
        steps   - Number of diffusion steps (default: 50)
        seed    - Random seed (default: 42)

    Returns:
        The inpainted result as PNG.
    """
    try:
        src_image = read_image_from_request("image")
        mask_image = read_image_from_request("mask")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    prompt = request.args.get("prompt", "a person in a hospital")
    steps = request.args.get("steps", 50, type=int)
    seed = request.args.get("seed", 42, type=int)

    logger.info("Inpainting with prompt='%s', steps=%d, seed=%d", prompt, steps, seed)
    t0 = time.time()

    # SD Inpainting expects: white = region to inpaint, black = region to keep
    # Our masks use: white = sensitive region to keep
    # So we invert the mask for the inpainting pipeline
    mask_array = np.array(mask_image.convert("L"))
    inverted_mask = Image.fromarray(255 - mask_array)

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        image=src_image,
        mask_image=inverted_mask,
        num_inference_steps=steps,
        generator=generator,
    ).images[0]

    elapsed = time.time() - t0
    logger.info("Inpainting complete in %.1fs", elapsed)

    return send_file(
        pil_to_bytes(result),
        mimetype="image/png",
        download_name="inpainted.png",
    )


# ---------------------------------------------------------------------------
# Fawkes cloaking endpoint
# ---------------------------------------------------------------------------

@app.route("/fawkes", methods=["POST"])
def fawkes_cloak():
    """
    Cloak an image using Fawkes (facial recognition protection).

    Expects multipart form data:
        image  - The source image to cloak (PNG/JPEG)

    Optional query params:
        mode   - Cloaking strength: 'low', 'mid', 'high' (default: 'mid')

    Returns:
        The cloaked image as PNG.
    """
    from fawkes_modern import cloak_image

    if "image" not in request.files:
        return jsonify({"error": "Missing required file field: 'image'"}), 400

    f = request.files["image"]
    img = Image.open(f.stream).convert("RGB")

    mode = request.args.get("mode", "mid")
    if mode not in ("low", "mid", "high"):
        return jsonify({"error": "mode must be 'low', 'mid', or 'high'"}), 400

    logger.info("Fawkes cloaking (mode=%s)...", mode)
    t0 = time.time()

    cloaked = cloak_image(img, mode=mode)

    elapsed = time.time() - t0
    logger.info("Fawkes cloaking complete in %.1fs", elapsed)

    return send_file(
        pil_to_bytes(cloaked),
        mimetype="image/png",
        download_name="fawkes_cloaked.png",
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    load_model()

    # Pre-load Fawkes models (lazy — loads on first request if skipped here)
    try:
        from fawkes_modern import init_fawkes
        init_fawkes("mid")
    except Exception as e:
        logger.warning("Fawkes init failed (will retry on first request): %s", e)

    logger.info("Starting DiffusionGuard + Fawkes API server on :8888")
    app.run(host="0.0.0.0", port=8888, debug=False)
