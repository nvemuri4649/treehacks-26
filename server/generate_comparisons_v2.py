"""
Generate comparison image sets — PAPER-FAITHFUL version.
=========================================================
Matches the original DiffusionGuard paper setup:
  - MTCNN face detection → tight oval face mask
  - 800 PGD iterations, eps=16/255, step_size=1/255
  - contour_shrink mask augmentation
  - Inpainting attacker uses the same face mask region

For each test, produces a 4-image set:
  1. Original image (center-cropped 512×512)
  2. Protected image (DiffusionGuard with face mask)
  3. Inpaint on UNPROTECTED original (attacker edits the face)
  4. Inpaint on PROTECTED image (attacker tries to edit face — should fail)

Also generates:
  - The face mask itself
  - Fawkes comparison
  - Side-by-side grid
"""

import os
import sys
import time
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Constrain TensorFlow GPU memory BEFORE any TF imports (MTCNN uses TF)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=2048)]
    )

# Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

os.environ['DIFFGUARD_REPO'] = '/workspace/DiffusionGuard'
sys.path.insert(0, '/workspace/DiffusionGuard')

import torch
from omegaconf import OmegaConf
from diffusers import StableDiffusionInpaintPipeline
from attacks import protect_image
from utils import overlay_images, get_mask_radius_list, tensor_to_pil_image

OUTPUT_DIR = "/workspace/comparison_outputs_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
DEVICE = "cuda"
DTYPE = torch.float16
MODEL_ID = "runwayml/stable-diffusion-inpainting"
IMG_SIZE = 512

logger.info("Loading SD inpainting pipeline...")
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    MODEL_ID, variant="fp16", torch_dtype=DTYPE
).to(DEVICE)
logger.info("SD pipeline loaded.")

# ---------------------------------------------------------------------------
# Face detection → mask generation
# ---------------------------------------------------------------------------
from mtcnn import MTCNN
_detector = MTCNN()


def detect_face_mask(img_512: Image.Image, margin_factor=0.35) -> Image.Image:
    """
    Detect the face in a 512×512 image and return an oval mask (white=face).
    Similar to the keanu_mask.png in the original paper.
    """
    img_array = np.array(img_512)
    results = _detector.detect_faces(img_array)

    if not results:
        logger.warning("No face detected! Using center oval fallback.")
        mask = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse([128, 64, 384, 384], fill=255)
        return mask

    # Pick largest face
    best = max(results, key=lambda r: r["box"][2] * r["box"][3])
    x, y, w, h = best["box"]

    # Add margin (the paper's masks have generous margins around the face)
    mx = int(w * margin_factor)
    my = int(h * margin_factor)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(IMG_SIZE, x + w + mx)
    y2 = min(IMG_SIZE, y + h + my)

    logger.info("Face detected: bbox=(%d,%d,%d,%d) with margin → (%d,%d,%d,%d)",
                x, y, x + w, y + h, x1, y1, x2, y2)

    # Draw oval mask
    mask = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([x1, y1, x2, y2], fill=255)

    # Slight Gaussian blur for soft edges (like a hand-drawn mask)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
    # Re-threshold to binary
    mask_arr = np.array(mask)
    mask_arr = (mask_arr > 128).astype(np.uint8) * 255
    mask = Image.fromarray(mask_arr)

    white_pct = (mask_arr > 128).sum() / mask_arr.size * 100
    logger.info("Face mask: %.1f%% white (paper's keanu mask was ~22%%)", white_pct)

    return mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def center_crop_square(img):
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def run_inpaint(image, mask_L, prompt, steps=50, seed=42):
    """
    Run SD inpainting.
    mask_L: PIL Image mode "L", white = face (region to KEEP in our convention).
    For SD inpainting: white = region to INPAINT.
    So we invert: white face → black (keep face), rest → white (inpaint around).

    BUT for the attack scenario: the ATTACKER wants to edit the FACE.
    So the attacker's mask should have white = face = region to inpaint.
    We pass the face mask directly as the SD inpaint mask (no inversion).
    """
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_L,  # white = region to inpaint (the face)
        num_inference_steps=steps,
        generator=generator,
    ).images[0]
    return result


def run_diffusionguard(src_image, mask_image_rgb, iters=800, eps=16/255, step_size=1/255):
    """
    Run DiffusionGuard with paper-faithful parameters.
    mask_image_rgb: PIL RGB, white = region to protect (the face).
    """
    config = OmegaConf.create({
        "exp_name": "paper_faithful",
        "method": "diffusionguard",
        "orig_image_name": "input.png",
        "mask_image_names": ["mask.png"],
        "model": {"inpainting": MODEL_ID},
        "training": {
            "size": IMG_SIZE,
            "iters": iters,
            "grad_reps": 1,
            "batch_size": 1,
            "eps": eps,
            "step_size": step_size,
            "num_inference_steps": 4,
            "mask": {
                "generation_method": "contour_shrink",
                "contour_strength": 1.1,
                "contour_iters": 15,
                "contour_smoothness": 0.1,
            },
        },
    })
    mask_list = [mask_image_rgb]
    mask_combined = overlay_images(mask_list)
    mask_radius_list = get_mask_radius_list(mask_list)

    adv_tensor = protect_image(
        config.method, pipe, src_image, mask_list, mask_combined, mask_radius_list, config
    )
    return tensor_to_pil_image(adv_tensor.detach().cpu())


def make_grid(images, labels, cols=4):
    """Create a labeled grid image."""
    from PIL import ImageFont
    w, h = images[0].size
    rows_needed = (len(images) + cols - 1) // cols
    label_h = 30
    grid = Image.new("RGB", (w * cols, (h + label_h) * rows_needed), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    for i, (img, label) in enumerate(zip(images, labels)):
        col = i % cols
        row = i // cols
        x = col * w
        y = row * (h + label_h)
        grid.paste(img, (x, y + label_h))
        draw.text((x + 5, y + 5), label, fill=(0, 0, 0))

    return grid


# ---------------------------------------------------------------------------
# Load + prepare test image
# ---------------------------------------------------------------------------
TEST_IMG_PATH = "/workspace/test_photo.jpg"
original_full = Image.open(TEST_IMG_PATH).convert("RGB")
original = center_crop_square(original_full).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
original.save(os.path.join(OUTPUT_DIR, "original_512.png"))
logger.info("Original image loaded and cropped to %s", original.size)

# ---------------------------------------------------------------------------
# Detect face → create mask
# ---------------------------------------------------------------------------
logger.info("Detecting face...")
face_mask_L = detect_face_mask(original)
face_mask_L.save(os.path.join(OUTPUT_DIR, "face_mask.png"))

# Convert mask to RGB for DiffusionGuard (it expects RGB mask images)
face_mask_rgb = Image.merge("RGB", [face_mask_L, face_mask_L, face_mask_L])

# ---------------------------------------------------------------------------
# DEMO 1: Paper-faithful DiffusionGuard (800 iters, eps=16/255)
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("DEMO 1: Paper-faithful DiffusionGuard (800 iters, eps=16/255)")
logger.info("=" * 60)

t0 = time.time()
protected = run_diffusionguard(original, face_mask_rgb, iters=800, eps=16/255, step_size=1/255)
dg_time = time.time() - t0
logger.info("DiffusionGuard took %.1fs", dg_time)

# Save protected image
demo1_dir = os.path.join(OUTPUT_DIR, "demo1_paper_faithful")
os.makedirs(demo1_dir, exist_ok=True)
original.save(os.path.join(demo1_dir, "1_original.png"))
face_mask_L.save(os.path.join(demo1_dir, "1b_face_mask.png"))
protected.save(os.path.join(demo1_dir, "2_protected.png"))

# Attacker inpaints FACE region on original (should succeed — clean face swap)
prompt_swap = "a different person's face, photorealistic"
logger.info("Inpainting original (face swap attack)...")
inpaint_orig = run_inpaint(original, face_mask_L, prompt_swap, seed=42)
inpaint_orig.save(os.path.join(demo1_dir, "3_inpaint_original.png"))

# Attacker inpaints FACE region on protected (should FAIL — garbled)
logger.info("Inpainting protected (face swap attack)...")
inpaint_prot = run_inpaint(protected, face_mask_L, prompt_swap, seed=42)
inpaint_prot.save(os.path.join(demo1_dir, "4_inpaint_protected.png"))

# Make grid
grid1 = make_grid(
    [original, protected, inpaint_orig, inpaint_prot],
    ["Original", "DiffusionGuard Protected", "Attack on Original", "Attack on Protected"],
)
grid1.save(os.path.join(demo1_dir, "grid.png"))
logger.info("DEMO 1 complete.")

# ---------------------------------------------------------------------------
# DEMO 2: Different attack prompts on same protected image
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("DEMO 2: Multiple attack prompts")
logger.info("=" * 60)

demo2_dir = os.path.join(OUTPUT_DIR, "demo2_multi_prompt")
os.makedirs(demo2_dir, exist_ok=True)

attack_prompts = [
    ("face_swap", "a completely different person's face, photorealistic", 42),
    ("aged", "an elderly person's face, wrinkled, photorealistic", 99),
    ("cartoon", "a cartoon character face, pixar style", 77),
    ("celebrity", "a famous movie star face, photorealistic portrait", 123),
]

for name, prompt, seed in attack_prompts:
    logger.info("Attack '%s': %s", name, prompt)
    inpaint_o = run_inpaint(original, face_mask_L, prompt, seed=seed)
    inpaint_p = run_inpaint(protected, face_mask_L, prompt, seed=seed)
    inpaint_o.save(os.path.join(demo2_dir, f"orig_{name}.png"))
    inpaint_p.save(os.path.join(demo2_dir, f"prot_{name}.png"))

logger.info("DEMO 2 complete.")

# ---------------------------------------------------------------------------
# DEMO 3: Fawkes + same inpainting attack
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("DEMO 3: Fawkes cloaking + inpainting attack")
logger.info("=" * 60)

demo3_dir = os.path.join(OUTPUT_DIR, "demo3_fawkes")
os.makedirs(demo3_dir, exist_ok=True)

try:
    from fawkes_modern import init_fawkes, cloak_image
    init_fawkes("low")

    t0 = time.time()
    fawkes_cloaked = cloak_image(original, mode="low")
    fawkes_time = time.time() - t0
    logger.info("Fawkes took %.1fs", fawkes_time)

    if fawkes_cloaked.size != (IMG_SIZE, IMG_SIZE):
        fawkes_cloaked = fawkes_cloaked.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    fawkes_cloaked.save(os.path.join(demo3_dir, "2_fawkes_cloaked.png"))

    inpaint_fawkes = run_inpaint(fawkes_cloaked, face_mask_L, prompt_swap, seed=42)
    inpaint_fawkes.save(os.path.join(demo3_dir, "4_inpaint_fawkes.png"))

    # Grid: original, DG protected, Fawkes cloaked; then their inpaint results
    grid3 = make_grid(
        [original, protected, fawkes_cloaked,
         inpaint_orig, inpaint_prot, inpaint_fawkes],
        ["Original", "DiffusionGuard", "Fawkes",
         "Attack: Original", "Attack: DiffusionGuard", "Attack: Fawkes"],
        cols=3,
    )
    grid3.save(os.path.join(demo3_dir, "grid_comparison.png"))
    logger.info("DEMO 3 complete.")

except Exception as e:
    logger.error("Fawkes demo failed: %s", e)
    import traceback
    traceback.print_exc()

# ---------------------------------------------------------------------------
# DEMO 4: Side-by-side with the paper's keanu example
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("DEMO 4: Paper's Keanu Reeves example")
logger.info("=" * 60)

demo4_dir = os.path.join(OUTPUT_DIR, "demo4_keanu")
os.makedirs(demo4_dir, exist_ok=True)

keanu = Image.open("/workspace/DiffusionGuard/assets/keanu.png").convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
keanu_mask = Image.open("/workspace/DiffusionGuard/assets/keanu_mask.png").convert("L").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
keanu_mask_rgb = Image.merge("RGB", [keanu_mask, keanu_mask, keanu_mask])

keanu.save(os.path.join(demo4_dir, "1_keanu_original.png"))
keanu_mask.save(os.path.join(demo4_dir, "1b_keanu_mask.png"))

t0 = time.time()
keanu_protected = run_diffusionguard(keanu, keanu_mask_rgb, iters=800, eps=16/255, step_size=1/255)
logger.info("Keanu DiffusionGuard took %.1fs", time.time() - t0)
keanu_protected.save(os.path.join(demo4_dir, "2_keanu_protected.png"))

keanu_prompt = "a different person, young woman, photorealistic"
inpaint_keanu_orig = run_inpaint(keanu, keanu_mask, keanu_prompt, seed=42)
inpaint_keanu_prot = run_inpaint(keanu_protected, keanu_mask, keanu_prompt, seed=42)

inpaint_keanu_orig.save(os.path.join(demo4_dir, "3_inpaint_original.png"))
inpaint_keanu_prot.save(os.path.join(demo4_dir, "4_inpaint_protected.png"))

grid4 = make_grid(
    [keanu, keanu_protected, inpaint_keanu_orig, inpaint_keanu_prot],
    ["Keanu Original", "DiffusionGuard Protected", "Attack on Original", "Attack on Protected"],
)
grid4.save(os.path.join(demo4_dir, "grid.png"))
logger.info("DEMO 4 complete.")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("ALL DEMOS COMPLETE")
logger.info("Output: %s", OUTPUT_DIR)
for d in sorted(os.listdir(OUTPUT_DIR)):
    full = os.path.join(OUTPUT_DIR, d)
    if os.path.isdir(full):
        files = os.listdir(full)
        logger.info("  %s/ (%d files)", d, len(files))
logger.info("=" * 60)
