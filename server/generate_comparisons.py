"""
Generate comparison image sets for DiffusionGuard + Fawkes demo.
Runs on the RunPod server directly (no proxy timeouts).

For each test, produces a 4-image set:
  1. Original image
  2. Protected/cloaked image
  3. Inpainting result on original
  4. Inpainting result on protected

Creates sets for:
  - DiffusionGuard protection with various prompts
  - Fawkes cloaking with inpainting attack
"""

import os
import sys
import time
import json
import logging
import numpy as np
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Setup paths
# ---------------------------------------------------------------------------
os.environ['DIFFGUARD_REPO'] = '/workspace/DiffusionGuard'
sys.path.insert(0, '/workspace/DiffusionGuard')

import torch
from omegaconf import OmegaConf
from diffusers import StableDiffusionInpaintPipeline
from attacks import protect_image
from utils import overlay_images, get_mask_radius_list, tensor_to_pil_image

OUTPUT_DIR = "/workspace/comparison_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load models
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
# Helpers
# ---------------------------------------------------------------------------

def center_crop_square(img):
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def run_inpaint(image, mask, prompt, steps=50, seed=42):
    """Run SD inpainting. mask: white=keep, black=edit."""
    mask_array = np.array(mask.convert("L"))
    inverted_mask = Image.fromarray(255 - mask_array)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=inverted_mask,
        num_inference_steps=steps,
        generator=generator,
    ).images[0]
    return result


def run_diffusionguard(src_image, mask_image, iters=500, eps=0.12549019607843137, step_size=0.00784313725490196):
    """Run DiffusionGuard protection."""
    config = OmegaConf.create({
        "exp_name": "batch",
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
    mask_list = [mask_image]
    mask_combined = overlay_images(mask_list)
    mask_radius_list = get_mask_radius_list(mask_list)

    adv_tensor = protect_image(
        config.method, pipe, src_image, mask_list, mask_combined, mask_radius_list, config
    )
    return tensor_to_pil_image(adv_tensor.detach().cpu())


def save_comparison_set(set_name, original, protected, inpaint_orig, inpaint_protected, extra_images=None):
    """Save a 4-image comparison set."""
    set_dir = os.path.join(OUTPUT_DIR, set_name)
    os.makedirs(set_dir, exist_ok=True)
    original.save(os.path.join(set_dir, "1_original.png"))
    protected.save(os.path.join(set_dir, "2_protected.png"))
    inpaint_orig.save(os.path.join(set_dir, "3_inpaint_original.png"))
    inpaint_protected.save(os.path.join(set_dir, "4_inpaint_protected.png"))
    if extra_images:
        for name, img in extra_images.items():
            img.save(os.path.join(set_dir, name))
    logger.info("Saved comparison set: %s", set_name)


# ---------------------------------------------------------------------------
# Load test image
# ---------------------------------------------------------------------------
TEST_IMG_PATH = "/workspace/test_photo.jpg"
if not os.path.exists(TEST_IMG_PATH):
    logger.error("Test image not found at %s", TEST_IMG_PATH)
    sys.exit(1)

original = Image.open(TEST_IMG_PATH).convert("RGB")
original = center_crop_square(original).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
original.save(os.path.join(OUTPUT_DIR, "original_512.png"))
logger.info("Original image: %s", original.size)

# Create masks
# Full white mask (protect everything)
mask_full = Image.new("L", (IMG_SIZE, IMG_SIZE), 255)
# Top 30% mask (for cat ears / hat edits)
mask_top = Image.new("L", (IMG_SIZE, IMG_SIZE), 255)
from PIL import ImageDraw
draw = ImageDraw.Draw(mask_top)
draw.rectangle([0, 0, 512, 155], fill=0)  # top 30% black = region to edit

# Bottom half mask (for outfit changes)
mask_bottom = Image.new("L", (IMG_SIZE, IMG_SIZE), 255)
draw2 = ImageDraw.Draw(mask_bottom)
draw2.rectangle([0, 256, 512, 512], fill=0)  # bottom half black = region to edit

# ---------------------------------------------------------------------------
# TEST SET 1: DiffusionGuard — cat ears
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("SET 1: DiffusionGuard — cat ears edit")
logger.info("=" * 60)

t0 = time.time()
protected_dg = run_diffusionguard(original, Image.fromarray(np.array(mask_top).astype(np.uint8)).convert("RGB"))
logger.info("DiffusionGuard protection took %.1fs", time.time() - t0)

prompt1 = "a person with cute cat ears, anime style"
inpaint_orig_1 = run_inpaint(original, mask_top, prompt1)
inpaint_prot_1 = run_inpaint(protected_dg, mask_top, prompt1)

save_comparison_set("set1_diffguard_catears", original, protected_dg, inpaint_orig_1, inpaint_prot_1)

# ---------------------------------------------------------------------------
# TEST SET 2: DiffusionGuard — outfit change
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("SET 2: DiffusionGuard — outfit change")
logger.info("=" * 60)

t0 = time.time()
protected_dg2 = run_diffusionguard(original, Image.fromarray(np.array(mask_bottom).astype(np.uint8)).convert("RGB"))
logger.info("DiffusionGuard protection took %.1fs", time.time() - t0)

prompt2 = "a person wearing a red superhero costume"
inpaint_orig_2 = run_inpaint(original, mask_bottom, prompt2)
inpaint_prot_2 = run_inpaint(protected_dg2, mask_bottom, prompt2)

save_comparison_set("set2_diffguard_outfit", original, protected_dg2, inpaint_orig_2, inpaint_prot_2)

# ---------------------------------------------------------------------------
# TEST SET 3: DiffusionGuard — face swap / scene change
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("SET 3: DiffusionGuard — full image edit")
logger.info("=" * 60)

# Reuse the protected image from set 1 (top mask) for a different prompt
prompt3 = "a person at the beach with sunglasses"
inpaint_orig_3 = run_inpaint(original, mask_top, prompt3, seed=123)
inpaint_prot_3 = run_inpaint(protected_dg, mask_top, prompt3, seed=123)

save_comparison_set("set3_diffguard_beach", original, protected_dg, inpaint_orig_3, inpaint_prot_3)

# ---------------------------------------------------------------------------
# TEST SET 4: Fawkes cloaking + inpainting attack
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("SET 4: Fawkes cloaking + inpainting")
logger.info("=" * 60)

try:
    from fawkes_modern import init_fawkes, cloak_image
    init_fawkes("low")  # use low mode for speed

    t0 = time.time()
    fawkes_cloaked = cloak_image(original, mode="low")
    logger.info("Fawkes cloaking took %.1fs", time.time() - t0)

    # Resize to 512x512 if needed
    if fawkes_cloaked.size != (IMG_SIZE, IMG_SIZE):
        fawkes_cloaked = fawkes_cloaked.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    prompt4 = "a person with cute cat ears, anime style"
    inpaint_orig_4 = inpaint_orig_1  # reuse from set 1
    inpaint_fawkes_4 = run_inpaint(fawkes_cloaked, mask_top, prompt4)

    save_comparison_set("set4_fawkes_catears", original, fawkes_cloaked, inpaint_orig_4, inpaint_fawkes_4)

except Exception as e:
    logger.error("Fawkes set failed: %s", e)
    import traceback
    traceback.print_exc()

# ---------------------------------------------------------------------------
# TEST SET 5: Fawkes cloaking + outfit change
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("SET 5: Fawkes cloaking + outfit change")
logger.info("=" * 60)

try:
    prompt5 = "a person wearing a red superhero costume"
    inpaint_fawkes_5 = run_inpaint(fawkes_cloaked, mask_bottom, prompt5)

    save_comparison_set("set5_fawkes_outfit", original, fawkes_cloaked, inpaint_orig_2, inpaint_fawkes_5)

except Exception as e:
    logger.error("Fawkes outfit set failed: %s", e)

# ---------------------------------------------------------------------------
# TEST SET 6: DiffusionGuard vs Fawkes side-by-side (same edit)
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("SET 6: DiffusionGuard vs Fawkes — side-by-side cat ears")
logger.info("=" * 60)

try:
    set6_dir = os.path.join(OUTPUT_DIR, "set6_comparison_dg_vs_fawkes")
    os.makedirs(set6_dir, exist_ok=True)

    original.save(os.path.join(set6_dir, "1_original.png"))
    protected_dg.save(os.path.join(set6_dir, "2a_diffguard_protected.png"))
    fawkes_cloaked.save(os.path.join(set6_dir, "2b_fawkes_cloaked.png"))
    inpaint_orig_1.save(os.path.join(set6_dir, "3_inpaint_original.png"))
    inpaint_prot_1.save(os.path.join(set6_dir, "4a_inpaint_diffguard.png"))
    inpaint_fawkes_4.save(os.path.join(set6_dir, "4b_inpaint_fawkes.png"))

    logger.info("Saved comparison set: set6_comparison_dg_vs_fawkes")
except Exception as e:
    logger.error("Set 6 failed: %s", e)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("ALL COMPARISON SETS COMPLETE")
logger.info("Output directory: %s", OUTPUT_DIR)
for d in sorted(os.listdir(OUTPUT_DIR)):
    full = os.path.join(OUTPUT_DIR, d)
    if os.path.isdir(full):
        files = os.listdir(full)
        logger.info("  %s/ (%d files)", d, len(files))
logger.info("=" * 60)
