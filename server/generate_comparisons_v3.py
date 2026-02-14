"""
Generate comparison sets — CORRECT THREAT MODEL
================================================
The DiffusionGuard threat model:
  - Attacker has a photo of someone
  - Attacker uses SD inpainting to KEEP the face but REPLACE the scene/body
  - e.g., put someone in prison, at a crime scene, in a compromising situation
  - DiffusionGuard adds adversarial noise to the face so when the attacker
    tries to keep it, the face gets corrupted/distorted

Mask conventions:
  - DiffusionGuard mask (white = face region to protect)
  - SD inpainting attack mask (white = region to REGENERATE = background/body)
    → attacker keeps the face (black), regenerates surroundings (white)

Output: 4-image comparison sets showing original, protected,
        attack-on-original (face intact, fake scene), 
        attack-on-protected (face corrupted, protection worked).
"""

import os
import sys
import time
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Constrain TF GPU memory BEFORE any TF imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=2048)]
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

os.environ['DIFFGUARD_REPO'] = '/workspace/DiffusionGuard'
sys.path.insert(0, '/workspace/DiffusionGuard')

import torch
from omegaconf import OmegaConf
from diffusers import StableDiffusionInpaintPipeline
from attacks import protect_image
from utils import overlay_images, get_mask_radius_list, tensor_to_pil_image

OUTPUT_DIR = "/workspace/comparison_outputs_v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
# Face detection
# ---------------------------------------------------------------------------
from mtcnn import MTCNN
_detector = MTCNN()


def detect_face_mask(img_512, margin_factor=0.35):
    """Detect face → oval mask. White = face region."""
    img_array = np.array(img_512)
    results = _detector.detect_faces(img_array)
    if not results:
        logger.warning("No face detected! Using center oval fallback.")
        mask = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse([128, 64, 384, 384], fill=255)
        return mask

    best = max(results, key=lambda r: r["box"][2] * r["box"][3])
    x, y, w, h = best["box"]
    mx = int(w * margin_factor)
    my = int(h * margin_factor)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(IMG_SIZE, x + w + mx)
    y2 = min(IMG_SIZE, y + h + my)

    logger.info("Face bbox: (%d,%d)→(%d,%d)", x1, y1, x2, y2)

    mask = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([x1, y1, x2, y2], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
    mask_arr = np.array(mask)
    mask_arr = (mask_arr > 128).astype(np.uint8) * 255

    white_pct = (mask_arr > 128).sum() / mask_arr.size * 100
    logger.info("Face mask: %.1f%% of image", white_pct)
    return Image.fromarray(mask_arr)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def center_crop_square(img):
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def run_incriminating_inpaint(image, face_mask_L, prompt, steps=50, seed=42):
    """
    Simulate the ATTACKER's action:
      - Keep the face (face_mask white region → SD keeps this)
      - Replace everything else with a new scene
    
    SD inpainting: white = region to REGENERATE.
    So the attack mask = INVERSE of the face mask:
      - Face = black (keep) 
      - Background/body = white (regenerate with prompt)
    """
    face_arr = np.array(face_mask_L)
    attack_mask_arr = 255 - face_arr  # Invert: background=white, face=black
    attack_mask = Image.fromarray(attack_mask_arr)

    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=attack_mask,  # white = background to regenerate
        num_inference_steps=steps,
        generator=generator,
    ).images[0]
    return result


def run_diffusionguard(src_image, mask_image_rgb, iters=800, eps=16/255, step_size=1/255):
    """Paper-faithful DiffusionGuard. mask white = face to protect."""
    config = OmegaConf.create({
        "exp_name": "v3",
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


# ---------------------------------------------------------------------------
# Load test image
# ---------------------------------------------------------------------------
TEST_IMG_PATH = "/workspace/test_photo.jpg"
original = Image.open(TEST_IMG_PATH).convert("RGB")
original = center_crop_square(original).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
original.save(os.path.join(OUTPUT_DIR, "original_512.png"))

# Detect face
face_mask_L = detect_face_mask(original)
face_mask_L.save(os.path.join(OUTPUT_DIR, "face_mask.png"))
face_mask_rgb = Image.merge("RGB", [face_mask_L, face_mask_L, face_mask_L])

# Also save the inverted mask (what the attacker uses)
attack_mask = Image.fromarray(255 - np.array(face_mask_L))
attack_mask.save(os.path.join(OUTPUT_DIR, "attack_mask_inverted.png"))

# ---------------------------------------------------------------------------
# PROTECT the image (once — reuse for all attacks)
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("Running DiffusionGuard (800 iters, eps=16/255, face mask)...")
logger.info("=" * 60)

t0 = time.time()
protected = run_diffusionguard(original, face_mask_rgb, iters=800, eps=16/255, step_size=1/255)
logger.info("DiffusionGuard took %.1fs", time.time() - t0)
protected.save(os.path.join(OUTPUT_DIR, "protected.png"))

# ---------------------------------------------------------------------------
# INCRIMINATING ATTACK SCENARIOS
# Each prompt puts the person in a fake compromising scene.
# The attacker KEEPS the face and replaces the background/body.
# ---------------------------------------------------------------------------

scenarios = [
    ("prison", "a prisoner in an orange jumpsuit in a prison cell, mugshot, photorealistic", 42),
    ("crime_scene", "a person at a crime scene with police tape, holding a weapon, photorealistic", 77),
    ("hospital", "a person lying in a hospital bed, sick, medical equipment, photorealistic", 99),
    ("protest", "a person at a violent protest with fire and smoke, photorealistic", 123),
    ("scandal", "a person in a seedy nightclub with neon lights, photorealistic", 55),
]

for name, prompt, seed in scenarios:
    logger.info("=" * 60)
    logger.info("SCENARIO: %s", name)
    logger.info("  Prompt: %s", prompt)
    logger.info("=" * 60)

    scene_dir = os.path.join(OUTPUT_DIR, f"scene_{name}")
    os.makedirs(scene_dir, exist_ok=True)

    # Attack on ORIGINAL (unprotected) — face stays intact, scene changes
    logger.info("  Attacking original (unprotected)...")
    attack_orig = run_incriminating_inpaint(original, face_mask_L, prompt, seed=seed)
    
    # Attack on PROTECTED — face should be corrupted/distorted
    logger.info("  Attacking protected...")
    attack_prot = run_incriminating_inpaint(protected, face_mask_L, prompt, seed=seed)

    # Save
    original.save(os.path.join(scene_dir, "1_original.png"))
    protected.save(os.path.join(scene_dir, "2_protected.png"))
    attack_orig.save(os.path.join(scene_dir, "3_attack_original.png"))
    attack_prot.save(os.path.join(scene_dir, "4_attack_protected.png"))

    logger.info("  Saved: %s/", name)

# ---------------------------------------------------------------------------
# KEANU example from the paper
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("KEANU: Paper's original example")
logger.info("=" * 60)

keanu_dir = os.path.join(OUTPUT_DIR, "keanu")
os.makedirs(keanu_dir, exist_ok=True)

keanu = Image.open("/workspace/DiffusionGuard/assets/keanu.png").convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
keanu_mask = Image.open("/workspace/DiffusionGuard/assets/keanu_mask.png").convert("L").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
keanu_mask_rgb = Image.merge("RGB", [keanu_mask, keanu_mask, keanu_mask])

keanu.save(os.path.join(keanu_dir, "1_keanu_original.png"))
keanu_mask.save(os.path.join(keanu_dir, "1b_face_mask.png"))

t0 = time.time()
keanu_protected = run_diffusionguard(keanu, keanu_mask_rgb, iters=800, eps=16/255, step_size=1/255)
logger.info("Keanu DiffusionGuard took %.1fs", time.time() - t0)
keanu_protected.save(os.path.join(keanu_dir, "2_keanu_protected.png"))

keanu_prompts = [
    ("hospital", "a person lying in a hospital bed, photorealistic", 42),
    ("prison", "a prisoner in orange jumpsuit in jail cell, mugshot, photorealistic", 77),
]

for name, prompt, seed in keanu_prompts:
    attack_o = run_incriminating_inpaint(keanu, keanu_mask, prompt, seed=seed)
    attack_p = run_incriminating_inpaint(keanu_protected, keanu_mask, prompt, seed=seed)
    attack_o.save(os.path.join(keanu_dir, f"3_attack_orig_{name}.png"))
    attack_p.save(os.path.join(keanu_dir, f"4_attack_prot_{name}.png"))

# ---------------------------------------------------------------------------
# FAWKES comparison
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("FAWKES: Cloaking comparison")
logger.info("=" * 60)

fawkes_dir = os.path.join(OUTPUT_DIR, "fawkes_comparison")
os.makedirs(fawkes_dir, exist_ok=True)

try:
    from fawkes_modern import init_fawkes, cloak_image
    init_fawkes("low")

    t0 = time.time()
    fawkes_cloaked = cloak_image(original, mode="low")
    logger.info("Fawkes took %.1fs", time.time() - t0)

    if fawkes_cloaked.size != (IMG_SIZE, IMG_SIZE):
        fawkes_cloaked = fawkes_cloaked.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    fawkes_cloaked.save(os.path.join(fawkes_dir, "2_fawkes_cloaked.png"))

    # Same incriminating attack on Fawkes-cloaked image
    prompt = "a prisoner in an orange jumpsuit in a prison cell, mugshot, photorealistic"
    attack_fawkes = run_incriminating_inpaint(fawkes_cloaked, face_mask_L, prompt, seed=42)
    attack_fawkes.save(os.path.join(fawkes_dir, "4_attack_fawkes.png"))

    # Also save DG comparison for same scenario
    original.save(os.path.join(fawkes_dir, "1_original.png"))
    protected.save(os.path.join(fawkes_dir, "2b_diffguard_protected.png"))

    attack_orig_prison = run_incriminating_inpaint(original, face_mask_L, prompt, seed=42)
    attack_dg_prison = run_incriminating_inpaint(protected, face_mask_L, prompt, seed=42)
    attack_orig_prison.save(os.path.join(fawkes_dir, "3_attack_original.png"))
    attack_dg_prison.save(os.path.join(fawkes_dir, "4b_attack_diffguard.png"))

    logger.info("Fawkes comparison complete.")

except Exception as e:
    logger.error("Fawkes failed: %s", e)
    import traceback
    traceback.print_exc()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
logger.info("=" * 60)
logger.info("ALL SCENARIOS COMPLETE")
logger.info("Output: %s", OUTPUT_DIR)
for d in sorted(os.listdir(OUTPUT_DIR)):
    full = os.path.join(OUTPUT_DIR, d)
    if os.path.isdir(full):
        logger.info("  %s/ (%d files)", d, len(os.listdir(full)))
logger.info("=" * 60)
