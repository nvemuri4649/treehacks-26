"""
Video glazing + Image-to-Image generation demo (RESUME VERSION)
Picks up from wherever it left off.
"""

import os, sys, time, logging, glob, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

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

sys.path.insert(0, '/workspace/DiffusionGuard')

import torch
from omegaconf import OmegaConf
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionImg2ImgPipeline
from attacks import protect_image
from utils import overlay_images, get_mask_radius_list, tensor_to_pil_image

OUTPUT_DIR = "/workspace/video_demo_output"
DEVICE = "cuda"
DTYPE = torch.float16
MODEL_ID = "runwayml/stable-diffusion-inpainting"
IMG2IMG_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
IMG_SIZE = 512
FPS = 5

FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames_original")
GLAZED_DIR = os.path.join(OUTPUT_DIR, "frames_glazed")
ATTACK_ORIG_DIR = os.path.join(OUTPUT_DIR, "frames_attack_original")
ATTACK_GLAZE_DIR = os.path.join(OUTPUT_DIR, "frames_attack_glazed")

# ---------------------------------------------------------------------------
logger.info("Loading SD inpainting pipeline...")
pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    MODEL_ID, variant="fp16", torch_dtype=DTYPE
).to(DEVICE)
logger.info("Inpainting pipeline loaded.")

from mtcnn import MTCNN
_detector = MTCNN()

def detect_face_mask(img_512, margin_factor=0.35):
    img_array = np.array(img_512)
    results = _detector.detect_faces(img_array)
    if not results:
        mask = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse([128, 64, 384, 384], fill=255)
        return mask
    best = max(results, key=lambda r: r["box"][2] * r["box"][3])
    x, y, w, h = best["box"]
    mx, my = int(w * margin_factor), int(h * margin_factor)
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2, y2 = min(IMG_SIZE, x + w + mx), min(IMG_SIZE, y + h + my)
    mask = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([x1, y1, x2, y2], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
    mask_arr = (np.array(mask) > 128).astype(np.uint8) * 255
    return Image.fromarray(mask_arr)

def run_diffusionguard(src_image, mask_rgb, iters=200):
    config = OmegaConf.create({
        "exp_name": "video", "method": "diffusionguard",
        "orig_image_name": "input.png", "mask_image_names": ["mask.png"],
        "model": {"inpainting": MODEL_ID},
        "training": {
            "size": IMG_SIZE, "iters": iters, "grad_reps": 1, "batch_size": 1,
            "eps": 16/255, "step_size": 1/255, "num_inference_steps": 4,
            "mask": {"generation_method": "contour_shrink", "contour_strength": 1.1,
                     "contour_iters": 15, "contour_smoothness": 0.1},
        },
    })
    mask_list = [mask_rgb]
    adv = protect_image(
        config.method, pipe_inpaint, src_image, mask_list,
        overlay_images(mask_list), get_mask_radius_list(mask_list), config
    )
    return tensor_to_pil_image(adv.detach().cpu())

def run_inpaint_attack(image, face_mask_L, prompt, seed=42):
    attack_mask = Image.fromarray(255 - np.array(face_mask_L))
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    return pipe_inpaint(
        prompt=prompt, image=image, mask_image=attack_mask,
        num_inference_steps=50, generator=gen
    ).images[0]

# ===================================================================
# PART 1: VIDEO GLAZING (RESUME)
# ===================================================================
logger.info("=" * 60)
logger.info("PART 1: VIDEO GLAZING (RESUME)")
logger.info("=" * 60)

frame_files = sorted(glob.glob(os.path.join(FRAMES_DIR, "frame_*.png")))
total_frames = len(frame_files)
logger.info("Total frames: %d", total_frames)

# Check what's already done
existing_glazed = set(os.listdir(GLAZED_DIR))
existing_attack_orig = set(os.listdir(ATTACK_ORIG_DIR))
existing_attack_glaze = set(os.listdir(ATTACK_GLAZE_DIR))
logger.info("Already glazed: %d, attack_orig: %d, attack_glazed: %d",
            len(existing_glazed), len(existing_attack_orig), len(existing_attack_glaze))

def center_crop_square(img):
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    return img.crop((left, top, left + side, top + side))

ATTACK_PROMPT = "a person in a prison cell wearing orange jumpsuit, mugshot, photorealistic"

for i, frame_path in enumerate(frame_files):
    frame_name = os.path.basename(frame_path)
    
    # Skip if all outputs exist for this frame
    if (frame_name in existing_glazed and 
        frame_name in existing_attack_orig and 
        frame_name in existing_attack_glaze):
        logger.info("Skipping frame %d/%d: %s (already done)", i+1, total_frames, frame_name)
        continue

    logger.info("--- Frame %d/%d: %s ---", i + 1, total_frames, frame_name)

    frame_raw = Image.open(frame_path).convert("RGB")
    # Ensure 512x512
    if frame_raw.size != (IMG_SIZE, IMG_SIZE):
        frame_512 = center_crop_square(frame_raw).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        frame_512.save(frame_path)
        logger.info("  Resized %dx%d -> 512x512", frame_raw.size[0], frame_raw.size[1])
    else:
        frame_512 = frame_raw
    face_mask = detect_face_mask(frame_512)
    face_mask_rgb = Image.merge("RGB", [face_mask, face_mask, face_mask])

    # Glaze
    if frame_name not in existing_glazed:
        t0 = time.time()
        glazed = run_diffusionguard(frame_512, face_mask_rgb, iters=200)
        logger.info("  DiffusionGuard: %.1fs", time.time() - t0)
        glazed.save(os.path.join(GLAZED_DIR, frame_name))
    else:
        glazed = Image.open(os.path.join(GLAZED_DIR, frame_name)).convert("RGB")
        logger.info("  Glazed already exists, loaded from disk")

    # Inpainting attack on original
    if frame_name not in existing_attack_orig:
        attack_orig = run_inpaint_attack(frame_512, face_mask, ATTACK_PROMPT, seed=42)
        attack_orig.save(os.path.join(ATTACK_ORIG_DIR, frame_name))
    
    # Inpainting attack on glazed
    if frame_name not in existing_attack_glaze:
        attack_glazed = run_inpaint_attack(glazed, face_mask, ATTACK_PROMPT, seed=42)
        attack_glazed.save(os.path.join(ATTACK_GLAZE_DIR, frame_name))

    logger.info("  Frame %d/%d done.", i + 1, total_frames)

# Reassemble videos
logger.info("Reassembling videos...")
for name, src_dir in [
    ("original", FRAMES_DIR),
    ("glazed", GLAZED_DIR),
    ("attack_original", ATTACK_ORIG_DIR),
    ("attack_glazed", ATTACK_GLAZE_DIR),
]:
    out_path = os.path.join(OUTPUT_DIR, f"video_{name}.mp4")
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", os.path.join(src_dir, "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        out_path
    ], capture_output=True)
    logger.info("  Created: %s", out_path)

logger.info("VIDEO GLAZING COMPLETE")

# ===================================================================
# PART 2: IMAGE-TO-IMAGE
# ===================================================================
logger.info("=" * 60)
logger.info("PART 2: IMG2IMG — glazing disrupts reference-based generation")
logger.info("=" * 60)

IMG2IMG_DIR = os.path.join(OUTPUT_DIR, "img2img")
os.makedirs(IMG2IMG_DIR, exist_ok=True)

del pipe_inpaint
torch.cuda.empty_cache()
import gc; gc.collect()

logger.info("Loading SD img2img pipeline...")
pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
    IMG2IMG_MODEL, variant="fp16", torch_dtype=DTYPE
).to(DEVICE)
logger.info("Img2img pipeline loaded.")

original_img = Image.open(os.path.join(FRAMES_DIR, "frame_0001.png")).convert("RGB")
glazed_img = Image.open(os.path.join(GLAZED_DIR, "frame_0001.png")).convert("RGB")

original_img.save(os.path.join(IMG2IMG_DIR, "1_original.png"))
glazed_img.save(os.path.join(IMG2IMG_DIR, "1_glazed.png"))

# Use stronger protection if available
if os.path.exists("/workspace/comparison_outputs_v3/protected.png"):
    protected_strong = Image.open("/workspace/comparison_outputs_v3/protected.png").convert("RGB")
    protected_strong.save(os.path.join(IMG2IMG_DIR, "1_protected_strong.png"))
else:
    protected_strong = glazed_img

img2img_tests = [
    ("portrait_painting", "oil painting portrait of a person, masterpiece, detailed", 0.6, 42),
    ("anime", "anime style portrait of a person, studio ghibli, detailed", 0.65, 77),
    ("zombie", "zombie version of this person, horror, photorealistic", 0.55, 99),
    ("aging", "elderly version of this person, aged 80, wrinkled, photorealistic", 0.5, 123),
    ("deepfake_celeb", "portrait of a famous celebrity, photorealistic, detailed face", 0.7, 55),
]

for test_name, prompt, strength, seed in img2img_tests:
    orig_path = os.path.join(IMG2IMG_DIR, f"orig_{test_name}.png")
    glazed_path = os.path.join(IMG2IMG_DIR, f"glazed_{test_name}.png")
    
    if os.path.exists(orig_path) and os.path.exists(glazed_path):
        logger.info("Img2img test: %s — already done, skipping", test_name)
        continue

    logger.info("Img2img test: %s (strength=%.2f)", test_name, strength)

    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    result_orig = pipe_img2img(
        prompt=prompt, image=original_img, strength=strength,
        num_inference_steps=50, generator=gen
    ).images[0]

    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    result_glazed = pipe_img2img(
        prompt=prompt, image=protected_strong, strength=strength,
        num_inference_steps=50, generator=gen
    ).images[0]

    result_orig.save(orig_path)
    result_glazed.save(glazed_path)
    logger.info("  Saved: %s", test_name)

logger.info("=" * 60)
logger.info("ALL DEMOS COMPLETE. Output: %s", OUTPUT_DIR)
for item in sorted(os.listdir(OUTPUT_DIR)):
    full = os.path.join(OUTPUT_DIR, item)
    if os.path.isdir(full):
        logger.info("  %s/ (%d files)", item, len(os.listdir(full)))
    else:
        logger.info("  %s", item)
logger.info("=" * 60)
