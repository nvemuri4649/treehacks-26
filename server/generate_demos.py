"""
Generate high-quality demo comparisons: original vs adversarial photo.

Image generation:  SDXL + IP-Adapter Plus Face (local GPU)
Video generation:  Replicate API (minimax video-01-live)

The IP-Adapter Plus Face model uses CLIP ViT-H/14 — the exact same encoder
our ensemble attack targets — so the adversarial perturbation directly
disrupts the face embedding pipeline.

Usage:
    python generate_demos.py \
        --original photo.jpg \
        --adversarial adversarial.png \
        --output-dir ./output \
        [--replicate-token TOKEN]   # or set REPLICATE_API_TOKEN env var
"""

import os
import sys
import time
import argparse
import logging
import base64
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def center_crop_square(img: Image.Image, size: int = 512) -> Image.Image:
    """Center-crop to square, then resize. No squashing."""
    w, h = img.size
    crop_dim = min(w, h)
    left = (w - crop_dim) // 2
    top = (h - crop_dim) // 2
    img = img.crop((left, top, left + crop_dim, top + crop_dim))
    return img.resize((size, size), Image.LANCZOS)


# ===================================================================
# Image generation — SDXL + IP-Adapter Plus Face
# ===================================================================

PROMPTS = [
    ("cyberpunk", "a person in a cyberpunk city at night, neon lights reflecting in rain puddles, holographic advertisements, blade runner aesthetic, cinematic 4k"),
    ("astronaut", "a person as an astronaut floating in outer space with earth in the background, detailed NASA spacesuit, stars, cinematic lighting, 8k"),
    ("medieval", "a person as a medieval knight in shining plate armor, standing in a grand stone castle hall, dramatic torchlight, epic fantasy painting"),
    ("boxing", "a person as a boxer in a boxing ring, dramatic spotlight from above, sweat glistening, action pose, sports photography, 4k"),
    ("underwater", "a person scuba diving over a vibrant coral reef, tropical fish, sunlight filtering through crystal blue water, national geographic photo"),
    ("samurai", "a person as a samurai warrior in traditional armor, cherry blossom petals falling, misty mountain temple background, cinematic japanese aesthetic"),
]

IP_ADAPTER_SCALES = [0.6, 0.85]


def generate_sdxl_ip_adapter(original: Image.Image, adversarial: Image.Image,
                              output_dir: str, seed: int = 42):
    """Generate images using SDXL + IP-Adapter Plus Face."""
    from diffusers import StableDiffusionXLPipeline, AutoencoderKL

    logger.info("Loading SDXL + IP-Adapter Plus Face...")

    # Load SDXL base
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(DEVICE)

    # Load IP-Adapter Plus Face (uses CLIP ViT-H/14 — our attack target)
    # Must point to models/image_encoder (ViT-H/14, 1280-dim) not sdxl_models/image_encoder (bigG, 1664-dim)
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter-plus-face_sdxl_vit-h.bin",
        image_encoder_folder="models/image_encoder",
    )

    logger.info("SDXL + IP-Adapter loaded on %s", DEVICE)

    total = len(PROMPTS) * len(IP_ADAPTER_SCALES) * 2
    idx = 0

    for scale in IP_ADAPTER_SCALES:
        pipe.set_ip_adapter_scale(scale)
        scale_str = f"{scale:.2f}"

        for tag, prompt in PROMPTS:
            # ---- Original photo ----
            idx += 1
            logger.info("  [%d/%d] ORIG  scale=%s  %s", idx, total, scale_str, tag)
            gen = torch.Generator(device=DEVICE).manual_seed(seed)
            out_orig = pipe(
                prompt=prompt,
                negative_prompt="blurry, low quality, distorted face, deformed",
                ip_adapter_image=original,
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=gen,
                height=1024, width=1024,
            ).images[0]
            out_orig.save(os.path.join(output_dir, f"orig_s{scale_str}_{tag}.png"))

            # ---- Adversarial photo ----
            idx += 1
            logger.info("  [%d/%d] ATTACK scale=%s  %s", idx, total, scale_str, tag)
            gen = torch.Generator(device=DEVICE).manual_seed(seed)
            out_atk = pipe(
                prompt=prompt,
                negative_prompt="blurry, low quality, distorted face, deformed",
                ip_adapter_image=adversarial,
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=gen,
                height=1024, width=1024,
            ).images[0]
            out_atk.save(os.path.join(output_dir, f"attack_s{scale_str}_{tag}.png"))

    # Build comparison grid
    logger.info("Building comparison grid...")
    build_grid(output_dir)

    del pipe, vae
    torch.cuda.empty_cache()
    logger.info("Image generation complete.")


def build_grid(output_dir: str):
    """Build a comparison grid: rows = scales, col pairs = (orig, attack) per prompt."""
    cell = 384
    n_prompts = len(PROMPTS)
    n_scales = len(IP_ADAPTER_SCALES)

    # Layout: header row + scale rows.  Columns: label + (orig, attack) * n_prompts
    cols = 1 + 2 * n_prompts
    rows = 1 + n_scales
    grid = Image.new("RGB", (cols * cell, rows * cell), (30, 30, 30))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    # Header row
    for j, (tag, _) in enumerate(PROMPTS):
        x_orig = (1 + 2 * j) * cell
        x_atk = (1 + 2 * j + 1) * cell
        draw.text((x_orig + 10, cell // 2 - 20), f"{tag.upper()}", fill=(100, 255, 100), font=font)
        draw.text((x_orig + 10, cell // 2 + 5), "original", fill=(150, 150, 150), font=font_sm)
        draw.text((x_atk + 10, cell // 2 - 20), f"{tag.upper()}", fill=(255, 100, 100), font=font)
        draw.text((x_atk + 10, cell // 2 + 5), "protected", fill=(150, 150, 150), font=font_sm)

    for si, scale in enumerate(IP_ADAPTER_SCALES):
        y = (si + 1) * cell
        scale_str = f"{scale:.2f}"

        # Row label
        draw.text((10, y + cell // 2 - 10), f"scale\n{scale_str}", fill=(200, 200, 200), font=font)

        for j, (tag, _) in enumerate(PROMPTS):
            orig_path = os.path.join(output_dir, f"orig_s{scale_str}_{tag}.png")
            atk_path = os.path.join(output_dir, f"attack_s{scale_str}_{tag}.png")

            if os.path.exists(orig_path):
                img = Image.open(orig_path).resize((cell, cell), Image.LANCZOS)
                grid.paste(img, ((1 + 2 * j) * cell, y))
            if os.path.exists(atk_path):
                img = Image.open(atk_path).resize((cell, cell), Image.LANCZOS)
                grid.paste(img, ((1 + 2 * j + 1) * cell, y))

    grid.save(os.path.join(output_dir, "comparison_grid.png"))
    logger.info("  Saved comparison_grid.png (%dx%d)", grid.width, grid.height)


# ===================================================================
# Video generation — Replicate API
# ===================================================================

def generate_videos(original_path: str, adversarial_path: str,
                    output_dir: str, api_token: str):
    """Generate image-to-video comparisons using Replicate API."""
    try:
        import replicate
    except ImportError:
        logger.info("Installing replicate package...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "replicate"])
        import replicate

    os.environ["REPLICATE_API_TOKEN"] = api_token
    client = replicate.Client(api_token=api_token)

    video_prompts = [
        ("talking", "a person talking naturally to the camera, subtle head movements, natural blinking, professional interview setting"),
        ("walking", "a person walking confidently through a cyberpunk city street at night, neon reflections, cinematic"),
        ("action", "a person in dynamic action, turning their head and smiling, natural movement, cinematic lighting"),
    ]

    for tag, prompt in video_prompts:
        for label, img_path in [("orig", original_path), ("attack", adversarial_path)]:
            logger.info("  Video: %s_%s — generating...", label, tag)
            try:
                output = client.run(
                    "minimax/video-01-live",
                    input={
                        "prompt": prompt,
                        "first_frame_image": open(img_path, "rb"),
                    },
                )
                # output is a URL or FileOutput
                out_url = str(output)
                video_path = os.path.join(output_dir, f"video_{label}_{tag}.mp4")

                # Download the video
                import urllib.request
                urllib.request.urlretrieve(out_url, video_path)
                logger.info("    Saved: %s", video_path)

            except Exception as e:
                logger.warning("    Video generation failed: %s", e)
                # Try fallback model
                try:
                    logger.info("    Trying fallback (stable-video-diffusion)...")
                    output = client.run(
                        "stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438",
                        input={
                            "input_image": open(img_path, "rb"),
                            "frames_per_second": 6,
                            "motion_bucket_id": 127,
                        },
                    )
                    out_url = str(output)
                    video_path = os.path.join(output_dir, f"video_{label}_{tag}.mp4")
                    import urllib.request
                    urllib.request.urlretrieve(out_url, video_path)
                    logger.info("    Saved (fallback): %s", video_path)
                except Exception as e2:
                    logger.warning("    Fallback also failed: %s", e2)


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True, help="Path to original photo")
    parser.add_argument("--adversarial", required=True, help="Path to adversarial photo")
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-images", action="store_true", help="Skip SDXL generation")
    parser.add_argument("--skip-videos", action="store_true", help="Skip video generation")
    parser.add_argument("--replicate-token", default=None,
                        help="Replicate API token (or set REPLICATE_API_TOKEN env var)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load images
    original = center_crop_square(Image.open(args.original).convert("RGB"), 512)
    adversarial = center_crop_square(Image.open(args.adversarial).convert("RGB"), 512)

    original.save(os.path.join(args.output_dir, "00_original.png"))
    adversarial.save(os.path.join(args.output_dir, "00_adversarial.png"))

    logger.info("=" * 70)
    logger.info("DEMO GENERATION")
    logger.info("  Image model: SDXL + IP-Adapter Plus Face (CLIP ViT-H/14)")
    logger.info("  Video model: Replicate API (minimax video-01-live)")
    logger.info("=" * 70)

    # --- Images ---
    if not args.skip_images:
        generate_sdxl_ip_adapter(original, adversarial, args.output_dir, args.seed)
    else:
        logger.info("Skipping image generation (--skip-images)")

    # --- Videos ---
    token = args.replicate_token or os.environ.get("REPLICATE_API_TOKEN")
    if not args.skip_videos and token:
        logger.info("\nGenerating videos via Replicate API...")
        orig_path = os.path.join(args.output_dir, "00_original.png")
        adv_path = os.path.join(args.output_dir, "00_adversarial.png")
        generate_videos(orig_path, adv_path, args.output_dir, token)
    elif not args.skip_videos:
        logger.info("\nSkipping video generation (no REPLICATE_API_TOKEN set)")
        logger.info("  Set via: --replicate-token TOKEN  or  export REPLICATE_API_TOKEN=...")

    logger.info("\n" + "=" * 70)
    logger.info("DONE. Output: %s", args.output_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
