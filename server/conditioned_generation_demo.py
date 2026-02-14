"""
Conditioned Generation Demo — IP-Adapter + Stable Video Diffusion
==================================================================
Demonstrates that DiffusionGuard protection disrupts modern AI generation
pipelines that use images as conditioning:

1) IP-ADAPTER: Uses a reference face image to condition text-to-image
   generation. The image's identity/style guides the output.
   → Original image: clean, identity-preserving generation
   → Protected image: corrupted/distorted generation

2) STABLE VIDEO DIFFUSION: Generates short video clips from a single
   reference image.
   → Original image: smooth, coherent video
   → Protected image: glitchy, distorted video

Usage:
    python conditioned_generation_demo.py [--images-dir PATH] [--output-dir PATH]
                                          [--skip-ip-adapter] [--skip-svd]
                                          [--ip-adapter-scale 0.7]
"""

import os
import sys
import time
import glob
import argparse
import logging
from pathlib import Path

import torch
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE in ("cuda", "mps") else torch.float32

# Model IDs
SD15_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
IP_ADAPTER_REPO = "h94/IP-Adapter"
SVD_MODEL = "stabilityai/stable-video-diffusion-img2vid-xt"

IMG_SIZE = 512


def parse_args():
    parser = argparse.ArgumentParser(description="Conditioned generation demo")
    parser.add_argument(
        "--images-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "comparison_outputs_v3"),
        help="Directory with original/protected images",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "conditioned_outputs"),
        help="Output directory for generated images/videos",
    )
    parser.add_argument("--skip-ip-adapter", action="store_true", help="Skip IP-Adapter demos")
    parser.add_argument("--skip-svd", action="store_true", help="Skip Stable Video Diffusion demos")
    parser.add_argument("--ip-adapter-scale", type=float, default=0.7, help="IP-Adapter conditioning strength (0-1)")
    parser.add_argument("--svd-num-frames", type=int, default=25, help="Number of video frames to generate")
    parser.add_argument("--svd-fps", type=int, default=7, help="Output video FPS")
    parser.add_argument("--use-sdxl", action="store_true", help="Use SDXL instead of SD1.5 for IP-Adapter")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_image_pair(images_dir: str):
    """
    Load original and protected image pairs from the comparison directory.
    Returns list of (name, original_img, protected_img) tuples.
    """
    pairs = []

    # 1) Top-level original/protected
    orig_path = os.path.join(images_dir, "original_512.png")
    prot_path = os.path.join(images_dir, "protected.png")
    if os.path.exists(orig_path) and os.path.exists(prot_path):
        orig = Image.open(orig_path).convert("RGB")
        prot = Image.open(prot_path).convert("RGB")
        pairs.append(("main_subject", orig, prot))

    # 2) Keanu example
    keanu_orig = os.path.join(images_dir, "keanu", "1_keanu_original.png")
    keanu_prot = os.path.join(images_dir, "keanu", "2_keanu_protected.png")
    if os.path.exists(keanu_orig) and os.path.exists(keanu_prot):
        orig = Image.open(keanu_orig).convert("RGB")
        prot = Image.open(keanu_prot).convert("RGB")
        pairs.append(("keanu", orig, prot))

    # 3) Scene subdirectories (each has 1_original.png and 2_protected.png)
    for scene_dir in sorted(glob.glob(os.path.join(images_dir, "scene_*"))):
        scene_name = os.path.basename(scene_dir)
        orig_file = os.path.join(scene_dir, "1_original.png")
        prot_file = os.path.join(scene_dir, "2_protected.png")
        if os.path.exists(orig_file) and os.path.exists(prot_file):
            orig = Image.open(orig_file).convert("RGB")
            prot = Image.open(prot_file).convert("RGB")
            pairs.append((scene_name, orig, prot))

    return pairs


def ensure_size(img: Image.Image, size: int = IMG_SIZE) -> Image.Image:
    """Resize if needed."""
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    return img


# ===================================================================
# IP-ADAPTER: Image-conditioned text-to-image
# ===================================================================

def run_ip_adapter_demo(pairs, output_dir, args):
    """
    Use IP-Adapter to generate images conditioned on original vs protected faces.
    IP-Adapter encodes the reference image into CLIP image embeddings and injects
    them into the cross-attention layers alongside text embeddings.
    """
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

    ip_dir = os.path.join(output_dir, "ip_adapter")
    os.makedirs(ip_dir, exist_ok=True)

    # Load the appropriate pipeline
    if args.use_sdxl:
        logger.info("Loading SDXL + IP-Adapter...")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            SDXL_MODEL, torch_dtype=DTYPE, variant="fp16"
        ).to(DEVICE)
        pipe.load_ip_adapter(
            IP_ADAPTER_REPO,
            subfolder="sdxl_models",
            weight_name="ip-adapter_sdxl.bin",
        )
        gen_size = 1024
    else:
        logger.info("Loading SD 1.5 + IP-Adapter...")
        pipe = StableDiffusionPipeline.from_pretrained(
            SD15_MODEL, torch_dtype=DTYPE, variant="fp16"
        ).to(DEVICE)
        pipe.load_ip_adapter(
            IP_ADAPTER_REPO,
            subfolder="models",
            weight_name="ip-adapter_sd15.bin",
        )
        gen_size = 512

    pipe.set_ip_adapter_scale(args.ip_adapter_scale)
    logger.info("IP-Adapter loaded (scale=%.2f)", args.ip_adapter_scale)

    # Prompts for generation — each creates a new scene guided by the face
    prompts = [
        ("professional_headshot", "professional corporate headshot photo, studio lighting, "
         "clean background, high quality portrait, 4k"),
        ("oil_painting", "oil painting portrait in the style of Rembrandt, dramatic lighting, "
         "masterpiece, museum quality, detailed brushstrokes"),
        ("anime_portrait", "anime style portrait, studio ghibli aesthetic, "
         "detailed, vibrant colors, beautiful illustration"),
        ("cyberpunk", "cyberpunk portrait, neon lights, futuristic city background, "
         "blade runner style, rain, photorealistic"),
        ("medieval_knight", "portrait of a medieval knight in armor, castle background, "
         "cinematic lighting, epic fantasy, photorealistic"),
        ("astronaut", "portrait of an astronaut on the moon, earth in background, "
         "NASA spacesuit, photorealistic, detailed"),
    ]

    # Also test different IP-Adapter scales to show impact
    scales_to_test = [0.4, 0.7, 1.0]

    for pair_name, orig_img, prot_img in pairs:
        pair_dir = os.path.join(ip_dir, pair_name)
        os.makedirs(pair_dir, exist_ok=True)

        # Save reference images
        orig_img.save(os.path.join(pair_dir, "0_ref_original.png"))
        prot_img.save(os.path.join(pair_dir, "0_ref_protected.png"))

        # Generate with each prompt
        for prompt_name, prompt_text in prompts:
            logger.info("  IP-Adapter [%s] prompt=%s", pair_name, prompt_name)

            generator = torch.Generator(device=DEVICE).manual_seed(args.seed)
            result_orig = pipe(
                prompt=prompt_text,
                ip_adapter_image=ensure_size(orig_img),
                num_inference_steps=50,
                generator=generator,
                height=gen_size,
                width=gen_size,
            ).images[0]

            generator = torch.Generator(device=DEVICE).manual_seed(args.seed)
            result_prot = pipe(
                prompt=prompt_text,
                ip_adapter_image=ensure_size(prot_img),
                num_inference_steps=50,
                generator=generator,
                height=gen_size,
                width=gen_size,
            ).images[0]

            result_orig.save(os.path.join(pair_dir, f"orig_{prompt_name}.png"))
            result_prot.save(os.path.join(pair_dir, f"prot_{prompt_name}.png"))

            # Create side-by-side comparison
            comparison = create_comparison_grid(
                orig_img, prot_img, result_orig, result_prot,
                title=f"{pair_name} — {prompt_name}"
            )
            comparison.save(os.path.join(pair_dir, f"compare_{prompt_name}.png"))

        # Scale comparison (one prompt, multiple adapter strengths)
        prompt_text = prompts[0][1]  # use the headshot prompt
        for scale in scales_to_test:
            pipe.set_ip_adapter_scale(scale)

            generator = torch.Generator(device=DEVICE).manual_seed(args.seed)
            res_o = pipe(
                prompt=prompt_text,
                ip_adapter_image=ensure_size(orig_img),
                num_inference_steps=50,
                generator=generator,
                height=gen_size,
                width=gen_size,
            ).images[0]

            generator = torch.Generator(device=DEVICE).manual_seed(args.seed)
            res_p = pipe(
                prompt=prompt_text,
                ip_adapter_image=ensure_size(prot_img),
                num_inference_steps=50,
                generator=generator,
                height=gen_size,
                width=gen_size,
            ).images[0]

            res_o.save(os.path.join(pair_dir, f"scale_{scale:.1f}_orig.png"))
            res_p.save(os.path.join(pair_dir, f"scale_{scale:.1f}_prot.png"))

        # Reset scale
        pipe.set_ip_adapter_scale(args.ip_adapter_scale)

    # Cleanup
    del pipe
    torch.cuda.empty_cache() if DEVICE == "cuda" else None
    logger.info("IP-Adapter demo complete.")


def create_comparison_grid(
    ref_orig, ref_prot, gen_orig, gen_prot, title="", cell_size=256
):
    """
    Create a 2x2 comparison grid:
        [ref_original]  [generated_from_original]
        [ref_protected] [generated_from_protected]
    """
    from PIL import ImageDraw, ImageFont

    imgs = [ref_orig, gen_orig, ref_prot, gen_prot]
    imgs = [img.resize((cell_size, cell_size), Image.LANCZOS) for img in imgs]

    header_h = 30
    label_h = 20
    grid = Image.new("RGB", (cell_size * 2 + 10, cell_size * 2 + header_h + label_h * 2 + 20), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except (IOError, OSError):
        font = ImageFont.load_default()
        font_sm = font

    # Title
    if title:
        draw.text((10, 5), title, fill=(0, 0, 0), font=font)

    y_start = header_h
    labels = [
        ("Original Reference", "Generated from Original"),
        ("Protected Reference", "Generated from Protected"),
    ]

    for row in range(2):
        y = y_start + row * (cell_size + label_h + 5)
        for col in range(2):
            x = col * (cell_size + 10)
            draw.text((x + 5, y), labels[row][col], fill=(80, 80, 80), font=font_sm)
            grid.paste(imgs[row * 2 + col], (x, y + label_h))

    return grid


# ===================================================================
# STABLE VIDEO DIFFUSION: Image-conditioned video generation
# ===================================================================

def run_svd_demo(pairs, output_dir, args):
    """
    Use Stable Video Diffusion to generate short video clips from
    original vs protected reference images.
    """
    from diffusers import StableVideoDiffusionPipeline
    from diffusers.utils import export_to_video

    svd_dir = os.path.join(output_dir, "svd_video")
    os.makedirs(svd_dir, exist_ok=True)

    logger.info("Loading Stable Video Diffusion pipeline...")
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        SVD_MODEL,
        torch_dtype=DTYPE,
        variant="fp16",
    ).to(DEVICE)

    # Enable memory optimizations
    if DEVICE == "cuda":
        try:
            pipe.enable_model_cpu_offload()
        except (ImportError, Exception) as e:
            logger.warning("CPU offload not available (%s), loading to GPU directly", e)
            pipe = pipe.to(DEVICE)
    logger.info("SVD pipeline loaded.")

    for pair_name, orig_img, prot_img in pairs:
        pair_dir = os.path.join(svd_dir, pair_name)
        os.makedirs(pair_dir, exist_ok=True)

        # SVD expects specific input sizes
        svd_size = (1024, 576)  # SVD's native resolution

        orig_resized = orig_img.resize(svd_size, Image.LANCZOS)
        prot_resized = prot_img.resize(svd_size, Image.LANCZOS)

        # Save reference images
        orig_resized.save(os.path.join(pair_dir, "0_ref_original.png"))
        prot_resized.save(os.path.join(pair_dir, "0_ref_protected.png"))

        # Generate video from original
        logger.info("  SVD [%s] generating from original...", pair_name)
        generator = torch.Generator(device=DEVICE).manual_seed(args.seed)
        t0 = time.time()
        frames_orig = pipe(
            orig_resized,
            num_frames=args.svd_num_frames,
            decode_chunk_size=8,
            generator=generator,
        ).frames[0]
        logger.info("    Original video: %.1fs, %d frames", time.time() - t0, len(frames_orig))

        # Save frames and video
        frames_orig_dir = os.path.join(pair_dir, "frames_original")
        os.makedirs(frames_orig_dir, exist_ok=True)
        for i, frame in enumerate(frames_orig):
            frame.save(os.path.join(frames_orig_dir, f"frame_{i:04d}.png"))

        video_orig_path = os.path.join(pair_dir, "video_original.mp4")
        export_to_video(frames_orig, video_orig_path, fps=args.svd_fps)

        # Generate video from protected
        logger.info("  SVD [%s] generating from protected...", pair_name)
        generator = torch.Generator(device=DEVICE).manual_seed(args.seed)
        t0 = time.time()
        frames_prot = pipe(
            prot_resized,
            num_frames=args.svd_num_frames,
            decode_chunk_size=8,
            generator=generator,
        ).frames[0]
        logger.info("    Protected video: %.1fs, %d frames", time.time() - t0, len(frames_prot))

        # Save frames and video
        frames_prot_dir = os.path.join(pair_dir, "frames_protected")
        os.makedirs(frames_prot_dir, exist_ok=True)
        for i, frame in enumerate(frames_prot):
            frame.save(os.path.join(frames_prot_dir, f"frame_{i:04d}.png"))

        video_prot_path = os.path.join(pair_dir, "video_protected.mp4")
        export_to_video(frames_prot, video_prot_path, fps=args.svd_fps)

        # Create side-by-side frame comparisons (sample a few frames)
        sample_indices = np.linspace(0, len(frames_orig) - 1, min(6, len(frames_orig)), dtype=int)
        for idx in sample_indices:
            comp = Image.new("RGB", (1024 * 2 + 10, 576), (255, 255, 255))
            comp.paste(frames_orig[idx].resize((1024, 576), Image.LANCZOS), (0, 0))
            comp.paste(frames_prot[idx].resize((1024, 576), Image.LANCZOS), (1024 + 10, 0))
            comp.save(os.path.join(pair_dir, f"compare_frame_{idx:04d}.png"))

        logger.info("  SVD [%s] done. Videos: %s, %s", pair_name, video_orig_path, video_prot_path)

    # Cleanup
    del pipe
    torch.cuda.empty_cache() if DEVICE == "cuda" else None
    logger.info("SVD demo complete.")


# ===================================================================
# MAIN
# ===================================================================

def main():
    args = parse_args()

    images_dir = os.path.abspath(args.images_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("CONDITIONED GENERATION DEMO")
    logger.info("  Images dir: %s", images_dir)
    logger.info("  Output dir: %s", output_dir)
    logger.info("  Device: %s (%s)", DEVICE, DTYPE)
    logger.info("  IP-Adapter: %s", "SKIP" if args.skip_ip_adapter else "ON")
    logger.info("  SVD Video:  %s", "SKIP" if args.skip_svd else "ON")
    logger.info("=" * 60)

    # Load image pairs
    pairs = load_image_pair(images_dir)
    if not pairs:
        logger.error("No original/protected image pairs found in %s", images_dir)
        sys.exit(1)

    logger.info("Found %d image pairs:", len(pairs))
    for name, orig, prot in pairs:
        logger.info("  %s: orig=%s, prot=%s", name, orig.size, prot.size)

    # Limit to first 2 pairs for faster demo (can be overridden)
    if len(pairs) > 2:
        logger.info("Using first 2 pairs for demo (edit script to use all)")
        pairs = pairs[:2]

    # Part 1: IP-Adapter
    if not args.skip_ip_adapter:
        logger.info("=" * 60)
        logger.info("PART 1: IP-ADAPTER — Image-conditioned text-to-image")
        logger.info("=" * 60)
        try:
            run_ip_adapter_demo(pairs, output_dir, args)
        except Exception as e:
            logger.error("IP-Adapter demo failed: %s", e)
            import traceback
            traceback.print_exc()
            logger.info("Hint: You may need diffusers >= 0.25.0 for IP-Adapter support.")
            logger.info("  pip install --upgrade diffusers transformers")

    # Part 2: Stable Video Diffusion
    if not args.skip_svd:
        logger.info("=" * 60)
        logger.info("PART 2: STABLE VIDEO DIFFUSION — Image-to-video")
        logger.info("=" * 60)
        try:
            run_svd_demo(pairs, output_dir, args)
        except Exception as e:
            logger.error("SVD demo failed: %s", e)
            import traceback
            traceback.print_exc()
            logger.info("Hint: You may need diffusers >= 0.25.0 for SVD support.")
            logger.info("  pip install --upgrade diffusers transformers")

    # Summary
    logger.info("=" * 60)
    logger.info("DEMO COMPLETE. Output: %s", output_dir)
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, "").count(os.sep)
        indent = "  " * level
        logger.info("%s%s/ (%d files)", indent, os.path.basename(root), len(files))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
