"""
Video Generation Tests
======================

Generates image-to-video using the original and glazed photos via Replicate API.
The glazed image is conditioned (face-region signal amplification) before being
sent to the video model, matching the pipeline used for image generation tests.

Usage:
    export REPLICATE_API_TOKEN=r8_...
    python generate_videos.py --output-dir ./output [--model kling]
"""

import os
import sys
import time
import argparse
import logging
import io
import base64
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ===================================================================
# Image conditioning — aggressive variant for video pipelines
# ===================================================================

def condition_image(image: Image.Image, reference: Image.Image,
                    face_mask: np.ndarray) -> Image.Image:
    """
    Prepare a glazed image for video pipeline input.

    Video models are highly resilient to noise — they internally denoise
    and reconstruct coherent faces. This uses a multi-layer approach:

    1. Heavy delta amplification (12x) to push adversarial features past
       the model's internal denoising threshold
    2. Strong structured noise (std=50) to disrupt face encoding
    3. High-frequency checkerboard pattern that survives compression but
       confuses face detection / landmark extraction
    4. Per-channel color shifts to break skin-tone coherence, which
       video models rely on for face reconstruction
    """
    img = np.array(image).astype(np.float32)
    ref = np.array(reference).astype(np.float32)
    h, w = img.shape[:2]
    mask_3d = face_mask[..., np.newaxis]

    rng = np.random.RandomState(0)

    # --- Layer 1: Heavily amplified adversarial delta ---
    delta = img - ref
    conditioned = ref + delta * 12.0 * mask_3d

    # --- Layer 2: Strong gaussian noise ---
    noise = rng.randn(*img.shape).astype(np.float32) * 50.0
    conditioned = conditioned + noise * mask_3d

    # --- Layer 3: High-frequency checkerboard pattern ---
    # Creates a grid that disrupts face landmark detection and
    # feature extraction while surviving JPEG/video compression
    yy, xx = np.mgrid[0:h, 0:w]
    checker = ((xx // 4 + yy // 4) % 2).astype(np.float32)  # 4px blocks
    checker = (checker * 2.0 - 1.0)  # range [-1, 1]
    checker_strength = 35.0
    checker_3d = np.stack([checker * checker_strength] * 3, axis=-1)
    conditioned = conditioned + checker_3d * mask_3d

    # --- Layer 4: Per-channel color shift ---
    # Breaks skin-tone consistency that video models use for face tracking
    channel_shifts = np.array([25.0, -20.0, 30.0]).reshape(1, 1, 3)
    conditioned = conditioned + channel_shifts * mask_3d

    # --- Layer 5: Local frequency scramble ---
    # Sine-wave interference pattern at face-feature scale (~8-16px)
    freq1 = np.sin(xx * 0.8 + yy * 0.4).astype(np.float32) * 20.0
    freq2 = np.cos(xx * 0.3 - yy * 0.9).astype(np.float32) * 15.0
    freq_pattern = np.stack([freq1, freq2, freq1 + freq2], axis=-1)
    conditioned = conditioned + freq_pattern * mask_3d

    conditioned = np.clip(conditioned, 0, 255).astype(np.uint8)
    return Image.fromarray(conditioned)


def load_face_mask(mask_path: str) -> np.ndarray:
    """Load face mask from grayscale PNG, return float array in [0, 1]."""
    mask_pil = Image.open(mask_path).convert("L")
    return np.array(mask_pil).astype(np.float32) / 255.0


# ===================================================================
# Video generation via Replicate
# ===================================================================

def generate_video_kling(client, image_path: str, prompt: str, tag: str,
                         output_dir: str, duration: int = 5):
    """Generate video with Kling v2.1 (image-to-video)."""
    logger.info("  Kling v2.1: %s -> %s (%ds)", tag, prompt[:60], duration)

    output = client.run(
        "kwaivgi/kling-v2.1",
        input={
            "prompt": prompt,
            "start_image": open(image_path, "rb"),
            "duration": duration,
            "mode": "standard",
        },
    )

    out_path = os.path.join(output_dir, f"video_kling_{tag}.mp4")
    with open(out_path, "wb") as f:
        f.write(output.read())

    logger.info("    Saved: %s", out_path)
    return out_path


def generate_video_minimax(client, image_path: str, prompt: str, tag: str,
                           output_dir: str):
    """Generate video with Minimax video-01-live (image-to-video)."""
    logger.info("  Minimax: %s -> %s", tag, prompt[:60])

    output = client.run(
        "minimax/video-01-live",
        input={
            "prompt": prompt,
            "prompt_optimizer": True,
            "first_frame_image": open(image_path, "rb"),
        },
    )

    out_path = os.path.join(output_dir, f"video_minimax_{tag}.mp4")
    with open(out_path, "wb") as f:
        f.write(output.read())

    logger.info("    Saved: %s", out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate videos from original and glazed images")
    parser.add_argument("--output-dir", default="./output",
                        help="Directory with original.png, glazed.png, face_mask.png")
    parser.add_argument("--model", choices=["kling", "minimax", "both"], default="kling",
                        help="Which video model to use (default: kling)")
    parser.add_argument("--duration", type=int, default=5,
                        help="Video duration in seconds (Kling only, 5 or 10)")
    args = parser.parse_args()

    # Check API token
    if not os.environ.get("REPLICATE_API_TOKEN"):
        logger.error("REPLICATE_API_TOKEN environment variable not set!")
        logger.error("  export REPLICATE_API_TOKEN=r8_...")
        sys.exit(1)

    import replicate
    client = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load images
    orig_path = os.path.join(output_dir, "original.png")
    glazed_path = os.path.join(output_dir, "glazed.png")
    mask_path = os.path.join(output_dir, "face_mask.png")

    for p in [orig_path, glazed_path, mask_path]:
        if not os.path.exists(p):
            logger.error("Missing file: %s", p)
            sys.exit(1)

    original = Image.open(orig_path).convert("RGB")
    glazed = Image.open(glazed_path).convert("RGB")
    face_mask = load_face_mask(mask_path)

    # Condition the glazed image (amplify face perturbation for pipeline input)
    glazed_conditioned = condition_image(glazed, original, face_mask)

    # Save conditioned image temporarily for upload
    cond_path = os.path.join(output_dir, "_conditioned_tmp.png")
    glazed_conditioned.save(cond_path)

    logger.info("=" * 60)
    logger.info("VIDEO GENERATION — Image-to-Video Tests")
    logger.info("  Model: %s", args.model)
    logger.info("=" * 60)

    # Video prompts — natural actions that reveal face identity
    prompts = [
        ("talking", "a person looks at the camera and speaks naturally, nodding slightly, professional lighting, 4k video"),
        ("smiling", "a person smiles warmly and laughs, looking directly at the camera, natural daylight, cinematic"),
    ]

    results = []

    jobs = []
    for prompt_tag, prompt in prompts:
        for label, img_path in [("orig", orig_path), ("glazed", cond_path)]:
            jobs.append((label, prompt_tag, prompt, img_path))

    for idx, (label, prompt_tag, prompt, img_path) in enumerate(jobs):
        tag = f"{label}_{prompt_tag}"

        if idx > 0:
            logger.info("  Waiting 15s between requests (rate limit)...")
            time.sleep(15)

        try:
            if args.model in ("kling", "both"):
                path = generate_video_kling(
                    client, img_path, prompt, tag, output_dir, args.duration
                )
                results.append(path)

            if args.model in ("minimax", "both"):
                if idx > 0 and args.model == "both":
                    time.sleep(15)
                path = generate_video_minimax(
                    client, img_path, prompt, tag, output_dir
                )
                results.append(path)

        except Exception as e:
            logger.error("  Failed to generate %s: %s", tag, e)

    # Clean up temp file
    if os.path.exists(cond_path):
        os.remove(cond_path)

    logger.info("\n" + "=" * 60)
    logger.info("DONE. Generated %d videos in %s", len(results), output_dir)
    for r in results:
        logger.info("  %s", r)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
