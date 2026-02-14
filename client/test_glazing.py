#!/usr/bin/env python3
"""
DiffusionGuard Testing Script — Prove That Glazing Works
=========================================================
This script runs a full end-to-end test:

1. Sends the ORIGINAL image to the server for DiffusionGuard protection (glazing).
2. Sends BOTH the original AND protected image to the server's inpainting endpoint.
3. Saves all four images side by side for visual comparison:
   - original.png         : The source image
   - protected.png        : The glazed image (should look nearly identical to original)
   - inpaint_original.png : Inpainting result on UNPROTECTED image (should succeed - bad!)
   - inpaint_protected.png: Inpainting result on PROTECTED image (should fail - good!)

If DiffusionGuard works, inpaint_protected.png should be clearly degraded/broken
compared to inpaint_original.png, proving the image is defended against malicious editing.

Usage:
    python client/test_glazing.py --image test_images/face.png --mask test_images/face_mask.png
    python client/test_glazing.py --image face.png --mask mask.png --prompt "a person being arrested"
    python client/test_glazing.py --image face.png --mask mask.png --server http://192.168.1.42:5000
"""

import argparse
import io
import os
import sys
import time

import requests
from PIL import Image, ImageDraw, ImageFont


DEFAULT_SERVER = os.environ.get("DIFFGUARD_SERVER", "http://localhost:5000")


def check_health(server: str):
    """Verify the server is up."""
    try:
        resp = requests.get(f"{server}/health", timeout=5)
        resp.raise_for_status()
        info = resp.json()
        print(f"Server OK — GPU: {info.get('gpu', '?')}, Memory: {info.get('gpu_memory', '?')}")
        return info
    except Exception as e:
        print(f"ERROR: Cannot reach server at {server}: {e}")
        sys.exit(1)


def protect(server: str, image_path: str, mask_path: str, iters: int) -> Image.Image:
    """Send image + mask to /protect and return the protected PIL image."""
    with open(image_path, "rb") as img_f, open(mask_path, "rb") as mask_f:
        resp = requests.post(
            f"{server}/protect",
            files={"image": img_f, "mask": mask_f},
            params={"iters": iters},
            timeout=600,
        )
    if resp.status_code != 200:
        print(f"ERROR: /protect returned {resp.status_code}: {resp.text[:300]}")
        sys.exit(1)
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def inpaint(server: str, image: Image.Image, mask_path: str, prompt: str, seed: int = 42) -> Image.Image:
    """Send an image + mask to /test-inpaint and return the inpainted PIL image."""
    img_buf = io.BytesIO()
    image.save(img_buf, format="PNG")
    img_buf.seek(0)

    with open(mask_path, "rb") as mask_f:
        resp = requests.post(
            f"{server}/test-inpaint",
            files={"image": ("image.png", img_buf, "image/png"), "mask": mask_f},
            params={"prompt": prompt, "seed": seed},
            timeout=300,
        )
    if resp.status_code != 200:
        print(f"ERROR: /test-inpaint returned {resp.status_code}: {resp.text[:300]}")
        sys.exit(1)
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def add_label(image: Image.Image, text: str) -> Image.Image:
    """Add a text label at the top of an image."""
    w, h = image.size
    label_h = 32
    canvas = Image.new("RGB", (w, h + label_h), (30, 30, 30))
    canvas.paste(image, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((w - tw) // 2, 6), text, fill=(255, 255, 255), font=font)
    return canvas


def make_comparison_grid(images: list, labels: list, output_path: str):
    """Create a 2x2 comparison grid from 4 images with labels."""
    labeled = [add_label(img, lbl) for img, lbl in zip(images, labels)]

    w, h = labeled[0].size
    grid = Image.new("RGB", (w * 2 + 4, h * 2 + 4), (60, 60, 60))
    grid.paste(labeled[0], (0, 0))
    grid.paste(labeled[1], (w + 4, 0))
    grid.paste(labeled[2], (0, h + 4))
    grid.paste(labeled[3], (w + 4, h + 4))
    grid.save(output_path)
    return grid


def compute_pixel_diff(img_a: Image.Image, img_b: Image.Image) -> float:
    """Compute mean absolute pixel difference between two images."""
    import numpy as np
    a = np.array(img_a.resize((512, 512))).astype(float)
    b = np.array(img_b.resize((512, 512))).astype(float)
    return float(np.mean(np.abs(a - b)))


def main():
    parser = argparse.ArgumentParser(description="Test DiffusionGuard protection effectiveness")
    parser.add_argument("--image", required=True, help="Path to the source image")
    parser.add_argument("--mask", required=True, help="Path to the mask image")
    parser.add_argument("--server", default=DEFAULT_SERVER, help=f"Server URL (default: {DEFAULT_SERVER})")
    parser.add_argument("--prompt", default="a person in a hospital", help="Inpainting prompt")
    parser.add_argument("--iters", type=int, default=200, help="PGD iterations for protection")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for inpainting")
    parser.add_argument("--output-dir", default="test_results", help="Output directory")
    args = parser.parse_args()

    # Validate inputs
    for path in [args.image, args.mask]:
        if not os.path.isfile(path):
            print(f"ERROR: File not found: {path}")
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 0: Health check
    # ------------------------------------------------------------------
    print("=" * 60)
    print("DiffusionGuard Protection Test")
    print("=" * 60)
    check_health(args.server)
    print()

    # ------------------------------------------------------------------
    # Step 1: Load original image
    # ------------------------------------------------------------------
    original = Image.open(args.image).convert("RGB").resize((512, 512))
    original.save(os.path.join(args.output_dir, "original.png"))
    print(f"[1/4] Original image: {args.image} ({original.size})")

    # ------------------------------------------------------------------
    # Step 2: Protect the image (glaze it)
    # ------------------------------------------------------------------
    print(f"[2/4] Protecting image ({args.iters} PGD iterations)...")
    t0 = time.time()
    protected = protect(args.server, args.image, args.mask, args.iters)
    protect_time = time.time() - t0
    protected.save(os.path.join(args.output_dir, "protected.png"))
    print(f"       Done in {protect_time:.1f}s")

    # Measure how different the protected image looks from the original
    diff_orig_prot = compute_pixel_diff(original, protected)
    print(f"       Pixel diff (original vs protected): {diff_orig_prot:.2f}")
    print(f"       (should be small — protection should be visually imperceptible)")

    # ------------------------------------------------------------------
    # Step 3: Inpaint the ORIGINAL (unprotected) image
    # ------------------------------------------------------------------
    print(f"[3/4] Inpainting ORIGINAL image with prompt: \"{args.prompt}\"...")
    t0 = time.time()
    inpainted_original = inpaint(args.server, original, args.mask, args.prompt, args.seed)
    inpaint_orig_time = time.time() - t0
    inpainted_original.save(os.path.join(args.output_dir, "inpaint_original.png"))
    print(f"       Done in {inpaint_orig_time:.1f}s")

    # ------------------------------------------------------------------
    # Step 4: Inpaint the PROTECTED (glazed) image
    # ------------------------------------------------------------------
    print(f"[4/4] Inpainting PROTECTED image with prompt: \"{args.prompt}\"...")
    t0 = time.time()
    inpainted_protected = inpaint(args.server, protected, args.mask, args.prompt, args.seed)
    inpaint_prot_time = time.time() - t0
    inpainted_protected.save(os.path.join(args.output_dir, "inpaint_protected.png"))
    print(f"       Done in {inpaint_prot_time:.1f}s")

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    print()
    print("-" * 60)
    print("RESULTS")
    print("-" * 60)

    # Compare inpainting quality: how much did inpainting change the image?
    diff_inpaint_orig = compute_pixel_diff(original, inpainted_original)
    diff_inpaint_prot = compute_pixel_diff(protected, inpainted_protected)
    diff_between_inpaints = compute_pixel_diff(inpainted_original, inpainted_protected)

    print(f"  Pixel diff (original vs inpaint_original):     {diff_inpaint_orig:.2f}")
    print(f"  Pixel diff (protected vs inpaint_protected):   {diff_inpaint_prot:.2f}")
    print(f"  Pixel diff (inpaint_original vs inpaint_prot): {diff_between_inpaints:.2f}")
    print()

    if diff_between_inpaints > 20:
        print("  RESULT: Protection EFFECTIVE!")
        print("  The inpainting results differ significantly, meaning DiffusionGuard")
        print("  successfully disrupted the diffusion model's ability to edit the image.")
    elif diff_between_inpaints > 10:
        print("  RESULT: Protection PARTIALLY effective.")
        print("  Some disruption detected. Try increasing --iters for stronger protection.")
    else:
        print("  RESULT: Protection may be WEAK.")
        print("  Try increasing --iters (e.g., 400 or 800) for stronger protection.")

    # ------------------------------------------------------------------
    # Generate comparison grid
    # ------------------------------------------------------------------
    grid_path = os.path.join(args.output_dir, "comparison_grid.png")
    make_comparison_grid(
        [original, protected, inpainted_original, inpainted_protected],
        ["Original", "Protected (Glazed)", "Inpaint: Original", "Inpaint: Protected"],
        grid_path,
    )
    print()
    print(f"  Comparison grid saved to: {grid_path}")
    print()
    print("  Output files:")
    for fname in ["original.png", "protected.png", "inpaint_original.png",
                   "inpaint_protected.png", "comparison_grid.png"]:
        fpath = os.path.join(args.output_dir, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {fpath} ({size_kb:.0f} KB)")

    print()
    print("Open comparison_grid.png to visually compare all four images side by side.")


if __name__ == "__main__":
    main()
