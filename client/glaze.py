#!/usr/bin/env python3
"""
DiffusionGuard Client — Glaze Images via the GX10
==================================================
Sends an image + mask to the DiffusionGuard API server running on the
ASUS Ascent GX10 and saves the protected (glazed) image locally.

Usage:
    python client/glaze.py --image photo.png --mask mask.png
    python client/glaze.py --image photo.png --mask mask.png --server http://192.168.1.42:5000
    python client/glaze.py --image photo.png --mask mask.png --iters 400 --output glazed.png
"""

import argparse
import os
import sys
import time

import requests
from PIL import Image


DEFAULT_SERVER = os.environ.get("DIFFGUARD_SERVER", "http://localhost:5000")


def check_health(server: str) -> dict:
    """Check if the server is reachable and return status info."""
    try:
        resp = requests.get(f"{server}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        print(f"ERROR: Cannot connect to server at {server}")
        print("Make sure the GX10 is on and the server is running.")
        print("  ssh <user>@<gx10> 'docker exec diffguard bash -c \"cd /workspace/project/server && python app.py\"'")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Health check failed: {e}")
        sys.exit(1)


def protect_image(server: str, image_path: str, mask_path: str, iters: int) -> bytes:
    """Send image + mask to the server and return the protected image bytes."""
    url = f"{server}/protect"
    params = {"iters": iters}

    with open(image_path, "rb") as img_f, open(mask_path, "rb") as mask_f:
        files = {
            "image": (os.path.basename(image_path), img_f, "image/png"),
            "mask": (os.path.basename(mask_path), mask_f, "image/png"),
        }
        print(f"Uploading image and mask to {url} ({iters} PGD iterations)...")
        t0 = time.time()
        resp = requests.post(url, files=files, params=params, timeout=600)

    if resp.status_code != 200:
        print(f"ERROR: Server returned {resp.status_code}")
        try:
            print(resp.json())
        except Exception:
            print(resp.text[:500])
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"Protection complete in {elapsed:.1f}s ({len(resp.content)} bytes)")
    return resp.content


def main():
    parser = argparse.ArgumentParser(description="Protect images using DiffusionGuard on GX10")
    parser.add_argument("--image", required=True, help="Path to the source image")
    parser.add_argument("--mask", required=True, help="Path to the mask image (white=sensitive region)")
    parser.add_argument("--output", default=None, help="Output path (default: <image>_protected.png)")
    parser.add_argument("--server", default=DEFAULT_SERVER, help=f"Server URL (default: {DEFAULT_SERVER})")
    parser.add_argument("--iters", type=int, default=200, help="PGD iterations (default: 200)")
    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.image):
        print(f"ERROR: Image not found: {args.image}")
        sys.exit(1)
    if not os.path.isfile(args.mask):
        print(f"ERROR: Mask not found: {args.mask}")
        sys.exit(1)

    # Default output path
    if args.output is None:
        base, ext = os.path.splitext(args.image)
        args.output = f"{base}_protected.png"

    # Health check
    print(f"Checking server at {args.server}...")
    info = check_health(args.server)
    print(f"  Server OK — GPU: {info.get('gpu', '?')}, Memory: {info.get('gpu_memory', '?')}")

    # Protect
    protected_bytes = protect_image(args.server, args.image, args.mask, args.iters)

    # Save
    with open(args.output, "wb") as f:
        f.write(protected_bytes)
    print(f"Saved protected image to: {args.output}")

    # Quick visual sanity check
    orig = Image.open(args.image).convert("RGB")
    prot = Image.open(args.output).convert("RGB")
    print(f"  Original size: {orig.size}")
    print(f"  Protected size: {prot.size}")


if __name__ == "__main__":
    main()
