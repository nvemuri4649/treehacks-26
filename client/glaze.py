#!/usr/bin/env python3
"""
Cena Encryption Client — Glaze Images via Any GPU Backend
=========================================================
Sends an image + mask to the Cena encryption server and saves the
protected (glazed) image locally. Supports multiple GPU backends.

Usage:
    # Use default backend (from backends.json)
    python client/glaze.py --image photo.png --mask mask.png

    # Use a named backend
    python client/glaze.py --image photo.png --mask mask.png --backend gx10
    python client/glaze.py --image photo.png --mask mask.png --backend runpod
    python client/glaze.py --image photo.png --mask mask.png --backend local

    # Use a direct URL (overrides backend name)
    python client/glaze.py --image photo.png --mask mask.png --server http://192.168.1.42:5000

    # RunPod with pod ID from env
    RUNPOD_POD_ID=abc123 python client/glaze.py --image photo.png --mask mask.png --backend runpod
"""

import argparse
import os
import sys
import time

import requests
from PIL import Image

from backends import add_backend_args, get_server_url, print_backend_info


def check_health(server: str) -> dict:
    """Check if the server is reachable and return status info."""
    try:
        resp = requests.get(f"{server}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        print(f"ERROR: Cannot connect to server at {server}")
        print("Possible fixes:")
        print("  - Check that the GPU machine is on and the server is running")
        print("  - Verify the URL in backends.json or --server flag")
        print("  - For RunPod, make sure the pod is running and port 5000 is exposed")
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
        print(f"  Uploading to {url} ({iters} PGD iterations)...")
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
    print(f"  Protection complete in {elapsed:.1f}s ({len(resp.content)} bytes)")
    return resp.content


def main():
    parser = argparse.ArgumentParser(
        description="Protect images using Cena encryption on a remote GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python client/glaze.py --image photo.png --mask mask.png --backend gx10
  python client/glaze.py --image photo.png --mask mask.png --backend runpod
  python client/glaze.py --image photo.png --mask mask.png --server http://my-gpu:5000
        """,
    )
    parser.add_argument("--image", required=True, help="Path to the source image")
    parser.add_argument("--mask", required=True, help="Path to the mask image (white=sensitive region)")
    parser.add_argument("--output", default=None, help="Output path (default: <image>_protected.png)")
    parser.add_argument("--iters", type=int, default=200, help="PGD iterations (default: 200)")
    add_backend_args(parser)
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

    # Resolve backend
    server = get_server_url(args)
    print(f"Connecting to GPU backend...")
    print_backend_info(server, args.backend)

    # Health check
    info = check_health(server)
    print(f"  Server OK — GPU: {info.get('gpu', '?')}, Memory: {info.get('gpu_memory', '?')}")

    # Protect
    protected_bytes = protect_image(server, args.image, args.mask, args.iters)

    # Save
    with open(args.output, "wb") as f:
        f.write(protected_bytes)
    print(f"  Saved: {args.output}")

    # Quick visual sanity check
    orig = Image.open(args.image).convert("RGB")
    prot = Image.open(args.output).convert("RGB")
    print(f"  Original size: {orig.size}, Protected size: {prot.size}")


if __name__ == "__main__":
    main()
