#!/usr/bin/env python3
"""
Generate a simple elliptical mask for testing.

Usage:
    python client/make_test_mask.py --output test_images/face_mask.png
    python client/make_test_mask.py --output mask.png --cx 256 --cy 200 --rx 120 --ry 150
"""

import argparse
import os

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description="Generate an elliptical mask for testing")
    parser.add_argument("--output", default="test_images/face_mask.png", help="Output mask path")
    parser.add_argument("--size", type=int, default=512, help="Image size (square)")
    parser.add_argument("--cx", type=int, default=256, help="Ellipse center X")
    parser.add_argument("--cy", type=int, default=200, help="Ellipse center Y")
    parser.add_argument("--rx", type=int, default=110, help="Ellipse radius X")
    parser.add_argument("--ry", type=int, default=140, help="Ellipse radius Y")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Create black image, draw white ellipse (white = sensitive region to protect)
    mask = Image.new("RGB", (args.size, args.size), (0, 0, 0))
    draw = ImageDraw.Draw(mask)
    bbox = [args.cx - args.rx, args.cy - args.ry, args.cx + args.rx, args.cy + args.ry]
    draw.ellipse(bbox, fill=(255, 255, 255))

    mask.save(args.output)
    print(f"Saved mask to {args.output} (ellipse at ({args.cx},{args.cy}), r=({args.rx},{args.ry}))")


if __name__ == "__main__":
    main()
