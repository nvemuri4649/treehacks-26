#!/usr/bin/env python3
"""
Adversarial Agent Loop for DiffusionGuard Testing
==================================================
This script implements an adversarial testing loop to find the optimal glazing
strength needed to defeat deepfakes:

1. GLAZE: Apply DiffusionGuard protection to the image
2. DEEPFAKE: Generate a deepfake from the protected image
3. RATE: Use Claude to evaluate how convincing the deepfake is (1-10 scale)
4. ITERATE: If score > threshold, increase glazing strength and repeat

The loop continues until the deepfake quality drops below the threshold,
indicating that the glazing is strong enough to effectively protect the image.

Usage:
    # Basic usage with default backend
    python client/agent_loop.py \
        --image face.png \
        --mask mask.png \
        --prompt "a person in a hospital" \
        --threshold 6

    # With specific backend
    python client/agent_loop.py \
        --image face.png \
        --mask mask.png \
        --prompt "a person being arrested" \
        --threshold 6 \
        --backend gx10

    # Custom iteration schedule
    python client/agent_loop.py \
        --image face.png \
        --mask mask.png \
        --prompt "a person in jail" \
        --threshold 5 \
        --start-iters 100 \
        --iter-increment 100 \
        --max-iters 1000

Environment variables:
    ANTHROPIC_API_KEY    Required for Claude rater agent
    DIFFGUARD_BACKEND    Default backend name (can override with --backend)
"""

import argparse
import io
import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont

from backends import add_backend_args, get_server_url, print_backend_info
from rater_agent import rate_deepfake


# ============================================================================
# Server Communication
# ============================================================================

def check_server_health(server: str) -> dict:
    """Verify the GPU server is reachable and return status info."""
    try:
        resp = requests.get(f"{server}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"ERROR: Cannot reach server at {server}: {e}")
        sys.exit(1)


def protect_image(
    server: str,
    image_path: str,
    mask_path: str,
    iters: int,
) -> Tuple[Image.Image, float]:
    """
    Send image + mask to /protect and return the protected image + time taken.

    Returns:
        (protected_image, elapsed_time)
    """
    with open(image_path, "rb") as img_f, open(mask_path, "rb") as mask_f:
        t0 = time.time()
        resp = requests.post(
            f"{server}/protect",
            files={"image": img_f, "mask": mask_f},
            params={"iters": iters},
            timeout=900,  # 15 min max for heavy glazing
        )
        elapsed = time.time() - t0

    if resp.status_code != 200:
        print(f"ERROR: /protect returned {resp.status_code}: {resp.text[:300]}")
        sys.exit(1)

    protected = Image.open(io.BytesIO(resp.content)).convert("RGB")
    return protected, elapsed


def generate_deepfake(
    server: str,
    image: Image.Image,
    mask_path: str,
    prompt: str,
    seed: int = 42,
) -> Tuple[Image.Image, float]:
    """
    Send a protected image to /test-inpaint to generate a deepfake.

    Returns:
        (deepfake_image, elapsed_time)
    """
    img_buf = io.BytesIO()
    image.save(img_buf, format="PNG")
    img_buf.seek(0)

    with open(mask_path, "rb") as mask_f:
        t0 = time.time()
        resp = requests.post(
            f"{server}/test-inpaint",
            files={"image": ("image.png", img_buf, "image/png"), "mask": mask_f},
            params={"prompt": prompt, "seed": seed},
            timeout=600,
        )
        elapsed = time.time() - t0

    if resp.status_code != 200:
        print(f"ERROR: /test-inpaint returned {resp.status_code}: {resp.text[:300]}")
        sys.exit(1)

    deepfake = Image.open(io.BytesIO(resp.content)).convert("RGB")
    return deepfake, elapsed


# ============================================================================
# Visualization
# ============================================================================

def create_iteration_summary(
    iteration: int,
    original: Image.Image,
    protected: Image.Image,
    deepfake: Image.Image,
    score: int,
    iters: int,
    reasoning: str,
    output_path: str,
):
    """Create a visual summary of one iteration with all images and metrics."""
    # Resize all to same size
    size = 384
    imgs = [img.resize((size, size), Image.LANCZOS) for img in [original, protected, deepfake]]

    # Create canvas
    header_h = 60
    footer_h = 120
    padding = 10
    canvas_w = size * 3 + padding * 4
    canvas_h = size + header_h + footer_h + padding * 2

    canvas = Image.new("RGB", (canvas_w, canvas_h), (40, 40, 40))
    draw = ImageDraw.Draw(canvas)

    # Load font
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        text_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except (IOError, OSError):
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except (IOError, OSError):
            title_font = label_font = text_font = ImageFont.load_default()

    # Header
    title = f"Iteration {iteration} — {iters} PGD Iterations — Score: {score}/10"
    draw.text((padding, 15), title, fill=(255, 255, 255), font=title_font)

    # Images
    labels = ["Original", f"Protected ({iters} iters)", f"Deepfake (Score: {score}/10)"]
    y_pos = header_h + padding

    for i, (img, label) in enumerate(zip(imgs, labels)):
        x_pos = padding + i * (size + padding)

        # Label
        draw.text((x_pos, y_pos - 20), label, fill=(200, 200, 200), font=label_font)

        # Image
        canvas.paste(img, (x_pos, y_pos))

        # Color-coded border based on position
        color = [(100, 100, 255), (255, 200, 100), (255, 100, 100)][i]
        draw.rectangle([x_pos - 2, y_pos - 2, x_pos + size + 2, y_pos + size + 2], outline=color, width=2)

    # Footer with reasoning
    footer_y = y_pos + size + padding + 10
    draw.text((padding, footer_y), "Claude's Assessment:", fill=(200, 200, 200), font=label_font)

    # Wrap reasoning text
    max_chars = 110
    reasoning_short = reasoning[:300] if len(reasoning) > 300 else reasoning
    if len(reasoning_short) > max_chars:
        line1 = reasoning_short[:max_chars]
        line2 = reasoning_short[max_chars:max_chars*2]
        draw.text((padding, footer_y + 25), line1, fill=(255, 255, 255), font=text_font)
        draw.text((padding, footer_y + 45), line2, fill=(255, 255, 255), font=text_font)
    else:
        draw.text((padding, footer_y + 25), reasoning_short, fill=(255, 255, 255), font=text_font)

    # Score indicator
    score_color = (100, 255, 100) if score <= 6 else (255, 200, 100) if score <= 8 else (255, 100, 100)
    draw.text((padding, footer_y + 70), f"Protection: {'EFFECTIVE' if score <= 6 else 'PARTIAL' if score <= 8 else 'WEAK'}",
              fill=score_color, font=label_font)

    canvas.save(output_path)
    return canvas


# ============================================================================
# Main Loop
# ============================================================================

def run_agent_loop(
    server: str,
    image_path: str,
    mask_path: str,
    prompt: str,
    threshold: int,
    start_iters: int,
    iter_increment: int,
    max_iters: int,
    max_iterations: int,
    output_dir: str,
    anthropic_api_key: Optional[str] = None,
    seed: int = 42,
):
    """
    Run the adversarial agent loop.

    Args:
        server: GPU server URL
        image_path: Path to original image
        mask_path: Path to mask
        prompt: Inpainting prompt for deepfake generation
        threshold: Target score threshold (loop until score <= threshold)
        start_iters: Starting PGD iterations
        iter_increment: How much to increase iters each loop
        max_iters: Maximum PGD iterations to try
        max_iterations: Maximum number of loop iterations
        output_dir: Directory to save results
        anthropic_api_key: API key for Claude rater
        seed: Random seed for deepfake generation
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load original image
    original = Image.open(image_path).convert("RGB")
    original_resized = original.resize((512, 512), Image.LANCZOS)
    original_resized.save(os.path.join(output_dir, "original.png"))

    # Tracking
    results = []
    current_iters = start_iters
    iteration = 1

    print("=" * 80)
    print("ADVERSARIAL GLAZING AGENT LOOP")
    print("=" * 80)
    print(f"  Target threshold: {threshold}/10")
    print(f"  Starting PGD iterations: {start_iters}")
    print(f"  Iteration increment: +{iter_increment}")
    print(f"  Max PGD iterations: {max_iters}")
    print(f"  Max loop iterations: {max_iterations}")
    print(f"  Deepfake prompt: \"{prompt}\"")
    print(f"  Output directory: {output_dir}")
    print("=" * 80)
    print()

    while iteration <= max_iterations:
        print(f"{'='*80}")
        print(f"ITERATION {iteration} — Testing with {current_iters} PGD iterations")
        print(f"{'='*80}")

        # Step 1: Glaze the image
        print(f"[1/3] Glazing image ({current_iters} iterations)...")
        protected, glaze_time = protect_image(server, image_path, mask_path, current_iters)
        protected_path = os.path.join(output_dir, f"iter{iteration:02d}_protected_{current_iters}iters.png")
        protected.save(protected_path)
        print(f"      ✓ Complete in {glaze_time:.1f}s → {protected_path}")

        # Step 2: Generate deepfake
        print(f"[2/3] Generating deepfake...")
        deepfake, deepfake_time = generate_deepfake(server, protected, mask_path, prompt, seed)
        deepfake_path = os.path.join(output_dir, f"iter{iteration:02d}_deepfake_{current_iters}iters.png")
        deepfake.save(deepfake_path)
        print(f"      ✓ Complete in {deepfake_time:.1f}s → {deepfake_path}")

        # Step 3: Rate with Claude
        print(f"[3/3] Rating deepfake quality with Claude...")
        try:
            score, reasoning = rate_deepfake(
                original_image_path=os.path.join(output_dir, "original.png"),
                deepfake_image_path=deepfake_path,
                mask_path=mask_path,
                api_key=anthropic_api_key,
                prompt_context=prompt,
            )
            print(f"      ✓ Score: {score}/10")
            print(f"      Reasoning: {reasoning[:150]}...")
        except Exception as e:
            print(f"      ✗ ERROR rating deepfake: {e}")
            print(f"      Continuing with manual inspection required.")
            score = None
            reasoning = f"Error: {e}"

        # Save iteration summary
        summary_path = os.path.join(output_dir, f"iter{iteration:02d}_summary.png")
        create_iteration_summary(
            iteration, original_resized, protected, deepfake,
            score if score else 0, current_iters, reasoning, summary_path
        )
        print(f"      ✓ Summary saved → {summary_path}")

        # Track results
        results.append({
            "iteration": iteration,
            "pgd_iters": current_iters,
            "score": score,
            "reasoning": reasoning,
            "glaze_time": glaze_time,
            "deepfake_time": deepfake_time,
            "protected_path": protected_path,
            "deepfake_path": deepfake_path,
            "summary_path": summary_path,
        })

        print()

        # Check if we've reached the target
        if score is not None and score <= threshold:
            print("=" * 80)
            print("🎉 SUCCESS! Glazing is effective!")
            print("=" * 80)
            print(f"  Final score: {score}/10 (threshold: {threshold}/10)")
            print(f"  Optimal PGD iterations: {current_iters}")
            print(f"  Deepfake quality is now below threshold.")
            print(f"  Protection is working effectively!")
            print()
            break

        # Check if we've hit max PGD iterations
        if current_iters >= max_iters:
            print("=" * 80)
            print("⚠️  Reached maximum PGD iterations")
            print("=" * 80)
            print(f"  Current score: {score}/10 (threshold: {threshold}/10)")
            print(f"  Tried up to {current_iters} PGD iterations.")
            print(f"  Consider:")
            print(f"    - Increasing --max-iters")
            print(f"    - Adjusting the threshold")
            print(f"    - Checking if glazing parameters need tuning")
            print()
            break

        # Prepare for next iteration
        current_iters += iter_increment
        iteration += 1
        print(f"Score {score}/10 > threshold {threshold}/10. Increasing to {current_iters} iterations...\n")

    # Save results summary
    results_json_path = os.path.join(output_dir, "results.json")
    with open(results_json_path, "w") as f:
        json.dump({
            "config": {
                "image": image_path,
                "mask": mask_path,
                "prompt": prompt,
                "threshold": threshold,
                "start_iters": start_iters,
                "iter_increment": iter_increment,
                "max_iters": max_iters,
                "max_iterations": max_iterations,
                "seed": seed,
            },
            "results": results,
            "success": score is not None and score <= threshold,
            "final_score": score,
            "optimal_iters": current_iters if score and score <= threshold else None,
            "total_iterations": iteration,
        }, f, indent=2)

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Total iterations run: {iteration}")
    print(f"  Results saved to: {results_json_path}")
    print()
    print("Iteration breakdown:")
    for r in results:
        status = "✓ EFFECTIVE" if r["score"] and r["score"] <= threshold else "✗ WEAK"
        print(f"  [{r['iteration']}] {r['pgd_iters']} iters → Score {r['score']}/10 {status}")
    print("=" * 80)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Adversarial agent loop to find optimal glazing strength",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python client/agent_loop.py \\
      --image face.png --mask mask.png \\
      --prompt "a person in jail" --threshold 6

  # Custom iteration schedule
  python client/agent_loop.py \\
      --image face.png --mask mask.png \\
      --prompt "a person being arrested" \\
      --threshold 5 \\
      --start-iters 50 --iter-increment 100 --max-iters 800

  # With specific backend
  python client/agent_loop.py \\
      --image face.png --mask mask.png \\
      --prompt "a person in a hospital" \\
      --backend gx10 --threshold 6

Environment:
  ANTHROPIC_API_KEY    Required for Claude rater agent
  DIFFGUARD_BACKEND    Default GPU backend
        """,
    )

    # Required arguments
    parser.add_argument("--image", required=True, help="Path to the original image")
    parser.add_argument("--mask", required=True, help="Path to the mask (white = sensitive region)")
    parser.add_argument("--prompt", required=True, help="Inpainting prompt for deepfake generation")

    # Loop parameters
    parser.add_argument("--threshold", type=int, default=6,
                        help="Target score threshold (1-10, default: 6). Loop until score <= threshold")
    parser.add_argument("--start-iters", type=int, default=100,
                        help="Starting PGD iterations (default: 100)")
    parser.add_argument("--iter-increment", type=int, default=100,
                        help="How much to increase iterations each loop (default: 100)")
    parser.add_argument("--max-iters", type=int, default=1000,
                        help="Maximum PGD iterations to try (default: 1000)")
    parser.add_argument("--max-iterations", type=int, default=10,
                        help="Maximum number of loop iterations (default: 10)")

    # Output
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: agent_loop_results_TIMESTAMP)")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deepfake generation")
    parser.add_argument("--anthropic-api-key", default=None,
                        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")

    # Backend selection
    add_backend_args(parser)

    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.image):
        print(f"ERROR: Image not found: {args.image}")
        sys.exit(1)
    if not os.path.isfile(args.mask):
        print(f"ERROR: Mask not found: {args.mask}")
        sys.exit(1)

    if not (1 <= args.threshold <= 10):
        print(f"ERROR: Threshold must be between 1 and 10 (got {args.threshold})")
        sys.exit(1)

    # Check for Anthropic API key
    api_key = args.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Anthropic API key required.")
        print("  Set ANTHROPIC_API_KEY environment variable or use --anthropic-api-key")
        sys.exit(1)

    # Default output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"agent_loop_results_{timestamp}"

    # Resolve backend
    server = get_server_url(args)
    print("Connecting to GPU backend...")
    print_backend_info(server, args.backend)

    # Health check
    info = check_server_health(server)
    print(f"  Server OK — GPU: {info.get('gpu', '?')}, Memory: {info.get('gpu_memory', '?')}")
    print()

    # Run the loop
    run_agent_loop(
        server=server,
        image_path=args.image,
        mask_path=args.mask,
        prompt=args.prompt,
        threshold=args.threshold,
        start_iters=args.start_iters,
        iter_increment=args.iter_increment,
        max_iters=args.max_iters,
        max_iterations=args.max_iterations,
        output_dir=args.output_dir,
        anthropic_api_key=api_key,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
