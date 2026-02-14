"""
Small-epsilon CLIP attack sweep
================================
Fine-grained sweep of tiny perturbation budgets to find the threshold
where IP-Adapter conditioning starts breaking down.

Tests: eps = 1, 2, 3, 4, 5, 6, 7, 8, 10, 12 (out of 255)

For reference:
  - eps=1/255 means each pixel can change by at most 1 value (invisible)
  - eps=4/255 means each pixel can change by at most 4 values (imperceptible)
  - eps=8/255 means each pixel can change by at most 8 values (very subtle)
  - DiffusionGuard default is eps=16/255

Usage:
    python clip_small_eps_sweep.py
"""

import os
import sys
import time
import argparse
import logging

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
IMG_SIZE = 512

SD15_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
IP_ADAPTER_REPO = "h94/IP-Adapter"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", default="/workspace/comparison_outputs_v3")
    parser.add_argument("--output-dir", default="/workspace/conditioned_outputs_v3_small_eps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pgd-iters", type=int, default=200)
    parser.add_argument("--ip-adapter-scale", type=float, default=0.7)
    return parser.parse_args()


# ===================================================================
# CLIP encoder + PGD attack (same as ip_adapter_attack_demo.py)
# ===================================================================

def load_clip_image_encoder():
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

    logger.info("Loading CLIP image encoder...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        IP_ADAPTER_REPO,
        subfolder="models/image_encoder",
        torch_dtype=torch.float32,
    ).to(DEVICE)
    image_encoder.eval()

    feature_extractor = CLIPImageProcessor(
        size={"shortest_edge": 224},
        crop_size={"height": 224, "width": 224},
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
    )
    logger.info("CLIP encoder loaded.")
    return image_encoder, feature_extractor


def clip_preprocess_tensor(image_tensor, feature_extractor):
    mean = torch.tensor(feature_extractor.image_mean, device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor(feature_extractor.image_std, device=DEVICE).view(1, 3, 1, 1)
    clip_size = feature_extractor.size.get("shortest_edge", 224)
    resized = F.interpolate(image_tensor, size=(clip_size, clip_size), mode="bilinear", align_corners=False)
    return (resized - mean) / std


def pgd_attack_clip(image_pil, image_encoder, feature_extractor, eps, num_iters=200):
    step_size = max(eps / 5, 0.5 / 255)  # Ensure reasonable step size even for tiny eps

    img_np = np.array(image_pil.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        orig_clip_input = clip_preprocess_tensor(img_tensor, feature_extractor)
        orig_embedding = image_encoder(orig_clip_input).image_embeds.detach()

    delta = torch.zeros_like(img_tensor, requires_grad=True)
    delta.data.uniform_(-eps, eps)
    delta.data.clamp_(-img_tensor, 1.0 - img_tensor)

    best_delta = delta.data.clone()
    best_loss = float("-inf")

    for i in range(num_iters):
        delta.requires_grad_(True)
        adv_tensor = (img_tensor + delta).clamp(0, 1)
        clip_input = clip_preprocess_tensor(adv_tensor, feature_extractor)
        adv_embedding = image_encoder(clip_input).image_embeds

        # Maximize distance: minimize cosine similarity + maximize L2
        cos_sim = F.cosine_similarity(adv_embedding, orig_embedding)
        loss = cos_sim.mean() - 0.01 * F.mse_loss(adv_embedding, orig_embedding)

        loss.backward()

        with torch.no_grad():
            delta.data = delta.data - step_size * delta.grad.sign()
            delta.data.clamp_(-eps, eps)
            delta.data.clamp_(-img_tensor, 1.0 - img_tensor)

            current_metric = -cos_sim.item()
            if current_metric > best_loss:
                best_loss = current_metric
                best_delta = delta.data.clone()

        delta.grad.zero_()

    adv_image_tensor = (img_tensor + best_delta).clamp(0, 1)
    adv_np = (adv_image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(adv_np)


def get_clip_embedding(img, image_encoder, feature_extractor):
    t = torch.from_numpy(
        np.array(img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    clip_in = clip_preprocess_tensor(t, feature_extractor)
    return image_encoder(clip_in).image_embeds


# ===================================================================
# IP-Adapter generation
# ===================================================================

def run_ip_adapter_gen(pipe, ref_image, prompt, seed, scale=0.7):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    pipe.set_ip_adapter_scale(scale)
    return pipe(
        prompt=prompt,
        ip_adapter_image=ref_image.resize((IMG_SIZE, IMG_SIZE)),
        num_inference_steps=50,
        generator=generator,
        height=IMG_SIZE,
        width=IMG_SIZE,
    ).images[0]


# ===================================================================
# Visualization
# ===================================================================

def make_labeled_strip(images, labels, cell_size=180, label_h=36):
    """Row of images with two-line labels."""
    n = len(images)
    gap = 3
    w = cell_size * n + gap * (n - 1)
    strip = Image.new("RGB", (w, cell_size + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        font_b = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
    except (IOError, OSError):
        font = font_b = ImageFont.load_default()

    for i, (img, label) in enumerate(zip(images, labels)):
        x = i * (cell_size + gap)
        # Split label at newline
        lines = label.split("\n")
        for j, line in enumerate(lines):
            f = font_b if j == 0 else font
            draw.text((x + 3, 2 + j * 13), line, fill=(30, 30, 30), font=f)
        strip.paste(img.resize((cell_size, cell_size), Image.LANCZOS), (x, label_h))
    return strip


def stack_rows(rows, gap=4):
    if not rows:
        return Image.new("RGB", (10, 10))
    w = max(r.size[0] for r in rows)
    h = sum(r.size[1] for r in rows) + gap * (len(rows) - 1)
    out = Image.new("RGB", (w, h), (245, 245, 245))
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.size[1] + gap
    return out


# ===================================================================
# MAIN
# ===================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Fine-grained epsilon sweep: 1 through 12 out of 255
    eps_values = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]

    logger.info("=" * 70)
    logger.info("SMALL EPSILON CLIP ATTACK SWEEP")
    logger.info("  Epsilon values: %s (out of 255)", eps_values)
    logger.info("  PGD iters: %d", args.pgd_iters)
    logger.info("=" * 70)

    # Load keanu
    orig_path = os.path.join(args.images_dir, "keanu", "1_keanu_original.png")
    if not os.path.exists(orig_path):
        orig_path = os.path.join(args.images_dir, "original_512.png")
    original = Image.open(orig_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    original.save(os.path.join(args.output_dir, "00_original.png"))

    # ---------------------------------------------------------------
    # STEP 1: CLIP attacks at each epsilon
    # ---------------------------------------------------------------
    image_encoder, feature_extractor = load_clip_image_encoder()

    with torch.no_grad():
        orig_emb = get_clip_embedding(original, image_encoder, feature_extractor)

    attacked_images = {}
    clip_metrics = {}

    for eps_int in eps_values:
        eps = eps_int / 255.0
        logger.info("CLIP attack eps=%d/255 (%d iters)...", eps_int, args.pgd_iters)

        t0 = time.time()
        attacked = pgd_attack_clip(original, image_encoder, feature_extractor, eps, args.pgd_iters)
        elapsed = time.time() - t0

        attacked_images[eps_int] = attacked
        attacked.save(os.path.join(args.output_dir, f"attacked_eps{eps_int}.png"))

        # Measure
        with torch.no_grad():
            atk_emb = get_clip_embedding(attacked, image_encoder, feature_extractor)
            cos_sim = F.cosine_similarity(atk_emb, orig_emb).item()
            l2_dist = (atk_emb - orig_emb).norm().item()

        clip_metrics[eps_int] = {"cos_sim": cos_sim, "l2": l2_dist}
        logger.info("  eps=%d/255: cos_sim=%.4f, L2=%.2f (%.1fs)", eps_int, cos_sim, l2_dist, elapsed)

        # Also compute pixel-level PSNR
        orig_np = np.array(original).astype(float)
        atk_np = np.array(attacked).astype(float)
        mse = np.mean((orig_np - atk_np) ** 2)
        psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-10))
        clip_metrics[eps_int]["psnr"] = psnr
        logger.info("  pixel PSNR=%.1f dB, max_diff=%d/255", psnr, int(np.max(np.abs(orig_np - atk_np))))

    # Print summary table
    logger.info("")
    logger.info("=" * 70)
    logger.info("CLIP EMBEDDING DISTANCE SUMMARY")
    logger.info("%-10s %-12s %-10s %-10s", "eps", "cos_sim", "L2_dist", "PSNR(dB)")
    logger.info("-" * 45)
    for eps_int in eps_values:
        m = clip_metrics[eps_int]
        logger.info("%-10s %-12.4f %-10.2f %-10.1f", f"{eps_int}/255", m["cos_sim"], m["l2"], m["psnr"])
    logger.info("=" * 70)
    logger.info("")

    # Free CLIP encoder
    del image_encoder
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # STEP 2: IP-Adapter generation at each epsilon
    # ---------------------------------------------------------------
    from diffusers import StableDiffusionPipeline

    logger.info("Loading SD 1.5 + IP-Adapter...")
    pipe = StableDiffusionPipeline.from_pretrained(
        SD15_MODEL, torch_dtype=DTYPE, variant="fp16"
    ).to(DEVICE)
    pipe.load_ip_adapter(
        IP_ADAPTER_REPO, subfolder="models", weight_name="ip-adapter_sd15.bin"
    )
    pipe.set_ip_adapter_scale(args.ip_adapter_scale)
    logger.info("Pipeline loaded.")

    prompts = [
        ("headshot", "professional corporate headshot photo, studio lighting, clean background, 4k portrait"),
        ("cyberpunk", "cyberpunk portrait, neon lights, futuristic city, blade runner, photorealistic"),
        ("oil_painting", "oil painting portrait in the style of Rembrandt, dramatic lighting, masterpiece"),
    ]

    for prompt_name, prompt_text in prompts:
        prompt_dir = os.path.join(args.output_dir, prompt_name)
        os.makedirs(prompt_dir, exist_ok=True)

        logger.info("Prompt: %s", prompt_name)

        # Generate from original
        logger.info("  ref=original")
        gen_orig = run_ip_adapter_gen(pipe, original, prompt_text, args.seed, args.ip_adapter_scale)
        gen_orig.save(os.path.join(prompt_dir, "gen_original.png"))

        # Generate from each attacked image
        gen_results = [("original\neps=0", original, gen_orig)]

        for eps_int in eps_values:
            logger.info("  ref=eps%d", eps_int)
            gen = run_ip_adapter_gen(pipe, attacked_images[eps_int], prompt_text, args.seed, args.ip_adapter_scale)
            gen.save(os.path.join(prompt_dir, f"gen_eps{eps_int}.png"))

            m = clip_metrics[eps_int]
            label = f"eps={eps_int}/255\ncos={m['cos_sim']:.2f}"
            gen_results.append((label, attacked_images[eps_int], gen))

        # ---------------------------------------------------------------
        # Create comparison images
        # ---------------------------------------------------------------

        # Row 1: reference images (original + all eps levels)
        ref_row = make_labeled_strip(
            [r[1] for r in gen_results],
            [r[0] for r in gen_results],
        )
        # Row 2: generated outputs
        gen_row = make_labeled_strip(
            [r[2] for r in gen_results],
            ["→ " + r[0].split("\n")[0] for r in gen_results],
        )

        full_comparison = stack_rows([ref_row, gen_row])
        full_comparison.save(os.path.join(prompt_dir, "full_sweep.png"))

        # Also a "zoom" comparison of just the small eps values (1-5)
        zoom_results = [gen_results[0]] + gen_results[1:6]  # orig + eps 1-5
        zoom_ref = make_labeled_strip(
            [r[1] for r in zoom_results], [r[0] for r in zoom_results], cell_size=220,
        )
        zoom_gen = make_labeled_strip(
            [r[2] for r in zoom_results],
            ["→ " + r[0].split("\n")[0] for r in zoom_results],
            cell_size=220,
        )
        zoom_comparison = stack_rows([zoom_ref, zoom_gen])
        zoom_comparison.save(os.path.join(prompt_dir, "zoom_eps1_to_5.png"))

        logger.info("  Saved: %s/", prompt_name)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("DONE. Output: %s", args.output_dir)
    for root, dirs, files in os.walk(args.output_dir):
        level = root.replace(args.output_dir, "").count(os.sep)
        indent = "  " * level
        logger.info("%s%s/ (%d files)", indent, os.path.basename(root), len(files))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
