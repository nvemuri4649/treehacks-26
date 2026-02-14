"""
IP-Adapter CLIP-Targeted Adversarial Attack Demo
==================================================
DiffusionGuard targets SD inpainting, but IP-Adapter works via CLIP image
embeddings. This script creates adversarial perturbations that directly
attack the CLIP image encoder used by IP-Adapter, producing dramatically
worse conditioned generation.

Attack strategy:
  1) Load the same CLIP image encoder IP-Adapter uses
  2) PGD (Projected Gradient Descent) to MAXIMIZE distortion in CLIP space
  3) The perturbed image looks nearly identical to a human, but the CLIP
     embedding is completely different → IP-Adapter can't extract identity

We test multiple epsilon levels:
  - eps=8/255   (very subtle, near-invisible)
  - eps=16/255  (subtle, same as DiffusionGuard default)
  - eps=32/255  (moderate, slightly visible)
  - eps=48/255  (strong, noticeable if you look closely)
  - eps=64/255  (very strong)

We also test combining DiffusionGuard (anti-inpaint) + CLIP attack (anti-IP-Adapter).

Usage:
    python ip_adapter_attack_demo.py [--images-dir PATH] [--output-dir PATH]
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
IMG_SIZE = 512

# IP-Adapter models
SD15_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
IP_ADAPTER_REPO = "h94/IP-Adapter"


def parse_args():
    parser = argparse.ArgumentParser(description="CLIP-targeted adversarial attack for IP-Adapter")
    parser.add_argument("--images-dir", default="/workspace/comparison_outputs_v3")
    parser.add_argument("--output-dir", default="/workspace/conditioned_outputs_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pgd-iters", type=int, default=200, help="PGD optimization steps")
    parser.add_argument("--ip-adapter-scale", type=float, default=0.7)
    return parser.parse_args()


# ===================================================================
# CLIP-TARGETED ADVERSARIAL PERTURBATION
# ===================================================================

def load_clip_image_encoder():
    """Load the exact CLIP image encoder that IP-Adapter uses."""
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

    logger.info("Loading CLIP image encoder (same as IP-Adapter)...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        IP_ADAPTER_REPO,
        subfolder="models/image_encoder",
        torch_dtype=torch.float32,  # Need float32 for gradient computation
    ).to(DEVICE)
    image_encoder.eval()

    # The IP-Adapter repo doesn't include a preprocessor config, so we load
    # it from the base CLIP model. SD1.5 IP-Adapter uses ViT-H/14.
    # We construct the processor manually with standard CLIP ViT-H/14 values.
    feature_extractor = CLIPImageProcessor(
        size={"shortest_edge": 224},
        crop_size={"height": 224, "width": 224},
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        do_resize=True,
        do_center_crop=True,
        do_normalize=True,
    )

    logger.info("CLIP encoder loaded.")
    return image_encoder, feature_extractor


def clip_preprocess_tensor(image_tensor, feature_extractor):
    """
    Apply CLIP preprocessing to a [1, 3, H, W] tensor in [0, 1] range.
    Returns a tensor ready for the CLIP encoder.
    """
    # CLIP expects specific normalization
    mean = torch.tensor(feature_extractor.image_mean, device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor(feature_extractor.image_std, device=DEVICE).view(1, 3, 1, 1)

    # Resize to CLIP input size (224x224)
    clip_size = feature_extractor.size.get("shortest_edge", 224)
    resized = F.interpolate(image_tensor, size=(clip_size, clip_size), mode="bilinear", align_corners=False)

    # Normalize
    normalized = (resized - mean) / std
    return normalized


def pgd_attack_clip(
    image_pil: Image.Image,
    image_encoder,
    feature_extractor,
    eps: float = 16 / 255,
    step_size: float = None,
    num_iters: int = 200,
    target: str = "maximize_distance",
):
    """
    PGD attack against CLIP image encoder.

    Strategies:
      - "maximize_distance": maximize L2 distance in CLIP embedding space
        from the original image's embedding (untargeted attack)
      - "random_target": push embedding toward a random target vector
      - "invert": try to push embedding toward the negative of the original

    Returns the adversarially perturbed PIL image.
    """
    if step_size is None:
        step_size = eps / 5  # Conservative step size

    # Convert PIL to tensor [1, 3, H, W] in [0, 1]
    img_np = np.array(image_pil.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # Get original CLIP embedding (our reference)
    with torch.no_grad():
        orig_clip_input = clip_preprocess_tensor(img_tensor, feature_extractor)
        orig_embedding = image_encoder(orig_clip_input).image_embeds.detach()

    # Initialize perturbation (random within eps-ball)
    delta = torch.zeros_like(img_tensor, requires_grad=True)
    delta.data.uniform_(-eps, eps)
    delta.data.clamp_(-img_tensor, 1.0 - img_tensor)  # Ensure x + delta in [0,1]

    best_delta = delta.data.clone()
    best_loss = float("-inf")

    for i in range(num_iters):
        delta.requires_grad_(True)

        # Perturbed image
        adv_tensor = (img_tensor + delta).clamp(0, 1)

        # Get CLIP embedding of perturbed image
        clip_input = clip_preprocess_tensor(adv_tensor, feature_extractor)
        adv_embedding = image_encoder(clip_input).image_embeds

        if target == "maximize_distance":
            # Maximize L2 distance from original embedding
            loss = -F.mse_loss(adv_embedding, orig_embedding)
            # Also maximize cosine distance
            cos_sim = F.cosine_similarity(adv_embedding, orig_embedding)
            loss = loss + cos_sim.mean()  # We want to MINIMIZE cosine similarity

        elif target == "invert":
            # Push toward negative of original embedding
            neg_target = -orig_embedding.detach()
            loss = F.mse_loss(adv_embedding, neg_target)

        elif target == "random_target":
            # Push toward random point in embedding space
            if i == 0:
                random_target = torch.randn_like(orig_embedding)
                random_target = F.normalize(random_target, dim=-1) * orig_embedding.norm()
            loss = F.mse_loss(adv_embedding, random_target)

        # Backprop
        loss.backward()

        with torch.no_grad():
            # PGD step (maximize loss = ascend gradient for maximize_distance)
            if target == "maximize_distance":
                grad_sign = delta.grad.sign()
                delta.data = delta.data - step_size * grad_sign  # Gradient ascent (loss is negative)
            else:
                grad_sign = delta.grad.sign()
                delta.data = delta.data - step_size * grad_sign

            # Project back to eps-ball
            delta.data.clamp_(-eps, eps)
            # Ensure valid pixel range
            delta.data.clamp_(-img_tensor, 1.0 - img_tensor)

            # Track best
            current_loss = -loss.item() if target == "maximize_distance" else loss.item()
            if current_loss > best_loss:
                best_loss = current_loss
                best_delta = delta.data.clone()

        delta.grad.zero_()

        if (i + 1) % 50 == 0:
            with torch.no_grad():
                adv_check = (img_tensor + best_delta).clamp(0, 1)
                clip_check = clip_preprocess_tensor(adv_check, feature_extractor)
                emb_check = image_encoder(clip_check).image_embeds
                cos_dist = 1 - F.cosine_similarity(emb_check, orig_embedding).item()
                l2_dist = (emb_check - orig_embedding).norm().item()
                pixel_l_inf = best_delta.abs().max().item() * 255
            logger.info(
                "  PGD iter %d/%d: cos_dist=%.4f, L2_dist=%.2f, pixel_Linf=%.1f/255",
                i + 1, num_iters, cos_dist, l2_dist, pixel_l_inf,
            )

    # Apply best perturbation
    adv_image_tensor = (img_tensor + best_delta).clamp(0, 1)
    adv_np = (adv_image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(adv_np)


# ===================================================================
# IP-ADAPTER GENERATION
# ===================================================================

def run_ip_adapter_generation(pipe, ref_image, prompt, seed, scale=0.7, size=512):
    """Generate an image conditioned on ref_image + prompt via IP-Adapter."""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    pipe.set_ip_adapter_scale(scale)
    result = pipe(
        prompt=prompt,
        ip_adapter_image=ref_image.resize((size, size)),
        num_inference_steps=50,
        generator=generator,
        height=size,
        width=size,
    ).images[0]
    return result


def create_row_comparison(images, labels, cell_size=256):
    """Create a horizontal strip of images with labels."""
    n = len(images)
    label_h = 22
    strip = Image.new("RGB", (cell_size * n + (n - 1) * 4, cell_size + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(strip)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for i, (img, label) in enumerate(zip(images, labels)):
        x = i * (cell_size + 4)
        draw.text((x + 4, 2), label, fill=(40, 40, 40), font=font)
        strip.paste(img.resize((cell_size, cell_size), Image.LANCZOS), (x, label_h))

    return strip


def create_grid_comparison(rows, row_labels=None, cell_size=256):
    """Stack multiple row comparisons vertically with row labels."""
    if not rows:
        return Image.new("RGB", (100, 100), (255, 255, 255))

    row_label_w = 90 if row_labels else 0
    row_h = rows[0].size[1]
    row_w = rows[0].size[0]
    total_h = row_h * len(rows) + (len(rows) - 1) * 6
    grid = Image.new("RGB", (row_w + row_label_w, total_h), (255, 255, 255))

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except (IOError, OSError):
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(grid)

    for i, row in enumerate(rows):
        y = i * (row_h + 6)
        if row_labels and i < len(row_labels):
            draw.text((4, y + row_h // 2 - 6), row_labels[i], fill=(0, 0, 0), font=font)
        grid.paste(row, (row_label_w, y))

    return grid


# ===================================================================
# MAIN
# ===================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("CLIP-TARGETED ADVERSARIAL ATTACK FOR IP-ADAPTER")
    logger.info("  Device: %s", DEVICE)
    logger.info("  PGD iters: %d", args.pgd_iters)
    logger.info("=" * 70)

    # Load images
    orig_path = os.path.join(args.images_dir, "keanu", "1_keanu_original.png")
    diffguard_path = os.path.join(args.images_dir, "keanu", "2_keanu_protected.png")

    if not os.path.exists(orig_path):
        # Fallback to top-level
        orig_path = os.path.join(args.images_dir, "original_512.png")
        diffguard_path = os.path.join(args.images_dir, "protected.png")

    original = Image.open(orig_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    diffguard_protected = Image.open(diffguard_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    # Also load main_subject if available
    subjects = [("keanu", original, diffguard_protected)]
    main_orig = os.path.join(args.images_dir, "original_512.png")
    main_prot = os.path.join(args.images_dir, "protected.png")
    if os.path.exists(main_orig) and os.path.exists(main_prot):
        subjects.append((
            "main_subject",
            Image.open(main_orig).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS),
            Image.open(main_prot).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS),
        ))

    # ---------------------------------------------------------------
    # STEP 1: Load CLIP encoder and create adversarial perturbations
    # ---------------------------------------------------------------
    image_encoder, feature_extractor = load_clip_image_encoder()

    eps_levels = [8 / 255, 16 / 255, 32 / 255, 48 / 255, 64 / 255]

    for subject_name, orig_img, dg_img in subjects:
        subj_dir = os.path.join(args.output_dir, subject_name)
        os.makedirs(subj_dir, exist_ok=True)

        orig_img.save(os.path.join(subj_dir, "00_original.png"))
        dg_img.save(os.path.join(subj_dir, "01_diffguard_only.png"))

        logger.info("=" * 60)
        logger.info("Subject: %s", subject_name)
        logger.info("=" * 60)

        clip_attacked = {}

        # Pure CLIP attack at each epsilon
        for eps in eps_levels:
            eps_name = f"eps{int(eps * 255)}"
            logger.info("CLIP attack: %s (eps=%d/255, %d iters)", subject_name, int(eps * 255), args.pgd_iters)

            t0 = time.time()
            attacked = pgd_attack_clip(
                orig_img, image_encoder, feature_extractor,
                eps=eps, num_iters=args.pgd_iters, target="maximize_distance",
            )
            elapsed = time.time() - t0
            logger.info("  Done in %.1fs", elapsed)

            clip_attacked[eps] = attacked
            attacked.save(os.path.join(subj_dir, f"02_clip_attack_{eps_name}.png"))

        # Combined: DiffusionGuard + CLIP attack (attack the DG-protected image further)
        logger.info("Combined attack: DiffusionGuard + CLIP (eps=32/255)")
        t0 = time.time()
        combined = pgd_attack_clip(
            dg_img, image_encoder, feature_extractor,
            eps=32 / 255, num_iters=args.pgd_iters, target="maximize_distance",
        )
        elapsed = time.time() - t0
        logger.info("  Combined done in %.1fs", elapsed)
        combined.save(os.path.join(subj_dir, "03_combined_dg_clip.png"))

        # ---------------------------------------------------------------
        # Measure CLIP embedding distances
        # ---------------------------------------------------------------
        logger.info("Measuring CLIP embedding distances...")

        with torch.no_grad():
            def get_clip_embedding(img):
                t = torch.from_numpy(
                    np.array(img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
                ).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                clip_in = clip_preprocess_tensor(t, feature_extractor)
                return image_encoder(clip_in).image_embeds

            orig_emb = get_clip_embedding(orig_img)

            # DiffusionGuard only
            dg_emb = get_clip_embedding(dg_img)
            dg_cos = F.cosine_similarity(dg_emb, orig_emb).item()
            dg_l2 = (dg_emb - orig_emb).norm().item()
            logger.info("  DiffusionGuard only:  cos_sim=%.4f, L2=%.2f", dg_cos, dg_l2)

            # CLIP attacks
            for eps in eps_levels:
                emb = get_clip_embedding(clip_attacked[eps])
                cos = F.cosine_similarity(emb, orig_emb).item()
                l2 = (emb - orig_emb).norm().item()
                logger.info("  CLIP eps=%d/255:      cos_sim=%.4f, L2=%.2f", int(eps * 255), cos, l2)

            # Combined
            comb_emb = get_clip_embedding(combined)
            comb_cos = F.cosine_similarity(comb_emb, orig_emb).item()
            comb_l2 = (comb_emb - orig_emb).norm().item()
            logger.info("  DG + CLIP combined:   cos_sim=%.4f, L2=%.2f", comb_cos, comb_l2)

    # ---------------------------------------------------------------
    # STEP 2: Free CLIP encoder, load IP-Adapter pipeline
    # ---------------------------------------------------------------
    del image_encoder
    torch.cuda.empty_cache()

    from diffusers import StableDiffusionPipeline
    logger.info("Loading SD 1.5 + IP-Adapter...")
    pipe = StableDiffusionPipeline.from_pretrained(
        SD15_MODEL, torch_dtype=DTYPE, variant="fp16"
    ).to(DEVICE)
    pipe.load_ip_adapter(
        IP_ADAPTER_REPO, subfolder="models", weight_name="ip-adapter_sd15.bin"
    )
    logger.info("Pipeline loaded.")

    # ---------------------------------------------------------------
    # STEP 3: Generate with IP-Adapter using each noise level
    # ---------------------------------------------------------------
    prompts = [
        ("headshot", "professional corporate headshot photo, studio lighting, clean background, high quality portrait, 4k"),
        ("cyberpunk", "cyberpunk portrait, neon lights, futuristic city, blade runner style, photorealistic"),
        ("oil_painting", "oil painting portrait in the style of Rembrandt, dramatic lighting, masterpiece"),
    ]

    for subject_name, orig_img, dg_img in subjects:
        subj_dir = os.path.join(args.output_dir, subject_name)

        # Reload CLIP-attacked images from disk
        clip_attacked_imgs = {}
        for eps in eps_levels:
            eps_name = f"eps{int(eps * 255)}"
            path = os.path.join(subj_dir, f"02_clip_attack_{eps_name}.png")
            clip_attacked_imgs[eps] = Image.open(path).convert("RGB")
        combined_img = Image.open(os.path.join(subj_dir, "03_combined_dg_clip.png")).convert("RGB")

        for prompt_name, prompt_text in prompts:
            logger.info("Generating: %s / %s", subject_name, prompt_name)

            gen_dir = os.path.join(subj_dir, prompt_name)
            os.makedirs(gen_dir, exist_ok=True)

            # All reference images to test
            ref_images = [
                ("original", orig_img),
                ("diffguard", dg_img),
            ]
            for eps in eps_levels:
                ref_images.append((f"clip_eps{int(eps * 255)}", clip_attacked_imgs[eps]))
            ref_images.append(("combined_dg_clip", combined_img))

            gen_results = []

            for ref_name, ref_img in ref_images:
                logger.info("  ref=%s", ref_name)
                result = run_ip_adapter_generation(
                    pipe, ref_img, prompt_text, args.seed, scale=args.ip_adapter_scale
                )
                result.save(os.path.join(gen_dir, f"gen_{ref_name}.png"))
                gen_results.append((ref_name, ref_img, result))

            # Create comparison strips
            # Row 1: Reference images
            ref_row = create_row_comparison(
                [r[1] for r in gen_results],
                [r[0] for r in gen_results],
                cell_size=200,
            )
            # Row 2: Generated images
            gen_row = create_row_comparison(
                [r[2] for r in gen_results],
                [f"→ {r[0]}" for r in gen_results],
                cell_size=200,
            )
            # Stack
            full_grid = create_grid_comparison(
                [ref_row, gen_row],
                row_labels=["Reference", "Generated"],
            )
            full_grid.save(os.path.join(gen_dir, "comparison_grid.png"))

            # Also create a focused comparison: original vs best attacks
            key_refs = [
                gen_results[0],   # original
                gen_results[1],   # diffguard
                gen_results[-3],  # clip_eps48
                gen_results[-2],  # clip_eps64
                gen_results[-1],  # combined
            ]
            key_ref_row = create_row_comparison(
                [r[1] for r in key_refs], [r[0] for r in key_refs], cell_size=240
            )
            key_gen_row = create_row_comparison(
                [r[2] for r in key_refs], [f"→ {r[0]}" for r in key_refs], cell_size=240
            )
            key_grid = create_grid_comparison(
                [key_ref_row, key_gen_row],
                row_labels=["Input", "Output"],
            )
            key_grid.save(os.path.join(gen_dir, "key_comparison.png"))

            logger.info("  Saved: %s/%s/", subject_name, prompt_name)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("ALL DONE. Output: %s", args.output_dir)
    for root, dirs, files in os.walk(args.output_dir):
        level = root.replace(args.output_dir, "").count(os.sep)
        indent = "  " * level
        logger.info("%s%s/ (%d files)", indent, os.path.basename(root), len(files))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
