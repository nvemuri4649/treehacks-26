"""
Targeted CLIP concept attack
==============================
Instead of just maximizing CLIP distance (untargeted), push the image
embedding TOWARD a specific text concept — e.g. "tomato", "cat", "fire".

CLIP aligns text and image embeddings in the same space, so we can:
  1. Get text embedding for "a tomato" from CLIP text encoder
  2. PGD to minimize distance between image embedding and tomato text embedding
  3. IP-Adapter now "sees" a tomato instead of a face

Tests multiple concepts at small eps to show concept steering.

Usage:
    python clip_targeted_concept.py
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
# The IP-Adapter SD1.5 image encoder is CLIP ViT-H/14 from LAION
CLIP_MODEL_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", default="/workspace/comparison_outputs_v3")
    parser.add_argument("--output-dir", default="/workspace/conditioned_outputs_targeted")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pgd-iters", type=int, default=200)
    parser.add_argument("--ip-adapter-scale", type=float, default=0.7)
    return parser.parse_args()


# ===================================================================
# Load CLIP image encoder (from IP-Adapter) + text encoder (from LAION)
# ===================================================================

def load_clip_models():
    from transformers import (
        CLIPVisionModelWithProjection,
        CLIPTextModelWithProjection,
        CLIPTokenizer,
        CLIPImageProcessor,
    )

    # Image encoder — same one IP-Adapter uses
    logger.info("Loading CLIP image encoder (IP-Adapter's ViT-H/14)...")
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

    # Text encoder — matching LAION ViT-H/14
    logger.info("Loading CLIP text encoder (laion ViT-H/14)...")
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        CLIP_MODEL_ID,
        torch_dtype=torch.float32,
    ).to(DEVICE)
    text_encoder.eval()

    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_ID)

    logger.info("Both CLIP encoders loaded.")
    return image_encoder, feature_extractor, text_encoder, tokenizer


def get_text_embedding(text, text_encoder, tokenizer):
    """Get CLIP text embedding for a concept."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = text_encoder(**inputs)
    return outputs.text_embeds  # [1, embed_dim]


def clip_preprocess_tensor(image_tensor, feature_extractor):
    mean = torch.tensor(feature_extractor.image_mean, device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor(feature_extractor.image_std, device=DEVICE).view(1, 3, 1, 1)
    clip_size = feature_extractor.size.get("shortest_edge", 224)
    resized = F.interpolate(image_tensor, size=(clip_size, clip_size), mode="bilinear", align_corners=False)
    return (resized - mean) / std


def get_image_embedding(img, image_encoder, feature_extractor):
    t = torch.from_numpy(
        np.array(img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    clip_in = clip_preprocess_tensor(t, feature_extractor)
    return image_encoder(clip_in).image_embeds


# ===================================================================
# Targeted PGD: push image embedding toward a text concept
# ===================================================================

def pgd_targeted_concept(
    image_pil, target_text_emb, image_encoder, feature_extractor,
    eps, num_iters=200,
):
    """PGD to minimize distance between image embedding and target text embedding."""
    step_size = max(eps / 5, 0.5 / 255)

    img_np = np.array(image_pil.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # Normalize target
    target = F.normalize(target_text_emb.detach(), dim=-1)

    delta = torch.zeros_like(img_tensor, requires_grad=True)
    delta.data.uniform_(-eps, eps)
    delta.data.clamp_(-img_tensor, 1.0 - img_tensor)

    best_delta = delta.data.clone()
    best_cos = float("-inf")

    for i in range(num_iters):
        delta.requires_grad_(True)
        adv_tensor = (img_tensor + delta).clamp(0, 1)
        clip_input = clip_preprocess_tensor(adv_tensor, feature_extractor)
        adv_emb = image_encoder(clip_input).image_embeds
        adv_emb_norm = F.normalize(adv_emb, dim=-1)

        # MAXIMIZE cosine similarity to target concept
        cos_to_target = F.cosine_similarity(adv_emb_norm, target).mean()
        loss = -cos_to_target  # minimize negative = maximize similarity

        loss.backward()

        with torch.no_grad():
            delta.data = delta.data - step_size * delta.grad.sign()
            delta.data.clamp_(-eps, eps)
            delta.data.clamp_(-img_tensor, 1.0 - img_tensor)

            if cos_to_target.item() > best_cos:
                best_cos = cos_to_target.item()
                best_delta = delta.data.clone()

        delta.grad.zero_()

        if (i + 1) % 50 == 0:
            logger.info(
                "    iter %d/%d: cos_to_target=%.4f",
                i + 1, num_iters, best_cos,
            )

    adv_image_tensor = (img_tensor + best_delta).clamp(0, 1)
    adv_np = (adv_image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(adv_np), best_cos


# ===================================================================
# Visualization
# ===================================================================

def make_labeled_strip(images, labels, cell_size=200, label_h=36):
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

    # Target concepts to steer toward
    concepts = [
        ("tomato", "a ripe red tomato"),
        ("cat", "a cute cat face"),
        ("fire", "fire and flames"),
        ("skull", "a human skull"),
        ("flower", "a beautiful flower"),
        ("robot", "a metallic robot face"),
    ]

    # Small eps values — how little noise do we need?
    eps_values = [1, 2, 4, 6, 8]

    logger.info("=" * 70)
    logger.info("TARGETED CONCEPT CLIP ATTACK")
    logger.info("  Concepts: %s", [c[0] for c in concepts])
    logger.info("  Eps values: %s (out of 255)", eps_values)
    logger.info("=" * 70)

    # Load keanu
    orig_path = os.path.join(args.images_dir, "keanu", "1_keanu_original.png")
    if not os.path.exists(orig_path):
        orig_path = os.path.join(args.images_dir, "original_512.png")
    original = Image.open(orig_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    original.save(os.path.join(args.output_dir, "00_original.png"))

    # Load CLIP models
    image_encoder, feature_extractor, text_encoder, tokenizer = load_clip_models()

    # Get original image embedding
    with torch.no_grad():
        orig_emb = get_image_embedding(original, image_encoder, feature_extractor)

    # Get text embeddings for each concept
    concept_embeddings = {}
    for concept_name, concept_text in concepts:
        emb = get_text_embedding(concept_text, text_encoder, tokenizer)
        concept_embeddings[concept_name] = emb
        # How similar is the original to this concept?
        with torch.no_grad():
            orig_cos = F.cosine_similarity(
                F.normalize(orig_emb, dim=-1),
                F.normalize(emb, dim=-1),
            ).item()
        logger.info("  Original → '%s': cos_sim=%.4f", concept_name, orig_cos)

    # Free text encoder (don't need it anymore)
    del text_encoder, tokenizer
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Generate attacked images for each concept × eps
    # ---------------------------------------------------------------
    attacked = {}  # (concept, eps) → (image, cos_to_target, cos_to_original)

    for concept_name, concept_text in concepts:
        target_emb = concept_embeddings[concept_name]

        for eps_int in eps_values:
            eps = eps_int / 255.0
            logger.info("Attacking: concept='%s', eps=%d/255...", concept_name, eps_int)

            t0 = time.time()
            atk_img, cos_target = pgd_targeted_concept(
                original, target_emb, image_encoder, feature_extractor,
                eps=eps, num_iters=args.pgd_iters,
            )
            elapsed = time.time() - t0

            # Measure similarity back to original
            with torch.no_grad():
                atk_emb = get_image_embedding(atk_img, image_encoder, feature_extractor)
                cos_orig = F.cosine_similarity(
                    F.normalize(atk_emb, dim=-1),
                    F.normalize(orig_emb, dim=-1),
                ).item()

            attacked[(concept_name, eps_int)] = {
                "image": atk_img,
                "cos_target": cos_target,
                "cos_original": cos_orig,
            }

            atk_img.save(os.path.join(args.output_dir, f"atk_{concept_name}_eps{eps_int}.png"))
            logger.info(
                "  cos_to_%s=%.4f, cos_to_orig=%.4f (%.1fs)",
                concept_name, cos_target, cos_orig, elapsed,
            )

    # Print summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("CONCEPT TARGETING SUMMARY")
    logger.info("%-12s %-8s %-15s %-15s", "concept", "eps", "cos→target", "cos→original")
    logger.info("-" * 55)
    for concept_name, _ in concepts:
        for eps_int in eps_values:
            r = attacked[(concept_name, eps_int)]
            logger.info(
                "%-12s %-8s %-15.4f %-15.4f",
                concept_name, f"{eps_int}/255", r["cos_target"], r["cos_original"],
            )
        logger.info("")
    logger.info("=" * 70)

    # Free image encoder
    del image_encoder
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # IP-Adapter generation
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

    # Use a neutral prompt so IP-Adapter conditioning dominates
    gen_prompt = "a high quality detailed photograph, 4k, sharp"

    # Generate from original
    logger.info("Generating from original...")
    gen = torch.Generator(device=DEVICE).manual_seed(args.seed)
    gen_original = pipe(
        prompt=gen_prompt,
        ip_adapter_image=original,
        num_inference_steps=50,
        generator=gen,
        height=IMG_SIZE, width=IMG_SIZE,
    ).images[0]
    gen_original.save(os.path.join(args.output_dir, "gen_original.png"))

    # For each concept, generate at each eps
    for concept_name, _ in concepts:
        concept_dir = os.path.join(args.output_dir, concept_name)
        os.makedirs(concept_dir, exist_ok=True)

        gen_images = [("original\neps=0", original, gen_original)]

        for eps_int in eps_values:
            r = attacked[(concept_name, eps_int)]
            logger.info("  Generating: %s eps=%d...", concept_name, eps_int)

            gen = torch.Generator(device=DEVICE).manual_seed(args.seed)
            result = pipe(
                prompt=gen_prompt,
                ip_adapter_image=r["image"],
                num_inference_steps=50,
                generator=gen,
                height=IMG_SIZE, width=IMG_SIZE,
            ).images[0]
            result.save(os.path.join(concept_dir, f"gen_eps{eps_int}.png"))

            label = f"→{concept_name}\neps={eps_int}, cos={r['cos_target']:.2f}"
            gen_images.append((label, r["image"], result))

        # Comparison strips
        ref_row = make_labeled_strip(
            [g[1] for g in gen_images],
            [g[0] for g in gen_images],
        )
        gen_row = make_labeled_strip(
            [g[2] for g in gen_images],
            [f"generated" for g in gen_images],
        )
        comp = stack_rows([ref_row, gen_row])
        comp.save(os.path.join(concept_dir, "comparison.png"))
        logger.info("  Saved: %s/comparison.png", concept_name)

    # Cross-concept comparison at a single eps (eps=4)
    # Shows: same face image, steered to 6 different concepts
    logger.info("Creating cross-concept comparison at eps=4...")
    cross_eps = 4
    cross_dir = os.path.join(args.output_dir, "cross_concept_eps4")
    os.makedirs(cross_dir, exist_ok=True)

    cross_refs = [("original", original)]
    cross_gens = [("original", gen_original)]

    for concept_name, _ in concepts:
        r = attacked[(concept_name, cross_eps)]
        gen = torch.Generator(device=DEVICE).manual_seed(args.seed)
        result = pipe(
            prompt=gen_prompt,
            ip_adapter_image=r["image"],
            num_inference_steps=50,
            generator=gen,
            height=IMG_SIZE, width=IMG_SIZE,
        ).images[0]
        result.save(os.path.join(cross_dir, f"gen_{concept_name}.png"))
        cross_refs.append((f"→{concept_name}\ncos={r['cos_target']:.2f}", r["image"]))
        cross_gens.append((concept_name, result))

    ref_row = make_labeled_strip(
        [c[1] for c in cross_refs],
        [c[0] for c in cross_refs],
        cell_size=180,
    )
    gen_row = make_labeled_strip(
        [c[1] for c in cross_gens],
        [c[0] for c in cross_gens],
        cell_size=180,
    )
    comp = stack_rows([ref_row, gen_row])
    comp.save(os.path.join(cross_dir, "cross_concept.png"))

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
