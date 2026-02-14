"""
Ensemble CLIP Attack — Transferable adversarial perturbations
=============================================================

Optimizes perturbation against MULTIPLE vision encoders simultaneously,
so the adversarial effect transfers to closed-source models (GPT-4V,
Claude, Gemini) that use similar ViT architectures.

Key insight: if we fool 4-5 different CLIP/SigLIP models at once,
the perturbation exploits shared ViT features rather than model-specific
quirks, making it far more likely to transfer.

Ensemble members:
  1. CLIP ViT-H/14 (IP-Adapter's encoder, laion2b)
  2. CLIP ViT-L/14 (OpenAI, used by LLaVA)
  3. CLIP ViT-B/32 (OpenAI, smallest/fastest)
  4. SigLIP ViT-SO400M (Google, used by PaliGemma/Gemini-like)
  5. DINOv2 ViT-L/14 (Meta, self-supervised — different training!)

DINOv2 is especially important: it wasn't trained with text-image
contrastive learning, so fooling it AND CLIP models means we're
attacking fundamental ViT feature extraction, not just CLIP-specific
artifacts.

Usage:
    python ensemble_clip_attack.py [--image PATH] [--eps 8]
                                    [--target-concept "a tomato"]
                                    [--output-dir PATH]
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
IMG_SIZE = 512


def center_crop_square(img: Image.Image, size: int = IMG_SIZE) -> Image.Image:
    """Center-crop to square, then resize. Preserves aspect ratio — no squashing."""
    w, h = img.size
    crop_dim = min(w, h)
    left = (w - crop_dim) // 2
    top = (h - crop_dim) // 2
    img = img.crop((left, top, left + crop_dim, top + crop_dim))
    return img.resize((size, size), Image.LANCZOS)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="Path to input image")
    parser.add_argument("--images-dir", default="/workspace/comparison_outputs_v3")
    parser.add_argument("--output-dir", default="/workspace/ensemble_attack_output")
    parser.add_argument("--eps", type=int, default=8, help="Perturbation budget (out of 255)")
    parser.add_argument("--target-concept", default="a ripe red tomato",
                        help="Text concept to steer toward")
    parser.add_argument("--target-name", default="tomato", help="Short name for target")
    parser.add_argument("--pgd-iters", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ip-adapter", action="store_true",
                        help="Also generate IP-Adapter results to verify")
    return parser.parse_args()


# ===================================================================
# Model loading
# ===================================================================

class VisionModel:
    """Wrapper for a vision encoder."""
    def __init__(self, name, model, preprocess_fn, embed_fn, weight=1.0):
        self.name = name
        self.model = model
        self.preprocess_fn = preprocess_fn
        self.embed_fn = embed_fn
        self.weight = weight

    def get_embedding(self, image_tensor):
        """image_tensor: (1, 3, H, W) in [0, 1]"""
        preprocessed = self.preprocess_fn(image_tensor)
        return self.embed_fn(preprocessed)


def load_ensemble():
    """Load multiple vision encoders for ensemble attack."""
    models = []

    # --- 1. CLIP ViT-H/14 (IP-Adapter's encoder) ---
    try:
        from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
        logger.info("Loading CLIP ViT-H/14 (IP-Adapter)...")
        clip_h = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter", subfolder="models/image_encoder",
            torch_dtype=torch.float32
        ).to(DEVICE).eval()

        clip_h_proc = CLIPImageProcessor(
            size={"shortest_edge": 224}, crop_size={"height": 224, "width": 224},
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
        )
        mean_h = torch.tensor(clip_h_proc.image_mean, device=DEVICE).view(1, 3, 1, 1)
        std_h = torch.tensor(clip_h_proc.image_std, device=DEVICE).view(1, 3, 1, 1)

        def preprocess_clip_h(x):
            return (F.interpolate(x, (224, 224), mode="bilinear", align_corners=False) - mean_h) / std_h

        def embed_clip_h(x):
            return clip_h(x).image_embeds

        models.append(VisionModel("clip_vit_h14", clip_h, preprocess_clip_h, embed_clip_h, weight=2.0))
        logger.info("  ✓ CLIP ViT-H/14 loaded (weight=2.0, primary target)")
    except Exception as e:
        logger.warning("  ✗ CLIP ViT-H/14 failed: %s", e)

    # --- 2. CLIP ViT-L/14 (OpenAI, used by LLaVA) ---
    try:
        from transformers import CLIPVisionModelWithProjection
        logger.info("Loading CLIP ViT-L/14 (OpenAI)...")
        clip_l = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14", torch_dtype=torch.float32
        ).to(DEVICE).eval()

        mean_l = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=DEVICE).view(1, 3, 1, 1)
        std_l = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=DEVICE).view(1, 3, 1, 1)

        def preprocess_clip_l(x):
            return (F.interpolate(x, (224, 224), mode="bilinear", align_corners=False) - mean_l) / std_l

        def embed_clip_l(x):
            return clip_l(x).image_embeds

        models.append(VisionModel("clip_vit_l14", clip_l, preprocess_clip_l, embed_clip_l, weight=1.5))
        logger.info("  ✓ CLIP ViT-L/14 loaded (weight=1.5, LLaVA target)")
    except Exception as e:
        logger.warning("  ✗ CLIP ViT-L/14 failed: %s", e)

    # --- 3. CLIP ViT-B/32 (OpenAI, smallest) ---
    try:
        from transformers import CLIPVisionModelWithProjection
        logger.info("Loading CLIP ViT-B/32 (OpenAI)...")
        clip_b = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-base-patch32", torch_dtype=torch.float32
        ).to(DEVICE).eval()

        def preprocess_clip_b(x):
            return (F.interpolate(x, (224, 224), mode="bilinear", align_corners=False) - mean_l) / std_l

        def embed_clip_b(x):
            return clip_b(x).image_embeds

        models.append(VisionModel("clip_vit_b32", clip_b, preprocess_clip_b, embed_clip_b, weight=1.0))
        logger.info("  ✓ CLIP ViT-B/32 loaded (weight=1.0)")
    except Exception as e:
        logger.warning("  ✗ CLIP ViT-B/32 failed: %s", e)

    # --- 4. SigLIP ViT-SO400M (Google, Gemini-like) ---
    try:
        from transformers import SiglipVisionModel, SiglipImageProcessor
        logger.info("Loading SigLIP ViT-SO400M...")
        siglip = SiglipVisionModel.from_pretrained(
            "google/siglip-so400m-patch14-384", torch_dtype=torch.float32
        ).to(DEVICE).eval()

        siglip_proc = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        mean_s = torch.tensor(siglip_proc.image_mean, device=DEVICE).view(1, 3, 1, 1)
        std_s = torch.tensor(siglip_proc.image_std, device=DEVICE).view(1, 3, 1, 1)
        siglip_size = siglip_proc.size.get("height", 384)

        def preprocess_siglip(x):
            return (F.interpolate(x, (siglip_size, siglip_size), mode="bilinear", align_corners=False) - mean_s) / std_s

        def embed_siglip(x):
            out = siglip(x)
            # SigLIP doesn't have projection by default, use pooled output
            return out.pooler_output

        models.append(VisionModel("siglip_so400m", siglip, preprocess_siglip, embed_siglip, weight=1.5))
        logger.info("  ✓ SigLIP ViT-SO400M loaded (weight=1.5, Gemini target)")
    except Exception as e:
        logger.warning("  ✗ SigLIP failed: %s", e)

    # --- 5. DINOv2 ViT-L/14 (Meta, self-supervised) ---
    try:
        logger.info("Loading DINOv2 ViT-L/14 (Meta)...")
        dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", verbose=False)
        dinov2 = dinov2.to(DEVICE).eval()

        mean_d = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
        std_d = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

        def preprocess_dinov2(x):
            # DINOv2 uses 518x518 (14*37) but 224 works fine
            return (F.interpolate(x, (224, 224), mode="bilinear", align_corners=False) - mean_d) / std_d

        def embed_dinov2(x):
            return dinov2(x)  # CLS token embedding

        models.append(VisionModel("dinov2_vitl14", dinov2, preprocess_dinov2, embed_dinov2, weight=1.0))
        logger.info("  ✓ DINOv2 ViT-L/14 loaded (weight=1.0, self-supervised)")
    except Exception as e:
        logger.warning("  ✗ DINOv2 failed: %s", e)

    logger.info("Loaded %d/%d ensemble models", len(models), 5)
    return models


def load_text_encoder():
    """Load CLIP text encoder for targeted attacks."""
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer

    CLIP_MODEL_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    logger.info("Loading CLIP text encoder...")
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        CLIP_MODEL_ID, torch_dtype=torch.float32
    ).to(DEVICE).eval()
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_ID)

    return text_encoder, tokenizer


def get_text_embedding(text, text_encoder, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        return text_encoder(**inputs).text_embeds


# ===================================================================
# Ensemble PGD Attack
# ===================================================================

def ensemble_pgd_targeted(
    image_pil: Image.Image,
    target_text_emb: torch.Tensor,
    ensemble: list,
    eps: float,
    num_iters: int = 300,
    use_momentum: bool = True,
    momentum_decay: float = 0.9,
):
    """
    PGD attack that maximizes cosine similarity to target concept
    across ALL models in the ensemble simultaneously.

    Uses MI-FGSM (momentum iterative FGSM) for better transferability.
    """
    step_size = max(eps / 10, 0.5 / 255)

    img_np = np.array(center_crop_square(image_pil, IMG_SIZE)).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # Normalize target embedding
    target = F.normalize(target_text_emb.detach(), dim=-1)

    # Initialize perturbation
    delta = torch.zeros_like(img_tensor, requires_grad=True)
    delta.data.uniform_(-eps, eps)
    delta.data.clamp_(-img_tensor, 1.0 - img_tensor)

    best_delta = delta.data.clone()
    best_total_loss = float("inf")

    # Momentum buffer
    momentum = torch.zeros_like(img_tensor)

    # Get original embeddings for each model
    orig_embeddings = {}
    with torch.no_grad():
        for model in ensemble:
            orig_embeddings[model.name] = F.normalize(
                model.get_embedding(img_tensor), dim=-1
            )

    for i in range(num_iters):
        delta.requires_grad_(True)
        adv_tensor = (img_tensor + delta).clamp(0, 1)

        total_loss = torch.tensor(0.0, device=DEVICE)
        model_losses = {}

        for model in ensemble:
            adv_emb = model.get_embedding(adv_tensor)
            adv_emb_norm = F.normalize(adv_emb, dim=-1)

            # For CLIP models: maximize similarity to target concept
            if "clip" in model.name or "siglip" in model.name:
                # Target dimension might not match for SigLIP
                if adv_emb_norm.shape[-1] == target.shape[-1]:
                    cos_to_target = F.cosine_similarity(adv_emb_norm, target).mean()
                    loss = -cos_to_target * model.weight
                else:
                    # For dimension mismatch: maximize distance from original
                    cos_to_orig = F.cosine_similarity(
                        adv_emb_norm, orig_embeddings[model.name]
                    ).mean()
                    loss = cos_to_orig * model.weight
            else:
                # For non-CLIP models (DINOv2): maximize distance from original
                cos_to_orig = F.cosine_similarity(
                    adv_emb_norm, orig_embeddings[model.name]
                ).mean()
                loss = cos_to_orig * model.weight

            model_losses[model.name] = loss.item()
            total_loss = total_loss + loss

        total_loss.backward()

        with torch.no_grad():
            # MI-FGSM: use momentum for better transferability
            grad = delta.grad.data
            # L1 normalize gradient (improves transfer)
            grad = grad / (grad.abs().mean(dim=[1, 2, 3], keepdim=True) + 1e-8)

            if use_momentum:
                momentum = momentum_decay * momentum + grad
                update = momentum.sign()
            else:
                update = grad.sign()

            delta.data = delta.data - step_size * update
            delta.data.clamp_(-eps, eps)
            delta.data.clamp_(-img_tensor, 1.0 - img_tensor)

            if total_loss.item() < best_total_loss:
                best_total_loss = total_loss.item()
                best_delta = delta.data.clone()

        delta.grad.zero_()

        if (i + 1) % 50 == 0:
            losses_str = " | ".join(f"{k}={v:.4f}" for k, v in model_losses.items())
            logger.info("  iter %d/%d: total=%.4f | %s", i + 1, num_iters, total_loss.item(), losses_str)

    # Final image
    adv_tensor = (img_tensor + best_delta).clamp(0, 1)
    adv_np = (adv_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(adv_np)


# ===================================================================
# Evaluation
# ===================================================================

def evaluate_attack(original, adversarial, ensemble, target_emb=None):
    """Measure how much each model's embedding changed."""
    img_orig = torch.from_numpy(
        np.array(center_crop_square(original, IMG_SIZE)).astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    img_adv = torch.from_numpy(
        np.array(center_crop_square(adversarial, IMG_SIZE)).astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    results = {}
    with torch.no_grad():
        for model in ensemble:
            orig_emb = F.normalize(model.get_embedding(img_orig), dim=-1)
            adv_emb = F.normalize(model.get_embedding(img_adv), dim=-1)

            cos_shift = F.cosine_similarity(orig_emb, adv_emb).item()

            cos_target = None
            if target_emb is not None and ("clip" in model.name):
                t = F.normalize(target_emb, dim=-1)
                if adv_emb.shape[-1] == t.shape[-1]:
                    cos_target = F.cosine_similarity(adv_emb, t).item()

            results[model.name] = {
                "cos_to_original": cos_shift,
                "cos_to_target": cos_target,
            }

    return results


# ===================================================================
# Main
# ===================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("ENSEMBLE VISION MODEL ATTACK")
    logger.info("  Target concept: '%s' (%s)", args.target_concept, args.target_name)
    logger.info("  Eps: %d/255", args.eps)
    logger.info("  PGD iters: %d", args.pgd_iters)
    logger.info("=" * 70)

    # Load image
    if args.image:
        original = Image.open(args.image).convert("RGB")
    else:
        orig_path = os.path.join(args.images_dir, "keanu", "1_keanu_original.png")
        if not os.path.exists(orig_path):
            orig_path = os.path.join(args.images_dir, "original_512.png")
        original = Image.open(orig_path).convert("RGB")
    original = center_crop_square(original, IMG_SIZE)
    original.save(os.path.join(args.output_dir, "00_original.png"))

    # Load ensemble
    ensemble = load_ensemble()
    if not ensemble:
        logger.error("No models loaded!")
        sys.exit(1)

    # Load text encoder for targeted attack
    text_encoder, tokenizer = load_text_encoder()
    target_emb = get_text_embedding(args.target_concept, text_encoder, tokenizer)

    # Free text encoder
    del text_encoder, tokenizer
    torch.cuda.empty_cache()

    # Run ensemble attack
    logger.info("Running ensemble PGD attack...")
    t0 = time.time()
    adversarial = ensemble_pgd_targeted(
        original, target_emb, ensemble,
        eps=args.eps / 255.0,
        num_iters=args.pgd_iters,
    )
    elapsed = time.time() - t0
    logger.info("Ensemble attack complete in %.1fs", elapsed)

    adversarial.save(os.path.join(args.output_dir, f"01_ensemble_eps{args.eps}_{args.target_name}.png"))

    # Also run single-model attacks for comparison
    for model in ensemble:
        if "clip_vit_h" in model.name:
            logger.info("Running single-model attack (CLIP ViT-H only)...")
            single = ensemble_pgd_targeted(
                original, target_emb, [model],
                eps=args.eps / 255.0,
                num_iters=args.pgd_iters,
            )
            single.save(os.path.join(args.output_dir, f"02_single_clip_h_eps{args.eps}_{args.target_name}.png"))
            break

    # Evaluate
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION — Embedding shifts per model")
    logger.info("=" * 70)

    eval_ensemble = evaluate_attack(original, adversarial, ensemble, target_emb)
    logger.info("ENSEMBLE attack results:")
    for model_name, metrics in eval_ensemble.items():
        cos_o = metrics["cos_to_original"]
        cos_t = metrics["cos_to_target"]
        t_str = f", cos→target={cos_t:.4f}" if cos_t is not None else ""
        logger.info("  %-20s cos→original=%.4f%s", model_name, cos_o, t_str)

    if os.path.exists(os.path.join(args.output_dir, f"02_single_clip_h_eps{args.eps}_{args.target_name}.png")):
        single_img = Image.open(os.path.join(args.output_dir, f"02_single_clip_h_eps{args.eps}_{args.target_name}.png"))
        eval_single = evaluate_attack(original, single_img, ensemble, target_emb)
        logger.info("\nSINGLE (CLIP ViT-H only) attack results:")
        for model_name, metrics in eval_single.items():
            cos_o = metrics["cos_to_original"]
            cos_t = metrics["cos_to_target"]
            t_str = f", cos→target={cos_t:.4f}" if cos_t is not None else ""
            logger.info("  %-20s cos→original=%.4f%s", model_name, cos_o, t_str)

    # Optionally test with IP-Adapter
    if args.test_ip_adapter:
        logger.info("\nGenerating IP-Adapter test images...")
        from diffusers import StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            torch_dtype=torch.float16, variant="fp16"
        ).to(DEVICE)
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")

        # --- Multiple prompts to show attack works across scenarios ---
        test_prompts = [
            ("portrait", "a professional portrait photograph, studio lighting, 4k, sharp"),
            ("painting", "an oil painting in the style of Van Gogh, vivid colors, impasto"),
            ("fantasy", "a fantasy character portrait, magical glowing background, detailed"),
            ("photo", "a high quality detailed photograph, natural lighting, 4k"),
        ]

        # --- Multiple IP-adapter scales to show robustness ---
        test_scales = [0.5, 0.7, 1.0]

        GEN_SIZE = 512
        total = len(test_prompts) * len(test_scales)
        idx = 0

        for scale in test_scales:
            pipe.set_ip_adapter_scale(scale)
            for tag, prompt in test_prompts:
                idx += 1
                logger.info("  [%d/%d] scale=%.1f, prompt=%s", idx, total, scale, tag)

                # --- Generate from ORIGINAL image ---
                gen = torch.Generator(device=DEVICE).manual_seed(args.seed)
                out_orig = pipe(
                    prompt=prompt, ip_adapter_image=original,
                    num_inference_steps=50, generator=gen,
                    height=GEN_SIZE, width=GEN_SIZE,
                ).images[0]
                out_orig.save(os.path.join(
                    args.output_dir, f"ip_orig_s{scale:.1f}_{tag}.png"
                ))

                # --- Generate from ENSEMBLE-attacked image ---
                gen = torch.Generator(device=DEVICE).manual_seed(args.seed)
                out_ens = pipe(
                    prompt=prompt, ip_adapter_image=adversarial,
                    num_inference_steps=50, generator=gen,
                    height=GEN_SIZE, width=GEN_SIZE,
                ).images[0]
                out_ens.save(os.path.join(
                    args.output_dir, f"ip_attack_s{scale:.1f}_{tag}.png"
                ))

                logger.info("    saved: ip_orig_s%.1f_%s.png / ip_attack_s%.1f_%s.png",
                            scale, tag, scale, tag)

        # --- Also generate a comparison grid ---
        logger.info("  Building comparison grid...")
        try:
            cell_size = 256
            n_prompts = len(test_prompts)
            n_scales = len(test_scales)
            cols = 2 * n_prompts  # orig + attack for each prompt
            rows = n_scales + 1   # header row + one row per scale
            grid_w = cols * cell_size
            grid_h = rows * cell_size
            grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
            draw = ImageDraw.Draw(grid)

            # Header labels
            for j, (tag, _) in enumerate(test_prompts):
                x_orig = (2 * j) * cell_size
                x_atk = (2 * j + 1) * cell_size
                draw.text((x_orig + 10, 10), f"{tag}\nORIG", fill=(0, 128, 0))
                draw.text((x_atk + 10, 10), f"{tag}\nATTACK", fill=(200, 0, 0))

            for si, scale in enumerate(test_scales):
                y = (si + 1) * cell_size
                for j, (tag, _) in enumerate(test_prompts):
                    orig_path = os.path.join(args.output_dir, f"ip_orig_s{scale:.1f}_{tag}.png")
                    atk_path = os.path.join(args.output_dir, f"ip_attack_s{scale:.1f}_{tag}.png")

                    if os.path.exists(orig_path):
                        img_o = Image.open(orig_path).resize((cell_size, cell_size), Image.LANCZOS)
                        grid.paste(img_o, (2 * j * cell_size, y))
                    if os.path.exists(atk_path):
                        img_a = Image.open(atk_path).resize((cell_size, cell_size), Image.LANCZOS)
                        grid.paste(img_a, ((2 * j + 1) * cell_size, y))

                # Scale label on left margin
                draw.text((5, y + cell_size // 2 - 10), f"s={scale:.1f}", fill=(0, 0, 0))

            grid.save(os.path.join(args.output_dir, "comparison_grid.png"))
            logger.info("  Saved comparison_grid.png (%dx%d)", grid_w, grid_h)
        except Exception as e:
            logger.warning("  Grid generation failed: %s", e)

        del pipe
        torch.cuda.empty_cache()

    logger.info("\n" + "=" * 70)
    logger.info("DONE. Output: %s", args.output_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
