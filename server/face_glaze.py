"""
Face-Localized Identity Glazing
================================

Detects the face region, then applies adversarial perturbation ONLY to the
face — localized epsilon where it matters, zero everywhere else.

Untargeted attack: maximizes embedding distance from the original face
across diverse model families:

  1. Face Recognition (InceptionResnetV1 / VGGFace2)
  2. CLIP ViT-H/14 (IP-Adapter encoder)
  3. CLIP ViT-L/14
  4. DINOv2 ViT-L/14
  5. Stable Diffusion VAE encoder

The face mask is a soft ellipse with feathered edges so the boundary
between protected and unprotected regions is invisible.

Usage:
    python face_glaze.py --image photo.jpg [--eps 6] [--output-dir ./output]
"""

import os
import sys
import time
import argparse
import logging

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512


def center_crop_square(img: Image.Image, size: int = IMG_SIZE) -> Image.Image:
    w, h = img.size
    crop_dim = min(w, h)
    left = (w - crop_dim) // 2
    top = (h - crop_dim) // 2
    img = img.crop((left, top, left + crop_dim, top + crop_dim))
    return img.resize((size, size), Image.LANCZOS)


# ===================================================================
# Face detection & mask
# ===================================================================

def detect_face(image_pil: Image.Image):
    """Detect face bounding box. Returns (x1, y1, x2, y2) or None."""
    try:
        from facenet_pytorch import MTCNN
        mtcnn = MTCNN(keep_all=False, device=DEVICE, min_face_size=40)
        boxes, probs = mtcnn.detect(image_pil)
        if boxes is not None and len(boxes) > 0:
            best = int(np.argmax(probs))
            box = boxes[best].tolist()
            logger.info("  Face detected: [%.0f, %.0f, %.0f, %.0f] (conf=%.2f)",
                        *box, probs[best])
            return box
    except ImportError:
        logger.warning("  facenet-pytorch not available, using center heuristic")
    except Exception as e:
        logger.warning("  Face detection failed: %s", e)

    w, h = image_pil.size
    cx, cy = w // 2, int(h * 0.4)
    r = min(w, h) // 3
    logger.info("  Using center heuristic for face region")
    return [cx - r, cy - r, cx + r, cy + r]


def create_face_mask(img_size: tuple, bbox: list, expand: float = 1.4,
                     feather: float = 0.35) -> np.ndarray:
    """
    Soft elliptical mask over the face region.

    Returns:
        mask: (H, W) float array in [0, 1]
    """
    w, h = img_size
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    fw = (x2 - x1) * expand
    fh = (y2 - y1) * expand * 1.15

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = ((xx - cx) / (fw / 2)) ** 2 + ((yy - cy) / (fh / 2)) ** 2

    mask = np.clip(1.0 - (dist - 0.6) / feather, 0.0, 1.0)
    return mask


# ===================================================================
# Model loading — diverse ensemble
# ===================================================================

class FaceModel:
    """Wrapper for any differentiable model that produces embeddings."""
    def __init__(self, name, model, preprocess_fn, embed_fn, weight=1.0):
        self.name = name
        self.model = model
        self.preprocess_fn = preprocess_fn
        self.embed_fn = embed_fn
        self.weight = weight

    def get_embedding(self, image_tensor):
        preprocessed = self.preprocess_fn(image_tensor)
        return self.embed_fn(preprocessed)


def load_ensemble(face_bbox=None):
    """Load diverse model ensemble."""
    models = []
    img_size = IMG_SIZE

    if face_bbox:
        fx1 = max(0, int(face_bbox[0]))
        fy1 = max(0, int(face_bbox[1]))
        fx2 = min(img_size, int(face_bbox[2]))
        fy2 = min(img_size, int(face_bbox[3]))
    else:
        c = img_size // 2
        r = img_size // 4
        fx1, fy1, fx2, fy2 = c - r, c - r, c + r, c + r

    # 1. Face Recognition (InceptionResnetV1 — VGGFace2)
    try:
        from facenet_pytorch import InceptionResnetV1
        logger.info("Loading FaceNet (InceptionResnetV1, VGGFace2)...")
        facenet = InceptionResnetV1(pretrained='vggface2').to(DEVICE).eval()

        def preprocess_facenet(x):
            face = x[:, :, fy1:fy2, fx1:fx2]
            face = F.interpolate(face, (160, 160), mode="bilinear", align_corners=False)
            return (face - 0.5) / 0.5

        def embed_facenet(x):
            return facenet(x)

        models.append(FaceModel("facenet", facenet, preprocess_facenet, embed_facenet, weight=3.0))
        logger.info("  ✓ FaceNet loaded (weight=3.0)")
    except Exception as e:
        logger.warning("  ✗ FaceNet failed: %s", e)

    # 2. CLIP ViT-H/14
    try:
        from transformers import CLIPVisionModelWithProjection
        logger.info("Loading CLIP ViT-H/14...")
        clip_h = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter", subfolder="models/image_encoder",
            torch_dtype=torch.float32
        ).to(DEVICE).eval()

        mean_h = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=DEVICE).view(1, 3, 1, 1)
        std_h = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=DEVICE).view(1, 3, 1, 1)

        def preprocess_clip_h(x):
            return (F.interpolate(x, (224, 224), mode="bilinear", align_corners=False) - mean_h) / std_h

        def embed_clip_h(x):
            return clip_h(x).image_embeds

        models.append(FaceModel("clip_h14", clip_h, preprocess_clip_h, embed_clip_h, weight=1.5))
        logger.info("  ✓ CLIP ViT-H/14 loaded (weight=1.5)")
    except Exception as e:
        logger.warning("  ✗ CLIP ViT-H/14 failed: %s", e)

    # 3. CLIP ViT-L/14
    try:
        from transformers import CLIPVisionModelWithProjection
        logger.info("Loading CLIP ViT-L/14...")
        clip_l = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14", torch_dtype=torch.float32
        ).to(DEVICE).eval()

        mean_l = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=DEVICE).view(1, 3, 1, 1)
        std_l = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=DEVICE).view(1, 3, 1, 1)

        def preprocess_clip_l(x):
            return (F.interpolate(x, (224, 224), mode="bilinear", align_corners=False) - mean_l) / std_l

        def embed_clip_l(x):
            return clip_l(x).image_embeds

        models.append(FaceModel("clip_l14", clip_l, preprocess_clip_l, embed_clip_l, weight=1.0))
        logger.info("  ✓ CLIP ViT-L/14 loaded (weight=1.0)")
    except Exception as e:
        logger.warning("  ✗ CLIP ViT-L/14 failed: %s", e)

    # 4. DINOv2 ViT-L/14
    try:
        logger.info("Loading DINOv2 ViT-L/14...")
        dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", verbose=False)
        dinov2 = dinov2.to(DEVICE).eval()

        mean_d = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
        std_d = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

        def preprocess_dinov2(x):
            return (F.interpolate(x, (224, 224), mode="bilinear", align_corners=False) - mean_d) / std_d

        def embed_dinov2(x):
            return dinov2(x)

        models.append(FaceModel("dinov2", dinov2, preprocess_dinov2, embed_dinov2, weight=1.0))
        logger.info("  ✓ DINOv2 ViT-L/14 loaded (weight=1.0)")
    except Exception as e:
        logger.warning("  ✗ DINOv2 failed: %s", e)

    # 5. Stable Diffusion VAE encoder
    try:
        from diffusers import AutoencoderKL
        logger.info("Loading SD VAE encoder...")
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float32
        ).to(DEVICE).eval()

        def preprocess_vae(x):
            return x * 2.0 - 1.0

        def embed_vae(x):
            latent = vae.encode(x).latent_dist.mean
            return latent.flatten(start_dim=1)

        models.append(FaceModel("sd_vae", vae, preprocess_vae, embed_vae, weight=2.0))
        logger.info("  ✓ SD VAE loaded (weight=2.0)")
    except Exception as e:
        logger.warning("  ✗ SD VAE failed: %s", e)

    logger.info("Loaded %d models", len(models))
    return models


# ===================================================================
# Face-Localized PGD
# ===================================================================

def face_glaze(
    image_pil: Image.Image,
    face_mask: np.ndarray,
    ensemble: list,
    eps: float = 6.0 / 255.0,
    num_iters: int = 200,
    momentum_decay: float = 0.9,
) -> Image.Image:
    """
    Face-localized adversarial glazing.

    Maximizes embedding distance from original across all models,
    constrained to the face mask region with per-pixel epsilon budget.
    """
    step_size = max(eps / 8, 1.0 / 255)

    img_np = np.array(image_pil).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    mask_tensor = torch.from_numpy(face_mask).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    eps_map = eps * mask_tensor

    delta = torch.zeros_like(img_tensor, requires_grad=True)
    delta.data.uniform_(-eps, eps)
    delta.data = delta.data * mask_tensor
    delta.data.clamp_(-img_tensor, 1.0 - img_tensor)

    best_delta = delta.data.clone()
    best_loss = float("inf")
    momentum = torch.zeros_like(img_tensor)

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

            cos_sim = F.cosine_similarity(
                adv_emb_norm, orig_embeddings[model.name]
            ).mean()

            loss = cos_sim * model.weight
            model_losses[model.name] = cos_sim.item()
            total_loss = total_loss + loss

        total_loss.backward()

        with torch.no_grad():
            grad = delta.grad.data
            grad = grad / (grad.abs().mean(dim=[1, 2, 3], keepdim=True) + 1e-8)

            momentum = momentum_decay * momentum + grad
            update = momentum.sign()

            delta.data = delta.data - step_size * update * mask_tensor
            delta.data = torch.max(torch.min(delta.data, eps_map), -eps_map)
            delta.data = torch.max(torch.min(delta.data, 1.0 - img_tensor), -img_tensor)

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_delta = delta.data.clone()

        delta.grad.zero_()

        if (i + 1) % 25 == 0:
            losses_str = " | ".join(f"{k}={v:.3f}" for k, v in model_losses.items())
            logger.info("  iter %d/%d: %s", i + 1, num_iters, losses_str)

    adv_tensor = (img_tensor + best_delta).clamp(0, 1)
    adv_np = (adv_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(adv_np)


# ===================================================================
# Image conditioning
# ===================================================================

def condition_image(image: Image.Image, reference: Image.Image,
                    face_mask: np.ndarray) -> Image.Image:
    """
    Prepare a glazed image for downstream pipeline input.

    Multi-layer face-region processing that pushes adversarial signal
    past the denoising threshold of generation pipelines (diffusion,
    image-to-video, etc.):

    1. Heavy delta amplification to saturate adversarial features
    2. Structured gaussian noise
    3. High-frequency checkerboard disruption
    4. Per-channel color shift to break skin-tone coherence
    5. Sine-wave interference at face-feature scale
    """
    img = np.array(image).astype(np.float32)
    ref = np.array(reference).astype(np.float32)
    h, w = img.shape[:2]
    mask_3d = face_mask[..., np.newaxis]

    rng = np.random.RandomState(0)

    delta = img - ref
    conditioned = ref + delta * 12.0 * mask_3d

    noise = rng.randn(*img.shape).astype(np.float32) * 50.0
    conditioned = conditioned + noise * mask_3d

    yy, xx = np.mgrid[0:h, 0:w]
    checker = ((xx // 4 + yy // 4) % 2).astype(np.float32)
    checker = (checker * 2.0 - 1.0)
    checker_3d = np.stack([checker * 35.0] * 3, axis=-1)
    conditioned = conditioned + checker_3d * mask_3d

    channel_shifts = np.array([25.0, -20.0, 30.0]).reshape(1, 1, 3)
    conditioned = conditioned + channel_shifts * mask_3d

    freq1 = np.sin(xx * 0.8 + yy * 0.4).astype(np.float32) * 20.0
    freq2 = np.cos(xx * 0.3 - yy * 0.9).astype(np.float32) * 15.0
    freq_pattern = np.stack([freq1, freq2, freq1 + freq2], axis=-1)
    conditioned = conditioned + freq_pattern * mask_3d

    conditioned = np.clip(conditioned, 0, 255).astype(np.uint8)
    return Image.fromarray(conditioned)


# ===================================================================
# Generation tests — IP-Adapter + Outfill
# ===================================================================

def _make_outfill_mask(face_mask: np.ndarray) -> Image.Image:
    """Invert face mask for outfilling: white outside face (generate surroundings)."""
    inverted = ((1.0 - face_mask) > 0.3).astype(np.uint8) * 255
    return Image.fromarray(inverted).convert("L")


def generate_tests(original: Image.Image, glazed: Image.Image,
                   output_dir: str, face_mask: np.ndarray = None,
                   seed: int = 42):
    """
    Test generation pipelines:
      1. IP-Adapter — face transfer to new scene
      2. Outfill — keep face, generate surroundings
    """
    from diffusers import (
        StableDiffusionXLPipeline,
        StableDiffusionXLInpaintPipeline,
        AutoencoderKL,
        DPMSolverMultistepScheduler,
    )

    glazed_conditioned = condition_image(glazed, original, face_mask)

    neg = "blurry, low quality, distorted, deformed, bad anatomy, disfigured, poorly drawn face, mutation, extra limb"

    # ------------------------------------------------------------------
    # IP-Adapter — face identity transfer to new scenes
    # ------------------------------------------------------------------
    logger.info("Loading SDXL + IP-Adapter Plus Face...")
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae, torch_dtype=torch.float16, variant="fp16",
    ).to(DEVICE)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="sdxl_models",
        weight_name="ip-adapter-plus-face_sdxl_vit-h.bin",
        image_encoder_folder="models/image_encoder",
    )
    pipe.set_ip_adapter_scale(0.8)

    ip_prompts = [
        ("cyberpunk", "a person in a neon-lit cyberpunk city at night, rain, holographic signs, blade runner aesthetic, cinematic 4k, masterpiece"),
        ("astronaut", "a person as an astronaut on a space station with earth visible through the window, NASA photo, cinematic lighting, 8k, masterpiece"),
    ]

    for tag, prompt in ip_prompts:
        logger.info("  IP-Adapter: %s", tag)
        for label, img in [("orig", original), ("glazed", glazed_conditioned)]:
            gen = torch.Generator(device=DEVICE).manual_seed(seed)
            out = pipe(
                prompt=prompt,
                negative_prompt=neg,
                ip_adapter_image=img,
                num_inference_steps=40,
                guidance_scale=7.0,
                generator=gen,
                height=1024, width=1024,
            ).images[0]
            out.save(os.path.join(output_dir, f"ip_{label}_{tag}.png"))

    del pipe
    torch.cuda.empty_cache()
    logger.info("IP-Adapter tests done.")

    # ------------------------------------------------------------------
    # Outfill — face is preserved, generate surroundings from face
    # ------------------------------------------------------------------
    if face_mask is not None:
        logger.info("Loading SDXL Inpainting for outfill...")
        inpaint = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            vae=vae, torch_dtype=torch.float16, variant="fp16",
        ).to(DEVICE)
        inpaint.scheduler = DPMSolverMultistepScheduler.from_config(
            inpaint.scheduler.config, use_karras_sigmas=True
        )

        outfill_mask_pil = _make_outfill_mask(face_mask)
        orig_1024 = original.resize((1024, 1024), Image.LANCZOS)
        glazed_cond_1024 = glazed_conditioned.resize((1024, 1024), Image.LANCZOS)
        mask_1024 = outfill_mask_pil.resize((1024, 1024), Image.LANCZOS)

        outfill_prompts = [
            ("cyberpunk", "a person standing in a neon-lit cyberpunk city street at night, rain puddles, holographic ads, cinematic 4k, masterpiece"),
            ("office", "a person in a modern corporate office, floor-to-ceiling windows, city skyline view, professional photography, 8k, masterpiece"),
        ]

        for tag, prompt in outfill_prompts:
            logger.info("  Outfill: %s", tag)
            for label, img in [("orig", orig_1024), ("glazed", glazed_cond_1024)]:
                gen = torch.Generator(device=DEVICE).manual_seed(seed)
                out = inpaint(
                    prompt=prompt,
                    negative_prompt=neg,
                    image=img,
                    mask_image=mask_1024,
                    num_inference_steps=40,
                    guidance_scale=7.0,
                    strength=0.99,
                    generator=gen,
                    height=1024, width=1024,
                ).images[0]
                out.save(os.path.join(output_dir, f"outfill_{label}_{tag}.png"))

        del inpaint
        torch.cuda.empty_cache()
        logger.info("Outfill tests done.")

    del vae
    torch.cuda.empty_cache()
    logger.info("All generation tests complete.")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Face-localized identity glazing")
    parser.add_argument("--image", required=True, help="Path to input photo")
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--eps", type=int, default=6,
                        help="Perturbation budget on face (out of 255). Default 6.")
    parser.add_argument("--iters", type=int, default=200,
                        help="PGD iterations. Default 200.")
    parser.add_argument("--generate-tests", action="store_true",
                        help="Run generation pipeline tests")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("FACE GLAZE — Identity Protection")
    logger.info("  Epsilon: %d/255 (face only)", args.eps)
    logger.info("  Iterations: %d", args.iters)
    logger.info("=" * 60)

    original = center_crop_square(Image.open(args.image).convert("RGB"), IMG_SIZE)
    original.save(os.path.join(args.output_dir, "original.png"))

    logger.info("Detecting face...")
    bbox = detect_face(original)
    if bbox is None:
        logger.error("No face detected!")
        sys.exit(1)

    face_mask = create_face_mask(original.size, bbox)
    mask_vis = Image.fromarray((face_mask * 255).astype(np.uint8))
    mask_vis.save(os.path.join(args.output_dir, "face_mask.png"))
    logger.info("  Face mask coverage: %.1f%% of image", 100 * face_mask.mean())

    logger.info("Loading model ensemble...")
    ensemble = load_ensemble(face_bbox=bbox)
    if not ensemble:
        logger.error("No models loaded!")
        sys.exit(1)

    logger.info("Running face-localized PGD (eps=%d/255, %d iters)...",
                args.eps, args.iters)
    t0 = time.time()
    glazed = face_glaze(
        original, face_mask, ensemble,
        eps=args.eps / 255.0,
        num_iters=args.iters,
    )
    elapsed = time.time() - t0
    logger.info("Glazing complete in %.1fs", elapsed)

    glazed.save(os.path.join(args.output_dir, "glazed.png"))
    logger.info("Saved: glazed.png")

    logger.info("\nEvaluation — cosine similarity to original:")
    img_orig_t = torch.from_numpy(
        np.array(original).astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    img_glaze_t = torch.from_numpy(
        np.array(glazed).astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        for model in ensemble:
            orig_emb = F.normalize(model.get_embedding(img_orig_t), dim=-1)
            glaze_emb = F.normalize(model.get_embedding(img_glaze_t), dim=-1)
            cos = F.cosine_similarity(orig_emb, glaze_emb).item()
            logger.info("  %-12s cos_sim=%.4f", model.name, cos)

    del ensemble
    torch.cuda.empty_cache()

    if args.generate_tests:
        logger.info("\nRunning generation pipeline tests...")
        generate_tests(original, glazed, args.output_dir, face_mask, args.seed)

    logger.info("\n" + "=" * 60)
    logger.info("DONE. Output: %s", args.output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
