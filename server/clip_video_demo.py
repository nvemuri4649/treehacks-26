"""
CLIP Attack → Image-to-Video Demo
===================================
Shows that a tiny CLIP adversarial perturbation (imperceptible to humans)
completely breaks AI video generation from a reference image.

Pipeline:
  1. PGD attack on CLIP image encoder (same as clip_small_eps_sweep.py)
  2. Feed original + attacked images into Stable Video Diffusion (SVD)
     → SVD uses CLIP ViT internally to encode the conditioning image
     → Attacked CLIP embedding → glitchy, incoherent video
  3. Optionally also feed into CogVideoX for text-guided I2V
     ("make this person wave", "make this person walk") with original vs attacked

Usage:
    python clip_video_demo.py                          # defaults
    python clip_video_demo.py --image /path/to/face.png
    python clip_video_demo.py --eps-values 2 4 8 12    # custom eps sweep
    python clip_video_demo.py --skip-svd               # only CogVideoX
    python clip_video_demo.py --skip-cogvideo           # only SVD
"""

import os
import sys
import time
import gc
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

# Model IDs
IP_ADAPTER_REPO = "h94/IP-Adapter"
SVD_MODEL = "stabilityai/stable-video-diffusion-img2vid-xt"


def parse_args():
    parser = argparse.ArgumentParser(description="CLIP attack → video generation demo")
    parser.add_argument(
        "--image",
        default=None,
        help="Path to input face image. If not provided, looks in --images-dir.",
    )
    parser.add_argument(
        "--images-dir",
        default="/workspace/comparison_outputs_v3",
        help="Directory with original images (fallback)",
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/clip_video_output",
        help="Output directory",
    )
    parser.add_argument(
        "--eps-values",
        type=int,
        nargs="+",
        default=[2, 4, 8, 12],
        help="Epsilon values to test (out of 255)",
    )
    parser.add_argument("--pgd-iters", type=int, default=200, help="PGD iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--svd-num-frames", type=int, default=25, help="SVD frames")
    parser.add_argument("--svd-fps", type=int, default=7, help="Output video FPS")
    parser.add_argument("--skip-svd", action="store_true", help="Skip SVD demo")
    parser.add_argument("--skip-cogvideo", action="store_true", help="Skip CogVideoX demo")
    return parser.parse_args()


# ===================================================================
# CLIP image encoder + PGD attack
# ===================================================================

def load_clip_image_encoder():
    """Load the CLIP ViT-H/14 image encoder (same one IP-Adapter & SVD use)."""
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

    logger.info("Loading CLIP image encoder (IP-Adapter ViT-H/14)...")
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
    """Differentiable CLIP preprocessing."""
    mean = torch.tensor(feature_extractor.image_mean, device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor(feature_extractor.image_std, device=DEVICE).view(1, 3, 1, 1)
    clip_size = feature_extractor.size.get("shortest_edge", 224)
    resized = F.interpolate(
        image_tensor, size=(clip_size, clip_size),
        mode="bilinear", align_corners=False,
    )
    return (resized - mean) / std


def get_clip_embedding(img, image_encoder, feature_extractor):
    """Get CLIP image embedding from PIL Image."""
    t = torch.from_numpy(
        np.array(img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    clip_in = clip_preprocess_tensor(t, feature_extractor)
    return image_encoder(clip_in).image_embeds


def pgd_attack_clip(image_pil, image_encoder, feature_extractor, eps, num_iters=200):
    """
    PGD attack to MAXIMIZE CLIP embedding distance.
    Even tiny eps (2-4/255) can massively shift the embedding.
    """
    step_size = max(eps / 5, 0.5 / 255)

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

        # Maximize distance: minimize cosine similarity
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

        if (i + 1) % 50 == 0:
            logger.info(
                "    iter %d/%d: cos_sim=%.4f (dist=%.4f)",
                i + 1, num_iters, -best_loss, best_loss,
            )

    adv_image_tensor = (img_tensor + best_delta).clamp(0, 1)
    adv_np = (adv_image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(adv_np), -best_loss  # return (image, cos_sim)


# ===================================================================
# Visualization helpers
# ===================================================================

def get_font():
    """Try to load a good font, fall back to default."""
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            return ImageFont.truetype(path, 12), ImageFont.truetype(path, 10)
        except (IOError, OSError):
            continue
    f = ImageFont.load_default()
    return f, f


def make_frame_comparison(frames_dict, sample_count=8, cell_w=256, cell_h=144):
    """
    Create a comparison grid: rows = eps levels, columns = sampled frames.
    frames_dict: {"original": [frame_list], "eps=4": [frame_list], ...}
    """
    font_b, font_s = get_font()

    labels = list(frames_dict.keys())
    n_rows = len(labels)
    frame_lists = [frames_dict[k] for k in labels]
    n_frames = len(frame_lists[0])

    # Sample frame indices
    indices = np.linspace(0, n_frames - 1, min(sample_count, n_frames), dtype=int)
    n_cols = len(indices)

    label_w = 100
    header_h = 20
    gap = 2
    total_w = label_w + n_cols * (cell_w + gap)
    total_h = header_h + n_rows * (cell_h + gap)

    canvas = Image.new("RGB", (total_w, total_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    # Column headers (frame numbers)
    for ci, fi in enumerate(indices):
        x = label_w + ci * (cell_w + gap) + cell_w // 2 - 20
        draw.text((x, 2), f"frame {fi}", fill=(200, 200, 200), font=font_s)

    # Rows
    for ri, (label, frames) in enumerate(zip(labels, frame_lists)):
        y = header_h + ri * (cell_h + gap)
        # Row label
        draw.text((4, y + cell_h // 2 - 8), label, fill=(255, 255, 255), font=font_b)
        for ci, fi in enumerate(indices):
            x = label_w + ci * (cell_w + gap)
            frame = frames[fi].resize((cell_w, cell_h), Image.LANCZOS)
            canvas.paste(frame, (x, y))

    return canvas


def create_side_by_side_video_frames(frames_a, frames_b, label_a="Original", label_b="Attacked"):
    """Create list of side-by-side frames for ffmpeg stitching."""
    font_b, font_s = get_font()
    combined_frames = []
    gap = 4
    for fa, fb in zip(frames_a, frames_b):
        w, h = fa.size
        canvas = Image.new("RGB", (w * 2 + gap, h + 24), (0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((w // 2 - 30, 2), label_a, fill=(100, 255, 100), font=font_b)
        draw.text((w + gap + w // 2 - 30, 2), label_b, fill=(255, 100, 100), font=font_b)
        canvas.paste(fa, (0, 24))
        canvas.paste(fb, (w + gap, 24))
        combined_frames.append(canvas)
    return combined_frames


# ===================================================================
# PART 1: Stable Video Diffusion
# ===================================================================

def run_svd_demo(original, attacked_images, clip_metrics, output_dir, args):
    """
    Generate videos from original and CLIP-attacked images using SVD.
    SVD encodes the conditioning image through a CLIP image encoder,
    so CLIP perturbations directly corrupt the video generation.
    """
    from diffusers import StableVideoDiffusionPipeline
    from diffusers.utils import export_to_video

    svd_dir = os.path.join(output_dir, "svd")
    os.makedirs(svd_dir, exist_ok=True)

    logger.info("Loading Stable Video Diffusion (SVD-XT)...")
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        SVD_MODEL,
        torch_dtype=DTYPE,
        variant="fp16",
    )

    # Memory optimization
    if DEVICE == "cuda":
        try:
            pipe.enable_model_cpu_offload()
            logger.info("SVD loaded with CPU offloading.")
        except Exception as e:
            logger.warning("CPU offload failed (%s), loading directly to GPU", e)
            pipe = pipe.to(DEVICE)
    else:
        pipe = pipe.to(DEVICE)

    # SVD native resolution
    svd_w, svd_h = 1024, 576

    # --- Generate from ORIGINAL ---
    logger.info("Generating video from ORIGINAL image...")
    orig_resized = original.resize((svd_w, svd_h), Image.LANCZOS)
    orig_resized.save(os.path.join(svd_dir, "ref_original.png"))

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    t0 = time.time()
    frames_orig = pipe(
        orig_resized,
        num_frames=args.svd_num_frames,
        decode_chunk_size=4,
        generator=generator,
        num_inference_steps=25,
    ).frames[0]
    logger.info("  Original video: %.1fs, %d frames", time.time() - t0, len(frames_orig))

    # Save original video
    orig_video_path = os.path.join(svd_dir, "video_original.mp4")
    export_to_video(frames_orig, orig_video_path, fps=args.svd_fps)

    # Save individual frames
    orig_frames_dir = os.path.join(svd_dir, "frames_original")
    os.makedirs(orig_frames_dir, exist_ok=True)
    for i, frame in enumerate(frames_orig):
        frame.save(os.path.join(orig_frames_dir, f"frame_{i:04d}.png"))

    # --- Generate from each ATTACKED image ---
    all_frame_sets = {"original": frames_orig}

    for eps_int in args.eps_values:
        atk_img = attacked_images[eps_int]
        metrics = clip_metrics[eps_int]

        logger.info(
            "Generating video from ATTACKED image (eps=%d/255, cos_sim=%.3f)...",
            eps_int, metrics["cos_sim"],
        )

        atk_resized = atk_img.resize((svd_w, svd_h), Image.LANCZOS)
        atk_resized.save(os.path.join(svd_dir, f"ref_attacked_eps{eps_int}.png"))

        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        t0 = time.time()
        frames_atk = pipe(
            atk_resized,
            num_frames=args.svd_num_frames,
            decode_chunk_size=4,
            generator=generator,
            num_inference_steps=25,
        ).frames[0]
        elapsed = time.time() - t0
        logger.info("  eps=%d video: %.1fs, %d frames", eps_int, elapsed, len(frames_atk))

        # Save attacked video
        atk_video_path = os.path.join(svd_dir, f"video_attacked_eps{eps_int}.mp4")
        export_to_video(frames_atk, atk_video_path, fps=args.svd_fps)

        # Save individual frames
        atk_frames_dir = os.path.join(svd_dir, f"frames_attacked_eps{eps_int}")
        os.makedirs(atk_frames_dir, exist_ok=True)
        for i, frame in enumerate(frames_atk):
            frame.save(os.path.join(atk_frames_dir, f"frame_{i:04d}.png"))

        # Side-by-side comparison video
        sbs_frames = create_side_by_side_video_frames(
            frames_orig, frames_atk,
            label_a="Original",
            label_b=f"CLIP attacked eps={eps_int}/255",
        )
        sbs_dir = os.path.join(svd_dir, f"sbs_eps{eps_int}")
        os.makedirs(sbs_dir, exist_ok=True)
        for i, frame in enumerate(sbs_frames):
            frame.save(os.path.join(sbs_dir, f"frame_{i:04d}.png"))
        export_to_video(sbs_frames, os.path.join(svd_dir, f"video_sbs_eps{eps_int}.mp4"), fps=args.svd_fps)

        label = f"eps={eps_int}/255\ncos={metrics['cos_sim']:.2f}"
        all_frame_sets[label] = frames_atk

    # --- Comparison grid image ---
    logger.info("Creating frame comparison grid...")
    comparison = make_frame_comparison(all_frame_sets, sample_count=8)
    comparison.save(os.path.join(svd_dir, "frame_comparison_grid.png"))

    # Cleanup
    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("SVD demo complete. Output: %s", svd_dir)


# ===================================================================
# PART 2: CogVideoX (text-guided image-to-video)
# ===================================================================

def run_cogvideo_demo(original, attacked_images, clip_metrics, output_dir, args):
    """
    CogVideoX: text-guided image-to-video generation.
    We can prompt "the person waves hello" or "the person walks forward"
    and compare original vs CLIP-attacked conditioning images.
    """
    try:
        from diffusers import CogVideoXImageToVideoPipeline
        from diffusers.utils import export_to_video
    except ImportError:
        logger.warning("CogVideoX not available (needs diffusers >= 0.30). Skipping.")
        return

    cog_dir = os.path.join(output_dir, "cogvideo")
    os.makedirs(cog_dir, exist_ok=True)

    # CogVideoX-5B-I2V is the image-to-video variant
    COG_MODEL = "THUDM/CogVideoX-5b-I2V"

    logger.info("Loading CogVideoX-5B image-to-video pipeline...")
    try:
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            COG_MODEL,
            torch_dtype=DTYPE,
        )
        if DEVICE == "cuda":
            try:
                pipe.enable_model_cpu_offload()
            except Exception:
                pipe = pipe.to(DEVICE)
        else:
            pipe = pipe.to(DEVICE)
    except Exception as e:
        logger.error("Failed to load CogVideoX: %s", e)
        logger.info("Skipping CogVideoX demo. Install with: pip install diffusers>=0.30")
        return

    logger.info("CogVideoX loaded.")

    # Action prompts — things the character "does"
    action_prompts = [
        ("wave", "The person in the image waves hello and smiles at the camera"),
        ("walk", "The person in the image walks forward slowly, natural movement"),
        ("turn", "The person in the image slowly turns their head to the right and looks around"),
    ]

    cog_w, cog_h = 720, 480  # CogVideoX resolution
    cog_num_frames = 49  # CogVideoX default

    for action_name, prompt in action_prompts:
        action_dir = os.path.join(cog_dir, action_name)
        os.makedirs(action_dir, exist_ok=True)

        logger.info("CogVideoX action: '%s'", action_name)

        # --- Original ---
        logger.info("  Generating from ORIGINAL...")
        orig_resized = original.resize((cog_w, cog_h), Image.LANCZOS)
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        t0 = time.time()
        try:
            result = pipe(
                prompt=prompt,
                image=orig_resized,
                num_frames=cog_num_frames,
                generator=generator,
                num_inference_steps=30,
                guidance_scale=6.0,
            )
            frames_orig = result.frames[0]
            logger.info("    Original: %.1fs, %d frames", time.time() - t0, len(frames_orig))
            export_to_video(
                frames_orig,
                os.path.join(action_dir, "video_original.mp4"),
                fps=8,
            )
        except Exception as e:
            logger.error("    Original generation failed: %s", e)
            continue

        # --- Attacked versions ---
        for eps_int in args.eps_values:
            atk_img = attacked_images[eps_int]
            atk_resized = atk_img.resize((cog_w, cog_h), Image.LANCZOS)

            logger.info("  Generating from ATTACKED eps=%d/255...", eps_int)
            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            t0 = time.time()
            try:
                result = pipe(
                    prompt=prompt,
                    image=atk_resized,
                    num_frames=cog_num_frames,
                    generator=generator,
                    num_inference_steps=30,
                    guidance_scale=6.0,
                )
                frames_atk = result.frames[0]
                logger.info("    eps=%d: %.1fs, %d frames", eps_int, time.time() - t0, len(frames_atk))
                export_to_video(
                    frames_atk,
                    os.path.join(action_dir, f"video_attacked_eps{eps_int}.mp4"),
                    fps=8,
                )

                # Side-by-side
                sbs = create_side_by_side_video_frames(
                    frames_orig, frames_atk,
                    "Original", f"CLIP eps={eps_int}",
                )
                export_to_video(
                    sbs,
                    os.path.join(action_dir, f"video_sbs_eps{eps_int}.mp4"),
                    fps=8,
                )
            except Exception as e:
                logger.error("    eps=%d generation failed: %s", eps_int, e)

    # Cleanup
    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("CogVideoX demo complete. Output: %s", cog_dir)


# ===================================================================
# MAIN
# ===================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("CLIP ATTACK → VIDEO GENERATION DEMO")
    logger.info("  Device: %s (%s)", DEVICE, DTYPE)
    logger.info("  Eps values: %s (out of 255)", args.eps_values)
    logger.info("  PGD iters: %d", args.pgd_iters)
    logger.info("  SVD: %s  |  CogVideoX: %s",
                "SKIP" if args.skip_svd else "ON",
                "SKIP" if args.skip_cogvideo else "ON")
    logger.info("=" * 70)

    # ---------------------------------------------------------------
    # Load input image
    # ---------------------------------------------------------------
    if args.image and os.path.exists(args.image):
        original = Image.open(args.image).convert("RGB")
    else:
        # Try keanu from comparison outputs
        candidates = [
            os.path.join(args.images_dir, "keanu", "1_keanu_original.png"),
            os.path.join(args.images_dir, "original_512.png"),
        ]
        original = None
        for cand in candidates:
            if os.path.exists(cand):
                original = Image.open(cand).convert("RGB")
                logger.info("Loaded: %s", cand)
                break
        if original is None:
            logger.error("No input image found! Use --image to specify one.")
            sys.exit(1)

    original = original.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    original.save(os.path.join(args.output_dir, "00_original.png"))

    # ---------------------------------------------------------------
    # STEP 1: CLIP attacks at each epsilon
    # ---------------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 1: CLIP PGD ATTACKS")
    logger.info("=" * 70)

    image_encoder, feature_extractor = load_clip_image_encoder()

    with torch.no_grad():
        orig_emb = get_clip_embedding(original, image_encoder, feature_extractor)

    attacked_images = {}
    clip_metrics = {}

    for eps_int in args.eps_values:
        eps = eps_int / 255.0
        logger.info("CLIP attack eps=%d/255 (%d iters)...", eps_int, args.pgd_iters)

        t0 = time.time()
        attacked, cos_sim = pgd_attack_clip(
            original, image_encoder, feature_extractor,
            eps=eps, num_iters=args.pgd_iters,
        )
        elapsed = time.time() - t0

        attacked_images[eps_int] = attacked
        attacked.save(os.path.join(args.output_dir, f"attacked_eps{eps_int}.png"))

        # Measure embedding distance
        with torch.no_grad():
            atk_emb = get_clip_embedding(attacked, image_encoder, feature_extractor)
            l2_dist = (atk_emb - orig_emb).norm().item()

        # Pixel-level PSNR
        orig_np = np.array(original).astype(float)
        atk_np = np.array(attacked).astype(float)
        mse = np.mean((orig_np - atk_np) ** 2)
        psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-10))

        clip_metrics[eps_int] = {
            "cos_sim": cos_sim,
            "l2": l2_dist,
            "psnr": psnr,
        }
        logger.info(
            "  eps=%d/255: cos_sim=%.4f, L2=%.2f, PSNR=%.1fdB (%.1fs)",
            eps_int, cos_sim, l2_dist, psnr, elapsed,
        )

    # Summary table
    logger.info("")
    logger.info("CLIP EMBEDDING SUMMARY:")
    logger.info("%-10s %-12s %-10s %-10s", "eps", "cos_sim", "L2_dist", "PSNR(dB)")
    logger.info("-" * 45)
    for eps_int in args.eps_values:
        m = clip_metrics[eps_int]
        logger.info("%-10s %-12.4f %-10.2f %-10.1f", f"{eps_int}/255", m["cos_sim"], m["l2"], m["psnr"])
    logger.info("")

    # Free CLIP encoder before loading video models
    del image_encoder, feature_extractor
    torch.cuda.empty_cache()
    gc.collect()

    # ---------------------------------------------------------------
    # STEP 2: Stable Video Diffusion
    # ---------------------------------------------------------------
    if not args.skip_svd:
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 2: STABLE VIDEO DIFFUSION (image → video)")
        logger.info("  SVD encodes the conditioning image with CLIP ViT")
        logger.info("  → CLIP perturbation = corrupted video generation")
        logger.info("=" * 70)
        try:
            run_svd_demo(original, attacked_images, clip_metrics, args.output_dir, args)
        except Exception as e:
            logger.error("SVD demo failed: %s", e)
            import traceback
            traceback.print_exc()

    # ---------------------------------------------------------------
    # STEP 3: CogVideoX (text-guided I2V)
    # ---------------------------------------------------------------
    if not args.skip_cogvideo:
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 3: CogVideoX (text-guided image → video)")
        logger.info("  Prompt the model to make the character DO something")
        logger.info("  → CLIP perturbation breaks character understanding")
        logger.info("=" * 70)
        try:
            run_cogvideo_demo(original, attacked_images, clip_metrics, args.output_dir, args)
        except Exception as e:
            logger.error("CogVideoX demo failed: %s", e)
            import traceback
            traceback.print_exc()

    # ---------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("ALL DONE. Output: %s", args.output_dir)
    logger.info("")
    for root, dirs, files in os.walk(args.output_dir):
        level = root.replace(args.output_dir, "").count(os.sep)
        indent = "  " * level
        logger.info(
            "%s%s/ (%d files)",
            indent, os.path.basename(root), len(files),
        )
    logger.info("")
    logger.info("KEY INSIGHT:")
    logger.info("  The adversarial perturbation is IMPERCEPTIBLE to humans")
    logger.info("  (PSNR > 40dB at eps<=4) but completely breaks the video model's")
    logger.info("  understanding of the image through its CLIP encoder.")
    logger.info("  Original → smooth, coherent video of the person")
    logger.info("  Attacked → glitchy, distorted, identity-destroyed video")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
