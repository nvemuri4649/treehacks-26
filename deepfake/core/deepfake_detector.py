"""
Multi-signal deepfake detection pipeline.

Combines three independent analysis signals:
1. Claude Vision artifact analysis (weight: 0.50)
2. EXIF/metadata forensic analysis (weight: 0.20)
3. Frequency domain (DCT) spectral analysis (weight: 0.30)

Each signal produces a 0-1 probability score. The final deepfake_probability
is a weighted ensemble of all signals.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import anthropic
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

from deepfake.core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class VisionAnalysisResult:
    """Result from Claude Vision deepfake analysis."""

    is_likely_deepfake: bool
    confidence: float  # 0-1
    artifacts_found: list[str] = field(default_factory=list)
    reasoning: str = ""
    raw_response: str = ""


@dataclass
class MetadataAnalysisResult:
    """Result from EXIF/metadata forensic analysis."""

    score: float  # 0-1 deepfake probability
    has_camera_exif: bool = False
    has_ai_signatures: bool = False
    ai_tool_detected: str = ""
    exif_stripped: bool = False
    suspicious_fields: list[str] = field(default_factory=list)
    raw_metadata: dict = field(default_factory=dict)


@dataclass
class FrequencyAnalysisResult:
    """Result from DCT frequency domain analysis."""

    score: float  # 0-1 deepfake probability
    spectral_energy_ratio: float = 0.0  # High-freq / total energy
    has_periodic_artifacts: bool = False
    has_spectral_gap: bool = False
    details: str = ""


@dataclass
class DeepfakeAnalysisResult:
    """Combined result from all three analysis signals."""

    deepfake_probability: float  # 0-1 weighted ensemble score
    threat_level: str  # "low", "medium", "high", "critical"
    vision: VisionAnalysisResult | None = None
    metadata: MetadataAnalysisResult | None = None
    frequency: FrequencyAnalysisResult | None = None

    # Source info
    image_path: str = ""
    source_url: str = ""

    @staticmethod
    def compute_threat_level(prob: float) -> str:
        if prob >= 0.8:
            return "critical"
        elif prob >= 0.6:
            return "high"
        elif prob >= 0.4:
            return "medium"
        else:
            return "low"


# ---------------------------------------------------------------------------
# Signal 1: Claude Vision Analysis
# ---------------------------------------------------------------------------

VISION_ANALYSIS_PROMPT = """You are an expert forensic analyst specializing in detecting AI-generated and deepfake images. Analyze this image carefully for signs that it is AI-generated, a face swap, or otherwise synthetically manipulated.

Look specifically for these artifact categories:

1. **Skin Texture**: Unnatural smoothness, plastic/waxy appearance, lack of pores or fine details
2. **Lighting & Shadows**: Inconsistent lighting direction on face vs. background, missing or impossible shadows
3. **Facial Symmetry**: Unnatural perfect symmetry OR asymmetric distortions in ears, teeth, hairline
4. **Blending Boundary**: Visible seams, color mismatches, or blur artifacts where the face meets neck/hair/background
5. **Eyes**: Mismatched eye reflections/catchlights, irregular pupil shapes, different iris patterns
6. **Hair & Ears**: Melting/morphing hair strands, distorted ear shapes, inconsistent hair-skin boundary
7. **Background**: Warped geometry near face edges, impossible patterns, blurred regions that shouldn't be
8. **Overall Coherence**: Body proportions mismatch, resolution inconsistencies, temporal artifacts

Respond ONLY with valid JSON in this exact format:
{
  "is_likely_deepfake": true/false,
  "confidence": 0.0-1.0,
  "artifacts_found": ["brief description of each artifact"],
  "reasoning": "2-3 sentence explanation of your analysis"
}"""


class VisionAnalyzer:
    """Analyze images for deepfake artifacts using Claude's vision capabilities."""

    def __init__(self):
        self._client: Optional[anthropic.AsyncAnthropic] = None

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(
                api_key=settings.anthropic_api_key,
            )
        return self._client

    async def analyze(self, image_path: str | Path) -> VisionAnalysisResult:
        """
        Analyze an image for deepfake artifacts using Claude Vision.

        Args:
            image_path: path to the image file.

        Returns:
            VisionAnalysisResult with confidence score and detected artifacts.
        """
        logger.info("Vision analysis: %s", image_path)
        client = self._get_client()

        # Read and encode image
        path = Path(image_path)
        with open(path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # Determine media type
        suffix = path.suffix.lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_types.get(suffix, "image/jpeg")

        try:
            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": VISION_ANALYSIS_PROMPT,
                            },
                        ],
                    }
                ],
            )

            raw_text = response.content[0].text
            # Parse JSON from response (handle potential markdown wrapping)
            json_text = raw_text
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0]
            elif "```" in json_text:
                json_text = json_text.split("```")[1].split("```")[0]

            data = json.loads(json_text.strip())

            return VisionAnalysisResult(
                is_likely_deepfake=data.get("is_likely_deepfake", False),
                confidence=float(data.get("confidence", 0.0)),
                artifacts_found=data.get("artifacts_found", []),
                reasoning=data.get("reasoning", ""),
                raw_response=raw_text,
            )

        except json.JSONDecodeError as e:
            logger.warning("Failed to parse vision response as JSON: %s", e)
            return VisionAnalysisResult(
                is_likely_deepfake=False,
                confidence=0.0,
                reasoning=f"JSON parse error: {e}",
                raw_response=raw_text if "raw_text" in dir() else "",
            )
        except Exception as e:
            logger.error("Vision analysis failed: %s", e)
            return VisionAnalysisResult(
                is_likely_deepfake=False,
                confidence=0.0,
                reasoning=f"Analysis error: {e}",
            )


# ---------------------------------------------------------------------------
# Signal 2: EXIF / Metadata Analysis
# ---------------------------------------------------------------------------

# Known AI tool metadata signatures
_AI_SIGNATURES = {
    # Software / tool indicators
    "stable diffusion": "Stable Diffusion",
    "dall-e": "DALL-E",
    "dall·e": "DALL-E",
    "midjourney": "Midjourney",
    "comfyui": "ComfyUI",
    "automatic1111": "Automatic1111",
    "invokeai": "InvokeAI",
    "novelai": "NovelAI",
    "adobe firefly": "Adobe Firefly",
    "generative fill": "Adobe Generative Fill",
    "ai_generated": "Generic AI",
    "synthetic_media": "Synthetic Media",
    "deepfacelab": "DeepFaceLab",
    "faceswap": "FaceSwap",
    "reface": "Reface",
}

# Camera EXIF tags that indicate authentic photography
_CAMERA_TAGS = {"Make", "Model", "LensModel", "FocalLength", "ExposureTime", "ISOSpeedRatings"}


class MetadataAnalyzer:
    """Analyze image EXIF/metadata for AI generation signatures."""

    def analyze(self, image_path: str | Path) -> MetadataAnalysisResult:
        """
        Perform forensic analysis on image metadata.

        Checks for:
        - Known AI tool signatures in EXIF fields
        - Presence/absence of camera EXIF data
        - Suspicious or stripped metadata patterns
        - ComfyUI/A1111 workflow data in PNG chunks

        Args:
            image_path: path to the image file.

        Returns:
            MetadataAnalysisResult with score and findings.
        """
        logger.info("Metadata analysis: %s", image_path)
        path = Path(image_path)

        result = MetadataAnalysisResult(score=0.0, raw_metadata={})

        try:
            img = Image.open(path)
        except Exception as e:
            logger.warning("Cannot open image for metadata analysis: %s", e)
            return result

        # Extract all EXIF data
        exif_data = self._extract_exif(img)
        result.raw_metadata = exif_data

        # Check for camera EXIF (authentic photography indicator)
        camera_tags_found = set(exif_data.keys()) & _CAMERA_TAGS
        result.has_camera_exif = len(camera_tags_found) >= 2

        # Check for AI generation signatures
        all_values_str = " ".join(str(v).lower() for v in exif_data.values())
        for pattern, tool_name in _AI_SIGNATURES.items():
            if pattern in all_values_str:
                result.has_ai_signatures = True
                result.ai_tool_detected = tool_name
                result.suspicious_fields.append(f"AI signature detected: {tool_name}")
                break

        # Check PNG text chunks (ComfyUI, A1111 embed workflow data here)
        if path.suffix.lower() == ".png":
            png_text = self._extract_png_text(img)
            result.raw_metadata["_png_text_chunks"] = list(png_text.keys())

            png_combined = " ".join(str(v).lower() for v in png_text.values())
            for pattern, tool_name in _AI_SIGNATURES.items():
                if pattern in png_combined:
                    result.has_ai_signatures = True
                    result.ai_tool_detected = tool_name
                    result.suspicious_fields.append(
                        f"AI workflow in PNG text chunk: {tool_name}"
                    )
                    break

            # ComfyUI-specific: look for "prompt" or "workflow" keys
            if "prompt" in png_text or "workflow" in png_text:
                result.has_ai_signatures = True
                result.ai_tool_detected = result.ai_tool_detected or "ComfyUI"
                result.suspicious_fields.append("ComfyUI workflow data found in PNG")

        # Check if EXIF is completely stripped (suspicious for social media repost)
        if not exif_data and path.suffix.lower() in (".jpg", ".jpeg"):
            result.exif_stripped = True
            result.suspicious_fields.append("All EXIF data stripped (common in AI images)")

        # Compute metadata score
        result.score = self._compute_score(result)

        return result

    def _extract_exif(self, img: Image.Image) -> dict[str, Any]:
        """Extract EXIF data as a human-readable dictionary."""
        exif = {}
        raw_exif = img.getexif()
        if not raw_exif:
            return exif

        for tag_id, value in raw_exif.items():
            tag_name = TAGS.get(tag_id, str(tag_id))
            try:
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                exif[tag_name] = str(value)[:500]  # Truncate long values
            except Exception:
                exif[tag_name] = repr(value)[:500]

        return exif

    def _extract_png_text(self, img: Image.Image) -> dict[str, str]:
        """Extract text chunks from PNG metadata."""
        text_data = {}
        if hasattr(img, "text"):
            for key, value in img.text.items():
                text_data[key] = str(value)[:2000]
        return text_data

    @staticmethod
    def _compute_score(result: MetadataAnalysisResult) -> float:
        """Compute metadata-based deepfake probability score."""
        score = 0.3  # Base neutral score

        if result.has_ai_signatures:
            score = 0.95  # Near-certain if AI tool signature found

        elif result.has_camera_exif:
            score = 0.05  # Very likely authentic if camera data present

        elif result.exif_stripped:
            score = 0.45  # Slightly suspicious (many legitimate images also stripped)

        return min(1.0, max(0.0, score))


# ---------------------------------------------------------------------------
# Signal 3: Frequency Domain (DCT) Analysis
# ---------------------------------------------------------------------------

class FrequencyAnalyzer:
    """
    Analyze image frequency spectrum for GAN/diffusion artifacts.

    GAN-generated images often exhibit:
    - Periodic spectral artifacts (grid-like patterns in frequency domain)
    - Missing or suppressed high-frequency details
    - Characteristic spectral energy distribution differences
    """

    def analyze(self, image_path: str | Path) -> FrequencyAnalysisResult:
        """
        Perform DCT-based frequency analysis on an image.

        Args:
            image_path: path to the image file.

        Returns:
            FrequencyAnalysisResult with spectral metrics.
        """
        logger.info("Frequency analysis: %s", image_path)

        try:
            img = Image.open(image_path).convert("L")  # Grayscale
            img_array = np.array(img, dtype=np.float64)
        except Exception as e:
            logger.warning("Cannot load image for frequency analysis: %s", e)
            return FrequencyAnalysisResult(score=0.5, details=f"Load error: {e}")

        # Resize to standard size for consistent analysis
        from PIL import Image as PILImage
        if img_array.shape[0] != 256 or img_array.shape[1] != 256:
            img_resized = Image.fromarray(img_array.astype(np.uint8)).resize(
                (256, 256), Image.LANCZOS
            )
            img_array = np.array(img_resized, dtype=np.float64)

        # Compute 2D DCT via FFT
        spectrum = np.fft.fft2(img_array)
        spectrum_shifted = np.fft.fftshift(spectrum)
        magnitude = np.abs(spectrum_shifted)
        log_magnitude = np.log1p(magnitude)

        # Compute spectral energy distribution
        h, w = log_magnitude.shape
        center_y, center_x = h // 2, w // 2

        # Create radial distance map
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
        max_radius = math.sqrt(center_x ** 2 + center_y ** 2)

        # Divide spectrum into low (inner 30%), mid (30-70%), high (outer 30%) frequency bands
        low_mask = distances <= (max_radius * 0.3)
        mid_mask = (distances > max_radius * 0.3) & (distances <= max_radius * 0.7)
        high_mask = distances > max_radius * 0.7

        total_energy = np.sum(log_magnitude)
        low_energy = np.sum(log_magnitude[low_mask])
        mid_energy = np.sum(log_magnitude[mid_mask])
        high_energy = np.sum(log_magnitude[high_mask])

        if total_energy == 0:
            return FrequencyAnalysisResult(score=0.5, details="Zero total spectral energy")

        high_ratio = high_energy / total_energy
        low_ratio = low_energy / total_energy

        # Detect periodic artifacts: look for isolated peaks in the spectrum
        has_periodic = self._detect_periodic_artifacts(log_magnitude, center_y, center_x)

        # Detect spectral gap: GAN images often have a gap between mid and high frequencies
        has_gap = self._detect_spectral_gap(log_magnitude, distances, max_radius)

        # Compute score
        score = self._compute_score(high_ratio, low_ratio, has_periodic, has_gap)

        return FrequencyAnalysisResult(
            score=score,
            spectral_energy_ratio=float(high_ratio),
            has_periodic_artifacts=has_periodic,
            has_spectral_gap=has_gap,
            details=(
                f"Energy distribution: low={low_ratio:.3f}, "
                f"mid={mid_energy / total_energy:.3f}, high={high_ratio:.3f}. "
                f"Periodic artifacts: {'yes' if has_periodic else 'no'}. "
                f"Spectral gap: {'yes' if has_gap else 'no'}."
            ),
        )

    def _detect_periodic_artifacts(
        self,
        log_mag: np.ndarray,
        center_y: int,
        center_x: int,
    ) -> bool:
        """
        Detect periodic artifacts by looking for isolated peaks
        in the frequency spectrum (excluding DC component).
        """
        # Mask out the central DC component region
        h, w = log_mag.shape
        masked = log_mag.copy()
        dc_radius = min(h, w) // 20
        y_coords, x_coords = np.ogrid[:h, :w]
        dc_mask = (y_coords - center_y) ** 2 + (x_coords - center_x) ** 2 <= dc_radius ** 2
        masked[dc_mask] = 0

        # Look for peaks significantly above the mean
        mean_val = np.mean(masked[masked > 0]) if np.any(masked > 0) else 0
        std_val = np.std(masked[masked > 0]) if np.any(masked > 0) else 1

        if std_val == 0:
            return False

        # Count peaks more than 4 standard deviations above mean
        threshold = mean_val + 4.0 * std_val
        peak_count = np.sum(masked > threshold)

        # Periodic artifacts would create multiple symmetric peaks
        return peak_count > 8

    def _detect_spectral_gap(
        self,
        log_mag: np.ndarray,
        distances: np.ndarray,
        max_radius: float,
    ) -> bool:
        """
        Detect abnormal gaps in the spectral energy distribution.

        GANs sometimes produce images with sharp drop-offs in energy
        at certain frequency bands.
        """
        # Compute radial energy profile (average energy per radius band)
        num_bins = 50
        bin_edges = np.linspace(0, max_radius, num_bins + 1)
        radial_energy = np.zeros(num_bins)

        for i in range(num_bins):
            mask = (distances >= bin_edges[i]) & (distances < bin_edges[i + 1])
            if np.any(mask):
                radial_energy[i] = np.mean(log_mag[mask])

        # Smooth the profile
        kernel_size = 3
        if len(radial_energy) >= kernel_size:
            kernel = np.ones(kernel_size) / kernel_size
            smoothed = np.convolve(radial_energy, kernel, mode="same")
        else:
            smoothed = radial_energy

        # Look for sharp drops (> 60% decrease between adjacent bins)
        for i in range(1, len(smoothed) - 1):
            if smoothed[i - 1] > 0:
                drop_ratio = smoothed[i] / smoothed[i - 1]
                if drop_ratio < 0.4:
                    return True

        return False

    @staticmethod
    def _compute_score(
        high_ratio: float,
        low_ratio: float,
        has_periodic: bool,
        has_gap: bool,
    ) -> float:
        """Compute frequency-based deepfake probability score."""
        score = 0.3  # Base neutral-ish score

        # GAN images typically have lower high-frequency energy
        # Natural photos: high_ratio usually 0.05-0.15
        # GAN images: high_ratio usually 0.01-0.06
        if high_ratio < 0.03:
            score += 0.25  # Very low high-frequency content
        elif high_ratio < 0.05:
            score += 0.10  # Somewhat low

        # Concentrated low-frequency energy
        if low_ratio > 0.7:
            score += 0.15

        # Periodic artifacts are strong GAN indicators
        if has_periodic:
            score += 0.30

        # Spectral gap is suggestive
        if has_gap:
            score += 0.15

        return min(1.0, max(0.0, score))


# ---------------------------------------------------------------------------
# Combined Deepfake Detector
# ---------------------------------------------------------------------------

# Signal weights for the ensemble
WEIGHT_VISION = 0.50
WEIGHT_METADATA = 0.20
WEIGHT_FREQUENCY = 0.30


class DeepfakeDetector:
    """
    Multi-signal deepfake detection pipeline.

    Orchestrates three independent analysis signals and produces
    a weighted ensemble deepfake probability score.
    """

    def __init__(self):
        self.vision = VisionAnalyzer()
        self.metadata = MetadataAnalyzer()
        self.frequency = FrequencyAnalyzer()
        self._semaphore = asyncio.Semaphore(settings.deepfake_analysis_concurrency)

    async def analyze(
        self,
        image_path: str | Path,
        source_url: str = "",
        skip_vision: bool = False,
    ) -> DeepfakeAnalysisResult:
        """
        Run full multi-signal deepfake analysis on an image.

        Args:
            image_path: path to the local image file.
            source_url: original URL where the image was found.
            skip_vision: if True, skip Claude Vision analysis (saves API cost).

        Returns:
            DeepfakeAnalysisResult with ensemble score and individual signal results.
        """
        async with self._semaphore:
            logger.info("Full deepfake analysis: %s", image_path)

            # Run metadata and frequency analysis synchronously (fast)
            meta_result = self.metadata.analyze(image_path)
            freq_result = self.frequency.analyze(image_path)

            # Run vision analysis asynchronously (slower, costs API call)
            vision_result = None
            if not skip_vision:
                vision_result = await self.vision.analyze(image_path)

            # Compute weighted ensemble score
            if vision_result:
                vision_score = vision_result.confidence if vision_result.is_likely_deepfake else (1.0 - vision_result.confidence)
                ensemble = (
                    WEIGHT_VISION * vision_score
                    + WEIGHT_METADATA * meta_result.score
                    + WEIGHT_FREQUENCY * freq_result.score
                )
            else:
                # Without vision, re-weight metadata and frequency
                adjusted_meta_weight = WEIGHT_METADATA / (WEIGHT_METADATA + WEIGHT_FREQUENCY)
                adjusted_freq_weight = WEIGHT_FREQUENCY / (WEIGHT_METADATA + WEIGHT_FREQUENCY)
                ensemble = (
                    adjusted_meta_weight * meta_result.score
                    + adjusted_freq_weight * freq_result.score
                )

            ensemble = min(1.0, max(0.0, ensemble))
            threat_level = DeepfakeAnalysisResult.compute_threat_level(ensemble)

            return DeepfakeAnalysisResult(
                deepfake_probability=ensemble,
                threat_level=threat_level,
                vision=vision_result,
                metadata=meta_result,
                frequency=freq_result,
                image_path=str(image_path),
                source_url=source_url,
            )

    async def analyze_batch(
        self,
        images: list[tuple[str | Path, str]],
        skip_vision: bool = False,
    ) -> list[DeepfakeAnalysisResult]:
        """
        Analyze multiple images concurrently.

        Args:
            images: list of (image_path, source_url) tuples.
            skip_vision: if True, skip Claude Vision for all.

        Returns:
            List of DeepfakeAnalysisResult, one per image.
        """
        tasks = [
            self.analyze(path, url, skip_vision=skip_vision)
            for path, url in images
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        final = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Analysis failed for image %d: %s", i, result)
                final.append(
                    DeepfakeAnalysisResult(
                        deepfake_probability=0.0,
                        threat_level="low",
                        image_path=str(images[i][0]),
                        source_url=images[i][1],
                    )
                )
            else:
                final.append(result)

        return final
