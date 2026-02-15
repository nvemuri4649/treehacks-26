"""
Async image downloading, validation, deduplication, and preprocessing.

Handles fetching candidate images from the web with retry logic,
validates they are real images, deduplicates by content hash, and
resizes for analysis.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
from PIL import Image

from deepfake.core.config import settings

logger = logging.getLogger(__name__)

# Rotating User-Agent strings to avoid blocks
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:134.0) Gecko/20100101 Firefox/134.0",
]

VALID_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "GIF", "BMP", "TIFF"}
MAX_IMAGE_DIMENSION = 1024  # Max width or height for analysis


@dataclass
class DownloadedImage:
    """A downloaded and validated image."""

    source_url: str
    local_path: Path
    content_hash: str
    width: int
    height: int
    format: str
    file_size: int
    is_duplicate: bool = False


@dataclass
class DownloadStats:
    """Aggregated download statistics."""

    total_attempted: int = 0
    successful: int = 0
    failed: int = 0
    duplicates_skipped: int = 0
    invalid_format: int = 0


class ImageProcessor:
    """
    Async image downloader and processor with retry, validation, and deduplication.
    """

    def __init__(
        self,
        output_dir: Path | None = None,
        max_concurrency: int | None = None,
    ):
        self._output_dir = output_dir or settings.evidence_dir
        self._max_concurrency = max_concurrency or settings.image_download_concurrency
        self._seen_hashes: set[str] = set()
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore = asyncio.Semaphore(self._max_concurrency)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=self._max_concurrency * 2,
                    max_keepalive_connections=self._max_concurrency,
                ),
            )
        return self._client

    def _random_ua(self) -> str:
        return random.choice(_USER_AGENTS)

    async def download_image(
        self,
        url: str,
        max_retries: int = 3,
        backoff_base: float = 1.0,
    ) -> Optional[DownloadedImage]:
        """
        Download a single image with retry and exponential backoff.

        Args:
            url: image URL to download.
            max_retries: number of retry attempts.
            backoff_base: base delay for exponential backoff.

        Returns:
            DownloadedImage if successful, None if failed.
        """
        async with self._semaphore:
            client = await self._get_client()

            for attempt in range(max_retries):
                try:
                    headers = {"User-Agent": self._random_ua()}
                    response = await client.get(url, headers=headers)
                    response.raise_for_status()

                    content = response.content
                    content_type = response.headers.get("content-type", "")

                    # Quick content-type check
                    if content_type and not content_type.startswith("image/"):
                        logger.debug("Non-image content-type for %s: %s", url[:60], content_type)
                        return None

                    return await self._process_downloaded(url, content)

                except httpx.HTTPStatusError as e:
                    if e.response.status_code in (403, 404, 410, 451):
                        logger.debug("Permanent error %d for %s", e.response.status_code, url[:60])
                        return None  # Don't retry on permanent errors
                    logger.warning(
                        "HTTP %d for %s (attempt %d/%d)",
                        e.response.status_code, url[:60], attempt + 1, max_retries,
                    )
                except (httpx.TimeoutException, httpx.ConnectError) as e:
                    logger.warning(
                        "Connection error for %s (attempt %d/%d): %s",
                        url[:60], attempt + 1, max_retries, type(e).__name__,
                    )
                except Exception as e:
                    logger.warning(
                        "Unexpected error for %s (attempt %d/%d): %s",
                        url[:60], attempt + 1, max_retries, e,
                    )

                if attempt < max_retries - 1:
                    delay = backoff_base * (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(delay)

            return None

    async def _process_downloaded(
        self, url: str, content: bytes
    ) -> Optional[DownloadedImage]:
        """Validate, deduplicate, resize, and save a downloaded image."""
        # Content hash for dedup
        content_hash = hashlib.sha256(content).hexdigest()[:16]

        if content_hash in self._seen_hashes:
            logger.debug("Duplicate image skipped: %s", url[:60])
            return DownloadedImage(
                source_url=url,
                local_path=Path(""),
                content_hash=content_hash,
                width=0,
                height=0,
                format="",
                file_size=len(content),
                is_duplicate=True,
            )
        self._seen_hashes.add(content_hash)

        # Validate it's a real image
        try:
            img = Image.open(io.BytesIO(content))
            img.verify()
            # Re-open after verify (verify consumes the image)
            img = Image.open(io.BytesIO(content))
        except Exception:
            logger.debug("Invalid image data from %s", url[:60])
            return None

        if img.format not in VALID_IMAGE_FORMATS:
            logger.debug("Unsupported format '%s' from %s", img.format, url[:60])
            return None

        orig_width, orig_height = img.size

        # Resize if needed (preserve aspect ratio)
        img = self._resize_image(img, MAX_IMAGE_DIMENSION)

        # Save to disk
        ext = img.format.lower() if img.format else "jpg"
        if ext == "jpeg":
            ext = "jpg"
        filename = f"{content_hash}.{ext}"
        save_path = self._output_dir / filename

        img.save(str(save_path), quality=95)
        file_size = save_path.stat().st_size

        return DownloadedImage(
            source_url=url,
            local_path=save_path,
            content_hash=content_hash,
            width=img.size[0],
            height=img.size[1],
            format=img.format or "JPEG",
            file_size=file_size,
            is_duplicate=False,
        )

    @staticmethod
    def _resize_image(img: Image.Image, max_dim: int) -> Image.Image:
        """Resize image so the longest side is at most max_dim pixels."""
        w, h = img.size
        if max(w, h) <= max_dim:
            return img

        if w > h:
            new_w = max_dim
            new_h = int(h * (max_dim / w))
        else:
            new_h = max_dim
            new_w = int(w * (max_dim / h))

        return img.resize((new_w, new_h), Image.LANCZOS)

    async def download_batch(
        self,
        urls: list[str],
    ) -> tuple[list[DownloadedImage], DownloadStats]:
        """
        Download a batch of images concurrently.

        Args:
            urls: list of image URLs to download.

        Returns:
            Tuple of (list of successful downloads, download statistics).
        """
        stats = DownloadStats(total_attempted=len(urls))

        tasks = [self.download_image(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        downloaded: list[DownloadedImage] = []
        for result in results:
            if isinstance(result, Exception):
                logger.error("Batch download exception: %s", result)
                stats.failed += 1
            elif result is None:
                stats.failed += 1
            elif result.is_duplicate:
                stats.duplicates_skipped += 1
            else:
                downloaded.append(result)
                stats.successful += 1

        logger.info(
            "Batch download: %d/%d successful, %d duplicates, %d failed",
            stats.successful, stats.total_attempted,
            stats.duplicates_skipped, stats.failed,
        )
        return downloaded, stats

    def image_to_base64(self, image_path: str | Path) -> str:
        """Convert an image file to a base64-encoded string."""
        path = Path(image_path)
        with open(path, "rb") as f:
            return __import__("base64").b64encode(f.read()).decode("utf-8")

    def image_to_numpy(self, image_path: str | Path) -> np.ndarray:
        """Load an image as a BGR numpy array (OpenCV format)."""
        import cv2
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        return img

    def reset_dedup_cache(self):
        """Clear the content hash deduplication cache."""
        self._seen_hashes.clear()

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
