"""
ImageTransformer — Applies adversarial encryption to images
before they are sent to cloud LLMs.

This module connects the Cena pipeline to the encryption server running
on the local GPU (DGX Spark). When an image is uploaded through the chat,
it is sent to the encryption server for adversarial perturbation before
being forwarded to any cloud model.

The encryption uses a low epsilon for visual similarity, but the perturbation
is designed to defeat AI inpainting/deepfake generation when processed by
diffusion models.
"""

from __future__ import annotations

import io
import logging

import requests
from PIL import Image

from config.settings import ENCRYPTION_SERVER_URL, ENCRYPTION_DEFAULT_ITERS

logger = logging.getLogger(__name__)


def _create_full_mask(image_bytes: bytes) -> bytes:
    """Create a full-image white mask (protect the entire image)."""
    img = Image.open(io.BytesIO(image_bytes))
    mask = Image.new("L", img.size, 255)  # Full white = protect everything
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    return buf.getvalue()


def transform(image_bytes: bytes, mime_type: str) -> bytes:
    """
    Apply adversarial encryption to *image_bytes* and return
    the protected image as bytes.

    Sends the image to the encryption server running on the local
    GPU (DGX Spark / SSH backend). The server applies PGD adversarial
    perturbations that are visually imperceptible but defeat AI inpainting.

    Args:
        image_bytes: Raw bytes of the uploaded image.
        mime_type:   MIME type string, e.g. ``"image/png"``, ``"image/jpeg"``.

    Returns:
        Glazed image bytes that are safe to transmit to cloud LLMs.
        Output is always PNG format.

    Falls back to returning the original image if the glazing server is
    unreachable (to avoid blocking the chat pipeline).
    """
    try:
        #Create a full-image mask (protect everything)
        mask_bytes = _create_full_mask(image_bytes)

        #Determine file extension from mime type
        ext = "png" if "png" in mime_type else "jpg"

        #Send to the encryption server
        logger.info(
            "Sending image to encryption server at %s (%d iterations)",
            ENCRYPTION_SERVER_URL,
            ENCRYPTION_DEFAULT_ITERS,
        )

        response = requests.post(
            f"{ENCRYPTION_SERVER_URL}/protect",
            files={
                "image": (f"upload.{ext}", io.BytesIO(image_bytes), mime_type),
                "mask": ("mask.png", io.BytesIO(mask_bytes), "image/png"),
            },
            params={"iters": ENCRYPTION_DEFAULT_ITERS},
            timeout=600,  # 10 min max for encryption
        )

        if response.status_code == 200:
            logger.info("Image encrypted successfully (%d bytes)", len(response.content))
            return response.content
        else:
            logger.warning(
                "Encryption server returned %d: %s — falling back to original",
                response.status_code,
                response.text[:200],
            )
            return image_bytes

    except requests.ConnectionError:
        logger.warning(
            "Cannot reach encryption server at %s — passing image through unencrypted",
            ENCRYPTION_SERVER_URL,
        )
        return image_bytes

    except Exception as e:
        logger.error("Image encryption failed: %s — passing through unencrypted", e)
        return image_bytes
