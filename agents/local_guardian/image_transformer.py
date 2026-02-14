"""
ImageTransformer — Transforms images to remove / obfuscate personal information
before they are sent to cloud LLMs.

=== PLACEHOLDER IMPLEMENTATION ===
This module exposes a stable interface that the rest of the system depends on.
Replace the body of `transform()` with your real image-transformation algorithm
when ready.  The function signature and return contract must stay the same.
"""

from __future__ import annotations


def transform(image_bytes: bytes, mime_type: str) -> bytes:
    """
    Apply a privacy-preserving transformation to *image_bytes* and return
    the transformed image as bytes.

    Args:
        image_bytes: Raw bytes of the uploaded image.
        mime_type:   MIME type string, e.g. ``"image/png"``, ``"image/jpeg"``.

    Returns:
        Transformed image bytes that are safe to transmit to cloud LLMs.
        The output MIME type should match the input MIME type.

    Contract:
        * The returned bytes MUST be a valid image of the same MIME type.
        * The transformation should strip or obfuscate any visually
          identifiable personal information (faces, license plates,
          documents, etc.) depending on your algorithm.
        * Must be a pure function — no side effects or network calls.

    Example integration (once implemented):
        1. Decode `image_bytes` with PIL / OpenCV.
        2. Run your transformation pipeline (blur faces, redact text, etc.).
        3. Re-encode to the same format and return the bytes.
    """
    # ------------------------------------------------------------------
    # TODO: Replace this block with the real image transformation logic.
    #
    # Example skeleton using Pillow:
    #
    #   from PIL import Image
    #   import io
    #
    #   img = Image.open(io.BytesIO(image_bytes))
    #   img = your_transform_pipeline(img)
    #   buf = io.BytesIO()
    #   fmt = "PNG" if mime_type == "image/png" else "JPEG"
    #   img.save(buf, format=fmt)
    #   return buf.getvalue()
    # ------------------------------------------------------------------

    return image_bytes
