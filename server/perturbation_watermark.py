"""
Perturbation Watermarking — Embed hidden data inside adversarial noise
======================================================================

Encodes a string/ID into the adversarial perturbation δ added by DiffusionGuard.
The perturbation has ~786K degrees of freedom (512×512×3), so we can encode
hundreds of bits without significantly impacting adversarial effectiveness.

Approach: Spread-Spectrum Watermarking
- Each bit is encoded across many pixels (redundancy = robustness)
- A secret key determines which pixels encode which bit
- Bit value 1 → force perturbation positive at those pixels
- Bit value 0 → force perturbation negative at those pixels
- Decoding: measure average perturbation sign in each pixel group

Two modes:
1) POST-HOC: Modify perturbation after DiffusionGuard computes it
2) CONSTRAINED: Apply constraints during PGD optimization (monkey-patch)

Usage:
    from perturbation_watermark import PerturbationWatermark

    wm = PerturbationWatermark(key=12345)

    # Encode after protection
    protected = diffusionguard(image)
    watermarked = wm.encode(original_image, protected_image, message="USER-42069")

    # Decode
    message = wm.decode(original_image, watermarked_image)
    print(message)  # "USER-42069"
"""

import hashlib
import struct
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Union

try:
    import torch
except ImportError:
    torch = None


class PerturbationWatermark:
    """
    Embed and extract hidden messages in adversarial perturbations.

    Parameters:
        key:            Secret key for pseudorandom pixel assignment
        bits_per_pixel: How many pixels per message bit (higher = more robust)
        channel:        Which channel to use (0=R, 1=G, 2=B, None=all)
        strength:       How aggressively to encode (0-1). Higher = more robust
                        but may reduce adversarial effectiveness slightly.
    """

    def __init__(
        self,
        key: Union[int, str] = 42,
        bits_per_pixel: int = 512,
        channel: Optional[int] = None,
        strength: float = 0.8,
    ):
        # Convert string keys to int via hash
        if isinstance(key, str):
            key = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
        self.key = key
        self.bits_per_pixel = bits_per_pixel
        self.channel = channel
        self.strength = np.clip(strength, 0.1, 1.0)

    # ------------------------------------------------------------------
    # Message ↔ Bits conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _string_to_bits(message: str) -> list:
        """Convert string to list of bits with a length header."""
        data = message.encode("utf-8")
        # 16-bit length header (max 65535 bytes)
        length_bytes = struct.pack(">H", len(data))
        # 16-bit CRC for integrity check
        crc = _crc16(data)
        crc_bytes = struct.pack(">H", crc)
        full = length_bytes + crc_bytes + data
        bits = []
        for byte in full:
            for i in range(8):
                bits.append((byte >> (7 - i)) & 1)
        return bits

    @staticmethod
    def _bits_to_string(bits: list) -> Tuple[str, bool]:
        """Convert bits back to string. Returns (message, is_valid)."""
        if len(bits) < 32:  # Need at least 4 bytes header
            return "", False

        # Read 16-bit length
        length = 0
        for i in range(16):
            length = (length << 1) | bits[i]

        # Read 16-bit CRC
        crc = 0
        for i in range(16, 32):
            crc = (crc << 1) | bits[i]

        # Read message bytes
        msg_bits = bits[32: 32 + length * 8]
        if len(msg_bits) < length * 8:
            return "", False

        msg_bytes = bytearray()
        for i in range(0, len(msg_bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(msg_bits):
                    byte = (byte << 1) | msg_bits[i + j]
            msg_bytes.append(byte)

        # Verify CRC
        actual_crc = _crc16(bytes(msg_bytes))
        is_valid = actual_crc == crc

        try:
            message = msg_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return msg_bytes.decode("utf-8", errors="replace"), False

        return message, is_valid

    # ------------------------------------------------------------------
    # Pixel assignment
    # ------------------------------------------------------------------

    def _get_pixel_assignments(self, h: int, w: int, num_bits: int):
        """
        Assign pixels to bits using the secret key.
        Returns list of (row_indices, col_indices) for each bit.
        """
        rng = np.random.RandomState(self.key)
        total_pixels = h * w

        # Shuffle all pixel indices
        indices = np.arange(total_pixels)
        rng.shuffle(indices)

        # Assign pixels_per_bit pixels to each bit
        assignments = []
        for bit_idx in range(num_bits):
            start = bit_idx * self.bits_per_pixel
            end = start + self.bits_per_pixel
            if end > total_pixels:
                # Wrap around with different shuffle
                rng2 = np.random.RandomState(self.key + bit_idx + 1)
                indices2 = np.arange(total_pixels)
                rng2.shuffle(indices2)
                pixel_ids = indices2[:self.bits_per_pixel]
            else:
                pixel_ids = indices[start:end]

            rows = pixel_ids // w
            cols = pixel_ids % w
            assignments.append((rows, cols))

        return assignments

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(
        self,
        original: Union[Image.Image, np.ndarray],
        protected: Union[Image.Image, np.ndarray],
        message: str,
    ) -> Image.Image:
        """
        Encode a message into the perturbation (protected - original).
        Returns a new protected image with the watermark embedded.

        The perturbation magnitude is preserved (stays within original ε-ball).
        """
        orig_arr = np.array(original if isinstance(original, Image.Image)
                            else Image.fromarray(original)).astype(np.float64)
        prot_arr = np.array(protected if isinstance(protected, Image.Image)
                            else Image.fromarray(protected)).astype(np.float64)

        # Extract perturbation
        delta = prot_arr - orig_arr  # shape: (H, W, 3)
        h, w, c = delta.shape

        # Convert message to bits
        bits = self._string_to_bits(message)
        max_capacity = (h * w) // self.bits_per_pixel
        if len(bits) > max_capacity:
            raise ValueError(
                f"Message too long: {len(bits)} bits, max capacity: {max_capacity} bits "
                f"({max_capacity // 8 - 4} chars) with current bits_per_pixel={self.bits_per_pixel}"
            )

        # Get pixel assignments
        assignments = self._get_pixel_assignments(h, w, len(bits))

        # Determine which channels to encode in
        channels = [self.channel] if self.channel is not None else list(range(c))

        # Encode each bit
        encoded_delta = delta.copy()
        for bit_idx, bit_value in enumerate(bits):
            rows, cols = assignments[bit_idx]
            target_sign = 1.0 if bit_value == 1 else -1.0

            for ch in channels:
                vals = encoded_delta[rows, cols, ch]

                # Force sign to match the bit value
                # For strength=1.0: fully force sign
                # For strength<1.0: only flip a fraction of mismatched pixels
                wrong_sign = (vals * target_sign) < 0  # pixels with wrong sign
                num_to_flip = int(np.ceil(wrong_sign.sum() * self.strength))

                if num_to_flip > 0:
                    wrong_indices = np.where(wrong_sign)[0]
                    flip_indices = wrong_indices[:num_to_flip]
                    # Flip sign while preserving magnitude
                    encoded_delta[rows[flip_indices], cols[flip_indices], ch] *= -1.0

                # For zero-valued perturbations, add a tiny nudge
                zero_mask = vals == 0
                if zero_mask.any():
                    nudge = target_sign * 1.0  # 1/255 nudge
                    encoded_delta[rows[zero_mask], cols[zero_mask], ch] = nudge

        # Reconstruct protected image
        watermarked = np.clip(orig_arr + encoded_delta, 0, 255).astype(np.uint8)
        return Image.fromarray(watermarked)

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(
        self,
        original: Union[Image.Image, np.ndarray],
        watermarked: Union[Image.Image, np.ndarray],
        max_message_bytes: int = 256,
    ) -> Tuple[str, bool, float]:
        """
        Decode a message from the watermarked perturbation.
        Returns: (message, is_valid, confidence)
        """
        orig_arr = np.array(original if isinstance(original, Image.Image)
                            else Image.fromarray(original)).astype(np.float64)
        wm_arr = np.array(watermarked if isinstance(watermarked, Image.Image)
                          else Image.fromarray(watermarked)).astype(np.float64)

        delta = wm_arr - orig_arr
        h, w, c = delta.shape

        # Maximum bits we need to read (header + max message)
        max_bits = 32 + max_message_bytes * 8  # 4 bytes header + message
        max_capacity = (h * w) // self.bits_per_pixel
        max_bits = min(max_bits, max_capacity)

        assignments = self._get_pixel_assignments(h, w, max_bits)
        channels = [self.channel] if self.channel is not None else list(range(c))

        # Read bits by measuring average sign
        bits = []
        confidences = []
        for bit_idx in range(max_bits):
            rows, cols = assignments[bit_idx]
            vals = []
            for ch in channels:
                vals.extend(delta[rows, cols, ch].tolist())
            avg = np.mean(vals)
            bits.append(1 if avg > 0 else 0)
            # Confidence = how strongly the average leans one way
            confidences.append(abs(avg))

        # Parse header to find actual message length, then trim
        if len(bits) >= 16:
            msg_len = 0
            for i in range(16):
                msg_len = (msg_len << 1) | bits[i]

            needed = 32 + msg_len * 8
            if needed <= len(bits):
                bits = bits[:needed]
                confidences = confidences[:needed]

        message, is_valid = self._bits_to_string(bits)
        avg_confidence = np.mean(confidences) if confidences else 0.0

        return message, is_valid, avg_confidence

    # ------------------------------------------------------------------
    # Capacity info
    # ------------------------------------------------------------------

    def capacity(self, h: int = 512, w: int = 512) -> dict:
        """Report encoding capacity for given image dimensions."""
        total_pixels = h * w
        max_bits = total_pixels // self.bits_per_pixel
        header_bits = 32  # 2 bytes length + 2 bytes CRC
        payload_bits = max_bits - header_bits
        max_chars = payload_bits // 8

        return {
            "total_pixels": total_pixels,
            "bits_per_pixel": self.bits_per_pixel,
            "max_bits": max_bits,
            "header_bits": header_bits,
            "payload_bits": payload_bits,
            "max_chars": max_chars,
            "max_bytes": max_chars,
        }

    # ------------------------------------------------------------------
    # Torch integration (for constrained optimization)
    # ------------------------------------------------------------------

    def get_sign_constraint_mask(self, h: int, w: int, message: str):
        """
        Returns (mask, target_signs) tensors for constrained PGD optimization.

        mask: (1, C, H, W) binary tensor — 1 where constraint is active
        target_signs: (1, C, H, W) tensor — +1 or -1 for desired sign

        During PGD, after each step:
            constrained = torch.where(mask, delta.abs() * target_signs, delta)
        """
        if torch is None:
            raise ImportError("PyTorch required for constrained optimization")

        bits = self._string_to_bits(message)
        assignments = self._get_pixel_assignments(h, w, len(bits))
        channels = [self.channel] if self.channel is not None else [0, 1, 2]

        mask = np.zeros((1, 3, h, w), dtype=np.float32)
        signs = np.zeros((1, 3, h, w), dtype=np.float32)

        for bit_idx, bit_value in enumerate(bits):
            rows, cols = assignments[bit_idx]
            target = 1.0 if bit_value == 1 else -1.0
            for ch in channels:
                mask[0, ch, rows, cols] = 1.0
                signs[0, ch, rows, cols] = target

        return torch.tensor(mask), torch.tensor(signs)


# ------------------------------------------------------------------
# CRC-16 for integrity checking
# ------------------------------------------------------------------

def _crc16(data: bytes) -> int:
    """Simple CRC-16/CCITT."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc


# ===================================================================
# CLI Demo
# ===================================================================

def demo():
    """
    Standalone demo: create a synthetic perturbation, encode a message,
    decode it back, and verify integrity.
    """
    import os

    print("=" * 60)
    print("PERTURBATION WATERMARK DEMO")
    print("=" * 60)

    # Create a test image (or load one)
    test_dir = os.path.join(os.path.dirname(__file__), "..", "video_demo_output", "img2img")
    orig_path = os.path.join(test_dir, "1_original.png")
    prot_path = os.path.join(test_dir, "1_glazed.png")

    if os.path.exists(orig_path) and os.path.exists(prot_path):
        print(f"Loading real images from {test_dir}")
        original = Image.open(orig_path).convert("RGB")
        protected = Image.open(prot_path).convert("RGB")
    else:
        print("Creating synthetic test images...")
        np.random.seed(0)
        h, w = 512, 512
        original = Image.fromarray(np.random.randint(50, 200, (h, w, 3), dtype=np.uint8))
        # Simulate adversarial perturbation (eps=16/255 ≈ ±16 pixel values)
        delta = np.random.uniform(-16, 16, (h, w, 3))
        protected_arr = np.clip(np.array(original).astype(float) + delta, 0, 255).astype(np.uint8)
        protected = Image.fromarray(protected_arr)

    # Show capacity
    wm = PerturbationWatermark(key="my-secret-key-2026", bits_per_pixel=512, strength=0.9)
    cap = wm.capacity(*original.size[::-1])  # PIL gives (w, h), we need (h, w)
    print(f"\nCapacity for {original.size[0]}×{original.size[1]} image:")
    print(f"  {cap['max_bits']} bits total")
    print(f"  {cap['payload_bits']} payload bits (after 32-bit header)")
    print(f"  {cap['max_chars']} max characters")

    # Test messages
    test_messages = [
        "USER-42069",
        "nikhil@treehacks.com",
        "Protected by DiffusionGuard v1.0 | ID: a3f8c2e1",
    ]

    print("\n" + "-" * 60)
    for msg in test_messages:
        print(f"\nEncoding: \"{msg}\" ({len(msg)} chars, {len(msg)*8 + 32} bits)")

        # Encode
        watermarked = wm.encode(original, protected, msg)

        # Measure perturbation change
        orig_arr = np.array(original).astype(float)
        prot_arr = np.array(protected).astype(float)
        wm_arr = np.array(watermarked).astype(float)

        orig_delta = prot_arr - orig_arr
        new_delta = wm_arr - orig_arr
        l_inf_change = np.max(np.abs(new_delta - orig_delta))
        l2_change = np.sqrt(np.mean((new_delta - orig_delta) ** 2))
        psnr_orig_prot = 10 * np.log10(255**2 / np.mean((prot_arr - orig_arr)**2))
        psnr_orig_wm = 10 * np.log10(255**2 / np.mean((wm_arr - orig_arr)**2))

        print(f"  Perturbation change: L∞={l_inf_change:.1f}, L2={l2_change:.4f}")
        print(f"  PSNR: original→protected={psnr_orig_prot:.1f}dB, original→watermarked={psnr_orig_wm:.1f}dB")

        # Decode
        decoded, valid, confidence = wm.decode(original, watermarked)
        status = "VALID" if valid else "INVALID CRC"
        print(f"  Decoded: \"{decoded}\" [{status}] confidence={confidence:.2f}")

        # Try decoding with wrong key
        wrong_wm = PerturbationWatermark(key="wrong-key", bits_per_pixel=512)
        decoded_wrong, valid_wrong, conf_wrong = wrong_wm.decode(original, watermarked)
        print(f"  Wrong key: \"{decoded_wrong[:20]}...\" [{'VALID' if valid_wrong else 'INVALID'}] confidence={conf_wrong:.2f}")

    # Test with lower redundancy for longer messages
    print("\n" + "-" * 60)
    print("Lower redundancy (128 pixels/bit) for longer messages:")
    wm_long = PerturbationWatermark(key="my-secret-key-2026", bits_per_pixel=128, strength=0.9)
    cap_long = wm_long.capacity(*original.size[::-1])
    print(f"  Capacity: {cap_long['max_chars']} chars")
    long_msg = "Owner: Nikhil | Unauthorized AI use prohibited | ID: a3f8c2e1-9b7d"
    print(f"  Encoding: \"{long_msg}\" ({len(long_msg)} chars)")
    wm_long_img = wm_long.encode(original, protected, long_msg)
    decoded_long, valid_long, conf_long = wm_long.decode(original, wm_long_img)
    print(f"  Decoded: \"{decoded_long}\" [{'VALID' if valid_long else 'INVALID'}] confidence={conf_long:.2f}")

    # Test JPEG compression robustness
    print("\n" + "-" * 60)
    print("JPEG compression robustness test:")
    import io
    for quality in [95, 85, 75, 50]:
        buf = io.BytesIO()
        watermarked_test = wm.encode(original, protected, "USER-42069")
        watermarked_test.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        compressed = Image.open(buf).convert("RGB")
        dec_msg, dec_valid, dec_conf = wm.decode(original, compressed)
        print(f"  JPEG Q={quality}: decoded=\"{dec_msg}\" [{'VALID' if dec_valid else 'FAIL'}] conf={dec_conf:.2f}")

    # Also test torch integration if available
    if torch is not None:
        print("\n" + "-" * 60)
        print("Torch constrained optimization support:")
        mask, signs = wm.get_sign_constraint_mask(512, 512, "USER-42069")
        constrained_pixels = mask.sum().item()
        total_pixels = 512 * 512 * 3
        print(f"  Constrained pixels: {int(constrained_pixels)}/{int(total_pixels)} "
              f"({constrained_pixels/total_pixels*100:.1f}%)")
        print(f"  Mask shape: {mask.shape}, Signs shape: {signs.shape}")
        print("  Usage in PGD loop:")
        print("    delta = torch.where(mask.bool(), delta.abs() * signs, delta)")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    demo()
