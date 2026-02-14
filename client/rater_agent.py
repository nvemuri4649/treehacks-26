#!/usr/bin/env python3
"""
Claude-based Rater Agent for Deepfake Quality Assessment
==========================================================
Uses Claude's vision capabilities to evaluate how convincing a deepfake is.

The agent is shown:
  1. The original image
  2. The deepfake generated from it
  3. A mask showing the region that should have been modified

It then provides a rating from 1-10 where:
  1-3:  Deepfake is very poor quality, glazing worked excellently
  4-6:  Deepfake has significant artifacts, glazing worked well
  7-8:  Deepfake is mostly convincing, glazing partially worked
  9-10: Deepfake is highly convincing, glazing failed

Usage:
    from rater_agent import rate_deepfake

    score, reasoning = rate_deepfake(
        original_image_path="face.png",
        deepfake_image_path="generated.png",
        mask_path="mask.png",
        api_key="your-anthropic-api-key"
    )
"""

import os
import base64
import json
from typing import Tuple, Optional
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("ERROR: anthropic package not installed. Install with: pip install anthropic")
    raise


def encode_image_base64(image_path: str) -> str:
    """Read an image file and encode it as base64."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path: str) -> str:
    """Get the media type for an image based on its extension."""
    ext = Path(image_path).suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }
    return media_types.get(ext, "image/png")


def rate_deepfake(
    original_image_path: str,
    deepfake_image_path: str,
    mask_path: str,
    api_key: Optional[str] = None,
    model: str = "claude-opus-4-6",
    prompt_context: str = "",
) -> Tuple[int, str]:
    """
    Use Claude to rate the quality of a deepfake.

    Args:
        original_image_path: Path to the original (pre-glazing) image
        deepfake_image_path: Path to the generated deepfake
        mask_path: Path to the mask (white = region that should be modified)
        api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
        model: Claude model to use (default: claude-opus-4-6)
        prompt_context: Additional context about what was supposed to change (e.g., "person in jail")

    Returns:
        (score, reasoning) tuple where:
            score: Integer from 1-10 (1=terrible deepfake, 10=perfect deepfake)
            reasoning: Text explanation of the rating
    """
    # Get API key from env if not provided
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "API key required. Set ANTHROPIC_API_KEY environment variable "
            "or pass api_key parameter."
        )

    # Validate files exist
    for path in [original_image_path, deepfake_image_path, mask_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")

    # Encode images
    original_b64 = encode_image_base64(original_image_path)
    deepfake_b64 = encode_image_base64(deepfake_image_path)
    mask_b64 = encode_image_base64(mask_path)

    # Get media types
    original_type = get_image_media_type(original_image_path)
    deepfake_type = get_image_media_type(deepfake_image_path)
    mask_type = get_image_media_type(mask_path)

    # Build the prompt
    system_prompt = """You are an expert evaluator of AI-generated deepfake images.
Your job is to assess how convincing a deepfake is by comparing it to the original image.

You will be shown:
1. The ORIGINAL image (before any AI manipulation)
2. The DEEPFAKE generated from a protected version of the original
3. A MASK showing which region should have been changed (white = modified area)

Your task: Rate from 1-10 how CONVINCING the deepfake is:

Rating Scale:
• 1-3: Very poor deepfake — obvious artifacts, distortions, or failure to modify the masked region convincingly. Protection worked EXCELLENTLY.
• 4-6: Mediocre deepfake — noticeable quality issues, unnatural elements, or inconsistencies. Protection worked WELL.
• 7-8: Good deepfake — mostly convincing but with subtle artifacts or quality degradation. Protection PARTIALLY worked.
• 9-10: Excellent deepfake — highly realistic, natural-looking, hard to distinguish from real. Protection FAILED.

Consider:
- Photorealism and natural appearance
- Consistency of lighting, colors, and textures
- Artifacts, distortions, or unnatural elements
- How well the modification blends with the rest of the image
- Whether the content in the masked region looks convincing

Respond with a JSON object:
{
  "score": <integer 1-10>,
  "reasoning": "<brief explanation of your rating, 2-3 sentences>",
  "artifacts_observed": "<list specific artifacts if any>",
  "protection_effectiveness": "<EXCELLENT/GOOD/PARTIAL/FAILED>"
}"""

    context_info = f"\n\nContext: The inpainting was attempting to create: {prompt_context}" if prompt_context else ""

    user_prompt = f"""Please evaluate this deepfake image.{context_info}

Image 1 (ORIGINAL): The original unmodified image
Image 2 (DEEPFAKE): The AI-generated result from a protected version
Image 3 (MASK): White areas show what should have been modified

Rate the deepfake from 1-10 and explain your reasoning."""

    # Call Claude API
    client = anthropic.Anthropic(api_key=api_key)

    try:
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": original_type,
                                "data": original_b64,
                            },
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": deepfake_type,
                                "data": deepfake_b64,
                            },
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mask_type,
                                "data": mask_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": user_prompt,
                        }
                    ],
                }
            ],
            system=system_prompt,
        )

        # Parse response
        response_text = message.content[0].text

        # Try to extract JSON from the response
        try:
            # Sometimes Claude wraps JSON in markdown code blocks
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            else:
                # Try to find JSON object in the text
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_str = response_text[json_start:json_end]

            result = json.loads(json_str)
            score = int(result["score"])
            reasoning = result.get("reasoning", response_text)

            # Validate score range
            if not 1 <= score <= 10:
                print(f"WARNING: Score {score} out of range, clamping to 1-10")
                score = max(1, min(10, score))

            return score, reasoning

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"WARNING: Could not parse structured response: {e}")
            print(f"Raw response: {response_text}")
            # Fall back to extracting score from text
            import re
            match = re.search(r'\b([1-9]|10)\b', response_text)
            if match:
                score = int(match.group(1))
                return score, response_text
            else:
                # Default to middle score if we can't parse
                return 5, response_text

    except anthropic.APIError as e:
        raise RuntimeError(f"Claude API error: {e}")


def rate_deepfake_batch(
    pairs: list,
    api_key: Optional[str] = None,
    model: str = "claude-opus-4-6",
    prompt_context: str = "",
) -> list:
    """
    Rate multiple original/deepfake pairs.

    Args:
        pairs: List of tuples (original_path, deepfake_path, mask_path)
        api_key: Anthropic API key
        model: Claude model to use
        prompt_context: Context about the modification

    Returns:
        List of tuples (score, reasoning) for each pair
    """
    results = []
    for i, (orig, fake, mask) in enumerate(pairs, 1):
        print(f"Rating pair {i}/{len(pairs)}...")
        try:
            score, reasoning = rate_deepfake(
                orig, fake, mask, api_key, model, prompt_context
            )
            results.append((score, reasoning))
            print(f"  Score: {score}/10")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((None, str(e)))

    return results


if __name__ == "__main__":
    # Simple CLI for testing
    import argparse

    parser = argparse.ArgumentParser(description="Rate deepfake quality using Claude")
    parser.add_argument("--original", required=True, help="Original image")
    parser.add_argument("--deepfake", required=True, help="Deepfake image")
    parser.add_argument("--mask", required=True, help="Mask image")
    parser.add_argument("--context", default="", help="Context about the modification")
    parser.add_argument("--model", default="claude-opus-4-6", help="Claude model to use")
    args = parser.parse_args()

    print("Evaluating deepfake quality with Claude...")
    print(f"  Original: {args.original}")
    print(f"  Deepfake: {args.deepfake}")
    print(f"  Mask: {args.mask}")
    print()

    score, reasoning = rate_deepfake(
        args.original,
        args.deepfake,
        args.mask,
        prompt_context=args.context,
        model=args.model,
    )

    print("=" * 70)
    print(f"DEEPFAKE QUALITY SCORE: {score}/10")
    print("=" * 70)
    print(reasoning)
    print("=" * 70)
