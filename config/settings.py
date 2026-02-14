"""
Configuration settings for Cena.
Loads environment variables from .env and exposes them as module-level constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

# ── Local Nemotron (vLLM) ────────────────────────────────────────────────
NEMOTRON_ENDPOINT: str = os.getenv("NEMOTRON_ENDPOINT", "http://localhost:8001/v1")
NEMOTRON_MODEL: str = os.getenv("NEMOTRON_MODEL", "nvidia/NVIDIA-Nemotron-Nano-9B-v2")

# ── Cloud API Keys ───────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# ── Cloud model defaults ─────────────────────────────────────────────────
DEFAULT_CLOUD_MODEL: str = os.getenv("DEFAULT_CLOUD_MODEL", "claude-sonnet-4-20250514")

CLAUDE_MODELS = {
    "claude-sonnet-4-20250514",
    "claude-haiku-4-20250414",
    "claude-3-5-sonnet-20241022",
}
OPENAI_MODELS = {
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
}

# ── Glazing Server (DiffusionGuard) ───────────────────────────────────────
GLAZE_SERVER_URL: str = os.getenv("GLAZE_SERVER_URL", "http://spark-abcd.local:5000")
GLAZE_DEFAULT_ITERS: int = int(os.getenv("GLAZE_DEFAULT_ITERS", "200"))

# ── Server ────────────────────────────────────────────────────────────────
HOST: str = os.getenv("HOST", "127.0.0.1")
PORT: int = int(os.getenv("PORT", "8000"))


def get_provider(model: str) -> str:
    """Return 'claude' or 'openai' based on the cloud model name."""
    if model in CLAUDE_MODELS or model.startswith("claude"):
        return "claude"
    if model in OPENAI_MODELS or model.startswith("gpt"):
        return "openai"
    raise ValueError(f"Unknown cloud model: {model}. Must be a Claude or OpenAI GPT model.")
