"""
Centralized configuration loaded from environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    # API Keys
    anthropic_api_key: str = ""
    brightdata_api_token: str = ""
    brightdata_browser_auth: str = ""

    # Face matching
    face_match_threshold: float = 0.35

    # Deepfake detection
    deepfake_flag_threshold: float = 0.6
    deepfake_high_confidence_threshold: float = 0.8

    # Search limits
    max_candidates_per_search: int = 50

    # Concurrency
    image_download_concurrency: int = 10
    deepfake_analysis_concurrency: int = 3

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Paths
    project_root: Path = _project_root
    output_dir: Path = field(default_factory=lambda: _project_root / "output")
    upload_dir: Path = field(default_factory=lambda: _project_root / "output" / "uploads")
    evidence_dir: Path = field(default_factory=lambda: _project_root / "output" / "evidence")

    def __post_init__(self):
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", self.anthropic_api_key)
        self.brightdata_api_token = os.getenv("BRIGHTDATA_API_TOKEN", self.brightdata_api_token)
        self.brightdata_browser_auth = os.getenv(
            "BRIGHTDATA_BROWSER_AUTH", self.brightdata_browser_auth
        )
        self.face_match_threshold = float(
            os.getenv("FACE_MATCH_THRESHOLD", self.face_match_threshold)
        )
        self.deepfake_flag_threshold = float(
            os.getenv("DEEPFAKE_FLAG_THRESHOLD", self.deepfake_flag_threshold)
        )
        self.deepfake_high_confidence_threshold = float(
            os.getenv(
                "DEEPFAKE_HIGH_CONFIDENCE_THRESHOLD", self.deepfake_high_confidence_threshold
            )
        )
        self.max_candidates_per_search = int(
            os.getenv("MAX_CANDIDATES_PER_SEARCH", self.max_candidates_per_search)
        )
        self.image_download_concurrency = int(
            os.getenv("IMAGE_DOWNLOAD_CONCURRENCY", self.image_download_concurrency)
        )
        self.deepfake_analysis_concurrency = int(
            os.getenv("DEEPFAKE_ANALYSIS_CONCURRENCY", self.deepfake_analysis_concurrency)
        )
        self.host = os.getenv("HOST", self.host)
        self.port = int(os.getenv("PORT", self.port))

        # Ensure output directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)


# Singleton settings instance
settings = Settings()
