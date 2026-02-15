"""
Configuration bridge for the deepfake detection module.

Exposes a ``settings`` object with the same interface that the deepfake
module's code expects, but reads values from the project-wide
``config.settings`` and environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

#Load .env from project root (same as config/settings.py)
_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / ".env")


@dataclass
class DeepfakeSettings:
    """Settings specific to the deepfake detection module."""

    #API Keys (shared with main config)
    anthropic_api_key: str = ""
    brightdata_api_token: str = ""
    brightdata_browser_auth: str = ""

    #Face matching
    face_match_threshold: float = 0.35

    #Deepfake detection
    deepfake_flag_threshold: float = 0.6
    deepfake_high_confidence_threshold: float = 0.8

    #Search limits
    max_candidates_per_search: int = 50

    #Concurrency
    image_download_concurrency: int = 10
    deepfake_analysis_concurrency: int = 3

    #Server (inherited from main config)
    host: str = "127.0.0.1"
    port: int = 8000

    #Paths — scoped under the project's output directory
    project_root: Path = _project_root
    output_dir: Path = field(default_factory=lambda: _project_root / "output" / "deepfake")
    upload_dir: Path = field(
        default_factory=lambda: _project_root / "output" / "deepfake" / "uploads"
    )
    evidence_dir: Path = field(
        default_factory=lambda: _project_root / "output" / "deepfake" / "evidence"
    )

    def __post_init__(self) -> None:
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", self.anthropic_api_key)
        self.brightdata_api_token = os.getenv(
            "BRIGHTDATA_API_TOKEN", self.brightdata_api_token
        )
        self.brightdata_browser_auth = os.getenv(
            "BRIGHTDATA_BROWSER_AUTH", self.brightdata_browser_auth
        )
        self.face_match_threshold = float(
            os.getenv("FACE_MATCH_THRESHOLD", str(self.face_match_threshold))
        )
        self.deepfake_flag_threshold = float(
            os.getenv("DEEPFAKE_FLAG_THRESHOLD", str(self.deepfake_flag_threshold))
        )
        self.deepfake_high_confidence_threshold = float(
            os.getenv(
                "DEEPFAKE_HIGH_CONFIDENCE_THRESHOLD",
                str(self.deepfake_high_confidence_threshold),
            )
        )
        self.max_candidates_per_search = int(
            os.getenv("MAX_CANDIDATES_PER_SEARCH", str(self.max_candidates_per_search))
        )
        self.image_download_concurrency = int(
            os.getenv("IMAGE_DOWNLOAD_CONCURRENCY", str(self.image_download_concurrency))
        )
        self.deepfake_analysis_concurrency = int(
            os.getenv(
                "DEEPFAKE_ANALYSIS_CONCURRENCY",
                str(self.deepfake_analysis_concurrency),
            )
        )

        #Read host/port from main config
        self.host = os.getenv("HOST", self.host)
        self.port = int(os.getenv("PORT", str(self.port)))

        #Ensure output directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)


#Singleton settings instance
settings = DeepfakeSettings()
