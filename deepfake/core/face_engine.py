"""
Face analysis engine using InsightFace.

Provides face detection, 512-D ArcFace embedding extraction,
and cosine similarity comparison for identity matching.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from numpy.linalg import norm

from deepfake.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """A single detected face with bounding box and embedding."""

    bbox: np.ndarray  # [x1, y1, x2, y2]
    embedding: np.ndarray  # 512-D ArcFace embedding
    det_score: float  # detection confidence (0-1)
    age: Optional[int] = None
    gender: Optional[str] = None  # 'M' or 'F'

    @property
    def bbox_tuple(self) -> tuple[int, int, int, int]:
        return tuple(self.bbox.astype(int))

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return float((x2 - x1) * (y2 - y1))


@dataclass
class FaceComparisonResult:
    """Result of comparing two face embeddings."""

    cosine_distance: float
    cosine_similarity: float
    is_same_person: bool
    confidence: float  # 0-1 confidence that they are the same person

    @staticmethod
    def from_embeddings(
        emb1: np.ndarray,
        emb2: np.ndarray,
        threshold: float | None = None,
    ) -> "FaceComparisonResult":
        threshold = threshold or settings.face_match_threshold
        sim = float(np.dot(emb1, emb2) / (norm(emb1) * norm(emb2)))
        dist = 1.0 - sim
        is_same = dist <= threshold

        # Map distance to a 0-1 confidence score
        # At distance 0 -> confidence 1.0
        # At threshold -> confidence 0.5
        # At distance 1 -> confidence ~0
        if dist <= 0:
            confidence = 1.0
        elif dist >= 1.0:
            confidence = 0.0
        else:
            confidence = max(0.0, 1.0 - (dist / threshold) * 0.5) if dist <= threshold else max(
                0.0, 0.5 * (1.0 - (dist - threshold) / (1.0 - threshold))
            )

        return FaceComparisonResult(
            cosine_distance=dist,
            cosine_similarity=sim,
            is_same_person=is_same,
            confidence=confidence,
        )


class FaceEngine:
    """
    Singleton wrapper around InsightFace for face detection and embedding extraction.

    Thread-safe. The InsightFace model is loaded lazily on first use.
    """

    _instance: Optional["FaceEngine"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs) -> "FaceEngine":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = "buffalo_l", ctx_id: int = 0):
        if self._initialized:
            return
        self._model_name = model_name
        self._ctx_id = ctx_id
        self._app: Optional[FaceAnalysis] = None
        self._model_lock = threading.Lock()
        self._initialized = True
        logger.info("FaceEngine singleton created (model=%s)", model_name)

    def _ensure_model(self) -> FaceAnalysis:
        """Lazy-load the InsightFace model."""
        if self._app is None:
            with self._model_lock:
                if self._app is None:
                    logger.info("Loading InsightFace model '%s'...", self._model_name)
                    app = FaceAnalysis(
                        name=self._model_name,
                        providers=["CPUExecutionProvider"],
                    )
                    app.prepare(ctx_id=self._ctx_id, det_size=(640, 640))
                    self._app = app
                    logger.info("InsightFace model loaded successfully.")
        return self._app

    def _load_image(self, image_input: str | Path | np.ndarray) -> np.ndarray:
        """Load image from path or pass through if already an ndarray."""
        if isinstance(image_input, np.ndarray):
            return image_input
        path = str(image_input)
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not load image from: {path}")
        return img

    def detect_faces(self, image_input: str | Path | np.ndarray) -> list[FaceDetection]:
        """
        Detect all faces in an image and return their embeddings.

        Args:
            image_input: file path, Path object, or numpy BGR image array.

        Returns:
            List of FaceDetection objects sorted by detection score (highest first).
        """
        app = self._ensure_model()
        img = self._load_image(image_input)

        faces = app.get(img)
        if not faces:
            return []

        detections = []
        for face in faces:
            det = FaceDetection(
                bbox=face.bbox,
                embedding=face.normed_embedding,
                det_score=float(face.det_score),
                age=getattr(face, "age", None),
                gender="M" if getattr(face, "gender", None) == 1 else "F"
                if getattr(face, "gender", None) is not None
                else None,
            )
            detections.append(det)

        # Sort by detection score descending
        detections.sort(key=lambda d: d.det_score, reverse=True)
        return detections

    def extract_embedding(self, image_input: str | Path | np.ndarray) -> np.ndarray:
        """
        Extract the face embedding from an image.
        Uses the highest-confidence face if multiple are detected.

        Args:
            image_input: file path, Path object, or numpy BGR image array.

        Returns:
            512-D normalized embedding vector.

        Raises:
            ValueError: If no face is detected.
        """
        detections = self.detect_faces(image_input)
        if not detections:
            raise ValueError("No face detected in the image.")
        return detections[0].embedding

    def compare_faces(
        self,
        image_or_emb_1: str | Path | np.ndarray,
        image_or_emb_2: str | Path | np.ndarray,
        threshold: float | None = None,
    ) -> FaceComparisonResult:
        """
        Compare two faces and determine if they are the same person.

        Args:
            image_or_emb_1: file path, image array, or pre-computed 512-D embedding.
            image_or_emb_2: file path, image array, or pre-computed 512-D embedding.
            threshold: cosine distance threshold (default from settings).

        Returns:
            FaceComparisonResult with distance, similarity, and match verdict.
        """
        emb1 = self._resolve_embedding(image_or_emb_1)
        emb2 = self._resolve_embedding(image_or_emb_2)
        return FaceComparisonResult.from_embeddings(emb1, emb2, threshold)

    def _resolve_embedding(self, input_val: str | Path | np.ndarray) -> np.ndarray:
        """Resolve input to a 512-D embedding vector."""
        if isinstance(input_val, np.ndarray) and input_val.ndim == 1 and input_val.shape[0] == 512:
            return input_val
        return self.extract_embedding(input_val)

    def find_best_match(
        self,
        reference_embedding: np.ndarray,
        candidate_images: list[str | Path | np.ndarray],
        threshold: float | None = None,
    ) -> list[tuple[int, FaceComparisonResult]]:
        """
        Compare a reference face against multiple candidate images.

        Args:
            reference_embedding: 512-D embedding of the reference face.
            candidate_images: list of image paths or arrays to compare against.
            threshold: cosine distance threshold.

        Returns:
            List of (index, FaceComparisonResult) tuples for matches that pass the threshold,
            sorted by cosine distance ascending (best match first).
        """
        threshold = threshold or settings.face_match_threshold
        matches = []

        for idx, candidate in enumerate(candidate_images):
            try:
                detections = self.detect_faces(candidate)
                for det in detections:
                    result = FaceComparisonResult.from_embeddings(
                        reference_embedding, det.embedding, threshold
                    )
                    if result.is_same_person:
                        matches.append((idx, result))
                        break  # One match per candidate is enough
            except Exception as e:
                logger.warning("Failed to process candidate %d: %s", idx, e)
                continue

        matches.sort(key=lambda m: m[1].cosine_distance)
        return matches


# Module-level convenience accessor
def get_face_engine() -> FaceEngine:
    """Get or create the singleton FaceEngine instance."""
    return FaceEngine()
