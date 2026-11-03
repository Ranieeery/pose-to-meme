"""Pose tracking with MediaPipe's Tasks API (PoseLandmarker, .task model file)."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarksConnections,
    RunningMode,
)

from pose_meme.config import TrackerConfig

POSE_CONNECTIONS: tuple[tuple[int, int], ...] = tuple(
    (connection.start, connection.end) for connection in PoseLandmarksConnections.POSE_LANDMARKS
)

MS_PER_SECOND = 1000


class ModelNotFoundError(FileNotFoundError):
    """The .task model file has not been downloaded."""


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    visibility: float


def landmarks_from_result(result: Any) -> list[Landmark] | None:
    """Convert a PoseLandmarkerResult to plain landmarks; None when nobody was found.

    The API types visibility as optional, so a missing value counts as not visible.
    """
    if not result.pose_landmarks:
        return None
    return [Landmark(raw.x, raw.y, raw.visibility or 0.0) for raw in result.pose_landmarks[0]]


class PoseTracker:
    def __init__(self, detector: Any, clock: Callable[[], float] = time.monotonic) -> None:
        self._detector = detector
        self._clock = clock
        self._last_timestamp_ms = -1

    def detect(self, frame_bgr: np.ndarray) -> list[Landmark] | None:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        # VIDEO mode requires strictly increasing timestamps.
        timestamp_ms = max(int(self._clock() * MS_PER_SECOND), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        return landmarks_from_result(self._detector.detect_for_video(image, timestamp_ms))

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._detector.close()


def create_tracker(config: TrackerConfig) -> PoseTracker:
    """Load the configured .task model into a tracker."""
    if not config.model_path.is_file():
        raise ModelNotFoundError(f"model not found: {config.model_path}; see the download steps in README.md")
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(config.model_path)),
        running_mode=RunningMode.VIDEO,
        num_poses=config.num_poses,
    )
    return PoseTracker(PoseLandmarker.create_from_options(options))
