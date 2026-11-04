"""Body and finger tracking with MediaPipe's Tasks API (.task model files)."""

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarksConnections,
    RunningMode,
)

from pose_meme.config import HandTrackerConfig, TrackerConfig

POSE_CONNECTIONS: tuple[tuple[int, int], ...] = tuple(
    (connection.start, connection.end) for connection in PoseLandmarksConnections.POSE_LANDMARKS
)
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = tuple(
    (connection.start, connection.end) for connection in HandLandmarksConnections.HAND_CONNECTIONS
)

MS_PER_SECOND = 1000
DETECTED = 1.0


class ModelNotFoundError(FileNotFoundError):
    """A .task model file has not been downloaded."""


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    visibility: float


@dataclass(frozen=True)
class TrackedBody:
    """What was seen in one frame; pose is None when nobody was found."""

    pose: list[Landmark] | None
    hands: list[list[Landmark]]


def landmarks_from_result(result: Any) -> list[Landmark] | None:
    """Convert a PoseLandmarkerResult to plain landmarks; None when nobody was found.

    The API types visibility as optional, so a missing value counts as not visible.
    """
    if not result.pose_landmarks:
        return None
    return [Landmark(raw.x, raw.y, raw.visibility or 0.0) for raw in result.pose_landmarks[0]]


def hands_from_result(result: Any) -> list[list[Landmark]]:
    """Convert a HandLandmarkerResult; the hand model gives no per-point visibility."""
    return [[Landmark(raw.x, raw.y, DETECTED) for raw in hand] for hand in result.hand_landmarks]


class _VideoTask:
    """A MediaPipe detector in VIDEO mode, which needs strictly increasing timestamps."""

    def __init__(self, detector: Any, clock: Callable[[], float] = time.monotonic) -> None:
        self._detector = detector
        self._clock = clock
        self._last_timestamp_ms = -1

    def _run(self, frame_bgr: np.ndarray) -> Any:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        timestamp_ms = max(int(self._clock() * MS_PER_SECOND), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        return self._detector.detect_for_video(image, timestamp_ms)

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class PoseTracker(_VideoTask):
    def detect(self, frame_bgr: np.ndarray) -> list[Landmark] | None:
        return landmarks_from_result(self._run(frame_bgr))


class HandTracker(_VideoTask):
    def detect(self, frame_bgr: np.ndarray) -> list[list[Landmark]]:
        return hands_from_result(self._run(frame_bgr))


class HandHold:
    """Keeps the last hands through brief detection losses, so they don't flicker.

    Hands close to the face come and go from frame to frame, and each loss would also
    change which rule applies to that hand.
    """

    def __init__(self, hold_frames: int) -> None:
        self._hold_frames = hold_frames
        self._last: list[list[Landmark]] = []
        self._misses = 0

    def update(self, hands: list[list[Landmark]]) -> list[list[Landmark]]:
        if len(hands) < len(self._last) and self._misses < self._hold_frames:
            self._misses += 1
            return self._last
        self._last = hands
        self._misses = 0
        return hands


class BodyTracker:
    """Runs both models on each frame at once: one after the other, they miss 30 FPS."""

    def __init__(self, pose: PoseTracker, hands: HandTracker, hold_frames: int) -> None:
        self._pose = pose
        self._hands = hands
        self._hold = HandHold(hold_frames)
        self._executor = ThreadPoolExecutor(max_workers=1)

    def detect(self, frame_bgr: np.ndarray) -> TrackedBody:
        hands = self._executor.submit(self._hands.detect, frame_bgr)
        return TrackedBody(pose=self._pose.detect(frame_bgr), hands=self._hold.update(hands.result()))

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._executor.shutdown()
        self._pose.close()
        self._hands.close()


def _require_model(path: Path) -> None:
    if not path.is_file():
        raise ModelNotFoundError(f"model not found: {path}; see the download steps in README.md")


def create_tracker(config: TrackerConfig) -> PoseTracker:
    """Load the configured pose model."""
    _require_model(config.model_path)
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(config.model_path)),
        running_mode=RunningMode.VIDEO,
        num_poses=config.num_poses,
    )
    return PoseTracker(PoseLandmarker.create_from_options(options))


def create_hand_tracker(config: HandTrackerConfig) -> HandTracker:
    """Load the configured hand model."""
    _require_model(config.model_path)
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(config.model_path)),
        running_mode=RunningMode.VIDEO,
        num_hands=config.num_hands,
        min_hand_detection_confidence=config.min_detection_confidence,
        min_hand_presence_confidence=config.min_presence_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    )
    return HandTracker(HandLandmarker.create_from_options(options))


def create_body_tracker(tracker: TrackerConfig, hand_tracker: HandTrackerConfig) -> BodyTracker:
    """Load both models; checks them first, so a missing file doesn't leak a loaded model."""
    _require_model(tracker.model_path)
    _require_model(hand_tracker.model_path)
    return BodyTracker(create_tracker(tracker), create_hand_tracker(hand_tracker), hand_tracker.hold_frames)
