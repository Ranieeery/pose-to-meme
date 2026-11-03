from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from pose_meme.config import TrackerConfig
from pose_meme.tracker import (
    POSE_CONNECTIONS,
    Landmark,
    ModelNotFoundError,
    PoseTracker,
    create_tracker,
    landmarks_from_result,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class FakeDetector:
    def __init__(self, result: SimpleNamespace | None = None) -> None:
        self.result = result or SimpleNamespace(pose_landmarks=[])
        self.images: list[np.ndarray] = []
        self.timestamps: list[int] = []
        self.closed = False

    def detect_for_video(self, image, timestamp_ms: int) -> SimpleNamespace:
        self.images.append(image.numpy_view().copy())
        self.timestamps.append(timestamp_ms)
        return self.result

    def close(self) -> None:
        self.closed = True


def raw_landmark(x: float, y: float, visibility: float | None) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=0.0, visibility=visibility)


def black_frame() -> np.ndarray:
    return np.zeros((4, 4, 3), dtype=np.uint8)


def test_no_detected_person_gives_none() -> None:
    assert landmarks_from_result(SimpleNamespace(pose_landmarks=[])) is None


def test_result_is_converted_to_plain_landmarks() -> None:
    result = SimpleNamespace(pose_landmarks=[[raw_landmark(0.1, 0.2, 0.9), raw_landmark(0.3, 0.4, 0.5)]])

    assert landmarks_from_result(result) == [Landmark(0.1, 0.2, 0.9), Landmark(0.3, 0.4, 0.5)]


def test_missing_visibility_counts_as_not_visible() -> None:
    result = SimpleNamespace(pose_landmarks=[[raw_landmark(0.1, 0.2, None)]])

    assert landmarks_from_result(result) == [Landmark(0.1, 0.2, 0.0)]


def test_only_the_first_person_is_used() -> None:
    result = SimpleNamespace(pose_landmarks=[[raw_landmark(0.1, 0.1, 1.0)], [raw_landmark(0.9, 0.9, 1.0)]])

    assert landmarks_from_result(result) == [Landmark(0.1, 0.1, 1.0)]


def test_frame_reaches_the_detector_as_rgb() -> None:
    detector = FakeDetector()
    frame = black_frame()
    frame[..., 0] = 255  # pure blue in OpenCV's BGR order

    PoseTracker(detector, clock=lambda: 1.0).detect(frame)

    np.testing.assert_array_equal(detector.images[0], cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    assert detector.images[0][0, 0].tolist() == [0, 0, 255]


def test_timestamps_are_in_milliseconds_and_follow_the_clock() -> None:
    detector = FakeDetector()
    times = iter([1.0, 1.5])
    tracker = PoseTracker(detector, clock=lambda: next(times))

    tracker.detect(black_frame())
    tracker.detect(black_frame())

    assert detector.timestamps == [1000, 1500]


def test_timestamps_strictly_increase_even_if_the_clock_does_not() -> None:
    # MediaPipe's VIDEO mode rejects a repeated timestamp.
    detector = FakeDetector()
    tracker = PoseTracker(detector, clock=lambda: 1.0)

    tracker.detect(black_frame())
    tracker.detect(black_frame())

    assert detector.timestamps == [1000, 1001]


def test_detect_returns_the_converted_landmarks() -> None:
    detector = FakeDetector(SimpleNamespace(pose_landmarks=[[raw_landmark(0.1, 0.2, 0.9)]]))

    assert PoseTracker(detector, clock=lambda: 1.0).detect(black_frame()) == [Landmark(0.1, 0.2, 0.9)]


def test_context_manager_closes_the_detector() -> None:
    detector = FakeDetector()

    with PoseTracker(detector, clock=lambda: 1.0):
        assert not detector.closed

    assert detector.closed


def test_missing_model_file_raises_a_helpful_error(tmp_path: Path) -> None:
    config = TrackerConfig(model_variant="lite", model_dir=tmp_path, num_poses=1, min_visibility=0.5)

    with pytest.raises(ModelNotFoundError, match=r"pose_landmarker_lite\.task.*README"):
        create_tracker(config)


def test_connections_are_index_pairs_inside_the_33_landmarks() -> None:
    assert POSE_CONNECTIONS
    assert all(0 <= start < 33 and 0 <= end < 33 for start, end in POSE_CONNECTIONS)


@pytest.mark.integration
def test_real_model_finds_nobody_in_a_black_frame() -> None:
    config = TrackerConfig("full", REPO_ROOT / "models", num_poses=1, min_visibility=0.5)

    with create_tracker(config) as tracker:
        assert tracker.detect(np.zeros((480, 640, 3), dtype=np.uint8)) is None


@pytest.mark.integration
def test_real_model_finds_the_33_landmarks_of_a_person() -> None:
    image = cv2.imread(str(REPO_ROOT / "memes" / "bjork-pose.jpg"))
    if image is None:
        pytest.skip("sample image not available")
    config = TrackerConfig("full", REPO_ROOT / "models", num_poses=1, min_visibility=0.5)

    with create_tracker(config) as tracker:
        landmarks = tracker.detect(image)

    assert landmarks is not None
    assert len(landmarks) == 33
