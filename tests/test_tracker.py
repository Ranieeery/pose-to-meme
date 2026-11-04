from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from pose_meme.config import HandTrackerConfig, TrackerConfig
from pose_meme.tracker import (
    HAND_CONNECTIONS,
    POSE_CONNECTIONS,
    BodyTracker,
    HandHold,
    HandTracker,
    Landmark,
    ModelNotFoundError,
    PoseTracker,
    create_hand_tracker,
    create_tracker,
    hands_from_result,
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


def hand_config(model_path: Path) -> HandTrackerConfig:
    return HandTrackerConfig(
        model_path=model_path,
        num_hands=2,
        min_detection_confidence=0.5,
        min_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        hold_frames=0,
    )


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


def hand_result(*hands: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(hand_landmarks=list(hands))


def test_no_detected_hand_gives_an_empty_list() -> None:
    assert hands_from_result(hand_result()) == []


def test_hands_are_converted_and_count_as_visible() -> None:
    # The hand model fills no visibility value; being detected is what it reports.
    result = hand_result([raw_landmark(0.1, 0.2, None)], [raw_landmark(0.3, 0.4, 0.0)])

    assert hands_from_result(result) == [[Landmark(0.1, 0.2, 1.0)], [Landmark(0.3, 0.4, 1.0)]]


def test_hand_tracker_timestamps_strictly_increase() -> None:
    detector = FakeDetector(hand_result())
    tracker = HandTracker(detector, clock=lambda: 1.0)

    tracker.detect(black_frame())
    tracker.detect(black_frame())

    assert detector.timestamps == [1000, 1001]


ONE = [Landmark(0.1, 0.1, 1.0)]
TWO = [Landmark(0.2, 0.2, 1.0)]
THREE = [Landmark(0.3, 0.3, 1.0)]


def test_hand_hold_passes_a_steady_detection_through() -> None:
    hold = HandHold(hold_frames=3)

    assert hold.update([ONE, TWO]) == [ONE, TWO]
    assert hold.update([THREE, TWO]) == [THREE, TWO]


def test_hand_hold_covers_a_brief_loss() -> None:
    hold = HandHold(hold_frames=2)
    hold.update([ONE, TWO])

    assert hold.update([ONE]) == [ONE, TWO]
    assert hold.update([]) == [ONE, TWO]


def test_hand_hold_lets_go_after_the_limit() -> None:
    hold = HandHold(hold_frames=2)
    hold.update([ONE, TWO])
    hold.update([])
    hold.update([])

    assert hold.update([]) == []


def test_hand_hold_starts_over_when_the_hand_comes_back() -> None:
    hold = HandHold(hold_frames=2)
    hold.update([ONE])
    hold.update([])
    hold.update([])
    assert hold.update([TWO]) == [TWO]

    assert hold.update([]) == [TWO]
    assert hold.update([]) == [TWO]
    assert hold.update([]) == []


def test_hand_hold_of_zero_frames_changes_nothing() -> None:
    hold = HandHold(hold_frames=0)
    hold.update([ONE])

    assert hold.update([]) == []


def test_body_tracker_holds_a_hand_lost_for_a_moment() -> None:
    pose_detector = FakeDetector(SimpleNamespace(pose_landmarks=[]))
    hand_detector = FakeDetector(hand_result([raw_landmark(0.3, 0.4, None)]))

    with BodyTracker(PoseTracker(pose_detector), HandTracker(hand_detector), hold_frames=1) as tracker:
        first = tracker.detect(black_frame())
        hand_detector.result = hand_result()
        second = tracker.detect(black_frame())
        third = tracker.detect(black_frame())

    assert first.hands == second.hands == [[Landmark(0.3, 0.4, 1.0)]]
    assert third.hands == []


def test_body_tracker_returns_pose_and_hands_from_the_same_frame() -> None:
    pose_detector = FakeDetector(SimpleNamespace(pose_landmarks=[[raw_landmark(0.1, 0.2, 0.9)]]))
    hand_detector = FakeDetector(hand_result([raw_landmark(0.3, 0.4, None)]))
    frame = black_frame()
    frame[0, 0] = (9, 9, 9)

    with BodyTracker(PoseTracker(pose_detector), HandTracker(hand_detector), hold_frames=0) as tracker:
        body = tracker.detect(frame)

    assert body.pose == [Landmark(0.1, 0.2, 0.9)]
    assert body.hands == [[Landmark(0.3, 0.4, 1.0)]]
    np.testing.assert_array_equal(pose_detector.images[0], hand_detector.images[0])


def test_body_tracker_closes_both_detectors() -> None:
    pose_detector, hand_detector = FakeDetector(), FakeDetector(hand_result())

    with BodyTracker(PoseTracker(pose_detector), HandTracker(hand_detector), hold_frames=0):
        pass

    assert pose_detector.closed
    assert hand_detector.closed


def test_missing_hand_model_raises_a_helpful_error(tmp_path: Path) -> None:
    config = hand_config(tmp_path / "hand_landmarker.task")

    with pytest.raises(ModelNotFoundError, match=r"hand_landmarker\.task.*README"):
        create_hand_tracker(config)


def test_hand_connections_are_index_pairs_inside_the_21_landmarks() -> None:
    assert HAND_CONNECTIONS
    assert all(0 <= start < 21 and 0 <= end < 21 for start, end in HAND_CONNECTIONS)


@pytest.mark.integration
def test_real_hand_model_finds_the_21_landmarks_of_a_hand() -> None:
    image = cv2.imread(str(REPO_ROOT / "memes" / "valverde-pose.webp"))
    if image is None:
        pytest.skip("sample image not available")
    config = hand_config(REPO_ROOT / "models" / "hand_landmarker.task")

    with create_hand_tracker(config) as tracker:
        hands = tracker.detect(image)

    assert hands
    assert all(len(hand) == 21 for hand in hands)


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
