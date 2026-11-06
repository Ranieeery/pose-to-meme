import numpy as np
import pytest

from pose_meme.dataset import Samples
from pose_meme.features import FEATURE_SIZE, extract_features
from scripts.demo_gif import body_landmarks, demo_indices
from tests.conftest import LEFT_ELBOW, LEFT_WRIST, RIGHT_WRIST, build_hand, build_pose

ASPECT = 4 / 3


def scene_features() -> np.ndarray:
    pose = build_pose({RIGHT_WRIST: (0.4, 0.3), LEFT_WRIST: (0.62, 0.75)}, visibility={LEFT_ELBOW: 0.1})
    return extract_features(pose, [build_hand((0.4, 0.3), extended=("index",))], ASPECT, min_visibility=0.5)


def test_landmarks_rebuilt_from_features_give_the_same_features_back() -> None:
    features = scene_features()

    pose, hands = body_landmarks(features, ASPECT)

    np.testing.assert_allclose(extract_features(pose, hands, ASPECT, min_visibility=0.5), features, atol=1e-9)


def test_rebuilt_body_fits_in_the_frame() -> None:
    pose, hands = body_landmarks(scene_features(), ASPECT)

    visible = [landmark for landmark in pose if landmark.visibility > 0] + [point for hand in hands for point in hand]
    assert all(0 <= landmark.x <= 1 and 0 <= landmark.y <= 1 for landmark in visible)


def test_hidden_points_stay_hidden() -> None:
    pose, _ = body_landmarks(scene_features(), ASPECT)

    assert pose[LEFT_ELBOW].visibility == 0


def make_samples() -> Samples:
    labels = ["neutro"] * 10 + ["pose_a"] * 10 + ["neutro"] * 10 + ["pose_a"] * 10
    sessions = ["s1"] * 20 + ["s2"] * 20
    return Samples(np.zeros((40, FEATURE_SIZE)), np.array(labels), np.array(sessions))


def test_demo_takes_consecutive_frames_of_each_pose_from_the_session_in_order() -> None:
    indexes = demo_indices(make_samples(), "s2", ["pose_a", "neutro"], start=2, count=3)

    assert indexes.tolist() == [32, 33, 34, 22, 23, 24]


def test_demo_rejects_a_pose_the_session_lacks() -> None:
    with pytest.raises(ValueError, match="pose_b"):
        demo_indices(make_samples(), "s2", ["pose_b"], start=0, count=3)


def test_demo_rejects_a_run_shorter_than_asked() -> None:
    with pytest.raises(ValueError, match="pose_a"):
        demo_indices(make_samples(), "s2", ["pose_a"], start=5, count=10)
