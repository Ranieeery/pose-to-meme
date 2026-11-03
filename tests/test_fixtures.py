"""Guards the synthetic poses: later tests are only as good as these fixtures."""

import pytest

from tests.conftest import (
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    NUM_LANDMARKS,
    RIGHT_INDEX,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)

POSE_FIXTURES = [
    "neutral_pose",
    "hand_on_mouth_pose",
    "hands_on_mouth_pose",
    "finger_up_pose",
    "pointing_at_self_pose",
    "hands_forward_pose",
    "hand_on_chest_pose",
]


@pytest.mark.parametrize("fixture_name", POSE_FIXTURES)
def test_pose_has_all_landmarks_inside_the_frame(fixture_name: str, request: pytest.FixtureRequest) -> None:
    pose = request.getfixturevalue(fixture_name)

    assert len(pose) == NUM_LANDMARKS
    assert all(0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0 for lm in pose)
    assert all(lm.visibility == 1.0 for lm in pose)


def test_y_grows_downward(neutral_pose) -> None:
    assert neutral_pose[NOSE].y < neutral_pose[LEFT_SHOULDER].y < neutral_pose[LEFT_WRIST].y


def test_mirrored_frame_puts_left_landmark_on_image_right(neutral_pose) -> None:
    assert neutral_pose[LEFT_SHOULDER].x > neutral_pose[RIGHT_SHOULDER].x


def test_one_hand_on_mouth_leaves_the_other_arm_relaxed(hand_on_mouth_pose, neutral_pose) -> None:
    assert hand_on_mouth_pose[RIGHT_WRIST].y < hand_on_mouth_pose[RIGHT_SHOULDER].y
    assert hand_on_mouth_pose[LEFT_WRIST] == neutral_pose[LEFT_WRIST]


def test_both_hands_on_mouth_are_above_the_shoulders_and_together(hands_on_mouth_pose) -> None:
    pose = hands_on_mouth_pose
    assert pose[LEFT_WRIST].y < pose[LEFT_SHOULDER].y
    assert pose[RIGHT_WRIST].y < pose[RIGHT_SHOULDER].y
    assert pose[RIGHT_SHOULDER].x < pose[RIGHT_WRIST].x < pose[LEFT_WRIST].x < pose[LEFT_SHOULDER].x


def test_raised_finger_points_up(finger_up_pose) -> None:
    assert finger_up_pose[RIGHT_INDEX].y < finger_up_pose[RIGHT_WRIST].y
    assert finger_up_pose[RIGHT_INDEX].x == finger_up_pose[RIGHT_WRIST].x


def test_finger_pointing_at_self_goes_toward_the_face(pointing_at_self_pose) -> None:
    pose = pointing_at_self_pose
    assert abs(pose[RIGHT_INDEX].x - pose[NOSE].x) < abs(pose[RIGHT_WRIST].x - pose[NOSE].x)


def test_hands_forward_are_wider_than_the_shoulders(hands_forward_pose) -> None:
    pose = hands_forward_pose
    assert pose[RIGHT_WRIST].x < pose[RIGHT_SHOULDER].x
    assert pose[LEFT_WRIST].x > pose[LEFT_SHOULDER].x


def test_hand_on_chest_is_below_the_shoulders_and_between_them(hand_on_chest_pose) -> None:
    pose = hand_on_chest_pose
    assert pose[RIGHT_WRIST].y > pose[RIGHT_SHOULDER].y
    assert pose[RIGHT_SHOULDER].x < pose[RIGHT_WRIST].x < pose[LEFT_SHOULDER].x


def test_make_pose_applies_overrides(make_pose) -> None:
    pose = make_pose(overrides={NOSE: (0.1, 0.2)}, visibility={LEFT_WRIST: 0.3})

    assert (pose[NOSE].x, pose[NOSE].y) == (0.1, 0.2)
    assert pose[LEFT_WRIST].visibility == 0.3
    assert pose[RIGHT_WRIST].visibility == 1.0
