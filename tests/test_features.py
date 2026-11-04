from dataclasses import replace

import numpy as np
import pytest

from pose_meme.features import FEATURE_SIZE, HAND_POINTS, FeatureView, extract_features
from tests.conftest import (
    LEFT_ELBOW,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    Landmark,
    build_hand,
    build_pose,
)

MIN_VISIBILITY = 0.5


def extract(pose, hands=(), aspect_ratio: float = 1.0):
    return extract_features(pose, list(hands), aspect_ratio, MIN_VISIBILITY)


def moved(landmarks, dx: float = 0.0, dy: float = 0.0, scale: float = 1.0):
    return [replace(lm, x=lm.x * scale + dx, y=lm.y * scale + dy) for lm in landmarks]


def as_frame_coordinates(points_px, width: int, height: int):
    return [Landmark(x / width, y / height, visibility) for x, y, visibility in points_px]


def test_vector_size_does_not_depend_on_the_detected_hands(neutral_pose) -> None:
    hand = build_hand((0.38, 0.70))

    assert extract(neutral_pose).shape == (FEATURE_SIZE,)
    assert extract(neutral_pose, [hand]).shape == (FEATURE_SIZE,)
    assert extract(neutral_pose, [hand, build_hand((0.62, 0.70))]).shape == (FEATURE_SIZE,)


def test_shoulders_become_the_origin_and_the_unit_of_distance(neutral_pose) -> None:
    view = FeatureView(extract(neutral_pose))

    assert view.point("left_shoulder") == pytest.approx((0.5, 0.0))
    assert view.point("right_shoulder") == pytest.approx((-0.5, 0.0))


def test_y_keeps_growing_downward(neutral_pose) -> None:
    view = FeatureView(extract(neutral_pose))

    assert view.point("nose")[1] < 0 < view.point("left_wrist")[1]


def test_features_do_not_change_with_position_in_the_frame(neutral_pose) -> None:
    hand = build_hand((0.38, 0.70))

    shifted = extract(moved(neutral_pose, dx=0.2, dy=-0.1), [moved(hand, dx=0.2, dy=-0.1)])

    np.testing.assert_allclose(shifted, extract(neutral_pose, [hand]), atol=1e-9)


def test_features_do_not_change_with_distance_from_the_camera(neutral_pose) -> None:
    hand = build_hand((0.38, 0.70))

    smaller = extract(moved(neutral_pose, scale=0.5), [moved(hand, scale=0.5)])

    np.testing.assert_allclose(smaller, extract(neutral_pose, [hand]), atol=1e-9)


def test_features_do_not_change_with_the_frame_aspect_ratio() -> None:
    # The same body in pixels, seen by a 16:9 and by a 4:3 camera. The shoulders are
    # tilted on purpose: horizontal shoulders would hide a wrong distance formula.
    body_px = [(320.0, 240.0, 1.0)] * 33
    body_px[NOSE] = (320.0, 100.0, 1.0)
    body_px[LEFT_SHOULDER] = (400.0, 180.0, 1.0)
    body_px[RIGHT_SHOULDER] = (240.0, 220.0, 1.0)
    body_px[LEFT_WRIST] = (420.0, 60.0, 1.0)

    wide = extract(as_frame_coordinates(body_px, 1280, 720), aspect_ratio=1280 / 720)
    narrow = extract(as_frame_coordinates(body_px, 640, 480), aspect_ratio=640 / 480)

    np.testing.assert_allclose(wide, narrow, atol=1e-9)


def test_low_visibility_point_is_reported_missing_and_zeroed(make_pose) -> None:
    features = extract(make_pose(visibility={LEFT_ELBOW: 0.1}))
    view = FeatureView(features)

    assert view.point("left_elbow") is None
    assert view.point("right_elbow") is not None
    assert np.isfinite(features).all()


def test_point_at_the_visibility_threshold_is_kept(make_pose) -> None:
    view = FeatureView(extract(make_pose(visibility={LEFT_ELBOW: MIN_VISIBILITY})))

    assert view.point("left_elbow") is not None


@pytest.mark.parametrize("hidden", [LEFT_SHOULDER, RIGHT_SHOULDER])
def test_no_features_without_both_shoulders(make_pose, hidden: int) -> None:
    assert extract(make_pose(visibility={hidden: 0.1})) is None


def test_no_features_when_the_shoulders_overlap(make_pose) -> None:
    pose = make_pose(overrides={LEFT_SHOULDER: (0.5, 0.4), RIGHT_SHOULDER: (0.5, 0.4)})

    assert extract(pose) is None


def test_undetected_hands_are_reported_missing_and_zeroed(neutral_pose) -> None:
    features = extract(neutral_pose)
    view = FeatureView(features)

    assert view.hand("left") is None
    assert view.hand("right") is None
    assert np.isfinite(features).all()


def test_hand_points_share_the_body_coordinates(neutral_pose) -> None:
    wrist = (neutral_pose[LEFT_WRIST].x, neutral_pose[LEFT_WRIST].y)
    view = FeatureView(extract(neutral_pose, [build_hand(wrist)]))

    hand = view.hand("left")

    assert hand.shape == (HAND_POINTS, 2)
    assert tuple(hand[0]) == pytest.approx(view.point("left_wrist"))


def test_hand_goes_to_the_arm_with_the_nearest_wrist(neutral_pose) -> None:
    wrist = (neutral_pose[RIGHT_WRIST].x, neutral_pose[RIGHT_WRIST].y)
    view = FeatureView(extract(neutral_pose, [build_hand(wrist)]))

    assert view.hand("right") is not None
    assert view.hand("left") is None


def test_hand_order_from_the_detector_does_not_matter(neutral_pose) -> None:
    left = build_hand((neutral_pose[LEFT_WRIST].x, neutral_pose[LEFT_WRIST].y))
    right = build_hand((neutral_pose[RIGHT_WRIST].x, neutral_pose[RIGHT_WRIST].y))

    np.testing.assert_array_equal(extract(neutral_pose, [left, right]), extract(neutral_pose, [right, left]))


def test_two_hands_never_land_on_the_same_arm(neutral_pose) -> None:
    # Both sit closer to the right wrist; the best overall match still splits them.
    near = build_hand((0.38, 0.70))
    less_near = build_hand((0.45, 0.70))

    view = FeatureView(extract(neutral_pose, [near, less_near]))

    assert view.hand("right")[0] == pytest.approx(FeatureView(extract(neutral_pose, [near])).hand("right")[0])
    assert view.hand("left") is not None
