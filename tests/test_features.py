from dataclasses import replace

import numpy as np
import pytest

from pose_meme.features import FEATURE_SIZE, HAND_POINTS, POSE_POINTS, FeatureView, extract_features, mirror_features
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


# --- mirroring, for data augmentation ---

ASPECT = 16 / 9


def mirrored_person(pose: list[Landmark], hands: list[list[Landmark]]) -> tuple[list[Landmark], list[list[Landmark]]]:
    """What the camera would see of the same person mirrored: x flipped, and the pose
    labels swapped, since the person's left arm now looks like a right arm."""
    partner = {
        index: POSE_POINTS[name.replace("left_", "right_") if name.startswith("left_") else name.replace("right_", "left_")]
        for name, index in POSE_POINTS.items()
    }
    flipped = [replace(landmark, x=1 - landmark.x) for landmark in pose]
    mirrored_pose = [flipped[partner.get(index, index)] for index in range(len(pose))]
    mirrored_hands = [[replace(landmark, x=1 - landmark.x) for landmark in hand] for hand in hands]
    return mirrored_pose, mirrored_hands


def asymmetric_scene() -> tuple[list[Landmark], list[list[Landmark]]]:
    """Right hand up by the face, left arm down with its elbow hidden: nothing symmetric."""
    pose = build_pose({RIGHT_WRIST: (0.4, 0.3), LEFT_WRIST: (0.62, 0.75)}, visibility={LEFT_ELBOW: 0.1})
    return pose, [build_hand((0.4, 0.3), extended=("index",))]


def test_mirrored_vector_matches_the_features_of_the_mirrored_person() -> None:
    pose, hands = asymmetric_scene()
    mirrored_pose, mirrored_hands = mirrored_person(pose, hands)

    expected = extract_features(mirrored_pose, mirrored_hands, ASPECT, min_visibility=0.5)
    mirrored = mirror_features(extract_features(pose, hands, ASPECT, min_visibility=0.5))

    np.testing.assert_allclose(mirrored, expected, atol=1e-9)


def test_mirroring_moves_the_hand_to_the_other_side() -> None:
    pose, hands = asymmetric_scene()

    mirrored = FeatureView(mirror_features(extract_features(pose, hands, ASPECT, min_visibility=0.5)))

    assert mirrored.hand("right") is None
    assert mirrored.hand("left") is not None
    assert mirrored.point("right_elbow") is None  # the hidden elbow keeps its flag, on the other side


def test_mirroring_twice_gives_back_the_original() -> None:
    pose, hands = asymmetric_scene()
    features = extract_features(pose, hands, ASPECT, min_visibility=0.5)

    np.testing.assert_array_equal(mirror_features(mirror_features(features)), features)


def test_mirroring_works_on_many_rows_at_once() -> None:
    pose, hands = asymmetric_scene()
    one = extract_features(pose, hands, ASPECT, min_visibility=0.5)
    rows = np.stack([one, mirror_features(one)])

    mirrored = mirror_features(rows)

    assert mirrored.shape == (2, FEATURE_SIZE)
    np.testing.assert_array_equal(mirrored[0], mirror_features(one))
    np.testing.assert_array_equal(mirrored[1], one)


def test_mirroring_does_not_change_its_input() -> None:
    pose, hands = asymmetric_scene()
    features = extract_features(pose, hands, ASPECT, min_visibility=0.5)
    before = features.copy()

    mirror_features(features)

    np.testing.assert_array_equal(features, before)
