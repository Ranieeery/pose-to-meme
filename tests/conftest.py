"""Shared fixtures: synthetic MediaPipe-like poses, no camera or model needed.

Conventions copied from MediaPipe, which real landmarks will follow:
- 33 landmarks with x/y normalized to [0, 1] and **y growing downward**;
- the app feeds a mirrored frame, so the landmark labeled "left" (odd index) is
  the one MediaPipe sees on the image's right side, hence the larger x below.

One-handed poses use the right arm; the classifier tests decide what to expect
from the other one.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import pytest

NUM_LANDMARKS = 33

NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_PINKY, RIGHT_PINKY = 17, 18
LEFT_INDEX, RIGHT_INDEX = 19, 20
LEFT_THUMB, RIGHT_THUMB = 21, 22

Point = tuple[float, float]
Pose = list["Landmark"]


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    visibility: float = 1.0


def _left_arm(elbow: Point, wrist: Point, index: Point | None = None) -> dict[int, Point]:
    return {
        LEFT_ELBOW: elbow,
        LEFT_WRIST: wrist,
        LEFT_PINKY: wrist,
        LEFT_INDEX: index or wrist,
        LEFT_THUMB: wrist,
    }


def _right_arm(elbow: Point, wrist: Point, index: Point | None = None) -> dict[int, Point]:
    return {
        RIGHT_ELBOW: elbow,
        RIGHT_WRIST: wrist,
        RIGHT_PINKY: wrist,
        RIGHT_INDEX: index or wrist,
        RIGHT_THUMB: wrist,
    }


# Standing, facing the camera, arms relaxed. Landmarks with no entry here (face
# details and lower body) stay at the fallback point, since features ignore them.
_BODY: dict[int, Point] = {
    NOSE: (0.50, 0.25),
    LEFT_SHOULDER: (0.60, 0.40),
    RIGHT_SHOULDER: (0.40, 0.40),
    **_left_arm((0.62, 0.55), (0.62, 0.70)),
    **_right_arm((0.38, 0.55), (0.38, 0.70)),
}
_FALLBACK: Point = (0.50, 0.80)


def build_pose(overrides: Mapping[int, Point] | None = None, visibility: Mapping[int, float] | None = None) -> Pose:
    """Build a 33-landmark pose: the relaxed body plus optional per-index changes."""
    points = {**_BODY, **(overrides or {})}
    visibility = visibility or {}
    return [
        Landmark(*points.get(index, _FALLBACK), visibility=visibility.get(index, 1.0))
        for index in range(NUM_LANDMARKS)
    ]


@pytest.fixture
def make_pose() -> Callable[..., Pose]:
    """Factory for custom poses, e.g. to lower the visibility of one landmark."""
    return build_pose


@pytest.fixture
def neutral_pose() -> Pose:
    return build_pose()


@pytest.fixture
def hand_on_mouth_pose() -> Pose:
    """mao_na_boca: one hand at the mouth."""
    return build_pose(_right_arm((0.40, 0.52), (0.49, 0.30)))


@pytest.fixture
def hands_on_mouth_pose() -> Pose:
    """maos_na_boca: both hands together over the mouth."""
    return build_pose({**_left_arm((0.56, 0.50), (0.52, 0.30)), **_right_arm((0.44, 0.50), (0.48, 0.30))})


@pytest.fixture
def finger_up_pose() -> Pose:
    """dedo_levantado: hand beside the head, index finger pointing up."""
    return build_pose(_right_arm((0.33, 0.50), (0.32, 0.30), index=(0.32, 0.20)))


@pytest.fixture
def pointing_at_self_pose() -> Pose:
    """apontando_para_si: hand beside the face, index finger pointing at it."""
    return build_pose(_right_arm((0.30, 0.42), (0.36, 0.28), index=(0.44, 0.27)))


@pytest.fixture
def hands_forward_pose() -> Pose:
    """maos_para_frente: both open hands reaching out, wider than the shoulders."""
    return build_pose({**_left_arm((0.70, 0.50), (0.80, 0.42)), **_right_arm((0.30, 0.50), (0.20, 0.42))})


@pytest.fixture
def hand_on_chest_pose() -> Pose:
    """mao_no_peito: one hand resting on the chest."""
    return build_pose(_right_arm((0.36, 0.55), (0.52, 0.47)))
