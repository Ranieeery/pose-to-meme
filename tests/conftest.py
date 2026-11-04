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


HAND_LANDMARKS = 21
FINGER_BASES = {"thumb": 1, "index": 5, "middle": 9, "ring": 13, "pinky": 17}
OPEN_HAND = ("thumb", "index", "middle", "ring", "pinky")
POINTING_HAND = ("index",)
ROCK_HAND = ("index", "pinky")


def build_hand(
    wrist: Point,
    direction: Point = (0.0, -1.0),
    size: float = 0.1,
    extended: tuple[str, ...] = OPEN_HAND,
    spread: float = 1.0,
) -> Pose:
    """Build the 21 landmarks of a hand; wrist to extended fingertip measures 1.1 * size.

    direction is a unit vector from the wrist to the fingers. Fingers not listed in
    extended fold back toward the palm, as in a real fist. spread is how many times
    wider the fingertips are than the knuckles: 1 for parallel fingers.
    """
    dx, dy = direction
    perpendicular = (-dy, dx)
    points: list[Landmark] = [Landmark(*wrist)] * HAND_LANDMARKS

    def at(along: float, across: float) -> Landmark:
        return Landmark(
            wrist[0] + size * (dx * along + perpendicular[0] * across),
            wrist[1] + size * (dy * along + perpendicular[1] * across),
        )

    for position, (finger, base) in enumerate(FINGER_BASES.items()):
        across = (position - 2) * 0.25
        alongs = (0.5, 0.7, 0.9, 1.1) if finger in extended else (0.5, 0.65, 0.55, 0.45)
        for joint, along in enumerate(alongs):
            fan = 1.0 + (spread - 1.0) * joint / 3
            points[base + joint] = at(along, across * fan)
    return points


@pytest.fixture
def make_hand() -> Callable[..., Pose]:
    return build_hand


@pytest.fixture
def make_pose() -> Callable[..., Pose]:
    """Factory for custom poses, e.g. to lower the visibility of one landmark."""
    return build_pose


@pytest.fixture
def neutral_pose() -> Pose:
    return build_pose()


@pytest.fixture
def hand_on_mouth_pose() -> Pose:
    """A hand at the mouth; open or closed depends on the detected fingers."""
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
    return build_pose(_right_arm((0.25, 0.42), (0.30, 0.28), index=(0.40, 0.27)))


@pytest.fixture
def hands_forward_pose() -> Pose:
    """Both hands reaching out, wider than the shoulders; open or claws depends on the fingers."""
    return build_pose({**_left_arm((0.70, 0.50), (0.80, 0.42)), **_right_arm((0.30, 0.50), (0.20, 0.42))})


@pytest.fixture
def hand_on_chest_pose() -> Pose:
    """mao_no_peito: one hand resting on the chest."""
    return build_pose(_right_arm((0.36, 0.55), (0.52, 0.47)))
