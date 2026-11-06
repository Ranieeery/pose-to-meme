"""Landmarks -> feature vector. The single source of truth for collection, training and inference.

Coordinates are "body units": x corrected by the frame's aspect ratio, origin at the
center of the shoulders, one unit = the distance between the shoulders. That makes
them independent of position in the frame, distance from the camera and camera shape.
y keeps MediaPipe's direction, growing downward.

Layout of the vector:
    pose coordinates | pose visibility flags | left hand coordinates, left hand flag
    | right hand coordinates, right hand flag
Anything missing is zeroed and flagged, so the size is always FEATURE_SIZE.
"""

from collections.abc import Sequence
from typing import Protocol

import numpy as np

POSE_POINTS: dict[str, int] = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
}
HAND_POINTS = 21
HAND_WRIST = 0
COORDINATES = 2
SIDES = ("left", "right")

_POSE_NAMES = tuple(POSE_POINTS)
_POSE_COORDINATES = len(POSE_POINTS) * COORDINATES
_HAND_COORDINATES = HAND_POINTS * COORDINATES
_FLAGS_START = _POSE_COORDINATES
_HANDS_START = _FLAGS_START + len(POSE_POINTS)
_HAND_BLOCK = _HAND_COORDINATES + 1  # coordinates + presence flag

FEATURE_SIZE = _HANDS_START + len(SIDES) * _HAND_BLOCK

_MIN_SHOULDER_DISTANCE = 1e-6  # guards the division; only overlapping shoulders fall below it


class LandmarkLike(Protocol):
    x: float
    y: float
    visibility: float


def extract_features(
    pose: Sequence[LandmarkLike],
    hands: Sequence[Sequence[LandmarkLike]],
    aspect_ratio: float,
    min_visibility: float,
) -> np.ndarray | None:
    """Build the feature vector; None if the shoulders can't define the body units."""
    left_shoulder = pose[POSE_POINTS["left_shoulder"]]
    right_shoulder = pose[POSE_POINTS["right_shoulder"]]
    if left_shoulder.visibility < min_visibility or right_shoulder.visibility < min_visibility:
        return None

    left = np.array([left_shoulder.x * aspect_ratio, left_shoulder.y])
    right = np.array([right_shoulder.x * aspect_ratio, right_shoulder.y])
    shoulder_distance = float(np.linalg.norm(left - right))
    if shoulder_distance < _MIN_SHOULDER_DISTANCE:
        return None
    center = (left + right) / 2

    def to_body_units(landmarks: Sequence[LandmarkLike]) -> np.ndarray:
        points = np.array([(lm.x * aspect_ratio, lm.y) for lm in landmarks])
        return (points - center) / shoulder_distance

    features = np.zeros(FEATURE_SIZE)

    indexes = list(POSE_POINTS.values())
    visible = np.array([pose[index].visibility >= min_visibility for index in indexes])
    coordinates = to_body_units([pose[index] for index in indexes])
    coordinates[~visible] = 0.0
    features[:_POSE_COORDINATES] = coordinates.ravel()
    features[_FLAGS_START:_HANDS_START] = visible

    for side, hand in assign_hands(pose, hands, aspect_ratio).items():
        start = _hand_start(side)
        features[start : start + _HAND_COORDINATES] = to_body_units(hand).ravel()
        features[start + _HAND_COORDINATES] = 1.0
    return features


def assign_hands(
    pose: Sequence[LandmarkLike],
    hands: Sequence[Sequence[LandmarkLike]],
    aspect_ratio: float,
) -> dict[str, Sequence[LandmarkLike]]:
    """Match hands to arms by the nearest pose wrist, with the best overall match for two."""

    def distance(hand: Sequence[LandmarkLike], side: str) -> float:
        wrist = pose[POSE_POINTS[f"{side}_wrist"]]
        return float(np.hypot((hand[HAND_WRIST].x - wrist.x) * aspect_ratio, hand[HAND_WRIST].y - wrist.y))

    if not hands:
        return {}
    if len(hands) == 1:
        return {min(SIDES, key=lambda side: distance(hands[0], side)): hands[0]}

    first, second = hands[: len(SIDES)]
    in_order = distance(first, "left") + distance(second, "right")
    swapped = distance(first, "right") + distance(second, "left")
    return {"left": first, "right": second} if in_order <= swapped else {"left": second, "right": first}


def _hand_start(side: str) -> int:
    return _HANDS_START + SIDES.index(side) * _HAND_BLOCK


def _other_side(name: str) -> str:
    for side, other in zip(SIDES, reversed(SIDES)):
        if name.startswith(f"{side}_"):
            return f"{other}_{name.removeprefix(f'{side}_')}"
    return name


def _mirror_plan() -> tuple[np.ndarray, np.ndarray]:
    """For each output position: which input position it reads, and the sign to apply."""
    source = np.arange(FEATURE_SIZE)
    sign = np.ones(FEATURE_SIZE)
    for position, name in enumerate(_POSE_NAMES):
        partner = _POSE_NAMES.index(_other_side(name))
        for axis in range(COORDINATES):
            source[position * COORDINATES + axis] = partner * COORDINATES + axis
        sign[position * COORDINATES] = -1.0  # x
        source[_FLAGS_START + position] = _FLAGS_START + partner
    for side in SIDES:
        start, partner_start = _hand_start(side), _hand_start(SIDES[1 - SIDES.index(side)])
        source[start : start + _HAND_BLOCK] = np.arange(partner_start, partner_start + _HAND_BLOCK)
        sign[start : start + _HAND_COORDINATES : COORDINATES] = -1.0  # x of every hand point
    return source, sign


_MIRROR_SOURCE, _MIRROR_SIGN = _mirror_plan()


def mirror_features(features: np.ndarray) -> np.ndarray:
    """The vector of the same pose mirrored left to right, for one vector or rows of them.

    Body units are centered on the shoulders, so mirroring negates x; the left and
    right pose points and hands also trade places, flags included. Missing points stay
    zero. Meant for training data only: the app never mirrors what it classifies.
    """
    return features[..., _MIRROR_SOURCE] * _MIRROR_SIGN


class FeatureView:
    """Named read access to a feature vector, so no other module depends on its layout."""

    def __init__(self, features: np.ndarray) -> None:
        self._features = features

    def point(self, name: str) -> tuple[float, float] | None:
        """A pose point in body units; None if it was below the visibility threshold."""
        position = _POSE_NAMES.index(name)
        if not self._features[_FLAGS_START + position]:
            return None
        x, y = self._features[position * COORDINATES : (position + 1) * COORDINATES]
        return float(x), float(y)

    def hand(self, side: str) -> np.ndarray | None:
        """A hand's points as a (21, 2) array in body units; None if it wasn't detected."""
        start = _hand_start(side)
        if not self._features[start + _HAND_COORDINATES]:
            return None
        return self._features[start : start + _HAND_COORDINATES].reshape(HAND_POINTS, COORDINATES)
