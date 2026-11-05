"""Pose classifiers behind a single interface."""

from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np

from pose_meme.config import RulesConfig
from pose_meme.features import FEATURE_SIZE, SIDES, FeatureView

POSE_CLASSES = (
    "mao_fechada_na_boca",
    "mao_aberta_na_boca",
    "maos_unidas",
    "dedo_levantado",
    "apontando_para_si",
    "sinal_de_rock",
    "maos_abertas_para_frente",
    "garras_para_frente",
    "mao_no_ombro",
    "dedo_na_boca",
)

# Rules either match or don't, so they carry no graded confidence.
RULE_CONFIDENCE = 1.0

# Hand landmark indexes. Fingers are (middle joint, tip). The thumb is left out
# because it bends sideways, so the distance test below doesn't apply to it.
HAND_WRIST = 0
INDEX_BASE = 5
PALM_CENTER = 9
PINKY_BASE = 17
INDEX_TIP = 8
PINKY_TIP = 20
INDEX_FINGER = (6, INDEX_TIP)
MIDDLE_FINGER = (10, 12)
RING_FINGER = (14, 16)
PINKY_FINGER = (18, PINKY_TIP)
LONG_FINGERS = (INDEX_FINGER, MIDDLE_FINGER, RING_FINGER, PINKY_FINGER)


class ModelLoadError(RuntimeError):
    """A trained model file exists but can't be used."""


class PoseClassifier(Protocol):
    def predict(self, features: np.ndarray) -> tuple[str, float]:
        """Return the class and a confidence in [0, 1]."""
        ...


class MLClassifier:
    """A trained scikit-learn pipeline; the confidence is the top class probability."""

    def __init__(self, pipeline: Any) -> None:
        self._pipeline = pipeline

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        probabilities = self._pipeline.predict_proba(features.reshape(1, -1))[0]
        best = int(probabilities.argmax())
        return str(self._pipeline.classes_[best]), float(probabilities[best])


def load_classifier(model_path: Path, rules: RulesConfig, neutral_class: str) -> PoseClassifier:
    """The trained model if there is one, otherwise the rules.

    A model that exists but can't run is an error: falling back to the rules would hide
    that the model the user trained is not the one running.
    """
    if not model_path.is_file():
        return RuleClassifier(rules, neutral_class)
    try:
        bundle = joblib.load(model_path)
        pipeline, feature_size = bundle["pipeline"], bundle["feature_size"]
    except Exception as error:  # unpickling a bad file can raise nearly anything
        raise ModelLoadError(f"could not load {model_path}: {error}; run scripts.train again") from error
    if feature_size != FEATURE_SIZE:
        raise ModelLoadError(
            f"{model_path} expects {feature_size} features, but features.py has {FEATURE_SIZE}; "
            "record new samples and train again"
        )
    return MLClassifier(pipeline)


def accept_pose(label: str, confidence: float, threshold: float, neutral_class: str) -> str:
    """Keep the class only when the classifier is confident enough."""
    return label if confidence >= threshold else neutral_class


def classify_frame(
    classifier: PoseClassifier,
    features: np.ndarray | None,
    threshold: float,
    neutral_class: str,
) -> tuple[str, float]:
    """Class to show for a frame; a frame without features (no body in view) is neutral."""
    if features is None:
        return neutral_class, RULE_CONFIDENCE
    label, confidence = classifier.predict(features)
    return accept_pose(label, confidence, threshold, neutral_class), confidence


def _distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(first - second))


class RuleClassifier:
    """Geometric rules over body units: the baseline, and the fallback without a trained model."""

    def __init__(self, rules: RulesConfig, neutral_class: str) -> None:
        self._rules = rules
        self._neutral = neutral_class

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        return self._label(FeatureView(features)), RULE_CONFIDENCE

    def _label(self, view: FeatureView) -> str:
        # Order matters: finger gestures also place a hand near the face or the chest,
        # so the specific ones must be checked before the generic position rules. Hands
        # toward the camera come first, since big open fingers can cross the mouth area.
        left, right = (view.hand(side) for side in SIDES)
        if left is not None and right is not None and self._are_toward_camera(left, right):
            spread_open = self._is_spread_open(left) and self._is_spread_open(right)
            return "maos_abertas_para_frente" if spread_open else "garras_para_frente"
        sides_at_mouth = [side for side in SIDES if self._is_at_mouth(view, side)]
        if len(sides_at_mouth) == len(SIDES) or (sides_at_mouth and self._wrists_together(view)):
            return "maos_unidas"
        if any(self._is_rock_sign(view, side) for side in SIDES):
            return "sinal_de_rock"
        if any(self._points_at_self(view, side) for side in SIDES):
            return "apontando_para_si"
        if any(self._finger_up(view, side) for side in SIDES):
            return "dedo_levantado"
        if sides_at_mouth:
            return self._mouth_class(view.hand(sides_at_mouth[0]))
        if any(self._is_on_chest(view, side) for side in SIDES):
            return "mao_no_ombro"
        return self._neutral

    def _wrists_together(self, view: FeatureView) -> bool:
        """Palms pressed together read as one hand, so only the pose wrists show the other one."""
        left, right = (view.point(f"{side}_wrist") for side in SIDES)
        if left is None or right is None:
            return False
        return _distance(np.array(left), np.array(right)) <= self._rules.hands_together_max_gap

    def _mouth_class(self, hand: np.ndarray | None) -> str:
        # Without finger tracking there is no way to tell the fingers apart, so a hand
        # hidden by the face counts as closed, the most generic of the three.
        if hand is None:
            return "mao_fechada_na_boca"
        if self._extended_fingers(hand) >= self._rules.open_hand_min_fingers:
            return "mao_aberta_na_boca"
        return "dedo_na_boca" if self._is_pointing(hand) else "mao_fechada_na_boca"

    def _is_extended(self, hand: np.ndarray, finger: tuple[int, int]) -> bool:
        """A straight finger's tip lies well past its middle joint, seen from the wrist."""
        joint, tip = finger
        wrist = hand[HAND_WRIST]
        return _distance(hand[tip], wrist) > self._rules.finger_extended_ratio * _distance(hand[joint], wrist)

    def _extended_fingers(self, hand: np.ndarray) -> int:
        return sum(self._is_extended(hand, finger) for finger in LONG_FINGERS)

    def _is_pointing(self, hand: np.ndarray) -> bool:
        return self._is_extended(hand, INDEX_FINGER) and self._extended_fingers(hand) == 1

    def _hand_points(self, view: FeatureView, side: str) -> np.ndarray:
        """The detected hand or, failing that, the pose model's rough hand points."""
        hand = view.hand(side)
        if hand is not None:
            return hand
        names = (f"{side}_{part}" for part in ("wrist", "index", "pinky", "thumb"))
        points = [point for point in map(view.point, names) if point is not None]
        return np.array(points).reshape(-1, 2)

    def _is_at_mouth(self, view: FeatureView, side: str) -> bool:
        nose = view.point("nose")
        points = self._hand_points(view, side)
        if nose is None or len(points) == 0:
            return False
        # Without a detected hand the nearest point is a knuckle, not a fingertip.
        radius = self._rules.mouth_radius if view.hand(side) is not None else self._rules.mouth_radius_without_fingers
        return float(np.linalg.norm(points - np.array(nose), axis=1).min()) < radius

    def _are_toward_camera(self, left: np.ndarray, right: np.ndarray) -> bool:
        """Palms look big next to the shoulders (even with curled fingers) and stay apart."""
        return (
            _distance(left[HAND_WRIST], left[PALM_CENTER]) >= self._rules.forward_hand_min_palm
            and _distance(right[HAND_WRIST], right[PALM_CENTER]) >= self._rules.forward_hand_min_palm
            and _distance(left[PALM_CENTER], right[PALM_CENTER]) >= self._rules.forward_hands_min_gap
        )

    def _is_spread_open(self, hand: np.ndarray) -> bool:
        knuckles = _distance(hand[INDEX_BASE], hand[PINKY_BASE])
        fingertips = _distance(hand[INDEX_TIP], hand[PINKY_TIP])
        return (
            self._extended_fingers(hand) >= self._rules.open_hand_min_fingers
            and fingertips >= self._rules.open_hand_min_spread * knuckles
        )

    def _is_rock_sign(self, view: FeatureView, side: str) -> bool:
        hand = view.hand(side)
        if hand is None or hand[HAND_WRIST][1] >= self._rules.chest_max_depth:  # an arm hanging down doesn't count
            return False
        return (
            self._is_extended(hand, INDEX_FINGER)
            and self._is_extended(hand, PINKY_FINGER)
            and self._extended_fingers(hand) == 2
        )

    def _points_at_self(self, view: FeatureView, side: str) -> bool:
        hand, nose = view.hand(side), view.point("nose")
        if hand is None or nose is None or not self._is_pointing(hand):
            return False
        face = np.array(nose)
        tip_to_face = _distance(hand[INDEX_TIP], face)
        dx, dy = hand[INDEX_TIP] - hand[INDEX_BASE]
        return bool(
            abs(dx) > self._rules.pointing_min_horizontal_ratio * abs(dy)
            and tip_to_face < _distance(hand[HAND_WRIST], face)
            # Closer than this, the finger is touching the mouth, not pointing at the face.
            and self._rules.self_point_min_radius <= tip_to_face < self._rules.self_point_max_radius
        )

    def _finger_up(self, view: FeatureView, side: str) -> bool:
        hand, nose = view.hand(side), view.point("nose")
        if hand is None or nose is None or not self._is_pointing(hand):
            return False
        dx, dy = hand[INDEX_TIP] - hand[INDEX_BASE]
        tip = hand[INDEX_TIP]
        return bool(-dy > abs(dx) and tip[1] < 0 and _distance(tip, np.array(nose)) >= self._rules.mouth_radius)

    def _is_on_chest(self, view: FeatureView, side: str) -> bool:
        hand = view.hand(side)
        position = hand[PALM_CENTER] if hand is not None else view.point(f"{side}_wrist")
        if position is None:
            return False
        x, y = position
        return bool(abs(x) < self._rules.chest_half_width and 0 < y < self._rules.chest_max_depth)
