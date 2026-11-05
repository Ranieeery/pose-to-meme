from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pose_meme.classifier import (
    POSE_CLASSES,
    MLClassifier,
    ModelLoadError,
    RuleClassifier,
    accept_pose,
    classify_frame,
    load_classifier,
)
from pose_meme.config import RulesConfig, load_config
from pose_meme.features import FEATURE_SIZE, extract_features
from tests.conftest import POINTING_HAND, ROCK_HAND, build_hand

REPO_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"
NEUTRAL = "neutro"
RULES = RulesConfig(
    mouth_radius=0.4,
    mouth_radius_without_fingers=0.55,
    self_point_min_radius=0.25,
    self_point_max_radius=0.6,
    pointing_min_horizontal_ratio=1.5,
    finger_extended_ratio=1.1,
    open_hand_min_fingers=4,
    open_hand_min_spread=2.0,
    forward_hand_min_palm=0.35,
    forward_hands_min_gap=0.8,
    chest_half_width=0.7,
    chest_max_depth=1.2,
)
UP = (0.0, -1.0)
RIGHT = (1.0, 0.0)
LEFT = (-1.0, 0.0)


def classify(pose, hands=()) -> tuple[str, float]:
    features = extract_features(pose, list(hands), aspect_ratio=1.0, min_visibility=0.5)
    return RuleClassifier(RULES, NEUTRAL).predict(features)


def mirrored(landmarks):
    """The same pose with the arms swapped, to check that either hand works."""
    swapped = list(landmarks)
    for left, right in ((11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22)):
        swapped[left], swapped[right] = landmarks[right], landmarks[left]
    return [replace(lm, x=1.0 - lm.x) for lm in swapped]


def test_rule_classes_match_the_classes_in_the_config() -> None:
    assert set(POSE_CLASSES) == set(load_config(REPO_CONFIG).poses.prompts)


def test_relaxed_arms_are_neutral(neutral_pose) -> None:
    assert classify(neutral_pose)[0] == NEUTRAL


def test_relaxed_arms_with_detected_open_hands_are_neutral(neutral_pose) -> None:
    hands = [build_hand((0.62, 0.70), (0.0, 1.0)), build_hand((0.38, 0.70), (0.0, 1.0))]

    assert classify(neutral_pose, hands)[0] == NEUTRAL


def test_one_open_hand_raised_beside_the_head_is_neutral(make_pose) -> None:
    pose = make_pose({16: (0.25, 0.30), 14: (0.25, 0.45)})

    assert classify(pose, [build_hand((0.25, 0.30), UP)])[0] == NEUTRAL


# Hand on the mouth: closed (bebe-pensativo), one finger (macaco-dedo-boca), open (maria-joaquina-unha)


def test_hand_without_finger_tracking_on_the_mouth_counts_as_closed(hand_on_mouth_pose) -> None:
    assert classify(hand_on_mouth_pose)[0] == "mao_fechada_na_boca"


def test_hand_on_the_mouth_works_with_the_other_arm(hand_on_mouth_pose) -> None:
    assert classify(mirrored(hand_on_mouth_pose))[0] == "mao_fechada_na_boca"


def test_fist_on_the_mouth_is_closed(make_pose) -> None:
    pose = make_pose({16: (0.47, 0.34), 14: (0.40, 0.55)})
    fist = build_hand((0.47, 0.34), UP, size=0.13, extended=())

    assert classify(pose, [fist])[0] == "mao_fechada_na_boca"


def test_one_finger_up_on_the_lips_is_finger_on_mouth(make_pose) -> None:
    # macaco-dedo-boca: the finger points up, but it is touching the mouth.
    pose = make_pose({16: (0.48, 0.40), 14: (0.40, 0.55)})
    hand = build_hand((0.48, 0.40), UP, size=0.12, extended=POINTING_HAND)

    assert classify(pose, [hand])[0] == "dedo_na_boca"


def test_finger_on_the_mouth_works_with_the_other_arm(make_pose) -> None:
    pose = make_pose({16: (0.48, 0.40), 14: (0.40, 0.55)})
    hand = build_hand((0.52, 0.40), UP, size=0.12, extended=POINTING_HAND)

    assert classify(mirrored(pose), [hand])[0] == "dedo_na_boca"


def test_open_hand_on_the_mouth_is_open(make_pose) -> None:
    # Fingertips reach the mouth with the wrist far below it, as in maria-joaquina-unha.
    pose = make_pose({16: (0.47, 0.42), 14: (0.40, 0.55)})
    hand = build_hand((0.47, 0.42), UP, size=0.13)

    assert classify(pose, [hand])[0] == "mao_aberta_na_boca"


def test_detected_fingers_do_not_get_the_wider_radius(make_pose) -> None:
    pose = make_pose({16: (0.40, 0.45)})
    hand = build_hand((0.40, 0.45), UP, size=0.10)  # fingertip 0.5 widths from the nose

    assert classify(pose, [hand])[0] not in ("mao_aberta_na_boca", "mao_fechada_na_boca")


# Both hands on the mouth (bjork-pose)


def test_both_hands_on_the_mouth(hands_on_mouth_pose) -> None:
    assert classify(hands_on_mouth_pose)[0] == "maos_na_boca"


def test_undetected_hands_reach_the_mouth_with_a_wider_radius(make_pose) -> None:
    # bjork-pose: hands over the face hide from the hand model, and the pose model's
    # "index" point is a knuckle, so it stops short of the mouth (0.47 widths there).
    pose = make_pose({20: (0.42, 0.30), 19: (0.58, 0.30), 16: (0.46, 0.45), 15: (0.54, 0.45)})

    assert classify(pose)[0] == "maos_na_boca"


# Raised finger (lily-nerd)


def test_raised_index_finger(finger_up_pose) -> None:
    hand = build_hand((0.32, 0.30), UP, extended=POINTING_HAND)

    assert classify(finger_up_pose, [hand])[0] == "dedo_levantado"


def test_raised_index_finger_works_with_the_other_arm(finger_up_pose) -> None:
    hand = build_hand((0.68, 0.30), UP, extended=POINTING_HAND)

    assert classify(mirrored(finger_up_pose), [hand])[0] == "dedo_levantado"


def test_raised_finger_needs_finger_tracking(finger_up_pose) -> None:
    assert classify(finger_up_pose)[0] == NEUTRAL


# Pointing at oneself (ryujin-eu)


def test_pointing_at_self(pointing_at_self_pose) -> None:
    hand = build_hand((0.30, 0.28), RIGHT, extended=POINTING_HAND)

    assert classify(pointing_at_self_pose, [hand])[0] == "apontando_para_si"


def test_pointing_at_self_works_with_the_other_arm(pointing_at_self_pose) -> None:
    hand = build_hand((0.70, 0.28), LEFT, extended=POINTING_HAND)

    assert classify(mirrored(pointing_at_self_pose), [hand])[0] == "apontando_para_si"


def test_pointing_away_from_the_face_is_not_pointing_at_self(pointing_at_self_pose) -> None:
    hand = build_hand((0.30, 0.28), LEFT, extended=POINTING_HAND)

    assert classify(pointing_at_self_pose, [hand])[0] != "apontando_para_si"


def test_finger_touching_the_lips_sideways_is_not_pointing_at_self(make_pose) -> None:
    # The confusion seen on camera: macaco read as ryujin.
    pose = make_pose({16: (0.38, 0.27), 14: (0.30, 0.45)})
    hand = build_hand((0.38, 0.27), RIGHT, extended=POINTING_HAND)  # tip about 0.13 from the nose

    assert classify(pose, [hand])[0] == "dedo_na_boca"


def test_finger_going_up_at_an_angle_is_not_pointing_at_self(make_pose) -> None:
    diagonal = (0.6, -0.8)
    pose = make_pose({16: (0.36, 0.40), 14: (0.30, 0.55)})
    hand = build_hand((0.36, 0.40), diagonal, extended=POINTING_HAND)

    assert classify(pose, [hand])[0] != "apontando_para_si"


# Rock sign (licurgo-rock)


def test_rock_sign(make_pose) -> None:
    pose = make_pose({16: (0.48, 0.48), 14: (0.36, 0.55)})
    hand = build_hand((0.48, 0.48), UP, extended=ROCK_HAND)

    assert classify(pose, [hand])[0] == "sinal_de_rock"


def test_rock_sign_near_the_mouth_is_still_a_rock_sign(make_pose) -> None:
    pose = make_pose({16: (0.47, 0.36), 14: (0.36, 0.55)})
    hand = build_hand((0.47, 0.36), UP, extended=ROCK_HAND)

    assert classify(pose, [hand])[0] == "sinal_de_rock"


def test_rock_shape_on_a_hanging_arm_does_not_count(neutral_pose) -> None:
    hand = build_hand((0.38, 0.70), (0.0, 1.0), extended=ROCK_HAND)

    assert classify(neutral_pose, [hand])[0] == NEUTRAL


# Both hands toward the camera: spread open (numetal-pose) vs. claws (ozzy-osbourne)


def forward_hands(**shape):
    return [build_hand((0.80, 0.42), UP, size=0.15, **shape), build_hand((0.20, 0.42), UP, size=0.15, **shape)]


def test_spread_open_hands_toward_the_camera(hands_forward_pose) -> None:
    assert classify(hands_forward_pose, forward_hands(spread=3.0))[0] == "maos_abertas_para_frente"


def test_curled_fingers_toward_the_camera_are_claws(hands_forward_pose) -> None:
    assert classify(hands_forward_pose, forward_hands(extended=()))[0] == "garras_para_frente"


def test_straight_fingers_kept_together_are_claws_not_spread_hands(hands_forward_pose) -> None:
    assert classify(hands_forward_pose, forward_hands(spread=1.0))[0] == "garras_para_frente"


def test_small_hands_are_not_toward_the_camera(hands_forward_pose) -> None:
    hands = [
        build_hand((0.80, 0.42), UP, size=0.05, spread=3.0),
        build_hand((0.20, 0.42), UP, size=0.05, spread=3.0),
    ]

    assert classify(hands_forward_pose, hands)[0] not in ("maos_abertas_para_frente", "garras_para_frente")


def test_big_hands_together_on_the_mouth_are_not_toward_the_camera(hands_on_mouth_pose) -> None:
    # bjork-pose up close: big palms too, but pressed together instead of apart.
    hands = [build_hand((0.54, 0.42), UP, size=0.15), build_hand((0.46, 0.42), UP, size=0.15)]

    assert classify(hands_on_mouth_pose, hands)[0] == "maos_na_boca"


def test_hands_toward_the_camera_need_both_hands(hands_forward_pose) -> None:
    hand = build_hand((0.20, 0.42), UP, size=0.15, spread=3.0)

    assert classify(hands_forward_pose, [hand])[0] not in ("maos_abertas_para_frente", "garras_para_frente")


# Hand on the chest (valverde-pose)


def test_hand_on_the_chest(hand_on_chest_pose) -> None:
    assert classify(hand_on_chest_pose)[0] == "mao_no_peito"


def test_hand_on_the_chest_works_with_the_other_arm(hand_on_chest_pose) -> None:
    assert classify(mirrored(hand_on_chest_pose))[0] == "mao_no_peito"


def test_detected_hand_on_the_chest_counts_by_the_palm_not_the_wrist(make_pose) -> None:
    # valverde-pose: the forearm comes from the side, so the wrist is out of the chest
    # while the palm rests on it.
    pose = make_pose({16: (0.28, 0.55), 14: (0.15, 0.55)})
    hand = build_hand((0.28, 0.55), RIGHT, size=0.2)

    assert classify(pose, [hand])[0] == "mao_no_peito"


def test_hidden_arm_is_ignored_instead_of_read_as_a_position(make_pose) -> None:
    # Zeroed coordinates mean "chest center"; a hidden wrist must not match the chest.
    hidden = {15: 0.1, 16: 0.1, 17: 0.1, 18: 0.1, 19: 0.1, 20: 0.1, 21: 0.1, 22: 0.1}

    assert classify(make_pose(visibility=hidden))[0] == NEUTRAL


# Interface


def test_prediction_is_a_class_and_a_confidence(neutral_pose) -> None:
    label, confidence = classify(neutral_pose)

    assert isinstance(label, str)
    assert 0.0 <= confidence <= 1.0


# MLClassifier: same interface, tiny model trained here


def tiny_pipeline():
    rng = np.random.default_rng(0)
    features = np.vstack(
        [rng.normal(center, 0.1, (20, FEATURE_SIZE)) for center in (-5.0, 5.0)],
    )
    labels = ["neutro"] * 20 + ["mao_no_peito"] * 20
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3)).fit(features, labels)


def test_ml_classifier_returns_the_class_and_its_probability() -> None:
    label, confidence = MLClassifier(tiny_pipeline()).predict(np.full(FEATURE_SIZE, 5.0))

    assert label == "mao_no_peito"
    assert confidence == pytest.approx(1.0)


def test_ml_classifier_confidence_varies_with_the_input() -> None:
    # Unlike the rules, which is what makes the confidence threshold mean something.
    rng = np.random.default_rng(0)
    features = rng.normal(0.0, 1.0, (40, FEATURE_SIZE))
    labels = ["neutro", "mao_no_peito"] * 20
    pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)).fit(features, labels)

    confidences = {MLClassifier(pipeline).predict(rng.normal(0.0, 1.0, FEATURE_SIZE))[1] for _ in range(30)}

    assert len(confidences) > 1


def test_ml_classifier_goes_through_the_same_threshold() -> None:
    classifier = MLClassifier(tiny_pipeline())

    label, _ = classify_frame(classifier, np.full(FEATURE_SIZE, 5.0), threshold=1.1, neutral_class=NEUTRAL)

    assert label == NEUTRAL


def save_model(path: Path, feature_size: int = FEATURE_SIZE) -> None:
    joblib.dump({"pipeline": tiny_pipeline(), "feature_size": feature_size}, path)


def test_trained_model_is_used_when_it_exists(tmp_path: Path) -> None:
    path = tmp_path / "clf.joblib"
    save_model(path)

    assert isinstance(load_classifier(path, RULES, NEUTRAL), MLClassifier)


def test_rules_are_the_fallback_without_a_trained_model(tmp_path: Path) -> None:
    assert isinstance(load_classifier(tmp_path / "nope.joblib", RULES, NEUTRAL), RuleClassifier)


def test_model_from_another_feature_layout_is_an_error_not_a_silent_fallback(tmp_path: Path) -> None:
    # Falling back here would hide that the model the user trained is not running.
    path = tmp_path / "clf.joblib"
    save_model(path, feature_size=3)

    with pytest.raises(ModelLoadError, match="retrain|train"):
        load_classifier(path, RULES, NEUTRAL)


def test_unreadable_model_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "clf.joblib"
    path.write_bytes(b"not a model")

    with pytest.raises(ModelLoadError):
        load_classifier(path, RULES, NEUTRAL)


def test_frame_without_features_is_neutral() -> None:
    classifier = RuleClassifier(RULES, NEUTRAL)

    assert classify_frame(classifier, None, threshold=0.8, neutral_class=NEUTRAL)[0] == NEUTRAL


def test_frame_below_the_threshold_is_neutral_but_keeps_its_confidence(hand_on_chest_pose) -> None:
    class Unsure:
        def predict(self, features):
            return "mao_no_peito", 0.5

    features = extract_features(hand_on_chest_pose, [], aspect_ratio=1.0, min_visibility=0.5)

    assert classify_frame(Unsure(), features, threshold=0.8, neutral_class=NEUTRAL) == (NEUTRAL, 0.5)


@pytest.mark.parametrize(
    ("label", "confidence", "expected"),
    [
        ("mao_no_peito", 0.9, "mao_no_peito"),
        ("mao_no_peito", 0.8, "mao_no_peito"),
        ("mao_no_peito", 0.79, NEUTRAL),
        (NEUTRAL, 0.99, NEUTRAL),
    ],
)
def test_pose_is_accepted_only_above_the_threshold(label: str, confidence: float, expected: str) -> None:
    assert accept_pose(label, confidence, threshold=0.8, neutral_class=NEUTRAL) == expected
