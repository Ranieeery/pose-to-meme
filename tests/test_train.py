import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from pose_meme.config import RulesConfig, TrainingConfig
from pose_meme.dataset import DatasetError, Samples
from pose_meme.features import FEATURE_SIZE
from scripts.train import (
    build_knn,
    build_mlp,
    evaluate_by_session,
    session_folds,
    train_and_report,
    sessions_missing_classes,
    validate_samples,
)

TRAINING = TrainingConfig(
    hidden_layers=[16],
    alpha=0.001,
    max_iter=300,
    knn_neighbors=1,
    random_seed=0,
    min_sessions=2,
    max_class_f1_drop=0.05,
)
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
    hands_together_max_gap=0.6,
    head_min_height=0.35,
    head_half_width=0.8,
    tie_max_crossing=-0.1,
)
CLASSES = ["neutro", "pose_a", "pose_b"]

# The tiny training runs here are about the flow, not about reaching convergence.
pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


def make_samples(sessions: tuple[str, ...] = ("s1", "s2", "s3"), per_class: int = 10) -> Samples:
    """Three well separated clusters, so a working pipeline must score high."""
    rng = np.random.default_rng(0)
    centers = {"neutro": -5.0, "pose_a": 0.0, "pose_b": 5.0}
    features, labels, session_ids = [], [], []
    for session in sessions:
        for label, center in centers.items():
            for _ in range(per_class):
                features.append(center + rng.normal(0.0, 0.1, FEATURE_SIZE))
                labels.append(label)
                session_ids.append(session)
    return Samples(np.array(features), np.array(labels), np.array(session_ids))


# Session split: the rule that keeps the metrics honest


def test_no_session_is_in_train_and_test_at_the_same_time() -> None:
    samples = make_samples()

    folds = list(session_folds(samples.sessions))

    assert folds
    for train, test in folds:
        assert not set(samples.sessions[train]) & set(samples.sessions[test])


def test_every_session_is_tested_exactly_once() -> None:
    samples = make_samples()

    tested = [session for _, test in session_folds(samples.sessions) for session in set(samples.sessions[test])]

    assert sorted(tested) == ["s1", "s2", "s3"]


def test_each_fold_tests_a_single_whole_session() -> None:
    samples = make_samples()

    for _, test in session_folds(samples.sessions):
        tested = samples.sessions[test]
        assert len(set(tested)) == 1
        assert len(test) == np.count_nonzero(samples.sessions == tested[0])


def test_evaluation_cannot_memorize_frames_of_the_tested_session() -> None:
    # The two sessions have opposite labels for the same features. With 1-NN, any frame
    # of the tested session leaking into training would give a perfect score here.
    rng = np.random.default_rng(0)
    features = np.vstack([rng.normal(-5, 0.1, (20, FEATURE_SIZE)), rng.normal(5, 0.1, (20, FEATURE_SIZE))])
    features = np.vstack([features, features + 0.01])
    labels = np.array(["a"] * 20 + ["b"] * 20 + ["b"] * 20 + ["a"] * 20)
    sessions = np.array(["s1"] * 40 + ["s2"] * 40)

    evaluation = evaluate_by_session(build_knn(TRAINING), Samples(features, labels, sessions))

    assert evaluation.accuracy == 0.0


def test_evaluation_scores_separable_data_on_unseen_sessions() -> None:
    evaluation = evaluate_by_session(build_mlp(TRAINING), make_samples())

    assert evaluation.accuracy == 1.0
    assert set(evaluation.per_session) == {"s1", "s2", "s3"}
    assert evaluation.predictions.shape == (90,)
    assert evaluation.confidences.shape == (90,)


# Validation


def test_single_session_is_rejected() -> None:
    with pytest.raises(DatasetError, match="2 sessions"):
        validate_samples(make_samples(sessions=("s1",)), CLASSES, min_sessions=2)


def test_label_outside_the_config_is_rejected() -> None:
    with pytest.raises(DatasetError, match="pose_a"):
        validate_samples(make_samples(), ["neutro", "pose_b"], min_sessions=2)


def test_missing_neutral_class_is_rejected() -> None:
    samples = make_samples()
    keep = samples.labels != "neutro"
    subset = Samples(samples.features[keep], samples.labels[keep], samples.sessions[keep])

    with pytest.raises(DatasetError, match="neutro"):
        validate_samples(subset, CLASSES, min_sessions=2, neutral_class="neutro")


def test_classes_without_samples_are_reported_not_fatal() -> None:
    assert validate_samples(make_samples(), [*CLASSES, "pose_c"], min_sessions=2) == ["pose_c"]


def session_without(samples: Samples, session: str, label: str) -> Samples:
    keep = ~((samples.sessions == session) & (samples.labels == label))
    return Samples(samples.features[keep], samples.labels[keep], samples.sessions[keep])


def test_complete_sessions_miss_nothing() -> None:
    assert sessions_missing_classes(make_samples()) == {}


def test_session_without_a_class_is_named_with_what_it_misses() -> None:
    # A session with only the neutral class gave an accuracy of 12.6% with no explanation.
    samples = session_without(session_without(make_samples(), "s2", "pose_a"), "s2", "pose_b")

    assert sessions_missing_classes(samples) == {"s2": ["pose_a", "pose_b"]}


def test_class_no_session_has_is_not_a_session_problem() -> None:
    # Classes absent from all the data are what validate_samples already reports.
    samples = make_samples()
    keep = samples.labels != "pose_b"

    assert sessions_missing_classes(Samples(samples.features[keep], samples.labels[keep], samples.sessions[keep])) == {}


# Pipelines


def test_both_pipelines_scale_the_features() -> None:
    assert "standardscaler" in build_mlp(TRAINING).named_steps
    assert "standardscaler" in build_knn(TRAINING).named_steps


def test_mlp_is_reproducible_with_the_same_seed() -> None:
    samples = make_samples()

    first = build_mlp(TRAINING).fit(samples.features, samples.labels)
    second = build_mlp(TRAINING).fit(samples.features, samples.labels)

    np.testing.assert_array_equal(first.predict_proba(samples.features), second.predict_proba(samples.features))


# End to end


@pytest.fixture
def trained(tmp_path: Path) -> tuple[Path, Path, dict]:
    model_path = tmp_path / "models" / "clf.joblib"
    reports = tmp_path / "reports"
    metrics = train_and_report(
        make_samples(),
        TRAINING,
        RULES,
        neutral_class="neutro",
        confidence_threshold=0.8,
        model_path=model_path,
        reports_dir=reports,
    )
    return model_path, reports, metrics


def test_final_model_is_saved_with_its_feature_size(trained) -> None:
    model_path, _, _ = trained

    bundle = joblib.load(model_path)

    assert bundle["feature_size"] == FEATURE_SIZE
    assert set(bundle["pipeline"].classes_) == set(CLASSES)


def test_reports_are_written(trained) -> None:
    _, reports, _ = trained

    text = (reports / "report.md").read_text(encoding="utf-8")

    assert "MLP" in text and "KNN" in text and "Regras" in text
    assert "precision" in text
    assert "Matriz de confus" in text
    assert json.loads((reports / "metrics.json").read_text(encoding="utf-8"))["sessions"] == ["s1", "s2", "s3"]


def test_report_warns_about_incomplete_sessions(tmp_path: Path) -> None:
    samples = session_without(make_samples(), "s2", "pose_a")

    metrics = train_and_report(
        samples,
        TRAINING,
        RULES,
        neutral_class="neutro",
        confidence_threshold=0.8,
        model_path=tmp_path / "clf.joblib",
        reports_dir=tmp_path,
    )

    assert metrics["sessions_missing_classes"] == {"s2": ["pose_a"]}
    assert "s2: pose_a" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_report_has_no_warning_with_complete_sessions(trained) -> None:
    _, reports, metrics = trained

    assert metrics["sessions_missing_classes"] == {}
    assert "incompletas" not in (reports / "report.md").read_text(encoding="utf-8")


def test_metrics_compare_the_three_classifiers(trained) -> None:
    _, _, metrics = trained

    assert set(metrics["classifiers"]) == {"mlp", "knn", "rules"}
    for name in ("mlp", "knn"):
        assert metrics["classifiers"][name]["accuracy"] == 1.0


@pytest.mark.parametrize(
    "field",
    ["accuracy", "macro_f1", "accuracy_after_threshold", "false_meme_rate", "worst_session_accuracy", "latency_ms"],
)
def test_metrics_cover_what_the_user_would_see(trained, field: str) -> None:
    _, _, metrics = trained

    assert field in metrics["classifiers"]["mlp"]


def test_metrics_have_the_f1_of_each_class(trained) -> None:
    _, _, metrics = trained

    per_class = metrics["classifiers"]["mlp"]["per_class_f1"]
    assert set(per_class) == set(metrics["samples_per_class"])
    assert all(0.0 <= value <= 1.0 for value in per_class.values())
