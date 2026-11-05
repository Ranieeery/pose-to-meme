"""Evaluates and trains the classifier. Run from the project root: python -m scripts.train

Every metric comes from sessions the evaluated model never saw (leave-one-session-out).
Consecutive frames are nearly identical, so a random split would leak test frames into
training and inflate the numbers. The saved model is then trained on all sessions.
"""

import json
import sys
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from pose_meme.classifier import RULE_CONFIDENCE, MLClassifier, PoseClassifier, RuleClassifier, accept_pose
from pose_meme.config import ConfigError, RulesConfig, TrainingConfig, load_config
from pose_meme.dataset import DatasetError, Samples, read_samples
from pose_meme.features import FEATURE_SIZE

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
LATENCY_SAMPLES = 200
MS_PER_SECOND = 1000
DISPLAY_NAMES = {"mlp": "MLP", "knn": "KNN", "rules": "Regras"}


@dataclass(frozen=True)
class SessionEvaluation:
    predictions: np.ndarray
    confidences: np.ndarray
    accuracy: float
    per_session: dict[str, float]


def build_mlp(config: TrainingConfig) -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=tuple(config.hidden_layers),
            alpha=config.alpha,
            max_iter=config.max_iter,
            random_state=config.random_seed,
        ),
    )


def build_knn(config: TrainingConfig) -> Pipeline:
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=config.knn_neighbors))


def session_folds(sessions: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """(train, test) indexes: each session is tested once, by a model that never saw it."""
    return LeaveOneGroupOut().split(sessions, groups=sessions)


def validate_samples(
    samples: Samples,
    classes: list[str],
    min_sessions: int,
    neutral_class: str | None = None,
) -> list[str]:
    """Reject samples that can't give an honest result; returns the classes without samples."""
    sessions = set(samples.sessions.tolist())
    if len(sessions) < min_sessions:
        raise DatasetError(
            f"found {len(sessions)} session(s), need at least {min_sessions} sessions: "
            "a model can only be evaluated on a session it did not train on"
        )
    labels = set(samples.labels.tolist())
    if unknown := labels - set(classes):
        raise DatasetError(f"labels not in config.yaml: {sorted(unknown)}")
    if neutral_class is not None and neutral_class not in labels:
        raise DatasetError(f"no samples of the neutral class '{neutral_class}'")
    return sorted(set(classes) - labels)


def _summarize(predictions: np.ndarray, confidences: np.ndarray, samples: Samples) -> SessionEvaluation:
    correct = predictions == samples.labels
    per_session = {
        str(session): float(correct[samples.sessions == session].mean()) for session in np.unique(samples.sessions)
    }
    return SessionEvaluation(predictions, confidences, float(correct.mean()), per_session)


def evaluate_by_session(pipeline: Pipeline, samples: Samples) -> SessionEvaluation:
    predictions = np.empty(len(samples.labels), dtype=samples.labels.dtype)
    confidences = np.empty(len(samples.labels))
    for train, test in session_folds(samples.sessions):
        model = clone(pipeline).fit(samples.features[train], samples.labels[train])
        probabilities = model.predict_proba(samples.features[test])
        predictions[test] = model.classes_[probabilities.argmax(axis=1)]
        confidences[test] = probabilities.max(axis=1)
    return _summarize(predictions, confidences, samples)


def _evaluate_rules(classifier: RuleClassifier, samples: Samples) -> SessionEvaluation:
    """Rules aren't trained, so all sessions are unseen by them already."""
    predictions = np.array([classifier.predict(features)[0] for features in samples.features])
    return _summarize(predictions, np.full(len(predictions), RULE_CONFIDENCE), samples)


def _latency_ms(classifier: PoseClassifier, samples: Samples) -> float:
    """Median time of one prediction, the way the frame loop calls it."""
    times = []
    for features in samples.features[:LATENCY_SAMPLES]:
        start = time.perf_counter()
        classifier.predict(features)
        times.append((time.perf_counter() - start) * MS_PER_SECOND)
    return float(np.median(times))


def _metrics(
    evaluation: SessionEvaluation,
    samples: Samples,
    neutral_class: str,
    threshold: float,
    latency_ms: float,
) -> dict[str, Any]:
    """Beyond accuracy: what the user sees after the threshold, and memes shown for no pose."""
    accepted = np.array(
        [
            accept_pose(label, confidence, threshold, neutral_class)
            for label, confidence in zip(evaluation.predictions, evaluation.confidences, strict=True)
        ]
    )
    is_neutral = samples.labels == neutral_class
    return {
        "accuracy": evaluation.accuracy,
        "macro_f1": float(f1_score(samples.labels, evaluation.predictions, average="macro", zero_division=0)),
        "accuracy_after_threshold": float((accepted == samples.labels).mean()),
        "false_meme_rate": float((accepted[is_neutral] != neutral_class).mean()) if is_neutral.any() else None,
        "worst_session_accuracy": min(evaluation.per_session.values()),
        "per_session": evaluation.per_session,
        "latency_ms": latency_ms,
    }


def _confusion_table(evaluation: SessionEvaluation, samples: Samples, labels: list[str]) -> str:
    matrix = confusion_matrix(samples.labels, evaluation.predictions, labels=labels)
    rows = [
        "| real \\ previsto | " + " | ".join(labels) + " |",
        "|---" * (len(labels) + 1) + "|",
    ]
    rows += [
        f"| {label} | " + " | ".join(str(value) for value in row) + " |"
        for label, row in zip(labels, matrix, strict=True)
    ]
    return "\n".join(rows)


def _report_text(metrics: dict[str, Any], evaluations: dict[str, SessionEvaluation], samples: Samples) -> str:
    labels = sorted(set(samples.labels.tolist()) | {str(p) for e in evaluations.values() for p in e.predictions})
    lines = [
        "# Relatório de treino",
        "",
        f"Sessões ({len(metrics['sessions'])}): {', '.join(metrics['sessions'])}. Amostras: {metrics['samples']}.",
        "",
        "Todas as métricas vêm de sessões que o modelo avaliado não viu (leave-one-session-out). "
        "As regras não são treinadas, então todas as sessões são inéditas para elas.",
        "",
        "## Comparação",
        "",
        "| Classificador | Acurácia | F1 macro | Pior sessão | Acurácia após threshold "
        "| Meme indevido em `neutro` | Latência (ms) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, values in metrics["classifiers"].items():
        false_memes = values["false_meme_rate"]
        lines.append(
            f"| {DISPLAY_NAMES[name]} | {values['accuracy']:.3f} | {values['macro_f1']:.3f} "
            f"| {values['worst_session_accuracy']:.3f} | {values['accuracy_after_threshold']:.3f} "
            f"| {'-' if false_memes is None else f'{false_memes:.3f}'} | {values['latency_ms']:.3f} |"
        )
    lines += ["", "## Amostras por classe", "", "| Classe | Amostras |", "|---|---|"]
    lines += [f"| {label} | {count} |" for label, count in sorted(metrics["samples_per_class"].items())]
    for name, evaluation in evaluations.items():
        report = classification_report(
            samples.labels, evaluation.predictions, labels=labels, digits=3, zero_division=0
        )
        lines += [
            "",
            f"## {DISPLAY_NAMES[name]}",
            "",
            "```",
            report.rstrip(),
            "```",
            "",
            "### Matriz de confusão",
            "",
            _confusion_table(evaluation, samples, labels),
        ]
    return "\n".join(lines) + "\n"


def train_and_report(
    samples: Samples,
    training: TrainingConfig,
    rules: RulesConfig,
    neutral_class: str,
    confidence_threshold: float,
    model_path: Path,
    reports_dir: Path,
) -> dict[str, Any]:
    """Evaluate the three classifiers, save the final MLP and write the reports."""
    pipelines = {"mlp": build_mlp(training), "knn": build_knn(training)}
    rule_classifier = RuleClassifier(rules, neutral_class)

    evaluations = {name: evaluate_by_session(pipeline, samples) for name, pipeline in pipelines.items()}
    evaluations["rules"] = _evaluate_rules(rule_classifier, samples)

    final = {name: clone(pipeline).fit(samples.features, samples.labels) for name, pipeline in pipelines.items()}
    latencies = {name: _latency_ms(MLClassifier(model), samples) for name, model in final.items()}
    latencies["rules"] = _latency_ms(rule_classifier, samples)

    metrics = {
        "sessions": sorted(set(samples.sessions.tolist())),
        "samples": len(samples.labels),
        "samples_per_class": dict(Counter(samples.labels.tolist())),
        "classifiers": {
            name: _metrics(evaluation, samples, neutral_class, confidence_threshold, latencies[name])
            for name, evaluation in evaluations.items()
        },
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final["mlp"], "feature_size": FEATURE_SIZE}, model_path)

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (reports_dir / "report.md").write_text(_report_text(metrics, evaluations, samples), encoding="utf-8")
    return metrics


def main() -> None:
    try:
        config = load_config(CONFIG_PATH)
        samples = read_samples(config.data.samples_dir)
        missing = validate_samples(
            samples,
            [config.poses.neutral_class, *config.poses.prompts],
            config.training.min_sessions,
            config.poses.neutral_class,
        )
    except (ConfigError, DatasetError) as error:
        sys.exit(f"error: {error}")

    if missing:
        print(f"warning: classes without samples, which the model won't learn: {missing}")
    metrics = train_and_report(
        samples,
        config.training,
        config.rules,
        config.poses.neutral_class,
        config.classifier.confidence_threshold,
        config.classifier.model_path,
        config.data.reports_dir,
    )
    for name, values in metrics["classifiers"].items():
        print(f"{DISPLAY_NAMES[name]:7} accuracy {values['accuracy']:.3f}  worst session {values['worst_session_accuracy']:.3f}")
    print(f"model: {config.classifier.model_path}\nreport: {config.data.reports_dir / 'report.md'}")


if __name__ == "__main__":
    main()
