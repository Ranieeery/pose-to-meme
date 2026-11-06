"""Compares the MLP's F1 per class between two training reports.

Run from the project root: python -m scripts.compare_metrics OLD.json [NEW.json]
(NEW defaults to the current reports/metrics.json). Exits with 1 when an existing
class dropped by more than training.max_class_f1_drop, so a new pose can't quietly
make the old ones worse.
"""

import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pose_meme.config import ConfigError, load_config

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
METRICS_FILE = "metrics.json"
MODEL = "mlp"  # the saved model; the baselines don't matter here


class MetricsError(ValueError):
    """A metrics file is missing or lacks the per-class F1."""


@dataclass(frozen=True)
class ClassChange:
    label: str
    old: float | None  # None: the class is new
    new: float | None  # None: the class was removed
    rejected: bool

    @property
    def delta(self) -> float | None:
        return None if self.old is None or self.new is None else self.new - self.old


def read_class_f1(path: Path) -> dict[str, float]:
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise MetricsError(f"metrics file not found: {path}") from None
    try:
        return dict(metrics["classifiers"][MODEL]["per_class_f1"])
    except KeyError:
        raise MetricsError(f"{path} has no per_class_f1; run python -m scripts.train to regenerate it") from None


def compare_class_f1(old: Mapping[str, float], new: Mapping[str, float], max_drop: float) -> list[ClassChange]:
    """One row per class in either report; only a class in both can be rejected."""
    changes = []
    for label in sorted(old.keys() | new.keys()):
        before, after = old.get(label), new.get(label)
        dropped = before is not None and after is not None and before - after > max_drop
        changes.append(ClassChange(label, before, after, dropped))
    return changes


def format_comparison(changes: Sequence[ClassChange], max_drop: float) -> list[str]:
    def value(number: float | None) -> str:
        return "-" if number is None else f"{number:.3f}"

    lines = ["| Classe | F1 antes | F1 depois | Diferença | |", "|---|---|---|---|---|"]
    for change in changes:
        if change.old is None:
            note = "nova"
        elif change.new is None:
            note = "removida"
        else:
            note = f"piorou mais que {max_drop}" if change.rejected else ""
        delta = "-" if change.delta is None else f"{change.delta:+.3f}"
        lines.append(f"| {change.label} | {value(change.old)} | {value(change.new)} | {delta} | {note} |")
    rejected = [change.label for change in changes if change.rejected]
    verdict = f"REJEITADO: {', '.join(rejected)}" if rejected else "APROVADO"
    lines.append(f"\n{verdict} (tolerância: queda de F1 de até {max_drop})")
    return lines


def main() -> None:
    try:
        config = load_config(CONFIG_PATH)
        old_path = Path(sys.argv[1])
        new_path = Path(sys.argv[2]) if len(sys.argv) > 2 else config.data.reports_dir / METRICS_FILE
        changes = compare_class_f1(
            read_class_f1(old_path), read_class_f1(new_path), config.training.max_class_f1_drop
        )
    except IndexError:
        sys.exit("usage: python -m scripts.compare_metrics OLD.json [NEW.json]")
    except (ConfigError, MetricsError) as error:
        sys.exit(f"error: {error}")

    print("\n".join(format_comparison(changes, config.training.max_class_f1_drop)))
    sys.exit(1 if any(change.rejected for change in changes) else 0)


if __name__ == "__main__":
    main()
