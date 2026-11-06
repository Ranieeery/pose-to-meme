import json
from pathlib import Path

import pytest

from scripts.compare_metrics import MetricsError, compare_class_f1, format_comparison, read_class_f1


def test_class_within_the_tolerance_passes() -> None:
    changes = compare_class_f1({"a": 0.90}, {"a": 0.86}, max_drop=0.05)

    assert [(change.label, change.rejected) for change in changes] == [("a", False)]
    assert changes[0].delta == pytest.approx(-0.04)


def test_class_that_drops_beyond_the_tolerance_is_rejected() -> None:
    (change,) = compare_class_f1({"a": 0.90}, {"a": 0.80}, max_drop=0.05)

    assert change.rejected


def test_drop_exactly_at_the_tolerance_passes() -> None:
    (change,) = compare_class_f1({"a": 0.75}, {"a": 0.5}, max_drop=0.25)

    assert not change.rejected


def test_improvement_is_never_rejected() -> None:
    (change,) = compare_class_f1({"a": 0.5}, {"a": 0.9}, max_drop=0.0)

    assert not change.rejected


def test_new_class_is_listed_but_never_rejected() -> None:
    changes = compare_class_f1({"a": 0.9}, {"a": 0.9, "nova": 0.3}, max_drop=0.05)

    new = next(change for change in changes if change.label == "nova")
    assert new.old is None
    assert new.delta is None
    assert not new.rejected


def test_removed_class_is_listed_but_never_rejected() -> None:
    changes = compare_class_f1({"a": 0.9, "antiga": 0.8}, {"a": 0.9}, max_drop=0.05)

    removed = next(change for change in changes if change.label == "antiga")
    assert removed.new is None
    assert not removed.rejected


def test_classes_come_in_name_order() -> None:
    changes = compare_class_f1({"b": 0.9, "a": 0.9}, {"c": 0.9, "a": 0.9, "b": 0.9}, max_drop=0.05)

    assert [change.label for change in changes] == ["a", "b", "c"]


def test_table_flags_rejections_and_gives_the_verdict() -> None:
    changes = compare_class_f1({"a": 0.9, "b": 0.9}, {"a": 0.7, "b": 0.92, "nova": 0.6}, max_drop=0.05)

    text = "\n".join(format_comparison(changes, max_drop=0.05))

    assert "REJEITADO" in text
    assert "piorou" in text
    assert "nova" in text


def test_table_approves_when_nothing_dropped_too_much() -> None:
    text = "\n".join(format_comparison(compare_class_f1({"a": 0.9}, {"a": 0.89}, 0.05), max_drop=0.05))

    assert "APROVADO" in text
    assert "REJEITADO" not in text


def write_metrics(path: Path, mlp: dict) -> Path:
    path.write_text(json.dumps({"classifiers": {"mlp": mlp}}), encoding="utf-8")
    return path


def test_reads_the_mlp_class_f1_from_a_metrics_file(tmp_path: Path) -> None:
    path = write_metrics(tmp_path / "metrics.json", {"per_class_f1": {"a": 0.5}})

    assert read_class_f1(path) == {"a": 0.5}


def test_metrics_from_before_per_class_f1_are_a_clear_error(tmp_path: Path) -> None:
    path = write_metrics(tmp_path / "metrics.json", {"accuracy": 0.9})

    with pytest.raises(MetricsError, match="per_class_f1"):
        read_class_f1(path)


def test_missing_metrics_file_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(MetricsError, match="nope.json"):
        read_class_f1(tmp_path / "nope.json")
