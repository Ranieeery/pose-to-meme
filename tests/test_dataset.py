from pathlib import Path

import numpy as np
import pytest

from pose_meme.dataset import (
    DatasetError,
    Recorder,
    SampleWriter,
    class_for_key,
    next_session_id,
    read_samples,
)
from pose_meme.features import FEATURE_SIZE
from scripts.collect import hud_lines

CLASSES = ["neutro", "pose_a", "pose_b"]


def vector(value: float = 0.5) -> np.ndarray:
    return np.full(FEATURE_SIZE, value)


def write_session(directory: Path, session_id: str, rows: list[tuple[str, float]]) -> None:
    with SampleWriter(directory, session_id) as writer:
        for label, value in rows:
            writer.write(label, vector(value))


# Sessions and files


def test_first_session_id(tmp_path: Path) -> None:
    assert next_session_id(tmp_path) == "sessao_01"


def test_session_id_never_reuses_an_existing_session(tmp_path: Path) -> None:
    write_session(tmp_path, "sessao_01", [("neutro", 0.1)])
    write_session(tmp_path, "sessao_07", [("neutro", 0.1)])

    assert next_session_id(tmp_path) == "sessao_08"


def test_samples_round_trip_with_session_and_label(tmp_path: Path) -> None:
    write_session(tmp_path, "sessao_01", [("neutro", 0.25), ("pose_a", -1.5)])

    samples = read_samples(tmp_path)

    assert samples.labels.tolist() == ["neutro", "pose_a"]
    assert samples.sessions.tolist() == ["sessao_01", "sessao_01"]
    assert samples.features.shape == (2, FEATURE_SIZE)
    np.testing.assert_allclose(samples.features[1], vector(-1.5))


def test_all_sessions_are_read(tmp_path: Path) -> None:
    write_session(tmp_path, "sessao_01", [("neutro", 0.1)])
    write_session(tmp_path, "sessao_02", [("pose_a", 0.2), ("pose_b", 0.3)])

    assert read_samples(tmp_path).sessions.tolist() == ["sessao_01", "sessao_02", "sessao_02"]


def test_file_is_only_created_when_something_is_recorded(tmp_path: Path) -> None:
    with SampleWriter(tmp_path, "sessao_01"):
        pass

    assert list(tmp_path.iterdir()) == []


def test_writer_refuses_to_overwrite_a_session(tmp_path: Path) -> None:
    write_session(tmp_path, "sessao_01", [("neutro", 0.1)])

    with SampleWriter(tmp_path, "sessao_01") as writer, pytest.raises(DatasetError, match="sessao_01"):
        writer.write("neutro", vector())


def test_no_samples_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="no samples"):
        read_samples(tmp_path)


def test_samples_from_an_old_feature_layout_are_rejected(tmp_path: Path) -> None:
    # Mixing vectors of another layout would train on garbage without any error.
    columns = ",".join(f"f{index}" for index in range(3))
    (tmp_path / "sessao_01.csv").write_text(f"session_id,label,{columns}\nsessao_01,neutro,1,2,3\n", encoding="utf-8")

    with pytest.raises(DatasetError, match="sessao_01.*feature"):
        read_samples(tmp_path)


# Keys


@pytest.mark.parametrize(("key", "expected"), [("0", "neutro"), ("1", "pose_a"), ("a", "pose_b")])
def test_keys_choose_classes_by_position(key: str, expected: str) -> None:
    assert class_for_key(ord(key), CLASSES, "01a") == expected


@pytest.mark.parametrize("key", [ord("2"), ord("9"), ord("b"), 255])
def test_other_keys_choose_nothing(key: int) -> None:
    assert class_for_key(key, CLASSES, "01a") is None


def test_key_beyond_the_last_class_chooses_nothing() -> None:
    assert class_for_key(ord("3"), CLASSES, "0123") is None


# Recording HUD


def test_hud_lists_the_classes_with_keys_counts_and_the_selected_one() -> None:
    rec = recorder(countdown=0.0)
    rec.select("pose_a")
    rec.toggle(now=0.0)
    rec.should_record(now=0.0)
    rec.saved()

    lines = hud_lines("sessao_03", rec, CLASSES, "01a", target=100, now=0.0, has_body=True)

    assert "sessao_03" in lines[0]
    assert "gravando" in lines[0]
    assert "  0 neutro: 0/100" in lines
    assert "> 1 pose_a: 1/100" in lines
    assert "  a pose_b: 0/100" in lines


def test_hud_warns_when_there_is_no_body_to_record() -> None:
    lines = hud_lines("sessao_03", recorder(), CLASSES, "01a", target=100, now=0.0, has_body=False)

    assert "ombros" in lines[0]


# Recorder


def recorder(countdown: float = 2.0, every: int = 1, target: int = 100) -> Recorder:
    return Recorder("neutro", countdown_seconds=countdown, sample_every=every, samples_per_class=target)


def test_nothing_is_recorded_before_the_start() -> None:
    rec = recorder()

    assert rec.should_record(now=10.0) is False
    assert rec.state == "parado"


def test_nothing_is_recorded_during_the_countdown() -> None:
    # Frames of the arm still moving into the pose would carry the wrong label.
    rec = recorder(countdown=2.0)
    rec.toggle(now=10.0)

    assert rec.should_record(now=11.9) is False
    assert rec.state == "preparando"
    assert rec.seconds_left(now=11.0) == pytest.approx(1.0)


def test_recording_starts_after_the_countdown() -> None:
    rec = recorder(countdown=2.0)
    rec.toggle(now=10.0)

    assert rec.should_record(now=12.0) is True
    assert rec.state == "gravando"


def test_only_one_in_n_frames_is_recorded() -> None:
    rec = recorder(countdown=0.0, every=3)
    rec.toggle(now=0.0)

    assert [rec.should_record(now=float(i)) for i in range(6)] == [True, False, False, True, False, False]


def test_toggle_stops_recording() -> None:
    rec = recorder(countdown=0.0)
    rec.toggle(now=0.0)
    rec.toggle(now=1.0)

    assert rec.should_record(now=2.0) is False
    assert rec.state == "parado"


def test_recording_stops_by_itself_at_the_class_target() -> None:
    rec = recorder(countdown=0.0, target=2)
    rec.toggle(now=0.0)
    for now in (0.0, 1.0):
        assert rec.should_record(now=now) is True
        rec.saved()

    assert rec.should_record(now=2.0) is False
    assert rec.state == "parado"
    assert rec.counts["neutro"] == 2


def test_class_already_at_its_target_does_not_start_again() -> None:
    rec = recorder(countdown=0.0, target=1)
    rec.toggle(now=0.0)
    rec.should_record(now=0.0)
    rec.saved()

    rec.toggle(now=1.0)

    assert rec.should_record(now=1.0) is False


def test_choosing_a_class_stops_recording() -> None:
    rec = recorder(countdown=0.0)
    rec.toggle(now=0.0)

    rec.select("pose_a")

    assert rec.state == "parado"
    assert rec.label == "pose_a"


def test_counts_are_per_class() -> None:
    rec = recorder(countdown=0.0)
    rec.toggle(now=0.0)
    rec.should_record(now=0.0)
    rec.saved()
    rec.select("pose_a")
    rec.toggle(now=1.0)
    rec.should_record(now=1.0)
    rec.saved()
    rec.should_record(now=2.0)
    rec.saved()

    assert rec.counts == {"neutro": 1, "pose_a": 2}
