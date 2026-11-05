"""Recorded samples: one CSV per session, and the logic of when a frame gets recorded."""

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Self

import numpy as np

from pose_meme.features import FEATURE_SIZE

SESSION_PREFIX = "sessao_"
SESSION_PATTERN = re.compile(rf"{SESSION_PREFIX}(\d+)\.csv")
METADATA_COLUMNS = ["session_id", "label"]

STOPPED, COUNTDOWN, RECORDING = "parado", "preparando", "gravando"


class DatasetError(ValueError):
    """The samples on disk can't be used."""


@dataclass(frozen=True)
class Samples:
    features: np.ndarray
    labels: np.ndarray
    sessions: np.ndarray


def next_session_id(directory: Path) -> str:
    """The number after the highest session on disk, so gaps from deleted files stay gaps."""
    numbers = [
        int(match.group(1))
        for match in (SESSION_PATTERN.fullmatch(path.name) for path in directory.glob("*.csv"))
        if match
    ]
    return f"{SESSION_PREFIX}{max(numbers, default=0) + 1:02d}"


class SampleWriter:
    """Appends a session's samples to its CSV; the file only appears with the first sample."""

    def __init__(self, directory: Path, session_id: str) -> None:
        self._path = directory / f"{session_id}.csv"
        self._session_id = session_id
        self._file: IO[str] | None = None
        self._writer: Any = None  # csv.writer's return type isn't public

    def write(self, label: str, features: np.ndarray) -> None:
        if self._file is None:
            self._open()
        self._writer.writerow([self._session_id, label, *features.tolist()])
        self._file.flush()  # a crash in the frame loop must not lose a recorded take

    def _open(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._file = self._path.open("x", newline="", encoding="utf-8")
        except FileExistsError:
            raise DatasetError(f"session already exists: {self._path}") from None
        self._writer = csv.writer(self._file)
        self._writer.writerow([*METADATA_COLUMNS, *(f"f{index}" for index in range(FEATURE_SIZE))])

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._file is not None:
            self._file.close()


def read_samples(directory: Path) -> Samples:
    """Read every session; samples from another feature layout are an error, not a mix."""
    features: list[list[float]] = []
    labels: list[str] = []
    sessions: list[str] = []
    expected_columns = len(METADATA_COLUMNS) + FEATURE_SIZE

    for path in sorted(directory.glob(f"{SESSION_PREFIX}*.csv")):
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            header = next(reader, None)
            if header is not None and len(header) != expected_columns:
                raise DatasetError(
                    f"{path.name}: has {len(header) - len(METADATA_COLUMNS)} features, expected "
                    f"{FEATURE_SIZE}; features.py changed after this session was recorded"
                )
            for session_id, label, *values in reader:
                sessions.append(session_id)
                labels.append(label)
                features.append([float(value) for value in values])

    if not features:
        raise DatasetError(f"no samples in {directory}; record some with scripts.collect")
    return Samples(np.array(features), np.array(labels), np.array(sessions))


def class_for_key(key: int, classes: list[str], class_keys: str) -> str | None:
    """Keys choose a class by position: the first key is the first class."""
    return next((name for char, name in zip(class_keys, classes) if ord(char) == key), None)


class Recorder:
    """Decides which frames get recorded for the chosen class."""

    def __init__(self, label: str, countdown_seconds: float, sample_every: int, samples_per_class: int) -> None:
        self.label = label
        self.state = STOPPED
        self.counts: Counter[str] = Counter()
        self._countdown_seconds = countdown_seconds
        self._sample_every = sample_every
        self._samples_per_class = samples_per_class
        self._starts_at = 0.0
        self._frames = 0

    def select(self, label: str) -> None:
        self.label = label
        self.state = STOPPED

    def toggle(self, now: float) -> None:
        if self.state != STOPPED:
            self.state = STOPPED
        elif self.counts[self.label] < self._samples_per_class:
            self.state = COUNTDOWN
            self._starts_at = now + self._countdown_seconds
            self._frames = 0

    def seconds_left(self, now: float) -> float:
        return max(0.0, self._starts_at - now)

    def should_record(self, now: float) -> bool:
        if self.state == COUNTDOWN and now >= self._starts_at:
            self.state = RECORDING
        if self.state != RECORDING:
            return False
        # Consecutive frames are nearly identical; skipping some gives more variety per sample.
        record = self._frames % self._sample_every == 0
        self._frames += 1
        return record

    def saved(self) -> None:
        self.counts[self.label] += 1
        if self.counts[self.label] >= self._samples_per_class:
            self.state = STOPPED
