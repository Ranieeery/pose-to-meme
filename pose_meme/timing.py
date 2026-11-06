"""Time measurement for the frame loop: FPS for the HUD and time per stage for the README."""

import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

MS_PER_SECOND = 1000
TAIL_PERCENTILE = 95  # the slow frames a mean hides


class FpsCounter:
    """Frames per second over a sliding window, so the HUD number stays readable."""

    def __init__(self, window: int) -> None:
        self._times: deque[float] = deque(maxlen=window)

    def tick(self, now: float) -> None:
        self._times.append(now)

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


@dataclass(frozen=True)
class StageStats:
    stage: str
    frames: int
    mean_ms: float
    p95_ms: float


class StageTimer:
    """Time spent per stage of the loop, skipping the first frames (model warm-up, camera start)."""

    def __init__(self, warmup_frames: int, clock: Callable[[], float] = time.perf_counter) -> None:
        self._warmup = warmup_frames
        self._clock = clock
        self._frame = 0
        self._durations: dict[str, list[float]] = {}  # insertion order is the loop order

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        start = self._clock()
        try:
            yield
        finally:
            if self._frame >= self._warmup:
                self._durations.setdefault(stage, []).append(self._clock() - start)

    def end_frame(self) -> None:
        self._frame += 1

    def stats(self) -> list[StageStats]:
        return [
            StageStats(
                stage,
                len(seconds),
                float(np.mean(seconds)) * MS_PER_SECOND,
                float(np.percentile(seconds, TAIL_PERCENTILE)) * MS_PER_SECOND,
            )
            for stage, seconds in self._durations.items()
        ]


def format_stats(stats: Sequence[StageStats]) -> list[str]:
    """A markdown table: mean and p95 per stage, its share of the frame, and the total as FPS."""
    if not stats:
        return []
    total = sum(item.mean_ms for item in stats)
    lines = [
        f"| Etapa | Média (ms) | p{TAIL_PERCENTILE} (ms) | Parte do frame |",
        "|---|---|---|---|",
    ]
    for item in stats:
        lines.append(f"| {item.stage} | {item.mean_ms:.2f} | {item.p95_ms:.2f} | {item.mean_ms / total:.0%} |")
    lines.append(f"| **total** | {total:.2f} | | {MS_PER_SECOND / total:.1f} FPS |")
    lines.append(f"\n{stats[0].frames} frames medidos.")
    return lines
