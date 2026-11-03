"""Time measurement for the frame loop."""

from collections import deque


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
