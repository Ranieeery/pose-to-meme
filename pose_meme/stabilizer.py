"""Debounce of the per-frame class, so noise doesn't make the meme flicker."""

from collections import Counter, deque


class PoseStabilizer:
    """Changes the pose only when a class fills the minimum share of the sliding window."""

    def __init__(self, window_size: int, min_ratio: float, initial: str) -> None:
        self._window: deque[str] = deque(maxlen=window_size)
        self._window_size = window_size
        self._min_ratio = min_ratio
        self.current = initial

    def update(self, label: str) -> str:
        self._window.append(label)
        leader, count = Counter(self._window).most_common(1)[0]
        if count / self._window_size >= self._min_ratio:
            self.current = leader
        return self.current
