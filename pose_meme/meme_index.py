"""Which meme goes with which pose. For now a static map; CLIP matching replaces it in M4."""

import random
from pathlib import Path

import cv2
import numpy as np

from pose_meme.config import MemesConfig
from pose_meme.renderer import fit_overlay


class MemeLoadError(RuntimeError):
    """A meme file is missing or can't be decoded."""


def _read_image(path: Path) -> np.ndarray:
    # cv2.imread can't open non-ASCII paths on Windows, so decode the bytes instead.
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError as error:
        raise MemeLoadError(f"could not read meme file: {path}") from error
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED) if data.size else None
    if image is None:
        raise MemeLoadError(f"could not decode meme file: {path}")
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image


def load_static_memes(
    config: MemesConfig,
    frame_size: tuple[int, int],
    height_ratio: float,
) -> dict[str, list[np.ndarray]]:
    """Load and resize every mapped meme once, so the frame loop never touches the disk."""
    return {
        pose: [fit_overlay(_read_image(config.dir / name), frame_size, height_ratio) for name in names]
        for pose, names in config.static_map.items()
    }


class MemeSelector:
    """Draws a meme when the pose changes and keeps it for as long as the pose lasts."""

    def __init__(self, memes: dict[str, list[np.ndarray]], neutral_class: str, rng: random.Random) -> None:
        self._memes = memes
        self._neutral = neutral_class
        self._rng = rng
        self._pose: str | None = None
        self._current: np.ndarray | None = None

    def update(self, pose: str) -> np.ndarray | None:
        if pose != self._pose:
            self._pose = pose
            options = self._memes.get(pose) if pose != self._neutral else None
            self._current = self._rng.choice(options) if options else None
        return self._current
