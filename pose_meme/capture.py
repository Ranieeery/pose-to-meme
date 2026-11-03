"""Webcam wrapper around OpenCV."""

from collections.abc import Callable
from types import TracebackType
from typing import Self

import cv2
import numpy as np

from pose_meme.config import CameraConfig

MIRROR_HORIZONTALLY = 1  # cv2.flip's flipCode for a flip around the vertical axis


class CameraError(RuntimeError):
    """The webcam could not be opened or stopped delivering frames."""


class Camera:
    def __init__(self, config: CameraConfig, opener: Callable[[int], cv2.VideoCapture] = cv2.VideoCapture) -> None:
        self._mirror = config.mirror
        self._capture = opener(config.index)
        if not self._capture.isOpened():
            self._capture.release()
            raise CameraError(f"could not open camera {config.index}; check camera.index in config.yaml")
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)

    def read(self) -> np.ndarray:
        """Return the next BGR frame, mirrored if configured."""
        ok, frame = self._capture.read()
        if not ok:
            raise CameraError("could not read a frame from the camera")
        return cv2.flip(frame, MIRROR_HORIZONTALLY) if self._mirror else frame

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._capture.release()
