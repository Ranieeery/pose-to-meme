"""Drawing on the video frame. Functions draw in place, to avoid copying each frame."""

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from pose_meme.tracker import Landmark

# BGR colors and sizes in pixels; purely cosmetic, so not in config.yaml.
SEGMENT_COLOR = (255, 255, 255)
SEGMENT_THICKNESS = 2
POINT_COLOR = (0, 200, 255)
POINT_RADIUS = 3
HUD_ORIGIN = (10, 25)
HUD_COLOR = (0, 255, 0)
HUD_OUTLINE_COLOR = (0, 0, 0)
HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_SCALE = 0.7
HUD_THICKNESS = 2
HUD_OUTLINE_THICKNESS = 4


def draw_skeleton(
    frame: np.ndarray,
    landmarks: Sequence["Landmark"],
    connections: Iterable[tuple[int, int]],
    min_visibility: float,
) -> None:
    """Draw visible landmarks, and the segments whose two ends are both visible."""
    height, width = frame.shape[:2]
    points = [
        (round(lm.x * (width - 1)), round(lm.y * (height - 1))) if lm.visibility >= min_visibility else None
        for lm in landmarks
    ]
    for start, end in connections:
        if points[start] is not None and points[end] is not None:
            cv2.line(frame, points[start], points[end], SEGMENT_COLOR, SEGMENT_THICKNESS, cv2.LINE_AA)
    for point in points:
        if point is not None:
            cv2.circle(frame, point, POINT_RADIUS, POINT_COLOR, cv2.FILLED, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, fps: float) -> None:
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, HUD_ORIGIN, HUD_FONT, HUD_SCALE, HUD_OUTLINE_COLOR, HUD_OUTLINE_THICKNESS, cv2.LINE_AA)
    cv2.putText(frame, text, HUD_ORIGIN, HUD_FONT, HUD_SCALE, HUD_COLOR, HUD_THICKNESS, cv2.LINE_AA)
