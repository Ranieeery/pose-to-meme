"""Drawing on the video frame. Functions draw in place, to avoid copying each frame."""

from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import cv2
import numpy as np

from pose_meme.features import POSE_POINTS, assign_hands

if TYPE_CHECKING:
    from pose_meme.tracker import Landmark

# BGR colors and sizes in pixels; purely cosmetic, so not in config.yaml.
SEGMENT_COLOR = (255, 255, 255)
SEGMENT_THICKNESS = 2
POINT_COLOR = (0, 200, 255)
POINT_RADIUS = 3
HUD_ORIGIN = (10, 25)
HUD_LINE_HEIGHT = 25
HUD_COLOR = (0, 255, 0)
HUD_OUTLINE_COLOR = (0, 0, 0)
HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_SCALE = 0.7
HUD_THICKNESS = 2
HUD_OUTLINE_THICKNESS = 4

MAX_ALPHA = 255.0
BGRA_CHANNELS = 4
HIDDEN = -1.0  # below any visibility threshold
POSE_HAND_PARTS = ("pinky", "index", "thumb")


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


def draw_body(
    frame: np.ndarray,
    pose: Sequence["Landmark"] | None,
    hands: Sequence[Sequence["Landmark"]],
    pose_connections: Iterable[tuple[int, int]],
    hand_connections: Iterable[tuple[int, int]],
    min_visibility: float,
) -> None:
    """Draw one skeleton from two models: a detected hand takes over from the pose's hand.

    The pose model has its own rough wrist and hand points. Drawn together with the
    hand model's, they would show a doubled wrist and an extra triangle over the palm.
    """
    if pose is not None:
        height, width = frame.shape[:2]
        merged = list(pose)
        for side, hand in assign_hands(pose, hands, width / height).items():
            wrist = POSE_POINTS[f"{side}_wrist"]
            merged[wrist] = replace(merged[wrist], x=hand[0].x, y=hand[0].y, visibility=hand[0].visibility)
            for part in POSE_HAND_PARTS:
                index = POSE_POINTS[f"{side}_{part}"]
                merged[index] = replace(merged[index], visibility=HIDDEN)
        draw_skeleton(frame, merged, pose_connections, min_visibility)
    for hand in hands:
        draw_skeleton(frame, hand, hand_connections, min_visibility)


def draw_hud(frame: np.ndarray, fps: float, pose: str, confidence: float) -> None:
    draw_text_lines(frame, [f"FPS: {fps:.1f}", f"Pose: {pose}", f"Confianca: {confidence:.2f}"])


def draw_text_lines(frame: np.ndarray, lines: Sequence[str]) -> None:
    """Outlined text from the top left corner; cv2 fonts have no accented characters."""
    x, y = HUD_ORIGIN
    for number, text in enumerate(lines):
        origin = (x, y + number * HUD_LINE_HEIGHT)
        cv2.putText(frame, text, origin, HUD_FONT, HUD_SCALE, HUD_OUTLINE_COLOR, HUD_OUTLINE_THICKNESS, cv2.LINE_AA)
        cv2.putText(frame, text, origin, HUD_FONT, HUD_SCALE, HUD_COLOR, HUD_THICKNESS, cv2.LINE_AA)


def fit_overlay(image: np.ndarray, frame_size: tuple[int, int], height_ratio: float) -> np.ndarray:
    """Resize an image to its display size, keeping proportions and within the frame width."""
    frame_height, frame_width = frame_size
    height, width = image.shape[:2]
    scale = min(frame_height * height_ratio / height, frame_width / width)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def overlay_centered(frame: np.ndarray, image: np.ndarray) -> None:
    """Paste an image on the frame's center, using its alpha channel when it has one."""
    frame_height, frame_width = frame.shape[:2]
    height, width = image.shape[:2]
    shown_height, shown_width = min(height, frame_height), min(width, frame_width)

    top, left = (frame_height - shown_height) // 2, (frame_width - shown_width) // 2
    crop_top, crop_left = (height - shown_height) // 2, (width - shown_width) // 2
    source = image[crop_top : crop_top + shown_height, crop_left : crop_left + shown_width]
    target = frame[top : top + shown_height, left : left + shown_width]

    if image.shape[2] < BGRA_CHANNELS:
        target[:] = source
        return
    alpha = source[..., 3:].astype(np.float32) / MAX_ALPHA
    target[:] = (alpha * source[..., :3] + (1.0 - alpha) * target + 0.5).astype(np.uint8)
