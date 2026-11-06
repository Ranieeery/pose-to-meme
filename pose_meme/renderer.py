"""Drawing: skeleton on the camera frame, and the view with a side panel for info and meme.

Skeleton and text functions draw in place, to avoid copying each frame."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
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
PANEL_COLOR = (40, 40, 40)
PANEL_PADDING = 12
SEPARATOR_COLOR = (90, 90, 90)
SEPARATOR_THICKNESS = 1

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


def info_lines(fps: float, pose: str, confidence: float) -> list[str]:
    """What the panel shows above the meme; always the same number of lines."""
    return [f"Pose: {pose}", f"Confianca: {confidence:.2f}", f"FPS: {fps:.1f}"]


def draw_text_lines(frame: np.ndarray, lines: Sequence[str], origin: tuple[int, int] = HUD_ORIGIN) -> None:
    """Outlined text going down from origin; cv2 fonts have no accented characters."""
    x, y = origin
    for number, text in enumerate(lines):
        position = (x, y + number * HUD_LINE_HEIGHT)
        cv2.putText(frame, text, position, HUD_FONT, HUD_SCALE, HUD_OUTLINE_COLOR, HUD_OUTLINE_THICKNESS, cv2.LINE_AA)
        cv2.putText(frame, text, position, HUD_FONT, HUD_SCALE, HUD_COLOR, HUD_THICKNESS, cv2.LINE_AA)


def fit_image(image: np.ndarray, box_size: tuple[int, int]) -> np.ndarray:
    """Resize an image to the largest size that fits the box, keeping its proportions."""
    box_height, box_width = box_size
    height, width = image.shape[:2]
    scale = min(box_height / height, box_width / width)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


@dataclass(frozen=True)
class PanelLayout:
    """The panel beside the camera, in panel coordinates: info lines on top, meme box below."""

    width: int
    info_height: int
    meme_box: tuple[int, int, int, int]  # top, left, height, width

    @property
    def meme_size(self) -> tuple[int, int]:
        return self.meme_box[2], self.meme_box[3]


def display_size(frame_size: tuple[int, int], width_ratio: float, max_width: int) -> tuple[int, int]:
    """Camera size on screen: shrunk by the same factor on both axes when camera + panel
    would be wider than max_width, so the window never needs resizing (which stretches)."""
    frame_height, frame_width = frame_size
    scale = min(1.0, max_width / (frame_width * (1 + width_ratio)))
    return max(1, int(frame_height * scale)), max(1, int(frame_width * scale))


def panel_layout(frame_size: tuple[int, int], width_ratio: float, line_count: int) -> PanelLayout:
    """Computed once from the camera's real size, since nothing may assume a fixed one."""
    frame_height, frame_width = frame_size
    width = max(1, round(frame_width * width_ratio))
    info_height = PANEL_PADDING + line_count * HUD_LINE_HEIGHT
    top = min(info_height + PANEL_PADDING, frame_height - 1)
    box_height = max(1, frame_height - top - PANEL_PADDING)
    left = min(PANEL_PADDING, width - 1)
    box_width = max(1, width - 2 * PANEL_PADDING)
    return PanelLayout(width, info_height, (top, left, box_height, box_width))


class ViewComposer:
    """Builds the window image: the camera on the left, the panel on the right.

    Built for speed, since it runs every frame: the view buffer is allocated once, and
    the panel with its meme (the alpha blend is the costly part) is drawn only when
    the meme changes. Each frame then copies the camera and redraws the text strip.
    """

    def __init__(self, frame_size: tuple[int, int], panel_width_ratio: float, max_width: int, line_count: int) -> None:
        self.camera_size = display_size(frame_size, panel_width_ratio, max_width)
        self.layout = panel_layout(self.camera_size, panel_width_ratio, line_count)
        height, width = self.camera_size
        self._view = np.empty((height, width + self.layout.width, 3), dtype=np.uint8)
        self._camera = self._view[:, :width]
        self._panel = self._view[:, width:]
        self._shrunk = np.empty((height, width, 3), dtype=np.uint8)  # cv2.resize needs a contiguous target
        self._panel_base = np.empty_like(self._panel)  # background + separator + meme, without text
        self._text_rows = self.layout.meme_box[0]  # the text never reaches the meme box
        self._meme: np.ndarray | None = None
        self._draw_panel_base(None)

    def compose(self, frame: np.ndarray, lines: Sequence[str], meme: np.ndarray | None) -> np.ndarray:
        """Returns the same buffer every frame; it is only valid until the next call."""
        if meme is not self._meme:
            self._draw_panel_base(meme)
        if frame.shape[:2] == self.camera_size:
            self._camera[:] = frame
        else:
            height, width = self.camera_size
            cv2.resize(frame, (width, height), dst=self._shrunk, interpolation=cv2.INTER_LINEAR)
            self._camera[:] = self._shrunk
        self._panel[: self._text_rows] = self._panel_base[: self._text_rows]
        if lines:
            draw_text_lines(self._panel, lines, (PANEL_PADDING, PANEL_PADDING + HUD_LINE_HEIGHT - HUD_LINE_HEIGHT // 4))
        return self._view

    def _draw_panel_base(self, meme: np.ndarray | None) -> None:
        self._meme = meme
        base = self._panel_base
        base[:] = PANEL_COLOR
        info_bottom = self.layout.info_height
        cv2.line(
            base, (PANEL_PADDING, info_bottom), (self.layout.width - PANEL_PADDING, info_bottom), SEPARATOR_COLOR,
            SEPARATOR_THICKNESS,
        )
        if meme is not None:
            top, left, height, width = self.layout.meme_box
            overlay_centered(base[top : top + height, left : left + width], meme)
        self._panel[:] = base


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
