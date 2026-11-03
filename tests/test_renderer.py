import numpy as np

from pose_meme.renderer import draw_hud, draw_skeleton
from tests.conftest import Landmark

HEIGHT, WIDTH = 100, 200


def black_frame() -> np.ndarray:
    return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


def is_lit(frame: np.ndarray, x: float, y: float) -> bool:
    return bool(frame[int(y * (HEIGHT - 1)), int(x * (WIDTH - 1))].any())


def test_skeleton_draws_landmarks_and_the_segment_between_them() -> None:
    frame = black_frame()
    landmarks = [Landmark(0.2, 0.5), Landmark(0.8, 0.5)]

    draw_skeleton(frame, landmarks, connections=[(0, 1)], min_visibility=0.5)

    assert is_lit(frame, 0.2, 0.5)
    assert is_lit(frame, 0.5, 0.5)  # middle of the segment
    assert is_lit(frame, 0.8, 0.5)
    assert not is_lit(frame, 0.5, 0.05)


def test_low_visibility_landmark_and_its_segments_are_not_drawn() -> None:
    frame = black_frame()
    landmarks = [Landmark(0.2, 0.5), Landmark(0.8, 0.5, visibility=0.1)]

    draw_skeleton(frame, landmarks, connections=[(0, 1)], min_visibility=0.5)

    assert is_lit(frame, 0.2, 0.5)
    assert not is_lit(frame, 0.5, 0.5)
    assert not is_lit(frame, 0.8, 0.5)


def test_landmark_exactly_at_the_threshold_is_drawn() -> None:
    frame = black_frame()

    draw_skeleton(frame, [Landmark(0.5, 0.5, visibility=0.5)], connections=[], min_visibility=0.5)

    assert is_lit(frame, 0.5, 0.5)


def test_landmarks_outside_the_frame_keep_the_shape_and_do_not_crash() -> None:
    # MediaPipe extrapolates body parts out of view to coordinates beyond [0, 1].
    frame = black_frame()
    landmarks = [Landmark(-0.5, 1.5), Landmark(1.5, -0.5)]

    draw_skeleton(frame, landmarks, connections=[(0, 1)], min_visibility=0.5)

    assert frame.shape == (HEIGHT, WIDTH, 3)


def test_hud_writes_on_the_frame_keeping_its_shape() -> None:
    frame = black_frame()

    draw_hud(frame, fps=29.97)

    assert frame.any()
    assert frame.shape == (HEIGHT, WIDTH, 3)


def test_hud_shows_the_fps_value() -> None:
    first, second = black_frame(), black_frame()

    draw_hud(first, fps=10.0)
    draw_hud(second, fps=88.0)

    assert not np.array_equal(first, second)
