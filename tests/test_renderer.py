import numpy as np
import pytest

from pose_meme.renderer import (
    HUD_LINE_HEIGHT,
    PANEL_COLOR,
    ViewComposer,
    display_size,
    draw_body,
    draw_skeleton,
    draw_text_lines,
    fit_image,
    info_lines,
    overlay_centered,
    panel_layout,
)
from tests.conftest import (
    RIGHT_ELBOW,
    RIGHT_INDEX,
    RIGHT_PINKY,
    RIGHT_THUMB,
    RIGHT_WRIST,
    Landmark,
    build_hand,
    build_pose,
)

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


FOREARM = [(RIGHT_ELBOW, RIGHT_WRIST), (RIGHT_WRIST, RIGHT_INDEX)]
POSE_ELBOW, POSE_WRIST, POSE_INDEX = (0.1, 0.5), (0.3, 0.3), (0.3, 0.1)
HAND_WRIST = (0.4, 0.5)


def arm_pose():
    return build_pose({RIGHT_ELBOW: POSE_ELBOW, RIGHT_WRIST: POSE_WRIST, RIGHT_INDEX: POSE_INDEX})


def draw_arm(frame: np.ndarray, hands: list) -> None:
    draw_body(frame, arm_pose(), hands, FOREARM, hand_connections=[], min_visibility=0.5)


def test_without_a_detected_hand_the_pose_hand_points_are_drawn() -> None:
    frame = black_frame()

    draw_arm(frame, [])

    assert is_lit(frame, *POSE_WRIST)
    assert is_lit(frame, *POSE_INDEX)


def test_detected_hand_replaces_the_pose_hand_points() -> None:
    # Two models drawing one hand was the doubled wrist and the extra triangle on camera.
    frame = black_frame()

    draw_arm(frame, [build_hand(HAND_WRIST)])

    assert not is_lit(frame, *POSE_INDEX)
    assert not is_lit(frame, *POSE_WRIST)


def test_arm_connects_to_the_wrist_of_the_detected_hand() -> None:
    frame = black_frame()

    draw_arm(frame, [build_hand(HAND_WRIST)])

    assert is_lit(frame, 0.25, 0.5)  # between the pose elbow and the hand wrist
    assert is_lit(frame, *HAND_WRIST)


def test_hand_points_of_the_other_arm_stay_when_only_one_hand_is_detected() -> None:
    frame = black_frame()
    pose = build_pose({RIGHT_WRIST: (0.1, 0.5), 15: (0.9, 0.5), 19: (0.9, 0.1)})

    draw_body(frame, pose, [build_hand((0.1, 0.5))], [], [], min_visibility=0.5)

    assert is_lit(frame, 0.9, 0.1)


def test_hidden_hand_points_stay_hidden_with_no_detected_hand() -> None:
    frame = black_frame()
    pose = build_pose(
        {RIGHT_THUMB: (0.2, 0.2), RIGHT_PINKY: (0.8, 0.2)},
        visibility={RIGHT_THUMB: 0.1, RIGHT_PINKY: 0.1},
    )

    draw_body(frame, pose, [], [], [], min_visibility=0.5)

    assert not is_lit(frame, 0.2, 0.2)
    assert not is_lit(frame, 0.8, 0.2)


def test_hand_connections_are_drawn() -> None:
    frame = black_frame()
    hand = [Landmark(0.2, 0.5), Landmark(0.8, 0.5)]

    draw_body(frame, None, [hand], [], hand_connections=[(0, 1)], min_visibility=0.5)

    assert is_lit(frame, 0.5, 0.5)


def test_hands_are_drawn_even_without_a_pose() -> None:
    frame = black_frame()

    draw_body(frame, None, [[Landmark(0.5, 0.5)]], [], [], min_visibility=0.5)

    assert is_lit(frame, 0.5, 0.5)


def test_info_lines_show_each_value() -> None:
    lines = info_lines(fps=29.97, pose="mao_no_ombro", confidence=0.876, sound_on=True, sound_key="m")

    text = " | ".join(lines)
    assert "30.0" in text
    assert "mao_no_ombro" in text
    assert "0.88" in text


def test_info_lines_have_a_fixed_count_so_the_layout_never_moves() -> None:
    assert len(info_lines(0.0, "neutro", 0.0, True, "m")) == len(info_lines(99.0, "garras_para_frente", 1.0, False, "m"))


def test_info_lines_show_the_sound_state_and_its_key() -> None:
    on = " | ".join(info_lines(30.0, "neutro", 1.0, sound_on=True, sound_key="m"))
    off = " | ".join(info_lines(30.0, "neutro", 1.0, sound_on=False, sound_key="m"))

    assert "ligado" in on
    assert "desligado" in off
    assert "[m]" in on


def test_text_lines_go_down_the_frame() -> None:
    frame = np.zeros((200, 400, 3), np.uint8)

    draw_text_lines(frame, ["first", "second", "third"])

    lit_rows = np.nonzero(frame.any(axis=(1, 2)))[0]
    assert lit_rows.max() - lit_rows.min() > 2 * HUD_LINE_HEIGHT - HUD_LINE_HEIGHT // 2


def test_text_lines_start_at_the_given_origin() -> None:
    frame = np.zeros((200, 400, 3), np.uint8)

    draw_text_lines(frame, ["text"], origin=(250, 100))

    lit_rows, lit_columns = np.nonzero(frame.any(axis=2))
    assert lit_columns.min() >= 240
    assert lit_rows.min() >= 80


def bgra(height: int, width: int, color: tuple[int, int, int], alpha: int) -> np.ndarray:
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[..., :3] = color
    image[..., 3] = alpha
    return image


def test_opaque_image_is_pasted_centered() -> None:
    frame = black_frame()

    overlay_centered(frame, np.full((20, 40, 3), 200, dtype=np.uint8))

    assert frame.shape == (HEIGHT, WIDTH, 3)
    lit_rows, lit_columns = np.nonzero(frame[..., 0])
    assert (lit_rows.min(), lit_rows.max()) == (40, 59)
    assert (lit_columns.min(), lit_columns.max()) == (80, 119)
    assert frame[50, 100].tolist() == [200, 200, 200]


def test_fully_transparent_pixels_keep_the_frame() -> None:
    frame = np.full((HEIGHT, WIDTH, 3), 77, dtype=np.uint8)

    overlay_centered(frame, bgra(20, 40, (255, 255, 255), alpha=0))

    assert (frame == 77).all()


def test_fully_opaque_alpha_replaces_the_frame() -> None:
    frame = np.full((HEIGHT, WIDTH, 3), 77, dtype=np.uint8)

    overlay_centered(frame, bgra(20, 40, (10, 20, 30), alpha=255))

    assert frame[50, 100].tolist() == [10, 20, 30]
    assert frame[0, 0].tolist() == [77, 77, 77]


def test_half_alpha_blends_image_and_frame() -> None:
    frame = np.full((HEIGHT, WIDTH, 3), 100, dtype=np.uint8)

    overlay_centered(frame, bgra(20, 40, (200, 200, 200), alpha=128))

    assert frame[50, 100].tolist() == pytest.approx([150, 150, 150], abs=1)


@pytest.mark.parametrize("size", [(300, 40), (20, 900), (300, 900)])
def test_image_larger_than_the_frame_is_clipped_to_it(size: tuple[int, int]) -> None:
    frame = black_frame()

    overlay_centered(frame, np.full((*size, 3), 200, dtype=np.uint8))

    assert frame.shape == (HEIGHT, WIDTH, 3)
    assert frame[50, 100].tolist() == [200, 200, 200]


def test_fit_image_fills_the_box_height_keeping_the_proportions() -> None:
    assert fit_image(np.zeros((100, 50, 3), dtype=np.uint8), box_size=(240, 640)).shape == (240, 120, 3)


def test_fit_image_is_limited_by_the_box_width() -> None:
    assert fit_image(np.zeros((10, 1000, 3), dtype=np.uint8), box_size=(480, 640)).shape == (6, 640, 3)


def test_fit_image_keeps_the_alpha_channel() -> None:
    fitted = fit_image(bgra(100, 100, (1, 2, 3), alpha=9), box_size=(50, 50))

    assert fitted.shape == (50, 50, 4)
    assert fitted[0, 0, 3] == 9


# --- camera on the left, panel on the right: info on top, meme below ---

LAYOUT_FRAME = (480, 640)


def test_panel_width_follows_the_ratio_of_the_camera_width() -> None:
    assert panel_layout(LAYOUT_FRAME, width_ratio=0.5, line_count=3).width == 320


def test_meme_box_sits_below_the_info_and_inside_the_panel() -> None:
    layout = panel_layout(LAYOUT_FRAME, width_ratio=0.5, line_count=3)
    top, left, height, width = layout.meme_box

    assert top >= 3 * HUD_LINE_HEIGHT
    assert left >= 0
    assert left + width <= layout.width
    assert top + height <= LAYOUT_FRAME[0]
    assert layout.meme_size == (height, width)


def test_more_info_lines_leave_less_room_for_the_meme() -> None:
    few = panel_layout(LAYOUT_FRAME, width_ratio=0.5, line_count=2)
    many = panel_layout(LAYOUT_FRAME, width_ratio=0.5, line_count=6)

    assert many.meme_box[2] < few.meme_box[2]


def test_tiny_frame_still_gives_a_meme_box() -> None:
    layout = panel_layout((20, 20), width_ratio=0.1, line_count=10)

    assert min(layout.meme_size) >= 1


def composer(frame_size: tuple[int, int] = LAYOUT_FRAME, max_width: int = 10_000) -> ViewComposer:
    return ViewComposer(frame_size, panel_width_ratio=0.5, max_width=max_width, line_count=3)


def camera() -> np.ndarray:
    return np.full((*LAYOUT_FRAME, 3), 50, dtype=np.uint8)


def test_view_puts_the_camera_untouched_on_the_left_with_the_panel_beside_it() -> None:
    view = composer().compose(camera(), [], None)

    assert view.shape == (480, 640 + 320, 3)
    assert np.array_equal(view[:, :640], camera())


def panel_without_separator(chosen: ViewComposer, view: np.ndarray) -> np.ndarray:
    return np.delete(view[:, 640:], chosen.layout.info_height, axis=0)


def test_panel_without_meme_or_text_is_plain_background_below_the_separator() -> None:
    chosen = composer()

    view = chosen.compose(camera(), [], None)

    assert (panel_without_separator(chosen, view) == PANEL_COLOR).all()
    assert (view[chosen.layout.info_height, 640 + chosen.layout.width // 2] != PANEL_COLOR).any()


def test_info_text_goes_on_the_panel_top_not_on_the_camera() -> None:
    chosen = composer()

    view = chosen.compose(camera(), ["Pose: neutro"], None)

    assert np.array_equal(view[:, :640], camera())
    lit_rows = np.nonzero((panel_without_separator(chosen, view) != PANEL_COLOR).any(axis=(1, 2)))[0]
    assert lit_rows.max() < 3 * HUD_LINE_HEIGHT + HUD_LINE_HEIGHT


def test_meme_is_centered_in_its_box_on_the_panel() -> None:
    chosen = composer()
    meme = fit_image(np.full((10, 10, 3), 200, dtype=np.uint8), chosen.layout.meme_size)

    view = chosen.compose(camera(), [], meme)

    top, left, height, width = chosen.layout.meme_box
    assert view[top + height // 2, 640 + left + width // 2].tolist() == [200, 200, 200]
    assert np.array_equal(view[:, :640], camera())


def test_transparent_meme_shows_the_panel_background() -> None:
    chosen = composer()

    view = chosen.compose(camera(), [], fit_image(bgra(10, 10, (255, 255, 255), alpha=0), chosen.layout.meme_size))

    assert (panel_without_separator(chosen, view) == PANEL_COLOR).all()


def test_composing_does_not_change_the_camera_frame() -> None:
    frame = camera()

    composer().compose(frame, ["text"], np.full((10, 10, 3), 200, dtype=np.uint8))

    assert (frame == 50).all()


def test_text_of_the_previous_frame_does_not_linger() -> None:
    chosen = composer()
    expected = composer().compose(camera(), ["FPS: 1"], None).copy()

    chosen.compose(camera(), ["Pose: garras_para_frente, a long line"], None)
    view = chosen.compose(camera(), ["FPS: 1"], None)

    assert np.array_equal(view, expected)


def test_meme_change_replaces_the_panel_and_its_end_clears_it() -> None:
    chosen = composer()
    red = fit_image(np.full((10, 10, 3), (0, 0, 200), dtype=np.uint8), chosen.layout.meme_size)
    blue = fit_image(np.full((10, 10, 3), (200, 0, 0), dtype=np.uint8), chosen.layout.meme_size)
    top, left, height, width = chosen.layout.meme_box
    center = (top + height // 2, 640 + left + width // 2)

    assert chosen.compose(camera(), [], red)[center].tolist() == [0, 0, 200]
    assert chosen.compose(camera(), [], blue)[center].tolist() == [200, 0, 0]
    assert (panel_without_separator(chosen, chosen.compose(camera(), [], None)) == PANEL_COLOR).all()


def test_camera_changes_reach_the_view_every_frame() -> None:
    chosen = composer()
    chosen.compose(camera(), [], None)

    view = chosen.compose(np.full((*LAYOUT_FRAME, 3), 99, dtype=np.uint8), [], None)

    assert (view[:, :640] == 99).all()


def test_wide_camera_is_shrunk_evenly_to_fit_the_window() -> None:
    chosen = composer(frame_size=(720, 1280), max_width=1440)

    view = chosen.compose(np.full((720, 1280, 3), 70, dtype=np.uint8), [], None)

    assert view.shape[1] <= 1440
    height, width = chosen.camera_size
    assert width / height == pytest.approx(1280 / 720, rel=0.01)
    assert view.shape[0] == height
    assert (view[:, :width] == 70).all()


# --- the window never gets wider than the screen, and nothing is stretched ---


def test_view_that_fits_keeps_the_camera_size() -> None:
    assert display_size((480, 640), width_ratio=0.5, max_width=1600) == (480, 640)


def test_view_too_wide_shrinks_the_camera_keeping_its_proportions() -> None:
    height, width = display_size((720, 1280), width_ratio=0.5, max_width=1440)

    assert width + round(width * 0.5) <= 1440
    assert width / height == pytest.approx(1280 / 720, rel=0.01)


def test_view_exactly_at_the_limit_is_kept() -> None:
    assert display_size((720, 1280), width_ratio=0.5, max_width=1920) == (720, 1280)
