import pytest

from pose_meme.timing import FpsCounter


def test_fps_is_zero_until_two_frames_were_seen() -> None:
    counter = FpsCounter(window=5)
    assert counter.fps == 0.0

    counter.tick(1.0)
    assert counter.fps == 0.0


def test_fps_comes_from_the_intervals_between_frames() -> None:
    counter = FpsCounter(window=5)

    for now in (0.0, 0.1, 0.2):
        counter.tick(now)

    assert counter.fps == pytest.approx(10.0)


def test_old_frames_leave_the_window() -> None:
    counter = FpsCounter(window=3)

    for now in (0.0, 1.0, 1.1, 1.2):  # the slow first interval falls out
        counter.tick(now)

    assert counter.fps == pytest.approx(10.0)


def test_repeated_timestamps_do_not_divide_by_zero() -> None:
    counter = FpsCounter(window=3)

    counter.tick(1.0)
    counter.tick(1.0)

    assert counter.fps == 0.0
