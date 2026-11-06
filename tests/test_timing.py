import pytest

from pose_meme.timing import FpsCounter, StageTimer, format_stats


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


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


def run_frames(timer: StageTimer, clock: FakeClock, durations: list[dict[str, float]]) -> None:
    for frame in durations:
        for stage, seconds in frame.items():
            with timer.measure(stage):
                clock.now += seconds
        timer.end_frame()


def test_measure_records_the_time_spent_inside() -> None:
    clock = FakeClock()
    timer = StageTimer(warmup_frames=0, clock=clock)

    run_frames(timer, clock, [{"tracking": 0.02}, {"tracking": 0.04}])

    (stats,) = timer.stats()
    assert stats.stage == "tracking"
    assert stats.frames == 2
    assert stats.mean_ms == pytest.approx(30.0)


def test_warmup_frames_are_discarded() -> None:
    clock = FakeClock()
    timer = StageTimer(warmup_frames=2, clock=clock)

    run_frames(timer, clock, [{"tracking": 1.0}, {"tracking": 1.0}, {"tracking": 0.01}])

    (stats,) = timer.stats()
    assert stats.frames == 1
    assert stats.mean_ms == pytest.approx(10.0)


def test_p95_shows_the_slow_frames_the_mean_hides() -> None:
    clock = FakeClock()
    timer = StageTimer(warmup_frames=0, clock=clock)

    run_frames(timer, clock, [{"render": 0.001}] * 90 + [{"render": 0.1}] * 10)

    (stats,) = timer.stats()
    assert stats.mean_ms == pytest.approx(10.9)
    assert stats.p95_ms == pytest.approx(100.0)


def test_stages_keep_the_order_of_the_loop() -> None:
    clock = FakeClock()
    timer = StageTimer(warmup_frames=0, clock=clock)

    run_frames(timer, clock, [{"capture": 0.01, "tracking": 0.02, "render": 0.003}])

    assert [stats.stage for stats in timer.stats()] == ["capture", "tracking", "render"]


def test_time_is_measured_even_when_the_stage_raises() -> None:
    clock = FakeClock()
    timer = StageTimer(warmup_frames=0, clock=clock)

    with pytest.raises(RuntimeError), timer.measure("tracking"):
        clock.now += 0.01
        raise RuntimeError

    assert timer.stats()[0].mean_ms == pytest.approx(10.0)


def test_nothing_measured_gives_no_table() -> None:
    assert format_stats([]) == []


def test_table_has_one_row_per_stage_and_the_total_with_fps() -> None:
    clock = FakeClock()
    timer = StageTimer(warmup_frames=0, clock=clock)
    run_frames(timer, clock, [{"tracking": 0.03, "render": 0.01}] * 4)  # 40 ms a frame

    lines = format_stats(timer.stats())

    assert lines[0].startswith("| Etapa")
    tracking = next(line for line in lines if line.startswith("| tracking"))
    assert "30.00" in tracking
    assert "75%" in tracking
    total = next(line for line in lines if line.startswith("| **total**"))
    assert "40.00" in total
    assert "25.0 FPS" in total
