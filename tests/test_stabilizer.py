from pose_meme.stabilizer import PoseStabilizer

NEUTRAL = "neutro"
POSE = "mao_na_boca"
OTHER = "mao_no_ombro"


def stabilizer(window_size: int = 8, min_ratio: float = 0.7) -> PoseStabilizer:
    return PoseStabilizer(window_size, min_ratio, initial=NEUTRAL)


def feed(stable: PoseStabilizer, labels: list[str]) -> str:
    current = stable.current
    for label in labels:
        current = stable.update(label)
    return current


def test_starts_on_the_initial_class() -> None:
    assert stabilizer().current == NEUTRAL


def test_noise_of_one_or_two_frames_does_not_change_the_pose() -> None:
    stable = stabilizer()

    assert feed(stable, [NEUTRAL] * 8 + [POSE, POSE] + [NEUTRAL] * 4) == NEUTRAL


def test_pose_changes_when_the_majority_holds() -> None:
    stable = stabilizer()

    assert feed(stable, [NEUTRAL] * 8 + [POSE] * 6) == POSE


def test_pose_does_not_change_one_frame_before_reaching_the_ratio() -> None:
    stable = stabilizer()  # 0.7 of 8 frames needs 6 of them

    assert feed(stable, [NEUTRAL] * 8 + [POSE] * 5) == NEUTRAL
    assert stable.update(POSE) == POSE


def test_pose_can_change_before_the_window_fills_once_the_ratio_is_reachable() -> None:
    assert feed(stabilizer(), [POSE] * 6) == POSE


def test_noise_at_startup_does_not_change_the_pose() -> None:
    assert feed(stabilizer(), [POSE] * 2) == NEUTRAL


def test_noise_does_not_interrupt_a_held_pose() -> None:
    stable = stabilizer()
    feed(stable, [POSE] * 8)

    assert feed(stable, [NEUTRAL, OTHER] + [POSE] * 3) == POSE


def test_without_a_majority_the_current_pose_stays() -> None:
    stable = stabilizer()
    feed(stable, [POSE] * 8)

    assert feed(stable, [OTHER, NEUTRAL] * 4) == POSE


def test_pose_returns_to_neutral_when_it_ends() -> None:
    stable = stabilizer()
    feed(stable, [POSE] * 8)

    assert feed(stable, [NEUTRAL] * 8) == NEUTRAL
