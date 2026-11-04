import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from pose_meme.config import MemesConfig
from pose_meme.meme_index import MemeLoadError, MemeSelector, load_static_memes

NEUTRAL = "neutro"
A = np.full((2, 2, 3), 1, dtype=np.uint8)
B = np.full((2, 2, 3), 2, dtype=np.uint8)
C = np.full((2, 2, 3), 3, dtype=np.uint8)


def selector(seed: int = 0) -> MemeSelector:
    return MemeSelector({"pose": [A, B, C], "other": [B]}, NEUTRAL, random.Random(seed))


def write_image(path: Path, height: int = 100, width: int = 200, channels: int = 3) -> None:
    ok, encoded = cv2.imencode(path.suffix, np.full((height, width, channels), 128, dtype=np.uint8))
    assert ok
    encoded.tofile(path)


def memes_config(directory: Path, static_map: dict[str, list[str]]) -> MemesConfig:
    return MemesConfig(directory, directory / "cache", "fake", 0.1, static_map)


def test_neutral_pose_shows_no_meme() -> None:
    assert selector().update(NEUTRAL) is None


def test_meme_stays_while_the_pose_lasts() -> None:
    chosen = selector()
    first = chosen.update("pose")

    assert all(chosen.update("pose") is first for _ in range(50))


def test_meme_is_drawn_again_when_the_pose_changes() -> None:
    chosen = selector()
    drawn = set()
    for _ in range(50):
        drawn.add(id(chosen.update("pose")))
        chosen.update(NEUTRAL)

    assert drawn == {id(A), id(B), id(C)}


def test_meme_disappears_when_the_pose_ends() -> None:
    chosen = selector()
    chosen.update("pose")

    assert chosen.update(NEUTRAL) is None


def test_changing_directly_between_poses_switches_the_meme() -> None:
    chosen = selector()
    chosen.update("pose")

    assert chosen.update("other") is B


def test_pose_without_memes_shows_nothing() -> None:
    assert selector().update("unmapped") is None


def test_static_map_loads_images_resized_for_the_frame(tmp_path: Path) -> None:
    write_image(tmp_path / "a.png")

    memes = load_static_memes(memes_config(tmp_path, {"pose": ["a.png"]}), (480, 640), height_ratio=0.5)

    assert memes["pose"][0].shape[:2] == (240, 480)


@pytest.mark.parametrize("name", ["a.png", "a.webp", "a.jpg", "a.jpeg", "A.PNG"])
def test_accepted_formats_load(tmp_path: Path, name: str) -> None:
    write_image(tmp_path / name)

    assert len(load_static_memes(memes_config(tmp_path, {"pose": [name]}), (100, 100), 0.5)["pose"]) == 1


def test_path_with_accents_loads(tmp_path: Path) -> None:
    write_image(tmp_path / "pensativo-ção.png")

    memes = load_static_memes(memes_config(tmp_path, {"pose": ["pensativo-ção.png"]}), (100, 100), 0.5)

    assert len(memes["pose"]) == 1


def test_alpha_channel_is_preserved(tmp_path: Path) -> None:
    write_image(tmp_path / "a.png", channels=4)

    memes = load_static_memes(memes_config(tmp_path, {"pose": ["a.png"]}), (100, 100), 0.5)

    assert memes["pose"][0].shape[2] == 4


def test_missing_file_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(MemeLoadError, match="nope.png"):
        load_static_memes(memes_config(tmp_path, {"pose": ["nope.png"]}), (100, 100), 0.5)


def test_unreadable_file_is_a_clear_error(tmp_path: Path) -> None:
    (tmp_path / "broken.png").write_bytes(b"not an image")

    with pytest.raises(MemeLoadError, match="broken.png"):
        load_static_memes(memes_config(tmp_path, {"pose": ["broken.png"]}), (100, 100), 0.5)
