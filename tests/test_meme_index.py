import json
import logging
import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from pose_meme.config import MemesConfig
from pose_meme.meme_index import (
    BUCKETS_FILE,
    MemeEmbeddings,
    MemeLoadError,
    MemeSelector,
    assign_memes,
    build_embeddings,
    group_by_pose,
    index_memes,
    list_meme_files,
    load_memes,
    to_rgb,
)

NEUTRAL = "neutro"
A = np.full((2, 2, 3), 1, dtype=np.uint8)
B = np.full((2, 2, 3), 2, dtype=np.uint8)
C = np.full((2, 2, 3), 3, dtype=np.uint8)

TEMPLATES = ["a person {}"]
PROMPTS = {"pose_x": "doing x", "pose_y": "doing y"}
# The fake encoder reads the red channel of an image as a key into this table, so
# each test image stands for a known direction in a 3D embedding space.
IMAGE_VECTORS = {
    10: (1.0, 0.0, 0.0),  # exactly pose_x
    20: (0.0, 1.0, 0.0),  # exactly pose_y
    30: (0.0, 0.0, 1.0),  # orthogonal to both poses: similarity 0
    40: (0.8, 0.6, 0.0),  # mostly pose_x, some pose_y
}
TEXT_VECTORS = {
    "a person doing x": (1.0, 0.0, 0.0),
    "a person doing y": (0.0, 1.0, 0.0),
    "a photo of doing x": (0.0, 0.0, 1.0),
}


class FakeEncoder:
    def __init__(self) -> None:
        self.images_encoded = 0
        self.texts_encoded = 0

    def encode_images(self, images: list[np.ndarray]) -> np.ndarray:
        self.images_encoded += len(images)
        return np.array([IMAGE_VECTORS[int(image[0, 0, 0])] for image in images], dtype=np.float32) * 3

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        self.texts_encoded += len(texts)
        return np.array([TEXT_VECTORS[text] for text in texts], dtype=np.float32) * 2


class CountingFactory:
    """Stands in for loading CLIP: tests check it is only called when the cache is stale."""

    def __init__(self) -> None:
        self.calls = 0
        self.encoder = FakeEncoder()

    def __call__(self) -> FakeEncoder:
        self.calls += 1
        return self.encoder


def selector(seed: int = 0) -> MemeSelector:
    return MemeSelector({"pose": [A, B, C], "other": [B]}, NEUTRAL, random.Random(seed))


def write_image(path: Path, height: int = 100, width: int = 200, channels: int = 3, red: int = 128) -> None:
    image = np.full((height, width, channels), 128, dtype=np.uint8)
    image[..., 2] = red  # OpenCV stores BGR(A), so red is channel 2
    ok, encoded = cv2.imencode(path.suffix, image)
    assert ok
    encoded.tofile(path)


def memes_config(
    directory: Path,
    model: str = "fake",
    templates: list[str] | None = None,
    min_similarity: float = 0.5,
    overrides: dict[str, str] | None = None,
) -> MemesConfig:
    return MemesConfig(
        directory, directory / "cache", model, templates or TEMPLATES, min_similarity, overrides or {}
    )


def embeddings(files: list[str], vectors: list[tuple[float, float, float]]) -> MemeEmbeddings:
    return MemeEmbeddings(
        tuple(files),
        np.array(vectors, dtype=np.float32).reshape(len(files), 3),
        tuple(PROMPTS),
        np.eye(3, dtype=np.float32)[:2],
    )


# --- selection while the app runs ---


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


# --- loading images for the overlay ---


def test_memes_load_resized_for_the_frame(tmp_path: Path) -> None:
    write_image(tmp_path / "a.png")

    memes = load_memes(tmp_path, {"pose": ["a.png"]}, (480, 640), height_ratio=0.5)

    assert memes["pose"][0].shape[:2] == (240, 480)


@pytest.mark.parametrize("name", ["a.png", "a.webp", "a.jpg", "a.jpeg", "A.PNG"])
def test_accepted_formats_load(tmp_path: Path, name: str) -> None:
    write_image(tmp_path / name)

    assert len(load_memes(tmp_path, {"pose": [name]}, (100, 100), 0.5)["pose"]) == 1


def test_path_with_accents_loads(tmp_path: Path) -> None:
    write_image(tmp_path / "pensativo-ção.png")

    assert len(load_memes(tmp_path, {"pose": ["pensativo-ção.png"]}, (100, 100), 0.5)["pose"]) == 1


def test_alpha_channel_is_preserved(tmp_path: Path) -> None:
    write_image(tmp_path / "a.png", channels=4)

    assert load_memes(tmp_path, {"pose": ["a.png"]}, (100, 100), 0.5)["pose"][0].shape[2] == 4


def test_missing_file_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(MemeLoadError, match="nope.png"):
        load_memes(tmp_path, {"pose": ["nope.png"]}, (100, 100), 0.5)


def test_unreadable_file_is_a_clear_error(tmp_path: Path) -> None:
    (tmp_path / "broken.png").write_bytes(b"not an image")

    with pytest.raises(MemeLoadError, match="broken.png"):
        load_memes(tmp_path, {"pose": ["broken.png"]}, (100, 100), 0.5)


# --- which files are memes ---


def test_only_accepted_formats_are_listed_in_name_order(tmp_path: Path) -> None:
    for name in ["b.webp", "a.PNG", "c.jpeg", "d.jpg"]:
        write_image(tmp_path / name)

    assert [path.name for path in list_meme_files(tmp_path)] == ["a.PNG", "b.webp", "c.jpeg", "d.jpg"]


def test_other_files_are_ignored_with_a_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    write_image(tmp_path / "a.png")
    (tmp_path / "notes.txt").write_text("hi", encoding="utf-8")
    (tmp_path / "sub").mkdir()

    with caplog.at_level(logging.WARNING):
        files = list_meme_files(tmp_path)

    assert [path.name for path in files] == ["a.png"]
    assert "notes.txt" in caplog.text
    assert "sub" not in caplog.text  # folders are not memes, nor something to warn about


def test_missing_meme_folder_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(MemeLoadError, match="nope"):
        list_meme_files(tmp_path / "nope")


# --- what CLIP sees ---


def test_color_order_becomes_rgb() -> None:
    bgr = np.zeros((1, 1, 3), dtype=np.uint8)
    bgr[0, 0] = (255, 0, 0)  # blue in OpenCV order

    assert to_rgb(bgr)[0, 0].tolist() == [0, 0, 255]


def test_transparent_pixels_become_white_and_opaque_ones_keep_their_color() -> None:
    bgra = np.zeros((1, 2, 4), dtype=np.uint8)
    bgra[0, 0] = (0, 0, 255, 0)  # red, fully transparent
    bgra[0, 1] = (0, 0, 255, 255)  # red, opaque

    rgb = to_rgb(bgra)

    assert rgb.shape == (1, 2, 3)
    assert rgb[0, 0].tolist() == [255, 255, 255]
    assert rgb[0, 1].tolist() == [255, 0, 0]


def test_class_embedding_is_the_mean_of_its_templates(tmp_path: Path) -> None:
    config = memes_config(tmp_path, templates=["a person {}", "a photo of {}"])

    result = build_embeddings(config, {"pose_x": "doing x"}, CountingFactory())

    expected = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)  # mean of the two template vectors, unit length
    np.testing.assert_allclose(result.texts[0], expected, atol=1e-6)


def test_embeddings_are_unit_length(tmp_path: Path) -> None:
    write_image(tmp_path / "a.png", red=40)

    result = build_embeddings(memes_config(tmp_path), PROMPTS, CountingFactory())

    np.testing.assert_allclose(np.linalg.norm(result.images, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(result.texts, axis=1), 1.0, atol=1e-6)


# --- zero-shot assignment ---


def test_each_meme_goes_to_its_most_similar_class() -> None:
    matches = assign_memes(embeddings(["x.png", "y.png"], [(1, 0, 0), (0, 1, 0)]), 0.5, {})

    assert group_by_pose(matches) == {"pose_x": ["x.png"], "pose_y": ["y.png"]}


def test_meme_below_the_minimum_similarity_is_ignored() -> None:
    matches = assign_memes(embeddings(["z.png"], [(0, 0, 1)]), 0.5, {})

    assert matches[0].pose is None
    assert group_by_pose(matches) == {}


def test_similarity_exactly_at_the_minimum_is_accepted() -> None:
    matches = assign_memes(embeddings(["xy.png"], [(0.8, 0.6, 0)]), 0.8, {})

    assert matches[0].pose == "pose_x"


def test_match_reports_the_runner_up_and_its_margin() -> None:
    match = assign_memes(embeddings(["xy.png"], [(0.8, 0.6, 0)]), 0.5, {})[0]

    assert (match.clip_pose, match.runner_up) == ("pose_x", "pose_y")
    assert match.similarity == pytest.approx(0.8)
    assert match.margin == pytest.approx(0.2)


def test_override_wins_over_clip_even_below_the_minimum() -> None:
    overrides = {"x.png": "pose_y", "z.png": "pose_x"}

    matches = assign_memes(embeddings(["x.png", "z.png"], [(1, 0, 0), (0, 0, 1)]), 0.5, overrides)

    assert group_by_pose(matches) == {"pose_y": ["x.png"], "pose_x": ["z.png"]}
    assert matches[0].clip_pose == "pose_x"  # what CLIP said stays visible
    assert all(match.overridden for match in matches)


def test_override_for_a_missing_file_is_warned(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assign_memes(embeddings(["x.png"], [(1, 0, 0)]), 0.5, {"gone.png": "pose_y"})

    assert "gone.png" in caplog.text


def test_single_class_has_no_runner_up() -> None:
    single = MemeEmbeddings(("x.png",), np.array([[1.0, 0, 0]]), ("pose_x",), np.array([[1.0, 0, 0]]))

    match = assign_memes(single, 0.5, {})[0]

    assert match.runner_up is None
    assert match.margin is None


def test_empty_folder_assigns_nothing(tmp_path: Path) -> None:
    factory = CountingFactory()

    result = build_embeddings(memes_config(tmp_path), PROMPTS, factory)

    assert assign_memes(result, 0.5, {}) == []
    assert factory.encoder.images_encoded == 0


# --- cache ---


def test_second_build_reads_the_cache_without_loading_the_model(tmp_path: Path) -> None:
    write_image(tmp_path / "x.png", red=10)
    build_embeddings(memes_config(tmp_path), PROMPTS, CountingFactory())
    factory = CountingFactory()

    result = build_embeddings(memes_config(tmp_path), PROMPTS, factory)

    assert factory.calls == 0
    assert result.files == ("x.png",)


def test_adding_a_meme_reencodes_the_images_but_not_the_prompts(tmp_path: Path) -> None:
    write_image(tmp_path / "x.png", red=10)
    build_embeddings(memes_config(tmp_path), PROMPTS, CountingFactory())
    write_image(tmp_path / "y.png", red=20)
    factory = CountingFactory()

    result = build_embeddings(memes_config(tmp_path), PROMPTS, factory)

    assert result.files == ("x.png", "y.png")
    assert factory.encoder.images_encoded == 2
    assert factory.encoder.texts_encoded == 0


def test_changing_a_meme_content_reencodes_the_images(tmp_path: Path) -> None:
    write_image(tmp_path / "x.png", red=10)
    build_embeddings(memes_config(tmp_path), PROMPTS, CountingFactory())
    write_image(tmp_path / "x.png", red=20)  # same name, new picture

    result = build_embeddings(memes_config(tmp_path), PROMPTS, CountingFactory())

    assert group_by_pose(assign_memes(result, 0.5, {})) == {"pose_y": ["x.png"]}


def test_removing_a_meme_drops_it(tmp_path: Path) -> None:
    write_image(tmp_path / "x.png", red=10)
    write_image(tmp_path / "y.png", red=20)
    build_embeddings(memes_config(tmp_path), PROMPTS, CountingFactory())
    (tmp_path / "y.png").unlink()

    assert build_embeddings(memes_config(tmp_path), PROMPTS, CountingFactory()).files == ("x.png",)


def test_changing_the_model_reencodes_everything(tmp_path: Path) -> None:
    write_image(tmp_path / "x.png", red=10)
    build_embeddings(memes_config(tmp_path), PROMPTS, CountingFactory())
    factory = CountingFactory()

    build_embeddings(memes_config(tmp_path, model="other"), PROMPTS, factory)

    assert factory.encoder.images_encoded == 1
    assert factory.encoder.texts_encoded == len(PROMPTS)


def test_changing_a_prompt_reencodes_only_the_prompts(tmp_path: Path) -> None:
    write_image(tmp_path / "x.png", red=10)
    build_embeddings(memes_config(tmp_path), PROMPTS, CountingFactory())
    factory = CountingFactory()

    result = build_embeddings(memes_config(tmp_path), {"pose_x": "doing y", "pose_y": "doing x"}, factory)

    assert factory.encoder.images_encoded == 0
    assert factory.encoder.texts_encoded == 2
    assert group_by_pose(assign_memes(result, 0.5, {})) == {"pose_y": ["x.png"]}


def test_changing_a_template_reencodes_the_prompts(tmp_path: Path) -> None:
    build_embeddings(memes_config(tmp_path), {"pose_x": "doing x"}, CountingFactory())
    factory = CountingFactory()

    build_embeddings(memes_config(tmp_path, templates=["a photo of {}"]), {"pose_x": "doing x"}, factory)

    assert factory.encoder.texts_encoded == 1


def test_corrupt_cache_is_rebuilt(tmp_path: Path) -> None:
    write_image(tmp_path / "x.png", red=10)
    build_embeddings(memes_config(tmp_path), PROMPTS, CountingFactory())
    for cache_file in (tmp_path / "cache").glob("*.npz"):
        cache_file.write_bytes(b"garbage")
    factory = CountingFactory()

    result = build_embeddings(memes_config(tmp_path), PROMPTS, factory)

    assert factory.calls == 1
    assert result.files == ("x.png",)


def test_unreadable_meme_is_a_clear_error_when_indexing(tmp_path: Path) -> None:
    (tmp_path / "broken.png").write_bytes(b"not an image")

    with pytest.raises(MemeLoadError, match="broken.png"):
        build_embeddings(memes_config(tmp_path), PROMPTS, CountingFactory())


# --- the whole indexing step ---


def test_index_writes_the_buckets_for_inspection(tmp_path: Path) -> None:
    write_image(tmp_path / "x.png", red=10)
    write_image(tmp_path / "z.png", red=30)

    matches = index_memes(memes_config(tmp_path), PROMPTS, CountingFactory())

    saved = json.loads((tmp_path / "cache" / BUCKETS_FILE).read_text(encoding="utf-8"))
    assert saved["buckets"] == {"pose_x": ["x.png"]}
    assert saved["ignored"] == ["z.png"]
    assert [match.file for match in matches] == ["x.png", "z.png"]


def test_changing_the_minimum_similarity_needs_no_model(tmp_path: Path) -> None:
    write_image(tmp_path / "xy.png", red=40)  # similarity 0.8 to pose_x
    index_memes(memes_config(tmp_path, min_similarity=0.5), PROMPTS, CountingFactory())
    factory = CountingFactory()

    matches = index_memes(memes_config(tmp_path, min_similarity=0.9), PROMPTS, factory)

    assert factory.calls == 0
    assert matches[0].pose is None
