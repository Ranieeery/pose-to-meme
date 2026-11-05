import copy
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
import yaml

from pose_meme.config import AppConfig, ConfigError, load_config

REPO_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"

# Deliberately different from config.yaml, so these tests don't break when a
# default is tuned. The repository file has its own test below.
VALID: dict[str, Any] = {
    "camera": {"index": 1, "width": 320, "height": 240, "mirror": False},
    "tracker": {"model_variant": "lite", "model_dir": "m", "num_poses": 1, "min_visibility": 0.25},
    "hand_tracker": {
        "model_path": "m/hand.task",
        "num_hands": 1,
        "min_detection_confidence": 0.6,
        "min_presence_confidence": 0.6,
        "min_tracking_confidence": 0.6,
        "hold_frames": 1,
    },
    "classifier": {"model_path": "m/clf.joblib", "confidence_threshold": 0.5},
    "rules": {
        "mouth_radius": 0.1,
        "mouth_radius_without_fingers": 0.15,
        "self_point_min_radius": 0.1,
        "self_point_max_radius": 0.2,
        "pointing_min_horizontal_ratio": 2.0,
        "finger_extended_ratio": 1.5,
        "open_hand_min_fingers": 3,
        "open_hand_min_spread": 1.5,
        "forward_hand_min_palm": 0.3,
        "forward_hands_min_gap": 0.5,
        "chest_half_width": 0.4,
        "chest_max_depth": 0.5,
    },
    "stabilizer": {"window_size": 4, "min_ratio": 0.5},
    "poses": {"neutral_class": "neutro", "prompts": {"maos_na_cabeca": "hands on head"}},
    "memes": {
        "dir": "imgs",
        "cache_dir": "cache",
        "clip_model": "fake",
        "min_similarity": 0.1,
        "static_map": {"maos_na_cabeca": ["a.png", "b.png"]},
    },
    "renderer": {"meme_height_ratio": 0.5, "show_skeleton": False},
    "data": {"samples_dir": "samples", "reports_dir": "out"},
    "ui": {"window_title": "test", "quit_key": "x", "skeleton_key": "y", "fps_window": 5},
    "collect": {
        "samples_per_class": 10,
        "sample_every": 1,
        "countdown_seconds": 0.5,
        "record_key": "r",
        "class_keys": "0123",
    },
    "training": {
        "hidden_layers": [8, 4],
        "alpha": 0.01,
        "max_iter": 50,
        "knn_neighbors": 3,
        "random_seed": 1,
        "min_sessions": 3,
    },
}


def write_config(tmp_path: Path, data: Any) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def with_value(section: str, key: str, value: Any) -> dict[str, Any]:
    data = copy.deepcopy(VALID)
    data[section][key] = value
    return data


def test_repo_config_has_the_documented_defaults() -> None:
    config = load_config(REPO_CONFIG)

    assert (config.camera.width, config.camera.height) == (640, 480)
    assert config.camera.mirror is True
    assert config.tracker.model_variant == "full"
    assert config.tracker.num_poses == 1
    assert config.classifier.confidence_threshold == 0.8
    assert config.stabilizer.window_size == 8
    assert config.stabilizer.min_ratio == 0.7
    assert config.poses.neutral_class == "neutro"
    assert set(config.poses.prompts) == {
        "mao_fechada_na_boca",
        "dedo_na_boca",
        "mao_aberta_na_boca",
        "maos_na_boca",
        "dedo_levantado",
        "apontando_para_si",
        "sinal_de_rock",
        "maos_abertas_para_frente",
        "garras_para_frente",
        "mao_no_peito",
    }
    assert config.memes.clip_model == "clip-ViT-B-32"
    assert config.renderer.meme_height_ratio == 0.4
    assert config.ui.quit_key == "q"
    assert config.training.min_sessions >= 2
    assert all(isinstance(size, int) for size in config.training.hidden_layers)
    assert config.hand_tracker.num_hands == 2
    assert config.hand_tracker.model_path == REPO_CONFIG.parent / "models" / "hand_landmarker.task"
    assert set(config.memes.static_map) == set(config.poses.prompts)
    mapped = {name for names in config.memes.static_map.values() for name in names}
    assert all((config.memes.dir / name).is_file() for name in mapped)


def test_values_are_loaded_into_typed_sections(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, VALID))

    assert isinstance(config, AppConfig)
    assert config.camera.index == 1
    assert config.camera.mirror is False
    assert config.tracker.min_visibility == 0.25
    assert config.poses.prompts == {"maos_na_cabeca": "hands on head"}


def test_relative_paths_resolve_against_the_config_folder(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, VALID))

    assert config.tracker.model_dir == tmp_path / "m"
    assert config.classifier.model_path == tmp_path / "m" / "clf.joblib"
    assert config.memes.dir == tmp_path / "imgs"
    assert config.data.reports_dir == tmp_path / "out"


def test_tracker_model_path_follows_the_configured_variant(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, VALID))

    assert config.tracker.model_path == tmp_path / "m" / "pose_landmarker_lite.task"


def test_absolute_paths_are_kept(tmp_path: Path) -> None:
    absolute = tmp_path / "elsewhere"
    config = load_config(write_config(tmp_path, with_value("memes", "dir", str(absolute))))

    assert config.memes.dir == absolute


def test_config_is_immutable(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, VALID))

    with pytest.raises(FrozenInstanceError):
        config.camera.width = 1  # type: ignore[misc]


def test_int_is_accepted_where_a_float_is_expected(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, with_value("stabilizer", "min_ratio", 1)))

    assert config.stabilizer.min_ratio == 1.0
    assert isinstance(config.stabilizer.min_ratio, float)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_empty_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ConfigError, match="mapping"):
        load_config(path)


def test_malformed_yaml_raises(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("camera: [unclosed", encoding="utf-8")

    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(path)


def test_missing_key_is_named_with_its_full_path(tmp_path: Path) -> None:
    data = copy.deepcopy(VALID)
    del data["camera"]["width"]

    with pytest.raises(ConfigError, match=r"camera\.width"):
        load_config(write_config(tmp_path, data))


def test_missing_section_is_named(tmp_path: Path) -> None:
    data = copy.deepcopy(VALID)
    del data["renderer"]

    with pytest.raises(ConfigError, match="renderer"):
        load_config(write_config(tmp_path, data))


def test_unknown_key_is_rejected_to_catch_typos(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"stabilizer\.window"):
        load_config(write_config(tmp_path, with_value("stabilizer", "window", 8)))


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("camera", "width", "640"),
        ("camera", "width", 640.5),
        ("camera", "width", True),  # bool is an int in Python, but never what was meant
        ("camera", "mirror", 1),
        ("stabilizer", "min_ratio", "high"),
        ("stabilizer", "min_ratio", True),
        ("memes", "dir", 3),
        ("memes", "clip_model", 3),
        ("poses", "prompts", "hands on head"),
        ("poses", "prompts", {"maos_na_cabeca": 1}),
        ("memes", "static_map", {"maos_na_cabeca": "a.png"}),
        ("memes", "static_map", {"maos_na_cabeca": [1]}),
        ("training", "hidden_layers", 8),
        ("training", "hidden_layers", ["8"]),
        ("training", "hidden_layers", [True]),
    ],
)
def test_wrong_type_is_rejected(tmp_path: Path, section: str, key: str, value: Any) -> None:
    with pytest.raises(ConfigError, match=rf"{section}\.{key}"):
        load_config(write_config(tmp_path, with_value(section, key, value)))


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("camera", "index", -1),
        ("camera", "width", 0),
        ("camera", "height", -480),
        ("tracker", "model_variant", "medium"),
        ("tracker", "num_poses", 0),
        ("tracker", "min_visibility", 1.5),
        ("classifier", "confidence_threshold", -0.1),
        ("stabilizer", "window_size", 0),
        ("stabilizer", "min_ratio", 0),
        ("stabilizer", "min_ratio", 1.1),
        ("memes", "min_similarity", 1.5),
        ("memes", "min_similarity", -1.5),
        ("renderer", "meme_height_ratio", 0),
        ("renderer", "meme_height_ratio", 1.1),
        ("hand_tracker", "num_hands", 0),
        ("hand_tracker", "min_detection_confidence", 1.5),
        ("hand_tracker", "min_presence_confidence", -0.1),
        ("hand_tracker", "min_tracking_confidence", 1.5),
        ("hand_tracker", "hold_frames", -1),
        ("rules", "mouth_radius", 0),
        ("rules", "mouth_radius_without_fingers", 0),
        ("rules", "self_point_min_radius", -1),
        ("rules", "self_point_max_radius", 0.05),  # below self_point_min_radius
        ("rules", "pointing_min_horizontal_ratio", 0),
        ("rules", "finger_extended_ratio", 0.5),
        ("rules", "open_hand_min_fingers", 0),
        ("rules", "open_hand_min_fingers", 5),
        ("rules", "open_hand_min_spread", 0),
        ("rules", "forward_hand_min_palm", 0),
        ("rules", "forward_hands_min_gap", 0),
        ("rules", "chest_half_width", 0),
        ("rules", "chest_max_depth", 0),
        ("memes", "static_map", {"unknown_pose": ["a.png"]}),
        ("memes", "static_map", {"maos_na_cabeca": []}),
        ("collect", "samples_per_class", 0),
        ("collect", "sample_every", 0),
        ("collect", "countdown_seconds", -1),
        ("collect", "record_key", "ab"),
        ("collect", "record_key", "3"),  # this key chooses a class
        ("collect", "record_key", "x"),  # same key as ui.quit_key
        ("collect", "record_key", "y"),  # same key as ui.skeleton_key
        ("collect", "class_keys", "0"),  # fewer keys than classes
        ("collect", "class_keys", "00"),  # the same key for two classes
        ("collect", "class_keys", "0x"),  # same key as ui.quit_key
        ("collect", "class_keys", "0y"),  # same key as ui.skeleton_key
        ("training", "hidden_layers", []),
        ("training", "hidden_layers", [0]),
        ("training", "alpha", -1),
        ("training", "max_iter", 0),
        ("training", "knn_neighbors", 0),
        ("training", "min_sessions", 1),
        ("ui", "quit_key", "quit"),
        ("ui", "quit_key", ""),
        ("ui", "skeleton_key", "ab"),
        ("ui", "skeleton_key", "x"),  # same key as quit_key
        ("ui", "fps_window", 1),
    ],
)
def test_out_of_range_value_is_rejected(tmp_path: Path, section: str, key: str, value: Any) -> None:
    with pytest.raises(ConfigError, match=rf"{section}\.{key}"):
        load_config(write_config(tmp_path, with_value(section, key, value)))


def test_prompts_cannot_be_empty(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"poses\.prompts"):
        load_config(write_config(tmp_path, with_value("poses", "prompts", {})))


def test_more_classes_than_collect_keys_are_rejected(tmp_path: Path) -> None:
    prompts = {f"pose_{index}": "prompt" for index in range(4)}  # plus the neutral class: 5 for 4 keys
    data = with_value("poses", "prompts", prompts)
    data["memes"]["static_map"] = {}

    with pytest.raises(ConfigError, match=r"collect\.class_keys"):
        load_config(write_config(tmp_path, data))


def test_repo_config_has_a_collect_key_for_every_class() -> None:
    config = load_config(REPO_CONFIG)

    assert len(config.collect.class_keys) >= len(config.poses.prompts) + 1


def test_neutral_class_cannot_have_a_prompt(tmp_path: Path) -> None:
    data = with_value("poses", "prompts", {"neutro": "a person standing", "maos_na_cabeca": "hands on head"})

    with pytest.raises(ConfigError, match=r"poses\.neutral_class"):
        load_config(write_config(tmp_path, data))
