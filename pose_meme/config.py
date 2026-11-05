"""Typed loader for config.yaml.

config.yaml is the single source of truth for values, so the dataclasses have no
defaults: a missing key is an error rather than a second default hidden in code.
"""

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml

T = TypeVar("T")

MODEL_VARIANTS = ("lite", "full", "heavy")
MODEL_FILE_TEMPLATE = "pose_landmarker_{variant}.task"
LONG_FINGERS = 4  # all but the thumb


class ConfigError(ValueError):
    """config.yaml is missing, malformed or holds an invalid value."""


@dataclass(frozen=True)
class CameraConfig:
    index: int
    width: int
    height: int
    mirror: bool


@dataclass(frozen=True)
class TrackerConfig:
    model_variant: str
    model_dir: Path
    num_poses: int
    min_visibility: float

    @property
    def model_path(self) -> Path:
        return self.model_dir / MODEL_FILE_TEMPLATE.format(variant=self.model_variant)


@dataclass(frozen=True)
class HandTrackerConfig:
    model_path: Path
    num_hands: int
    min_detection_confidence: float
    min_presence_confidence: float
    min_tracking_confidence: float
    hold_frames: int


@dataclass(frozen=True)
class ClassifierConfig:
    model_path: Path
    confidence_threshold: float


@dataclass(frozen=True)
class RulesConfig:
    """RuleClassifier thresholds; distances are in body units (shoulder widths)."""

    mouth_radius: float
    mouth_radius_without_fingers: float
    self_point_min_radius: float
    self_point_max_radius: float
    pointing_min_horizontal_ratio: float
    finger_extended_ratio: float
    open_hand_min_fingers: int
    open_hand_min_spread: float
    forward_hand_min_palm: float
    forward_hands_min_gap: float
    chest_half_width: float
    chest_max_depth: float


@dataclass(frozen=True)
class StabilizerConfig:
    window_size: int
    min_ratio: float


@dataclass(frozen=True)
class PosesConfig:
    neutral_class: str
    prompts: dict[str, str]


@dataclass(frozen=True)
class MemesConfig:
    dir: Path
    cache_dir: Path
    clip_model: str
    min_similarity: float
    static_map: dict[str, list[str]]


@dataclass(frozen=True)
class RendererConfig:
    meme_height_ratio: float
    show_skeleton: bool


@dataclass(frozen=True)
class DataConfig:
    samples_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class UiConfig:
    window_title: str
    quit_key: str
    skeleton_key: str
    fps_window: int


@dataclass(frozen=True)
class CollectConfig:
    samples_per_class: int
    sample_every: int
    countdown_seconds: float
    record_key: str
    class_keys: str


@dataclass(frozen=True)
class TrainingConfig:
    hidden_layers: list[int]
    alpha: float
    max_iter: int
    knn_neighbors: int
    random_seed: int
    min_sessions: int


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig
    tracker: TrackerConfig
    hand_tracker: HandTrackerConfig
    classifier: ClassifierConfig
    rules: RulesConfig
    stabilizer: StabilizerConfig
    poses: PosesConfig
    memes: MemesConfig
    renderer: RendererConfig
    data: DataConfig
    ui: UiConfig
    collect: CollectConfig
    training: TrainingConfig


def load_config(path: Path | str) -> AppConfig:
    """Read and validate a config file; relative paths resolve against its folder."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML in {path}: {error}") from error

    config = _parse_dataclass(AppConfig, raw, where="", base_dir=path.resolve().parent)
    _validate(config)
    return config


def _parse_dataclass(cls: type[T], raw: Any, where: str, base_dir: Path) -> T:
    if not isinstance(raw, dict):
        raise ConfigError(f"{where or 'config file'}: expected a mapping, got {type(raw).__name__}")

    prefix = f"{where}." if where else ""
    names = {field.name for field in fields(cls)}  # type: ignore[arg-type]
    if unknown := raw.keys() - names:
        raise ConfigError(f"unknown keys: {sorted(f'{prefix}{key}' for key in unknown)}")
    if missing := names - raw.keys():
        raise ConfigError(f"missing keys: {sorted(f'{prefix}{key}' for key in missing)}")

    hints = get_type_hints(cls)
    return cls(**{name: _parse_value(hints[name], raw[name], f"{prefix}{name}", base_dir) for name in names})


def _parse_value(expected: Any, value: Any, where: str, base_dir: Path) -> Any:
    if is_dataclass(expected):
        return _parse_dataclass(expected, value, where, base_dir)
    if expected is Path:
        return base_dir / _require(where, value, isinstance(value, str), "a path string")
    if expected is float:
        is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
        return float(_require(where, value, is_number, "a number"))
    if expected is int:
        return _require(where, value, isinstance(value, int) and not isinstance(value, bool), "an integer")
    if expected is bool:
        return _require(where, value, isinstance(value, bool), "true or false")
    if expected is str:
        return _require(where, value, isinstance(value, str), "a string")

    if get_origin(expected) is list:
        is_int_list = isinstance(value, list) and all(
            isinstance(item, int) and not isinstance(item, bool) for item in value
        )
        return _require(where, value, is_int_list, "a list of integers")

    # The remaining annotations are dict[str, str] and dict[str, list[str]].
    if get_args(expected)[1] is str:
        is_text_mapping = isinstance(value, dict) and all(
            isinstance(key, str) and isinstance(text, str) for key, text in value.items()
        )
        return _require(where, value, is_text_mapping, "a mapping of strings to strings")
    is_list_mapping = isinstance(value, dict) and all(
        isinstance(key, str) and isinstance(items, list) and all(isinstance(item, str) for item in items)
        for key, items in value.items()
    )
    return _require(where, value, is_list_mapping, "a mapping of strings to lists of strings")


def _require(where: str, value: Any, is_valid: bool, expected: str) -> Any:
    if not is_valid:
        raise ConfigError(f"{where}: expected {expected}, got {value!r}")
    return value


def _validate(config: AppConfig) -> None:
    camera, tracker, poses, ui = config.camera, config.tracker, config.poses, config.ui
    rules, static_map, hand = config.rules, config.memes.static_map, config.hand_tracker
    collect, training = config.collect, config.training
    checks = [
        (
            "collect.class_keys",
            collect.class_keys,
            len(collect.class_keys) >= len(poses.prompts) + 1,
            "one key per class, including the neutral class",
        ),
        (
            "collect.class_keys",
            collect.class_keys,
            len(set(collect.class_keys)) == len(collect.class_keys),
            "no repeated keys",
        ),
        (
            "collect.class_keys",
            collect.class_keys,
            not {ui.quit_key, ui.skeleton_key} & set(collect.class_keys),
            "different from the ui keys",
        ),
        ("collect.samples_per_class", collect.samples_per_class, collect.samples_per_class >= 1, ">= 1"),
        ("collect.sample_every", collect.sample_every, collect.sample_every >= 1, ">= 1"),
        ("collect.countdown_seconds", collect.countdown_seconds, collect.countdown_seconds >= 0, ">= 0"),
        ("collect.record_key", collect.record_key, len(collect.record_key) == 1, "a single character"),
        (
            "collect.record_key",
            collect.record_key,
            collect.record_key not in collect.class_keys,
            "not one of collect.class_keys",
        ),
        (
            "collect.record_key",
            collect.record_key,
            collect.record_key not in (ui.quit_key, ui.skeleton_key),
            "different from the ui keys",
        ),
        (
            "training.hidden_layers",
            training.hidden_layers,
            bool(training.hidden_layers) and all(size > 0 for size in training.hidden_layers),
            "a non-empty list of positive sizes",
        ),
        ("training.alpha", training.alpha, training.alpha >= 0, ">= 0"),
        ("training.max_iter", training.max_iter, training.max_iter >= 1, ">= 1"),
        ("training.knn_neighbors", training.knn_neighbors, training.knn_neighbors >= 1, ">= 1"),
        ("training.min_sessions", training.min_sessions, training.min_sessions >= 2, ">= 2"),
        ("hand_tracker.num_hands", hand.num_hands, hand.num_hands >= 1, ">= 1"),
        (
            "hand_tracker.min_detection_confidence",
            hand.min_detection_confidence,
            0 <= hand.min_detection_confidence <= 1,
            "in [0, 1]",
        ),
        (
            "hand_tracker.min_presence_confidence",
            hand.min_presence_confidence,
            0 <= hand.min_presence_confidence <= 1,
            "in [0, 1]",
        ),
        (
            "hand_tracker.min_tracking_confidence",
            hand.min_tracking_confidence,
            0 <= hand.min_tracking_confidence <= 1,
            "in [0, 1]",
        ),
        ("hand_tracker.hold_frames", hand.hold_frames, hand.hold_frames >= 0, ">= 0"),
        ("rules.mouth_radius", rules.mouth_radius, rules.mouth_radius > 0, "> 0"),
        (
            "rules.mouth_radius_without_fingers",
            rules.mouth_radius_without_fingers,
            rules.mouth_radius_without_fingers > 0,
            "> 0",
        ),
        ("rules.self_point_min_radius", rules.self_point_min_radius, rules.self_point_min_radius >= 0, ">= 0"),
        (
            "rules.self_point_max_radius",
            rules.self_point_max_radius,
            rules.self_point_max_radius > rules.self_point_min_radius,
            "above rules.self_point_min_radius",
        ),
        (
            "rules.pointing_min_horizontal_ratio",
            rules.pointing_min_horizontal_ratio,
            rules.pointing_min_horizontal_ratio > 0,
            "> 0",
        ),
        ("rules.finger_extended_ratio", rules.finger_extended_ratio, rules.finger_extended_ratio >= 1, ">= 1"),
        (
            "rules.open_hand_min_fingers",
            rules.open_hand_min_fingers,
            1 <= rules.open_hand_min_fingers <= LONG_FINGERS,
            f"in [1, {LONG_FINGERS}]",
        ),
        ("rules.open_hand_min_spread", rules.open_hand_min_spread, rules.open_hand_min_spread > 0, "> 0"),
        ("rules.forward_hand_min_palm", rules.forward_hand_min_palm, rules.forward_hand_min_palm > 0, "> 0"),
        ("rules.forward_hands_min_gap", rules.forward_hands_min_gap, rules.forward_hands_min_gap > 0, "> 0"),
        ("rules.chest_half_width", rules.chest_half_width, rules.chest_half_width > 0, "> 0"),
        ("rules.chest_max_depth", rules.chest_max_depth, rules.chest_max_depth > 0, "> 0"),
        ("memes.static_map", static_map, static_map.keys() <= poses.prompts.keys(), "only classes in poses.prompts"),
        ("memes.static_map", static_map, all(static_map.values()), "at least one file per class"),
        ("ui.quit_key", ui.quit_key, len(ui.quit_key) == 1, "a single character"),
        ("ui.skeleton_key", ui.skeleton_key, len(ui.skeleton_key) == 1, "a single character"),
        ("ui.skeleton_key", ui.skeleton_key, ui.skeleton_key != ui.quit_key, "different from ui.quit_key"),
        ("ui.fps_window", ui.fps_window, ui.fps_window >= 2, ">= 2"),
        ("camera.index", camera.index, camera.index >= 0, ">= 0"),
        ("camera.width", camera.width, camera.width > 0, "> 0"),
        ("camera.height", camera.height, camera.height > 0, "> 0"),
        ("tracker.model_variant", tracker.model_variant, tracker.model_variant in MODEL_VARIANTS, str(MODEL_VARIANTS)),
        ("tracker.num_poses", tracker.num_poses, tracker.num_poses >= 1, ">= 1"),
        ("tracker.min_visibility", tracker.min_visibility, 0 <= tracker.min_visibility <= 1, "in [0, 1]"),
        (
            "classifier.confidence_threshold",
            config.classifier.confidence_threshold,
            0 <= config.classifier.confidence_threshold <= 1,
            "in [0, 1]",
        ),
        ("stabilizer.window_size", config.stabilizer.window_size, config.stabilizer.window_size >= 1, ">= 1"),
        ("stabilizer.min_ratio", config.stabilizer.min_ratio, 0 < config.stabilizer.min_ratio <= 1, "in (0, 1]"),
        ("poses.prompts", poses.prompts, bool(poses.prompts), "not empty"),
        (
            "poses.neutral_class",
            poses.neutral_class,
            poses.neutral_class not in poses.prompts,
            "absent from poses.prompts",
        ),
        ("memes.min_similarity", config.memes.min_similarity, -1 <= config.memes.min_similarity <= 1, "in [-1, 1]"),
        (
            "renderer.meme_height_ratio",
            config.renderer.meme_height_ratio,
            0 < config.renderer.meme_height_ratio <= 1,
            "in (0, 1]",
        ),
    ]
    for where, value, is_valid, expected in checks:
        _require(where, value, is_valid, expected)
