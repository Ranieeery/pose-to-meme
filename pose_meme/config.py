"""Typed loader for config.yaml.

config.yaml is the single source of truth for values, so the dataclasses have no
defaults: a missing key is an error rather than a second default hidden in code.
"""

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml

T = TypeVar("T")

MODEL_VARIANTS = ("lite", "full", "heavy")
MODEL_FILE_TEMPLATE = "pose_landmarker_{variant}.task"


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
class ClassifierConfig:
    model_path: Path
    confidence_threshold: float


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
class AppConfig:
    camera: CameraConfig
    tracker: TrackerConfig
    classifier: ClassifierConfig
    stabilizer: StabilizerConfig
    poses: PosesConfig
    memes: MemesConfig
    renderer: RendererConfig
    data: DataConfig
    ui: UiConfig


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

    # The only remaining annotation is dict[str, str].
    is_text_mapping = isinstance(value, dict) and all(
        isinstance(key, str) and isinstance(text, str) for key, text in value.items()
    )
    return _require(where, value, is_text_mapping, "a mapping of strings to strings")


def _require(where: str, value: Any, is_valid: bool, expected: str) -> Any:
    if not is_valid:
        raise ConfigError(f"{where}: expected {expected}, got {value!r}")
    return value


def _validate(config: AppConfig) -> None:
    camera, tracker, poses, ui = config.camera, config.tracker, config.poses, config.ui
    checks = [
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
