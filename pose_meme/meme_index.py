"""Which meme goes with which pose: CLIP embeddings, matched zero-shot to the class prompts.

CLIP only runs here, while indexing, never in the frame loop. Embeddings are cached in
`cache_dir`, keyed by the folder content and the model, so a new meme is enough to
trigger a new index, and an unchanged folder starts without loading the model.
"""

import hashlib
import json
import logging
import random
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from pose_meme.config import PROMPT_PLACEHOLDER, MemesConfig
from pose_meme.renderer import fit_overlay

logger = logging.getLogger(__name__)

ACCEPTED_SUFFIXES = frozenset({".png", ".webp", ".jpg", ".jpeg"})
IMAGE_CACHE = "meme_embeddings.npz"
TEXT_CACHE = "prompt_embeddings.npz"
BUCKETS_FILE = "buckets.json"
_WHITE = 255.0
_CACHE_ERRORS = (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile)


class MemeLoadError(RuntimeError):
    """A meme file or folder is missing, can't be decoded, or the model can't be loaded."""


class Encoder(Protocol):
    """Maps RGB images and texts into the same embedding space (CLIP in the app)."""

    def encode_images(self, images: list[np.ndarray]) -> np.ndarray: ...

    def encode_texts(self, texts: list[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class MemeEmbeddings:
    files: tuple[str, ...]
    images: np.ndarray  # one unit vector per file
    classes: tuple[str, ...]
    texts: np.ndarray  # one unit vector per class: the mean of its templates


@dataclass(frozen=True)
class MemeMatch:
    file: str
    pose: str | None  # what the app shows it for; None when ignored
    clip_pose: str  # CLIP's best class, kept even when overridden or ignored
    similarity: float  # cosine similarity to clip_pose
    runner_up: str | None
    margin: float | None  # similarity gap to the runner-up: small means a coin toss
    overridden: bool


def _read_image(path: Path) -> np.ndarray:
    # cv2.imread can't open non-ASCII paths on Windows, so decode the bytes instead.
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError as error:
        raise MemeLoadError(f"could not read meme file: {path}") from error
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED) if data.size else None
    if image is None:
        raise MemeLoadError(f"could not decode meme file: {path}")
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image


def list_meme_files(directory: Path) -> list[Path]:
    """Images in the folder, in name order; anything else is skipped with a warning."""
    if not directory.is_dir():
        raise MemeLoadError(f"meme folder not found: {directory}")
    files = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        if path.suffix.lower() in ACCEPTED_SUFFIXES:
            files.append(path)
        else:
            logger.warning("ignoring %s: not one of %s", path.name, sorted(ACCEPTED_SUFFIXES))
    return files


def to_rgb(image: np.ndarray) -> np.ndarray:
    """OpenCV BGR(A) to the RGB CLIP expects, with transparency flattened onto white."""
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    alpha = image[..., 3:].astype(np.float32) / _WHITE
    flat = image[..., :3].astype(np.float32) * alpha + _WHITE * (1 - alpha)
    return cv2.cvtColor(np.round(flat).astype(np.uint8), cv2.COLOR_BGR2RGB)


def _unit(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def _hash(parts: Sequence[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "little"))  # length prefix: no two inputs collide by concatenation
        digest.update(part)
    return digest.hexdigest()


def _images_key(files: Sequence[Path], model: str) -> str:
    parts = [model.encode()]
    for path in files:
        parts += [path.name.encode(), path.read_bytes()]
    return _hash(parts)


def _texts_key(prompts: Mapping[str, str], templates: Sequence[str], model: str) -> str:
    return _hash([json.dumps([model, list(templates), list(prompts.items())]).encode()])


def _load_cache(path: Path, key: str) -> tuple[tuple[str, ...], np.ndarray] | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["key"]) != key:
                return None
            return tuple(str(name) for name in data["names"]), data["embeddings"]
    except _CACHE_ERRORS:
        return None


def _save_cache(path: Path, key: str, names: Sequence[str], embeddings: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        np.savez(file, key=np.array(key), names=np.array(list(names), dtype=str), embeddings=embeddings)


def build_embeddings(
    config: MemesConfig,
    prompts: Mapping[str, str],
    encoder_factory: Callable[[], Encoder],
) -> MemeEmbeddings:
    """Embeddings for every meme and class prompt, loading the model only for what the cache lacks."""
    files = list_meme_files(config.dir)
    classes = tuple(prompts)
    encoder: Encoder | None = None

    def get_encoder() -> Encoder:
        nonlocal encoder
        if encoder is None:
            encoder = encoder_factory()
        return encoder

    text_path = config.cache_dir / TEXT_CACHE
    text_key = _texts_key(prompts, config.prompt_templates, config.clip_model)
    cached_texts = _load_cache(text_path, text_key)
    if cached_texts is not None and cached_texts[0] == classes:
        texts = cached_texts[1]
    else:
        templates = config.prompt_templates
        sentences = [template.replace(PROMPT_PLACEHOLDER, prompts[name]) for name in classes for template in templates]
        encoded = _unit(get_encoder().encode_texts(sentences)).reshape(len(classes), len(templates), -1)
        texts = _unit(encoded.mean(axis=1))
        _save_cache(text_path, text_key, classes, texts)

    image_path = config.cache_dir / IMAGE_CACHE
    image_key = _images_key(files, config.clip_model)
    names = tuple(path.name for path in files)
    cached_images = _load_cache(image_path, image_key)
    if cached_images is not None and cached_images[0] == names:
        images = cached_images[1]
    elif files:
        pictures = [to_rgb(_read_image(path)) for path in files]  # decode first: a broken file fails fast
        images = _unit(get_encoder().encode_images(pictures))
        _save_cache(image_path, image_key, names, images)
    else:
        images = np.empty((0, texts.shape[1]), dtype=np.float32)
        _save_cache(image_path, image_key, names, images)

    return MemeEmbeddings(names, images, classes, texts)


def assign_memes(
    embeddings: MemeEmbeddings,
    min_similarity: float,
    overrides: Mapping[str, str],
) -> list[MemeMatch]:
    """Each meme goes to its most similar class, unless below min_similarity or overridden."""
    for name in sorted(overrides.keys() - set(embeddings.files)):
        logger.warning("override for %s ignored: no such meme", name)

    similarities = embeddings.images @ embeddings.texts.T
    matches = []
    for name, row in zip(embeddings.files, similarities):
        order = np.argsort(row)[::-1]
        best = embeddings.classes[order[0]]
        similarity = float(row[order[0]])
        has_runner_up = len(order) > 1
        clip_choice = best if similarity >= min_similarity else None
        matches.append(
            MemeMatch(
                file=name,
                pose=overrides.get(name, clip_choice),
                clip_pose=best,
                similarity=similarity,
                runner_up=embeddings.classes[order[1]] if has_runner_up else None,
                margin=similarity - float(row[order[1]]) if has_runner_up else None,
                overridden=name in overrides,
            )
        )
    return matches


def group_by_pose(matches: Sequence[MemeMatch]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {}
    for match in matches:
        if match.pose is not None:
            buckets.setdefault(match.pose, []).append(match.file)
    return buckets


def index_memes(
    config: MemesConfig,
    prompts: Mapping[str, str],
    encoder_factory: Callable[[], Encoder],
) -> list[MemeMatch]:
    """Embed (or read from cache), assign, and save the buckets to cache_dir for inspection."""
    matches = assign_memes(build_embeddings(config, prompts, encoder_factory), config.min_similarity, config.overrides)
    report = {
        "model": config.clip_model,
        "min_similarity": config.min_similarity,
        "buckets": group_by_pose(matches),
        "ignored": [match.file for match in matches if match.pose is None],
        "matches": [asdict(match) for match in matches],
    }
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    (config.cache_dir / BUCKETS_FILE).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return matches


class ClipEncoder:
    """CLIP through sentence-transformers. Imported lazily: torch takes seconds to load,
    and the app only needs it when the cache is stale."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        try:
            self._model = SentenceTransformer(model_name)
        except OSError as error:
            raise MemeLoadError(f"could not load CLIP model {model_name!r}: {error}") from error

    def encode_images(self, images: list[np.ndarray]) -> np.ndarray:
        from PIL import Image

        return np.asarray(self._model.encode([Image.fromarray(image) for image in images]))

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts))


def load_memes(
    directory: Path,
    buckets: Mapping[str, Sequence[str]],
    frame_size: tuple[int, int],
    height_ratio: float,
) -> dict[str, list[np.ndarray]]:
    """Load and resize every assigned meme once, so the frame loop never touches the disk."""
    return {
        pose: [fit_overlay(_read_image(directory / name), frame_size, height_ratio) for name in names]
        for pose, names in buckets.items()
    }


class MemeSelector:
    """Draws a meme when the pose changes and keeps it for as long as the pose lasts."""

    def __init__(self, memes: dict[str, list[np.ndarray]], neutral_class: str, rng: random.Random) -> None:
        self._memes = memes
        self._neutral = neutral_class
        self._rng = rng
        self._pose: str | None = None
        self._current: np.ndarray | None = None

    def update(self, pose: str) -> np.ndarray | None:
        if pose != self._pose:
            self._pose = pose
            options = self._memes.get(pose) if pose != self._neutral else None
            self._current = self._rng.choice(options) if options else None
        return self._current
