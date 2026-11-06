"""Builds docs/demo.gif from a recorded session, with no camera and no face on screen.

Run from the project root: python -m scripts.demo_gif

Each frame is a recorded sample: its feature vector is turned back into a stick figure,
then goes through the app's own steps (classifier, stabilizer, meme selector, panel).
The MLP is trained here without the demo session, so the GIF shows predictions on a
session the model never saw, like the metrics in the README.
"""

import random
import sys
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pose_meme.classifier import MLClassifier, classify_frame
from pose_meme.config import ConfigError, load_config
from pose_meme.dataset import DatasetError, Samples, read_samples
from pose_meme.features import POSE_POINTS, SIDES, FeatureView
from pose_meme.meme_index import ClipEncoder, MemeLoadError, MemeSelector, group_by_pose, index_memes, load_memes
from pose_meme.renderer import ViewComposer, draw_body
from pose_meme.stabilizer import PoseStabilizer
from pose_meme.tracker import HAND_CONNECTIONS, POSE_CONNECTIONS, Landmark
from scripts.train import build_mlp, training_data

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "demo.gif"

# What the demo shows; purely presentational, so not in config.yaml.
DEMO_SESSION = "sessao_04"
DEMO_POSES = (
    "neutro",
    "sinal_de_rock",
    "maos_unidas",
    "mao_fechada_na_boca",
    "dedo_levantado",
    "maos_na_cabeca",
    "mao_no_ombro",
    "neutro",
)
FIRST_FRAME = 60  # into each recorded run, past the start of the pose
FRAMES_PER_POSE = 26
FRAME_MS = 66  # samples were taken every other frame of a 30 FPS camera
CAMERA_SIZE = (360, 480)  # height, width of the stick-figure area
PANEL_WIDTH_RATIO = 0.9  # wide enough for the longest class name and its confidence
BACKGROUND = (30, 30, 30)
HEAD_COLOR = (200, 200, 200)
HEAD_RADIUS_SHOULDERS = 0.35  # head size, in shoulder widths
SHOULDER_WIDTH = 0.27  # of the frame height
BODY_CENTER = (0.5, 0.45)  # shoulder center, as shares of the frame width and height
MISSING = Landmark(0.0, 0.0, 0.0)
VISIBLE = 1.0
POSE_LANDMARKS = 33


def body_landmarks(features: np.ndarray, aspect_ratio: float) -> tuple[list[Landmark], list[list[Landmark]]]:
    """Normalized landmarks that extract_features turns back into these features.

    Features are in body units (origin at the shoulder center, unit = shoulder width),
    so the figure is drawn at a fixed place and size; only the pose itself is recorded.
    """
    view = FeatureView(features)
    center_x, center_y = BODY_CENTER[0] * aspect_ratio, BODY_CENTER[1]

    def to_frame(x: float, y: float) -> tuple[float, float]:
        return (center_x + x * SHOULDER_WIDTH) / aspect_ratio, center_y + y * SHOULDER_WIDTH

    pose = [MISSING] * POSE_LANDMARKS
    for name, index in POSE_POINTS.items():
        point = view.point(name)
        if point is not None:
            pose[index] = Landmark(*to_frame(*point), VISIBLE)
    hands = []
    for side in SIDES:
        hand = view.hand(side)
        if hand is not None:
            hands.append([Landmark(*to_frame(x, y), VISIBLE) for x, y in hand])
    return pose, hands


def demo_indices(samples: Samples, session: str, labels: Sequence[str], start: int, count: int) -> np.ndarray:
    """Consecutive recorded frames of each pose, in the order given."""
    picked = []
    for label in labels:
        rows = np.flatnonzero((samples.sessions == session) & (samples.labels == label))
        if len(rows) < start + count:
            raise ValueError(f"{session} has {len(rows)} frames of {label}, need {start + count}")
        picked.append(rows[start : start + count])
    return np.concatenate(picked)


def _draw_head(frame: np.ndarray, pose: list[Landmark]) -> None:
    nose = pose[POSE_POINTS["nose"]]
    if nose.visibility <= 0:
        return
    height, width = frame.shape[:2]
    radius = round(HEAD_RADIUS_SHOULDERS * SHOULDER_WIDTH * height)
    cv2.circle(frame, (round(nose.x * (width - 1)), round(nose.y * (height - 1))), radius, HEAD_COLOR, 2, cv2.LINE_AA)


def main() -> None:
    try:
        config = load_config(CONFIG_PATH)
        samples = read_samples(config.data.samples_dir)
        indexes = demo_indices(samples, DEMO_SESSION, DEMO_POSES, FIRST_FRAME, FRAMES_PER_POSE)
        matches = index_memes(config.memes, config.poses.prompts, lambda: ClipEncoder(config.memes.clip_model))
    except (ConfigError, DatasetError, MemeLoadError, ValueError) as error:
        sys.exit(f"error: {error}")

    neutral = config.poses.neutral_class
    seen = samples.sessions != DEMO_SESSION
    pipeline = build_mlp(config.training).fit(
        *training_data(samples.features[seen], samples.labels[seen], config.training.mirror_augmentation)
    )
    classifier = MLClassifier(pipeline)
    stabilizer = PoseStabilizer(config.stabilizer.window_size, config.stabilizer.min_ratio, initial=neutral)
    composer = ViewComposer(CAMERA_SIZE, PANEL_WIDTH_RATIO, max_width=10_000, line_count=3)
    memes = load_memes(config.memes.dir, group_by_pose(matches), composer.layout.meme_size)
    selector = MemeSelector(memes, neutral, random.Random(0))
    aspect = CAMERA_SIZE[1] / CAMERA_SIZE[0]

    images, correct, guessed = [], 0, 0
    for index in indexes:
        features = samples.features[index]
        guess, _ = classifier.predict(features)  # the model's top class, before the threshold
        label, confidence = classify_frame(classifier, features, config.classifier.confidence_threshold, neutral)
        correct += label == samples.labels[index]
        guessed += guess == samples.labels[index]
        pose = stabilizer.update(label)
        frame = np.full((*CAMERA_SIZE, 3), BACKGROUND, dtype=np.uint8)
        landmarks, hands = body_landmarks(features, aspect)
        draw_body(frame, landmarks, hands, POSE_CONNECTIONS, HAND_CONNECTIONS, config.tracker.min_visibility)
        _draw_head(frame, landmarks)
        # "Pose" is what the app shows: below the threshold, the guess counts as neutral.
        lines = [f"Pose: {pose}", f"Modelo: {guess} ({confidence:.2f})", f"Gravado: {samples.labels[index]}"]
        view = composer.compose(frame, lines, selector.update(pose))
        images.append(Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB)).quantize(colors=128))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(OUTPUT, save_all=True, append_images=images[1:], duration=FRAME_MS, loop=0, optimize=True)
    print(f"{OUTPUT}: {len(images)} frames, {OUTPUT.stat().st_size / 1024:.0f} KB")
    print(f"model's top class right: {guessed / len(indexes):.0%}")
    print(f"accepted pose right (after the threshold, before the stabilizer): {correct / len(indexes):.0%}")


if __name__ == "__main__":
    main()
