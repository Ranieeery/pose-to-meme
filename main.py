"""Frame loop: only wires the pose_meme modules together."""

import random
import sys
import time
from pathlib import Path

import cv2

from pose_meme.capture import Camera, CameraError
from pose_meme.classifier import ModelLoadError, classify_frame, load_classifier
from pose_meme.config import AppConfig, ConfigError, load_config
from pose_meme.features import extract_features
from pose_meme.meme_index import MemeLoadError, MemeSelector, load_static_memes
from pose_meme.renderer import draw_body, draw_hud, overlay_centered
from pose_meme.stabilizer import PoseStabilizer
from pose_meme.timing import FpsCounter
from pose_meme.tracker import (
    HAND_CONNECTIONS,
    POSE_CONNECTIONS,
    ModelNotFoundError,
    create_body_tracker,
)

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
KEY_WAIT_MS = 1
KEY_MASK = 0xFF


def run(config: AppConfig) -> None:
    title = config.ui.window_title
    neutral = config.poses.neutral_class
    min_visibility = config.tracker.min_visibility
    show_skeleton = config.renderer.show_skeleton
    fps = FpsCounter(config.ui.fps_window)
    classifier = load_classifier(config.classifier.model_path, config.rules, neutral)
    print(f"classifier: {type(classifier).__name__}")
    stabilizer = PoseStabilizer(config.stabilizer.window_size, config.stabilizer.min_ratio, initial=neutral)

    with Camera(config.camera) as camera, create_body_tracker(config.tracker, config.hand_tracker) as tracker:
        frame = camera.read()
        height, width = frame.shape[:2]  # whatever the camera delivers, not what was requested
        memes = load_static_memes(config.memes, (height, width), config.renderer.meme_height_ratio)
        selector = MemeSelector(memes, neutral, random.Random())

        try:
            while True:
                frame = camera.read()
                body = tracker.detect(frame)
                features = (
                    extract_features(body.pose, body.hands, width / height, min_visibility)
                    if body.pose is not None
                    else None
                )
                label, confidence = classify_frame(
                    classifier, features, config.classifier.confidence_threshold, neutral
                )
                pose = stabilizer.update(label)
                meme = selector.update(pose)
                fps.tick(time.perf_counter())

                if show_skeleton:
                    draw_body(frame, body.pose, body.hands, POSE_CONNECTIONS, HAND_CONNECTIONS, min_visibility)
                if meme is not None:
                    overlay_centered(frame, meme)
                draw_hud(frame, fps.fps, pose, confidence)
                cv2.imshow(title, frame)

                key = cv2.waitKey(KEY_WAIT_MS) & KEY_MASK
                if key == ord(config.ui.quit_key) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord(config.ui.skeleton_key):
                    show_skeleton = not show_skeleton
        finally:
            cv2.destroyAllWindows()


def main() -> None:
    try:
        run(load_config(CONFIG_PATH))
    except (ConfigError, CameraError, ModelNotFoundError, MemeLoadError, ModelLoadError) as error:
        sys.exit(f"error: {error}")


if __name__ == "__main__":
    main()
