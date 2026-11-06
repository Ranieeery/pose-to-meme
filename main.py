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
from pose_meme.meme_index import ClipEncoder, MemeLoadError, MemeSelector, group_by_pose, index_memes, load_memes
from pose_meme.renderer import ViewComposer, draw_body, info_lines
from pose_meme.sound import MemeCue, create_player
from pose_meme.stabilizer import PoseStabilizer
from pose_meme.timing import FpsCounter, StageTimer, format_stats
from pose_meme.tracker import (
    HAND_CONNECTIONS,
    POSE_CONNECTIONS,
    ModelNotFoundError,
    create_body_tracker,
)

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
KEY_WAIT_MS = 1
KEY_MASK = 0xFF
CHIME_FILE = "meme_chime.wav"  # generated, next to the meme index cache


def run(config: AppConfig) -> None:
    title = config.ui.window_title
    neutral = config.poses.neutral_class
    min_visibility = config.tracker.min_visibility
    show_skeleton = config.renderer.show_skeleton
    fps = FpsCounter(config.ui.fps_window)
    timer = StageTimer(config.ui.timing_warmup_frames)
    classifier = load_classifier(config.classifier.model_path, config.rules, neutral)
    print(f"classifier: {type(classifier).__name__}")
    stabilizer = PoseStabilizer(config.stabilizer.window_size, config.stabilizer.min_ratio, initial=neutral)
    # CLIP loads only if the meme folder or the prompts changed since the last index.
    matches = index_memes(config.memes, config.poses.prompts, lambda: ClipEncoder(config.memes.clip_model))
    buckets = group_by_pose(matches)
    print(f"memes: {sum(map(len, buckets.values()))} of {len(matches)} assigned to {len(buckets)} poses")

    with Camera(config.camera) as camera, create_body_tracker(config.tracker, config.hand_tracker) as tracker:
        frame = camera.read()
        height, width = frame.shape[:2]  # whatever the camera delivers, not what was requested
        composer = ViewComposer(
            (height, width),
            config.renderer.panel_width_ratio,
            config.ui.max_window_width,
            line_count=len(info_lines(0.0, neutral, 0.0, True, config.ui.sound_key)),
        )
        memes = load_memes(config.memes.dir, buckets, composer.layout.meme_size)
        selector = MemeSelector(memes, neutral, random.Random())
        player = create_player(config.memes.cache_dir / CHIME_FILE, config.sound.volume)
        cue = MemeCue(player, config.sound.cooldown_seconds)
        cue.muted = not config.sound.enabled  # the key still turns it on

        try:
            while True:
                with timer.measure("capture"):  # includes waiting for the camera's next frame
                    frame = camera.read()
                with timer.measure("tracking"):  # pose and hands, in parallel
                    body = tracker.detect(frame)
                with timer.measure("features"):
                    features = (
                        extract_features(body.pose, body.hands, width / height, min_visibility)
                        if body.pose is not None
                        else None
                    )
                with timer.measure("classifier"):
                    label, confidence = classify_frame(
                        classifier, features, config.classifier.confidence_threshold, neutral
                    )
                with timer.measure("stabilizer+meme"):
                    pose = stabilizer.update(label)
                    meme = selector.update(pose)
                    cue.update(meme, time.monotonic())
                fps.tick(time.perf_counter())

                with timer.measure("render"):
                    if show_skeleton:
                        draw_body(frame, body.pose, body.hands, POSE_CONNECTIONS, HAND_CONNECTIONS, min_visibility)
                    lines = info_lines(fps.fps, pose, confidence, not cue.muted, config.ui.sound_key)
                    view = composer.compose(frame, lines, meme)
                with timer.measure("display"):  # imshow + waitKey
                    cv2.imshow(title, view)
                    key = cv2.waitKey(KEY_WAIT_MS) & KEY_MASK
                timer.end_frame()

                if key == ord(config.ui.quit_key) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord(config.ui.skeleton_key):
                    show_skeleton = not show_skeleton
                if key == ord(config.ui.sound_key):
                    cue.muted = not cue.muted
        finally:
            cv2.destroyAllWindows()

    if table := format_stats(timer.stats()):
        print(f"\ntiming: model {config.tracker.model_variant}, frame {width}x{height}, {type(classifier).__name__}")
        print("\n".join(table))


def main() -> None:
    try:
        run(load_config(CONFIG_PATH))
    except (ConfigError, CameraError, ModelNotFoundError, MemeLoadError, ModelLoadError) as error:
        sys.exit(f"error: {error}")


if __name__ == "__main__":
    main()
