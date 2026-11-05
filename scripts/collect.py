"""Records labeled samples for one session. Run from the project root: python -m scripts.collect

Uses the same tracker and features.py as the app, so training sees what inference sees.
"""

import sys
import time
from pathlib import Path

import cv2

from pose_meme.capture import Camera, CameraError
from pose_meme.config import AppConfig, ConfigError, load_config
from pose_meme.dataset import COUNTDOWN, DatasetError, Recorder, SampleWriter, class_for_key, next_session_id
from pose_meme.features import extract_features
from pose_meme.renderer import draw_body, draw_text_lines
from pose_meme.tracker import HAND_CONNECTIONS, POSE_CONNECTIONS, ModelNotFoundError, create_body_tracker

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
KEY_WAIT_MS = 1
KEY_MASK = 0xFF


def hud_lines(
    session_id: str,
    recorder: Recorder,
    classes: list[str],
    class_keys: str,
    target: int,
    now: float,
    has_body: bool,
) -> list[str]:
    status = recorder.state
    if recorder.state == COUNTDOWN:
        status = f"{status} {recorder.seconds_left(now):.0f}s"
    lines = [f"{session_id}: {status}" + ("" if has_body else " (ombros fora do quadro)")]
    for key, name in zip(class_keys, classes):
        marker = ">" if name == recorder.label else " "
        lines.append(f"{marker} {key} {name}: {recorder.counts[name]}/{target}")
    return lines


def run(config: AppConfig) -> None:
    collect = config.collect
    neutral = config.poses.neutral_class
    classes = [neutral, *config.poses.prompts]
    min_visibility = config.tracker.min_visibility
    title = f"{config.ui.window_title} - coleta"
    show_skeleton = config.renderer.show_skeleton

    session_id = next_session_id(config.data.samples_dir)
    recorder = Recorder(neutral, collect.countdown_seconds, collect.sample_every, collect.samples_per_class)
    keys = (
        "tecla ao lado: classe | "
        f"{'espaco' if collect.record_key == ' ' else collect.record_key}: gravar/parar | "
        f"{config.ui.skeleton_key}: esqueleto | {config.ui.quit_key}: sair"
    )

    with (
        Camera(config.camera) as camera,
        create_body_tracker(config.tracker, config.hand_tracker) as tracker,
        SampleWriter(config.data.samples_dir, session_id) as writer,
    ):
        try:
            while True:
                frame = camera.read()
                height, width = frame.shape[:2]
                body = tracker.detect(frame)
                features = (
                    extract_features(body.pose, body.hands, width / height, min_visibility)
                    if body.pose is not None
                    else None
                )
                now = time.monotonic()
                if recorder.should_record(now) and features is not None:
                    writer.write(recorder.label, features)
                    recorder.saved()

                if show_skeleton:
                    draw_body(frame, body.pose, body.hands, POSE_CONNECTIONS, HAND_CONNECTIONS, min_visibility)
                lines = hud_lines(
                    session_id, recorder, classes, collect.class_keys, collect.samples_per_class, now, features is not None
                )
                draw_text_lines(frame, [*lines, keys])
                cv2.imshow(title, frame)

                key = cv2.waitKey(KEY_WAIT_MS) & KEY_MASK
                if key == ord(config.ui.quit_key) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord(config.ui.skeleton_key):
                    show_skeleton = not show_skeleton
                elif key == ord(collect.record_key):
                    recorder.toggle(now)
                elif (chosen := class_for_key(key, classes, collect.class_keys)) is not None:
                    recorder.select(chosen)
        finally:
            cv2.destroyAllWindows()

    total = sum(recorder.counts.values())
    print(f"{session_id}: {total} samples" + (f", {dict(recorder.counts)}" if total else " (no file created)"))


def main() -> None:
    try:
        run(load_config(CONFIG_PATH))
    except (ConfigError, CameraError, ModelNotFoundError, DatasetError) as error:
        sys.exit(f"error: {error}")


if __name__ == "__main__":
    main()
