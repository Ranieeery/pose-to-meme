"""Frame loop: only wires the pose_meme modules together."""

import sys
import time
from pathlib import Path

import cv2

from pose_meme.capture import Camera, CameraError
from pose_meme.config import AppConfig, ConfigError, load_config
from pose_meme.renderer import draw_hud, draw_skeleton
from pose_meme.timing import FpsCounter
from pose_meme.tracker import POSE_CONNECTIONS, ModelNotFoundError, create_tracker

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
KEY_WAIT_MS = 1
KEY_MASK = 0xFF


def run(config: AppConfig) -> None:
    title = config.ui.window_title
    show_skeleton = config.renderer.show_skeleton
    fps = FpsCounter(config.ui.fps_window)

    with Camera(config.camera) as camera, create_tracker(config.tracker) as tracker:
        try:
            while True:
                frame = camera.read()
                landmarks = tracker.detect(frame)
                fps.tick(time.perf_counter())

                if show_skeleton and landmarks is not None:
                    draw_skeleton(frame, landmarks, POSE_CONNECTIONS, config.tracker.min_visibility)
                draw_hud(frame, fps.fps)
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
    except (ConfigError, CameraError, ModelNotFoundError) as error:
        sys.exit(f"error: {error}")


if __name__ == "__main__":
    main()
