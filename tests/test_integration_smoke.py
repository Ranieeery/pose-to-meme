"""Checks against the real dependencies; deselected by default (see pytest.ini)."""

from pathlib import Path

import pytest


@pytest.mark.integration
def test_mediapipe_exposes_the_tasks_pose_landmarker() -> None:
    from mediapipe.tasks.python.vision import PoseLandmarker

    assert PoseLandmarker is not None


@pytest.mark.integration
def test_clip_encoder_puts_an_image_closest_to_its_description() -> None:
    import numpy as np

    from pose_meme.config import load_config
    from pose_meme.meme_index import ClipEncoder

    config = load_config(Path(__file__).resolve().parent.parent / "config.yaml")
    encoder = ClipEncoder(config.memes.clip_model)
    red = np.zeros((224, 224, 3), dtype=np.uint8)
    red[..., 0] = 255  # RGB

    image = encoder.encode_images([red])
    texts = encoder.encode_texts(["a plain red square", "a plain green square"])

    assert image.shape == (1, texts.shape[1])
    similarity = (image / np.linalg.norm(image)) @ (texts / np.linalg.norm(texts, axis=1, keepdims=True)).T
    assert similarity[0, 0] > similarity[0, 1]
