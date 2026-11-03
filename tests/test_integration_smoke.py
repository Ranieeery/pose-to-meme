"""Checks against the real dependencies; deselected by default (see pytest.ini)."""

import pytest


@pytest.mark.integration
def test_mediapipe_exposes_the_tasks_pose_landmarker() -> None:
    from mediapipe.tasks.python.vision import PoseLandmarker

    assert PoseLandmarker is not None
