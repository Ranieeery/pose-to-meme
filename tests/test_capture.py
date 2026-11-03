import cv2
import numpy as np
import pytest

from pose_meme.capture import Camera, CameraError
from pose_meme.config import CameraConfig

CONFIG = CameraConfig(index=2, width=320, height=240, mirror=False)


class FakeVideoCapture:
    """Stands in for cv2.VideoCapture, so no webcam is needed."""

    def __init__(self, frame: np.ndarray | None = None, opened: bool = True) -> None:
        self.frame = frame
        self.opened = opened
        self.props: dict[int, float] = {}
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def set(self, prop: int, value: float) -> bool:
        self.props[prop] = value
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        return (self.frame is not None, self.frame)

    def release(self) -> None:
        self.released = True


def asymmetric_frame() -> np.ndarray:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:, 0] = 255  # only the first column is lit
    return frame


def test_opens_the_configured_device_with_the_configured_size() -> None:
    fake = FakeVideoCapture()
    opened_indexes: list[int] = []

    def opener(index: int) -> FakeVideoCapture:
        opened_indexes.append(index)
        return fake

    Camera(CONFIG, opener=opener)

    assert opened_indexes == [2]
    assert fake.props[cv2.CAP_PROP_FRAME_WIDTH] == 320
    assert fake.props[cv2.CAP_PROP_FRAME_HEIGHT] == 240


def test_unavailable_camera_raises_and_names_the_index() -> None:
    fake = FakeVideoCapture(opened=False)

    with pytest.raises(CameraError, match="2"):
        Camera(CONFIG, opener=lambda _: fake)

    assert fake.released


def test_frame_is_returned_untouched_when_mirror_is_off() -> None:
    frame = asymmetric_frame()
    camera = Camera(CONFIG, opener=lambda _: FakeVideoCapture(frame))

    np.testing.assert_array_equal(camera.read(), frame)


def test_frame_is_flipped_horizontally_when_mirror_is_on() -> None:
    frame = asymmetric_frame()
    config = CameraConfig(index=0, width=320, height=240, mirror=True)
    camera = Camera(config, opener=lambda _: FakeVideoCapture(frame))

    mirrored = camera.read()

    assert mirrored[:, -1].min() == 255
    assert mirrored[:, 0].max() == 0


def test_failed_read_raises() -> None:
    camera = Camera(CONFIG, opener=lambda _: FakeVideoCapture(frame=None))

    with pytest.raises(CameraError, match="read"):
        camera.read()


def test_context_manager_releases_the_device() -> None:
    fake = FakeVideoCapture()

    with Camera(CONFIG, opener=lambda _: fake):
        assert not fake.released

    assert fake.released
