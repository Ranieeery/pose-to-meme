import io
import wave
from pathlib import Path

import numpy as np
import pytest

from pose_meme import sound
from pose_meme.sound import CHIME_SECONDS, SAMPLE_RATE, MemeCue, WindowsPlayer, chime, to_wav_bytes

A = np.zeros((2, 2, 3), dtype=np.uint8)
B = np.ones((2, 2, 3), dtype=np.uint8)


class FakePlayer:
    def __init__(self) -> None:
        self.plays = 0

    def play(self) -> None:
        self.plays += 1


def cue(cooldown: float = 1.0) -> tuple[MemeCue, FakePlayer]:
    player = FakePlayer()
    return MemeCue(player, cooldown_seconds=cooldown), player


# --- when to play ---


def test_plays_when_a_meme_appears() -> None:
    meme_cue, player = cue()

    meme_cue.update(A, now=10.0)

    assert player.plays == 1


def test_stays_quiet_while_the_same_meme_is_shown() -> None:
    meme_cue, player = cue()

    for frame in range(100):
        meme_cue.update(A, now=10.0 + frame / 30)

    assert player.plays == 1


def test_stays_quiet_without_a_meme() -> None:
    meme_cue, player = cue()

    for frame in range(10):
        meme_cue.update(None, now=float(frame))

    assert player.plays == 0


def test_plays_again_when_the_meme_comes_back_after_the_cooldown() -> None:
    meme_cue, player = cue(cooldown=1.0)
    meme_cue.update(A, now=10.0)
    meme_cue.update(None, now=10.5)

    meme_cue.update(A, now=12.0)

    assert player.plays == 2


def test_quick_changes_within_the_cooldown_play_once() -> None:
    meme_cue, player = cue(cooldown=1.0)

    meme_cue.update(A, now=10.0)
    meme_cue.update(B, now=10.3)
    meme_cue.update(None, now=10.5)
    meme_cue.update(A, now=10.9)

    assert player.plays == 1


def test_a_new_meme_after_the_cooldown_plays() -> None:
    meme_cue, player = cue(cooldown=1.0)
    meme_cue.update(A, now=10.0)

    meme_cue.update(B, now=11.0)

    assert player.plays == 2


def test_meme_shown_during_the_cooldown_does_not_play_later() -> None:
    meme_cue, player = cue(cooldown=1.0)
    meme_cue.update(A, now=10.0)
    meme_cue.update(B, now=10.5)  # suppressed

    for frame in range(60):  # B stays on screen past the cooldown
        meme_cue.update(B, now=10.5 + frame / 30)

    assert player.plays == 1


def test_muted_cue_never_plays() -> None:
    meme_cue, player = cue()
    meme_cue.muted = True

    meme_cue.update(A, now=10.0)

    assert player.plays == 0


def test_meme_shown_while_muted_does_not_play_on_unmute() -> None:
    meme_cue, player = cue()
    meme_cue.muted = True
    meme_cue.update(A, now=10.0)

    meme_cue.muted = False
    meme_cue.update(A, now=20.0)

    assert player.plays == 0


def test_no_player_is_a_silent_cue() -> None:
    meme_cue = MemeCue(None, cooldown_seconds=1.0)

    meme_cue.update(A, now=10.0)  # must not fail


# --- the sound itself ---


def test_chime_is_short_and_within_the_volume() -> None:
    samples = chime(volume=0.3)

    assert samples.dtype == np.int16
    assert len(samples) == pytest.approx(CHIME_SECONDS * SAMPLE_RATE, abs=1)
    assert np.abs(samples).max() <= 0.3 * np.iinfo(np.int16).max + 1


def test_chime_starts_and_ends_silent_so_it_does_not_click() -> None:
    samples = chime(volume=1.0).astype(np.float64) / np.iinfo(np.int16).max

    assert abs(samples[0]) < 0.01
    assert np.abs(samples[-20:]).max() < 0.01


def test_louder_volume_gives_a_louder_chime() -> None:
    assert np.abs(chime(volume=0.6)).max() > np.abs(chime(volume=0.3)).max()


def test_wav_bytes_are_a_valid_mono_16_bit_file() -> None:
    samples = chime(volume=0.3)

    with wave.open(io.BytesIO(to_wav_bytes(samples)), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, SAMPLE_RATE)
        assert wav.getnframes() == len(samples)


# --- playing on Windows ---


def test_player_writes_the_chime_once_and_plays_it_without_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(sound, "_play_file", lambda path, flags: calls.append((path, flags)))
    path = tmp_path / "cache" / "chime.wav"

    player = WindowsPlayer(path, volume=0.3)
    player.play()
    player.play()

    assert path.is_file()
    with wave.open(str(path), "rb") as wav:
        assert wav.getnframes() == len(chime(volume=0.3))
    assert len(calls) == 2
    assert calls[0][0] == str(path)
    assert calls[0][1] & sound.ASYNC_FLAG
