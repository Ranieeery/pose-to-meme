"""A short chime when a meme appears.

The same sound for every meme: it confirms that a pose was recognized, and stays out
of the way. Two rising notes with a soft start and a fade-out (no click), played
asynchronously so the frame loop never waits for it. The sound is synthesized here
and written to the cache, so no audio file lives in the repository.
"""

import io
import logging
import sys
import wave
from pathlib import Path
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

# Purely cosmetic, so not in config.yaml (like the renderer's colors).
SAMPLE_RATE = 44_100
CHIME_SECONDS = 0.22
NOTES_HZ = (1046.5, 1568.0)  # C6 then G6: a rising fifth reads as "found it"
NOTE_STARTS = (0.0, 0.07)  # seconds
ATTACK_SECONDS = 0.005  # a soft start, no click
DECAY_PER_SECOND = 18.0  # bell-like: loud at once, then fading
OVERTONE_RATIO, OVERTONE_GAIN = 2.0, 0.25  # a little brightness
FADE_OUT_SECONDS = 0.02  # ends exactly at zero, no click
INT16_MAX = np.iinfo(np.int16).max

if sys.platform == "win32":
    import winsound

    ASYNC_FLAG = winsound.SND_ASYNC
    _FLAGS = winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT
else:
    ASYNC_FLAG = 1
    _FLAGS = ASYNC_FLAG


class Player(Protocol):
    def play(self) -> None: ...


def chime(volume: float) -> np.ndarray:
    """The chime as 16-bit mono samples, peaking at volume (0 to 1) of full scale."""
    time = np.arange(round(CHIME_SECONDS * SAMPLE_RATE)) / SAMPLE_RATE
    signal = np.zeros_like(time)
    for frequency, start in zip(NOTES_HZ, NOTE_STARTS, strict=True):
        playing = time >= start
        local = time[playing] - start
        envelope = np.minimum(local / ATTACK_SECONDS, 1.0) * np.exp(-DECAY_PER_SECOND * local)
        tone = np.sin(2 * np.pi * frequency * local) + OVERTONE_GAIN * np.sin(
            2 * np.pi * frequency * OVERTONE_RATIO * local
        )
        signal[playing] += envelope * tone
    fade_samples = round(FADE_OUT_SECONDS * SAMPLE_RATE)
    signal[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples)
    signal /= np.abs(signal).max()
    return np.round(signal * volume * INT16_MAX).astype(np.int16)


def to_wav_bytes(samples: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(np.dtype(np.int16).itemsize)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(samples.astype("<i2").tobytes())
    return buffer.getvalue()


def _play_file(path: str, flags: int) -> None:
    winsound.PlaySound(path, flags)


class WindowsPlayer:
    """Plays the chime through winsound, which can't play from memory asynchronously,
    so the chime is written to a file once."""

    def __init__(self, path: Path, volume: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(to_wav_bytes(chime(volume)))
        self._path = str(path)

    def play(self) -> None:
        _play_file(self._path, _FLAGS)  # returns at once; a new chime cuts the previous one


def create_player(path: Path, volume: float) -> Player | None:
    """The chime player, or None (silence) where there is no winsound."""
    if sys.platform != "win32":
        logger.warning("meme sound needs Windows (winsound); running silent")
        return None
    return WindowsPlayer(path, volume)


class MemeCue:
    """Plays the chime when a meme appears: once per appearance, never more often than
    the cooldown, so flicking between poses doesn't turn into a burst of chimes."""

    def __init__(self, player: Player | None, cooldown_seconds: float) -> None:
        self.muted = False
        self._player = player
        self._cooldown = cooldown_seconds
        self._last_meme: np.ndarray | None = None
        self._last_play = -np.inf

    def update(self, meme: np.ndarray | None, now: float) -> None:
        appeared = meme is not None and meme is not self._last_meme
        self._last_meme = meme
        if not appeared or self.muted or self._player is None or now - self._last_play < self._cooldown:
            return
        self._player.play()
        self._last_play = now
