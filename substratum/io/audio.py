"""Audio file I/O: 48 kHz, 24-bit, mono WAV."""

from pathlib import Path

import numpy as np
import soundfile as sf


def write_wav(
    path: str | Path,
    audio: np.ndarray,
    sample_rate: int,
    subtype: str = "PCM_24",
) -> None:
    """Write a mono float array as a WAV file."""
    data = np.asarray(audio, dtype=np.float32)
    data = np.clip(data, -1.0, 1.0)
    sf.write(str(path), data, sample_rate, subtype=subtype)


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a WAV file, returning (audio, sample_rate)."""
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    return audio, sample_rate
