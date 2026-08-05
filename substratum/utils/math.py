"""Mathematical utilities for DSP."""

import numpy as np


def hz_to_midi(hz: float) -> float:
    """Convert frequency in Hz to MIDI note number."""
    return 69 + 12 * np.log2(hz / 440.0)


def midi_to_hz(midi: float) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def log_freq_mapping(value: float, min_hz: float = 25.0, max_hz: float = 70.0) -> float:
    """Map 0-1 value to logarithmic frequency range."""
    log_min = np.log(min_hz)
    log_max = np.log(max_hz)
    return np.exp(log_min + value * (log_max - log_min))


def db_to_linear(db: float) -> float:
    """Convert decibels to linear amplitude."""
    return 10.0 ** (db / 20.0)


def linear_to_db(linear: float) -> float:
    """Convert linear amplitude to decibels."""
    return 20.0 * np.log10(np.maximum(linear, 1e-10))


def normalize(audio: np.ndarray, headroom_db: float = -0.5) -> np.ndarray:
    """Normalize audio to specified headroom in dB."""
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    target = db_to_linear(headroom_db)
    return audio * (target / peak)
