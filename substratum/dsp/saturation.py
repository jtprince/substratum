"""Saturation and oversampling.

Analog-style soft saturation using tanh, processed at a higher sample rate
to keep harmonic fold-back (aliasing) above the audible band.
"""

import numpy as np
from scipy.signal import resample_poly


def oversample(signal: np.ndarray, factor: int = 4) -> np.ndarray:
    """Oversample by ``factor`` using polyphase resampling."""
    if factor <= 1:
        return signal
    return resample_poly(signal, factor, 1)


def downsample(signal: np.ndarray, factor: int = 4) -> np.ndarray:
    """Downsample by ``factor`` using polyphase resampling."""
    if factor <= 1:
        return signal
    return resample_poly(signal, 1, factor)


def tanh_saturate(signal: np.ndarray, drive: float = 0.5) -> np.ndarray:
    """Apply tanh saturation.

    ``drive`` in [0, 1]. At 0 the signal passes through unchanged;
    higher values apply more pre-gain before the tanh curve, generating
    progressively more harmonics. Never hard-clips.
    """
    drive = float(np.clip(drive, 0.0, 1.0))
    if drive <= 0.0:
        return signal
    gain = 1.0 + 8.0 * drive
    return np.tanh(gain * signal)


def atan_saturate(signal: np.ndarray, drive: float = 0.5) -> np.ndarray:
    """Apply atan saturation, an alternative to tanh. Bounded by +/-1."""
    drive = float(np.clip(drive, 0.0, 1.0))
    if drive <= 0.0:
        return signal
    gain = 1.0 + 8.0 * drive
    return np.arctan(gain * signal) * (2.0 / np.pi)


def curve_saturate(signal: np.ndarray, drive: float = 0.5, curve: float = 0.0) -> np.ndarray:
    """Saturate with a smooth tanh <-> atan blend.

    ``curve`` in [0, 1]: 0 is tanh (punchier), 1 is atan (softer, rounder).
    Both curves share the same pre-gain so the blend stays stable at any
    drive level.
    """
    drive = float(np.clip(drive, 0.0, 1.0))
    curve = float(np.clip(curve, 0.0, 1.0))
    if drive <= 0.0:
        return signal
    gain = 1.0 + 8.0 * drive
    tanh_out = np.tanh(gain * signal)
    atan_out = np.arctan(gain * signal) * (2.0 / np.pi)
    return (1.0 - curve) * tanh_out + curve * atan_out


def saturate(
    signal: np.ndarray,
    drive: float = 0.5,
    curve: str = "tanh",
    oversample_factor: int = 4,
) -> np.ndarray:
    """Oversampled saturation convenience wrapper.

    Applies the chosen saturation curve at ``oversample_factor`` x the input
    sample rate, then downsamples back. This keeps aliasing out of the
    audible spectrum.
    """
    up = oversample(signal, oversample_factor)
    sat = atan_saturate(up, drive) if curve == "atan" else tanh_saturate(up, drive)
    return downsample(sat, oversample_factor)
