"""Filters for tone shaping."""

import numpy as np
from scipy.signal import lfilter


def one_pole_lowpass(signal: np.ndarray, cutoff_hz: float, sample_rate: int) -> np.ndarray:
    """Simple one-pole low-pass filter.

    ``y[n] = y[n-1] + alpha * (x[n] - y[n-1])`` with
    ``alpha = 1 - exp(-2*pi*fc/fs)``.
    """
    alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff_hz / sample_rate)
    b = np.array([alpha])
    a = np.array([1.0, alpha - 1.0])
    return lfilter(b, a, signal)


def dc_blocker(signal: np.ndarray, r: float = 0.9995) -> np.ndarray:
    """High-pass to remove any DC offset (~4 Hz cutoff, preserves sub-bass)."""
    b = np.array([1.0, -1.0])
    a = np.array([1.0, -r])
    return lfilter(b, a, signal)
