"""Dynamics processors: soft clipper and look-ahead limiter."""

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.signal import lfilter, lfilter_zi


def soft_clip(signal: np.ndarray, threshold: float = 0.95, amount: float = 0.5) -> np.ndarray:
    """Very gentle soft clipper.

    Blends between passthrough (``amount=0``) and a tanh curve that
    asymptotes at ``threshold`` (``amount=1``). Only meant to catch peaks,
    not to color the tone.
    """
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.0:
        return signal
    shaped = threshold * np.tanh(signal / threshold)
    return (1.0 - amount) * signal + amount * shaped


def lookahead_limiter(
    signal: np.ndarray,
    threshold: float = 0.95,
    lookahead_ms: float = 5.0,
    release_ms: float = 120.0,
    makeup_db: float = 0.0,
    sample_rate: int = 48000,
) -> np.ndarray:
    """Simple look-ahead limiter.

    Peak-holds the magnitude over the look-ahead window, computes the gain
    needed to pull peaks to ``threshold``, applies it with instant attack and
    a one-pole release, then optionally applies makeup gain.

    Goal is preventing clipping, not loudness maximization.
    """
    lookahead = max(1, int(round(lookahead_ms / 1000.0 * sample_rate)))
    window = lookahead + 1

    # Peak-hold envelope over a centered window so peaks are caught early.
    env = maximum_filter1d(np.abs(signal), window, mode="nearest")
    gain = np.minimum(1.0, threshold / np.maximum(env, 1e-8))

    # A peak within the window must pull the whole window's gain down
    # (instant attack, no overshoot).
    gain = minimum_filter1d(gain, window, mode="nearest")

    # Slow release: smooth the recovery with a one-pole filter, but never
    # exceed the instantaneous gain so attack stays immediate. Initialized
    # at the first sample's gain so a steady signal passes unchanged.
    release_alpha = np.exp(-1.0 / (release_ms / 1000.0 * sample_rate))
    b = np.array([release_alpha])
    a = np.array([1.0, -(1.0 - release_alpha)])
    zi = lfilter_zi(b, a) * gain[0]
    smoothed, _ = lfilter(b, a, gain, zi=zi)
    gain = np.minimum(gain, smoothed)

    delayed = np.roll(signal, lookahead)
    delayed[:lookahead] = 0.0

    makeup = 10.0 ** (makeup_db / 20.0)
    return delayed * gain * makeup
