"""Master signal chain composition.

Signal flow (after the voice is built):

    Oversample (4x)
        -> Analog saturation (tanh)
        -> Downsample
        -> One-pole low-pass (tone filter)
        -> Soft clipper
        -> Look-ahead limiter
        -> DC blocker
        -> Normalize
"""

import numpy as np

from substratum.dsp.dynamics import lookahead_limiter, soft_clip
from substratum.dsp.filters import dc_blocker, one_pole_lowpass
from substratum.dsp.saturation import curve_saturate, downsample, oversample
from substratum.utils.math import normalize


def apply_master_chain(
    signal: np.ndarray,
    drive: float,
    lowpass_hz: float,
    sample_rate: int,
    oversample_factor: int = 4,
    limiter_threshold: float = 0.92,
    makeup_db: float = 0.0,
    soft_clip_amount: float = 0.4,
    curve: float = 0.0,
) -> np.ndarray:
    """Run the full post-processing chain on a generated voice."""
    sig = signal
    sig = oversample(sig, oversample_factor)
    sig = curve_saturate(sig, drive, curve)
    sig = downsample(sig, oversample_factor)
    sig = one_pole_lowpass(sig, lowpass_hz, sample_rate)
    sig = soft_clip(sig, amount=soft_clip_amount)
    sig = lookahead_limiter(
        sig,
        threshold=limiter_threshold,
        makeup_db=makeup_db,
        sample_rate=sample_rate,
    )
    sig = dc_blocker(sig)
    sig = normalize(sig, headroom_db=-0.5)
    return sig
