"""Master signal chain composition.

Signal flow (after the voice is built):

    Oversample (4x)
        -> Analog saturation (tanh)
        -> Wavefolder foldback (industrial distortion)
        -> Downsample
        -> Bit-crush (industrial)
        -> One-pole low-pass (tone filter)
        -> Soft clipper
        -> Look-ahead limiter
        -> DC blocker
        -> Normalize
"""

import numpy as np

from substratum.dsp.dynamics import lookahead_limiter, soft_clip
from substratum.dsp.filters import dc_blocker, one_pole_lowpass
from substratum.dsp.saturation import (
    bitcrush,
    curve_saturate,
    downsample,
    foldback_saturate,
    oversample,
)
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
    distortion: float = 0.0,
    crush: float = 0.0,
) -> np.ndarray:
    """Run the full post-processing chain on a generated voice.

    ``distortion`` (wavefolder foldback) runs inside the oversampled block so
    its dense harmonics fold above the audible band; ``crush`` (bit-depth
    reduction) runs at the base rate, before the tone filter.
    """
    sig = signal
    sig = oversample(sig, oversample_factor)
    sig = curve_saturate(sig, drive, curve)
    sig = foldback_saturate(sig, distortion)
    sig = downsample(sig, oversample_factor)
    sig = bitcrush(sig, crush)
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
