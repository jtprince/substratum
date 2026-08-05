"""Deep808Lab core: 808-style bass voice synthesis.

The voice is built from a sine fundamental whose pitch sweeps exponentially
down (punch), shaped by an ADSR amplitude envelope (weight affects decay),
and reinforced by warmth components (octave sine, triangle, low-order
harmonics). The result runs through the master chain in
``substratum.dsp.pipeline``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from substratum.dsp.envelopes import adsr_envelope, pitch_envelope_from_punch
from substratum.dsp.pipeline import apply_master_chain
from substratum.utils.math import db_to_linear

SAMPLE_RATE = 48000
FREQ_MIN = 25.0
FREQ_MAX = 70.0


@dataclass
class BassParams:
    """Parameters controlling the bass voice and its master chain.

    The five primary controls are ``freq``, ``punch``, ``drive``, ``warmth``
    and ``weight``. The remaining fields are "hidden" defaults that Advanced
    Mode will later expose.
    """

    freq: float = 38.0
    punch: float = 0.5
    drive: float = 0.35
    warmth: float = 0.25
    weight: float = 0.5

    # Hidden DSP parameters.
    attack: float = 0.002
    decay: float = 0.85
    sustain: float = 0.0
    release: float = 0.3
    duration: float = 2.0
    sample_rate: int = SAMPLE_RATE
    oversample_factor: int = 4

    # Human-readable description, used by presets and the gallery.
    description: str = ""

    def validated(self) -> BassParams:
        """Return a copy with all primary controls clamped to valid ranges."""
        return replace(
            self,
            freq=float(np.clip(self.freq, FREQ_MIN, FREQ_MAX)),
            punch=float(np.clip(self.punch, 0.0, 1.0)),
            drive=float(np.clip(self.drive, 0.0, 1.0)),
            warmth=float(np.clip(self.warmth, 0.0, 1.0)),
            weight=float(np.clip(self.weight, 0.0, 1.0)),
        )


def _weight_decay(base_decay: float, weight: float) -> float:
    """Weight lengthens the body of the note."""
    return base_decay * (1.0 + 0.5 * weight)


def _weight_sub_level(weight: float) -> float:
    """Weight brings the sub fundamental up toward unity."""
    return db_to_linear(-4.0 * (1.0 - weight))


def _weight_lowpass(drive: float, weight: float) -> float:
    """Tone filter cutoff, linked to drive and darkened by weight."""
    cutoff = 250.0 + 250.0 * drive - 150.0 * weight
    return float(np.clip(cutoff, 100.0, 600.0))


def _weight_limiter(weight: float) -> tuple[float, float]:
    """Weight lowers the limiter threshold (more glue) and adds makeup."""
    threshold = 0.95 - 0.06 * weight
    makeup_db = 2.0 * weight
    return threshold, makeup_db


def _warmth_components(phase: np.ndarray, warmth: float) -> np.ndarray:
    """Build the warmth layers: octave sine, triangle, 3rd and 4th harmonics.

    Each component gates in with warmth: at ``warmth=0`` the voice is a pure
    sine; higher warmth fades in an octave-up sine, a subtle triangle and
    low-order harmonics toward their target levels.
    """
    octave_amp = warmth * db_to_linear(-22.0 + 12.0 * warmth)
    tri_amp = 0.08 * warmth
    h3_amp = warmth * db_to_linear(-26.0 + 10.0 * warmth)
    h4_amp = warmth * db_to_linear(-30.0 + 12.0 * warmth)

    triangle = 2.0 / np.pi * np.arcsin(np.sin(phase))
    return (
        octave_amp * np.sin(2.0 * phase)
        + tri_amp * triangle
        + h3_amp * np.sin(3.0 * phase)
        + h4_amp * np.sin(4.0 * phase)
    )


def render(params: BassParams) -> np.ndarray:
    """Render a mono bass voice, post-master-chain, normalized to -0.5 dB."""
    p = params.validated()
    fs = p.sample_rate

    # Pitch envelope from punch -> instantaneous phase.
    pitch_env = pitch_envelope_from_punch(p.punch, p.duration, fs, base_freq=p.freq)
    inst_freq = p.freq * pitch_env
    phase = 2.0 * np.pi * np.cumsum(inst_freq) / fs

    # Amplitude envelope (decay lengthens with weight).
    decay = _weight_decay(p.decay, p.weight)
    env = adsr_envelope(
        p.duration,
        fs,
        attack=p.attack,
        decay=decay,
        sustain=p.sustain,
        release=p.release,
    )

    # Voice: sub fundamental + warmth layers, all under the amplitude envelope.
    sub_level = _weight_sub_level(p.weight)
    voice = sub_level * np.sin(phase) + _warmth_components(phase, p.warmth)
    voice *= env

    # Master chain.
    lowpass_hz = _weight_lowpass(p.drive, p.weight)
    limiter_threshold, makeup_db = _weight_limiter(p.weight)

    return apply_master_chain(
        voice,
        drive=p.drive,
        lowpass_hz=lowpass_hz,
        sample_rate=fs,
        oversample_factor=p.oversample_factor,
        limiter_threshold=limiter_threshold,
        makeup_db=makeup_db,
    )


def render_to(
    params: BassParams,
    output: str | Any,
) -> None:
    """Render and write to a WAV file. ``output`` accepts a path or file-like."""
    from substratum.io.audio import write_wav

    write_wav(output, render(params), params.sample_rate)
