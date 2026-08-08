"""Deep808Lab core: 808-style bass voice synthesis.

The voice is built from a sine fundamental whose pitch sweeps exponentially
down (punch), shaped by an ADSR amplitude envelope (weight affects decay),
and reinforced by warmth components (octave sine, triangle, low-order
harmonics). The result runs through the master chain in
``substratum.dsp.pipeline``.

Beyond the five classic controls (``freq``, ``punch``, ``drive``, ``warmth``,
``weight``) the voice exposes a set of "advanced" shaping parameters used by
the GUI and pattern renderer: envelope times, tone filter cutoff, sub level,
transient snap, stereo width, limiter makeup, saturation curve and glide.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from substratum.dsp.dynamics import soft_clip
from substratum.dsp.envelopes import adsr_envelope, pitch_envelope_from_punch
from substratum.dsp.pipeline import apply_master_chain
from substratum.utils.math import db_to_linear

SAMPLE_RATE = 48000
FREQ_MIN = 25.0
FREQ_MAX = 70.0

#: How many loop iterations to master when ``apply_master(periodic=True)``.
#: The extra copies give the chain's IIR filters and the limiter look-ahead
#: window converged state at the loop seam; the middle copy is returned.
_LOOP_PERIODS = 3

TONE_MIN = 100.0
TONE_MAX = 600.0
ATTACK_MAX = 0.5
DECAY_MAX = 5.0
MAKEUP_MAX = 12.0


@dataclass
class BassParams:
    """Parameters controlling the bass voice and its master chain.

    The five primary controls are ``freq``, ``punch``, ``drive``, ``warmth``
    and ``weight``. The remaining fields shape the envelope, tone and stereo
    character; ``transpose`` is used by the pattern renderer to shift notes.
    """

    freq: float = 38.0
    punch: float = 0.5
    drive: float = 0.35
    warmth: float = 0.25
    weight: float = 0.5

    # Envelope.
    attack: float = 0.002
    decay: float = 0.85
    sustain: float = 0.0
    release: float = 0.3
    duration: float = 2.0

    # Advanced shaping.
    tone_hz: float | None = None
    sub_level: float = 1.0
    snap: float = 0.0
    width: float = 0.0
    curve: float = 0.0
    glide: float = 0.0
    makeup_db: float = 0.0
    transpose: float = 0.0

    # Rendering.
    sample_rate: int = SAMPLE_RATE
    oversample_factor: int = 4

    # Human-readable description, used by presets and the gallery.
    description: str = ""

    def validated(self) -> BassParams:
        """Return a copy with all controls clamped to valid ranges."""
        tone = self.tone_hz
        return replace(
            self,
            freq=float(np.clip(self.freq, FREQ_MIN, FREQ_MAX)),
            punch=float(np.clip(self.punch, 0.0, 1.0)),
            drive=float(np.clip(self.drive, 0.0, 1.0)),
            warmth=float(np.clip(self.warmth, 0.0, 1.0)),
            weight=float(np.clip(self.weight, 0.0, 1.0)),
            attack=float(np.clip(self.attack, 0.0005, ATTACK_MAX)),
            decay=float(np.clip(self.decay, 0.02, DECAY_MAX)),
            sustain=float(np.clip(self.sustain, 0.0, 1.0)),
            release=float(np.clip(self.release, 0.01, 3.0)),
            duration=float(np.clip(self.duration, 0.05, 30.0)),
            tone_hz=float(np.clip(tone, TONE_MIN, TONE_MAX)) if tone is not None else None,
            sub_level=float(np.clip(self.sub_level, 0.0, 1.0)),
            snap=float(np.clip(self.snap, 0.0, 1.0)),
            width=float(np.clip(self.width, 0.0, 1.0)),
            curve=float(np.clip(self.curve, 0.0, 1.0)),
            glide=float(np.clip(self.glide, 0.0, 1.0)),
            makeup_db=float(np.clip(self.makeup_db, 0.0, MAKEUP_MAX)),
            transpose=float(np.clip(self.transpose, -24.0, 24.0)),
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


def _transient_click(n_samples: int, sample_rate: int, snap: float) -> np.ndarray:
    """A short onset click, scaled by ``snap`` (0-1)."""
    if snap <= 0.0 or n_samples <= 0:
        return np.zeros(n_samples)
    k = min(int(0.012 * sample_rate), n_samples)
    t = np.arange(k) / sample_rate
    blip = np.sin(2.0 * np.pi * 900.0 * t) * np.exp(-t / 0.003)
    click = np.zeros(n_samples)
    click[:k] = blip * snap
    return click


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


@dataclass(frozen=True)
class VoiceParts:
    """The pre-master voice split into its audible layers.

    ``center`` holds the sub fundamental and the transient snap, ``side``
    holds the warmth harmonics (for stereo width); ``sub``, ``warmth`` and
    ``snap`` are those layers individually and ``envelope`` is the raw ADSR
    shape. ``end_phase`` lets callers chain notes with phase continuity.
    """

    center: np.ndarray
    side: np.ndarray
    sub: np.ndarray
    warmth: np.ndarray
    snap: np.ndarray
    envelope: np.ndarray
    end_phase: float


def decompose_voice(
    params: BassParams,
    freq: float,
    duration: float,
    *,
    start_phase: float = 0.0,
    glide_from: float | None = None,
    glide_time: float = 0.25,
) -> tuple[VoiceParts, float]:
    """Build the pre-master-chain voice layers individually.

    Returns ``(parts, end_phase)`` where ``parts`` exposes the sub
    fundamental, warmth harmonics, transient snap and ADSR envelope
    separately, plus the combined ``center``/``side`` layers. The two-tuple
    shape mirrors :func:`build_voice` so pattern renderers can chain phases.
    """
    p = params.validated()
    fs = p.sample_rate
    n = int(duration * fs)
    if n <= 0:
        empty = VoiceParts(
            center=np.zeros(0),
            side=np.zeros(0),
            sub=np.zeros(0),
            warmth=np.zeros(0),
            snap=np.zeros(0),
            envelope=np.zeros(0),
            end_phase=float(start_phase),
        )
        return empty, empty.end_phase

    t = np.arange(n) / fs

    pitch_env = pitch_envelope_from_punch(p.punch, duration, fs, base_freq=freq)
    if glide_from is not None and p.glide > 0.0:
        ratio = glide_from / freq
        glide_env = 1.0 + (ratio - 1.0) * np.exp(-t / glide_time)
        pitch_env = pitch_env * glide_env

    inst_freq = freq * pitch_env
    phase = start_phase + 2.0 * np.pi * np.cumsum(inst_freq) / fs

    decay = _weight_decay(p.decay, p.weight)
    env = adsr_envelope(
        duration,
        fs,
        attack=p.attack,
        decay=decay,
        sustain=p.sustain,
        release=p.release,
    )

    sub_level = _weight_sub_level(p.weight) * p.sub_level
    sub = sub_level * np.sin(phase)
    snap = _transient_click(n, fs, p.snap)
    warmth = _warmth_components(phase, p.warmth)

    parts = VoiceParts(
        center=(sub + snap) * env,
        side=warmth * env,
        sub=sub * env,
        warmth=warmth * env,
        snap=snap * env,
        envelope=env,
        end_phase=float(phase[-1]),
    )
    return parts, parts.end_phase


def build_voice(
    params: BassParams,
    freq: float,
    duration: float,
    *,
    start_phase: float = 0.0,
    glide_from: float | None = None,
    glide_time: float = 0.25,
) -> tuple[tuple[np.ndarray, np.ndarray], float]:
    """Build the pre-master-chain voice layers.

    Returns ``((center, side), end_phase)``. The ``center`` layer holds the
    sub fundamental and the transient snap; the ``side`` layer holds the
    warmth harmonics so they can be panned for stereo width. ``end_phase``
    lets callers chain notes with phase continuity (for glide).
    """
    parts, end_phase = decompose_voice(
        params,
        freq,
        duration,
        start_phase=start_phase,
        glide_from=glide_from,
        glide_time=glide_time,
    )
    return (parts.center, parts.side), end_phase


def _master_channel(
    signal: np.ndarray,
    params: BassParams,
    lowpass_hz: float,
    limiter_threshold: float,
    limiter_makeup_db: float,
) -> np.ndarray:
    return apply_master_chain(
        signal,
        drive=params.drive,
        lowpass_hz=lowpass_hz,
        sample_rate=params.sample_rate,
        oversample_factor=params.oversample_factor,
        limiter_threshold=limiter_threshold,
        makeup_db=limiter_makeup_db,
        curve=params.curve,
    )


def apply_master(params: BassParams, signal: np.ndarray, *, periodic: bool = False) -> np.ndarray:
    """Run the master chain (optionally per stereo channel) on a voice.

    The limiter makeup knob is applied after normalization as an output gain
    with a safety soft clip, so it is audible instead of being cancelled out.

    When ``periodic`` is set, ``signal`` is treated as one iteration of a
    looping pattern: it is repeated ``_LOOP_PERIODS`` times, the master chain
    runs over the extended buffer, and the middle copy is returned. This lets
    the tone low-pass, limiter release and DC blocker states converge (and
    moves the limiter's zeroed look-ahead window into the discarded pre-roll),
    so the loop wraps without a click at the seam.
    """
    p = params.validated()
    lowpass_hz = p.tone_hz if p.tone_hz is not None else _weight_lowpass(p.drive, p.weight)
    limiter_threshold, limiter_makeup_db = _weight_limiter(p.weight)

    periods = _LOOP_PERIODS if periodic else 1
    if periods > 1 and len(signal) > 0:
        n = len(signal)
        ext = np.tile(signal, (periods, 1)) if signal.ndim == 2 else np.tile(signal, periods)
    else:
        ext = signal
        n = 0

    if ext.ndim == 2:
        out = np.stack(
            [
                _master_channel(ext[:, i], p, lowpass_hz, limiter_threshold, limiter_makeup_db)
                for i in range(ext.shape[1])
            ],
            axis=1,
        )
    else:
        out = _master_channel(ext, p, lowpass_hz, limiter_threshold, limiter_makeup_db)

    if p.makeup_db > 0.0:
        out = soft_clip(out * db_to_linear(p.makeup_db), threshold=0.98, amount=1.0)

    if n > 0:
        start = (len(ext) - n) // 2
        out = out[start : start + n]
    return out


def render(
    params: BassParams, *, freq: float | None = None, duration: float | None = None
) -> np.ndarray:
    """Render a mono bass voice (or stereo when ``width > 0``), master-chained.

    Defaults to ``params.freq`` and ``params.duration`` unless overridden.
    """
    p = params.validated()
    f = p.freq if freq is None else float(freq)
    d = p.duration if duration is None else float(duration)

    (center, side), _ = build_voice(p, f, d)
    if p.width > 0.0:
        w = p.width * 0.9
        audio = np.stack([center + side * w, center - side * w], axis=1)
    else:
        audio = center + side
    return apply_master(p, audio)


def render_to(
    params: BassParams,
    output: str | Any,
) -> None:
    """Render and write to a WAV file. ``output`` accepts a path or file-like."""
    from substratum.io.audio import write_wav

    write_wav(output, render(params), params.sample_rate)
