"""Arrange a parsed note pattern into a rendered audio buffer.

Each note is voiced with ``synth.decompose_voice`` (sharing phase across
notes so glide stays smooth), summed, then run through the master chain once.
The pattern loops are **tempo-locked**: the buffer is trimmed to exactly
``total_beats * 60 / bpm`` samples and note tails wrap around into the next
iteration, so every hit lands on the beat grid and playback loops without a
click.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from substratum.bass.synth import BassParams, apply_master, build_voice, decompose_voice
from substratum.pattern.notation import Note
from substratum.utils.math import hz_to_midi, midi_to_hz

#: Safety cap on pattern length (in beats) to avoid huge buffers.
MAX_BEATS = 128.0


@dataclass(frozen=True)
class PatternParts:
    """The tempo-locked loop split into its musical layers.

    ``sub``, ``warmth`` and ``snap`` are the pre-master summed layers
    (``sub + snap`` is the center voice, ``warmth`` the side voice);
    ``envelope`` is the summed ADSR shape; ``mastered`` is the final
    stereo (or mono) output after the master chain and gain.
    """

    sub: np.ndarray
    warmth: np.ndarray
    snap: np.ndarray
    envelope: np.ndarray
    mastered: np.ndarray
    note_midis: list[float | None] = field(default_factory=list)
    note_beats: list[float] = field(default_factory=list)


def _loop_length(params: BassParams, notes: list[Note], bpm: float) -> int:
    """Number of samples in one tempo-locked loop iteration."""
    p = params.validated()
    beats = min(max(sum(note.beats for note in notes), 1.0), MAX_BEATS)
    return max(int(round(beats * 60.0 / max(float(bpm), 1.0) * p.sample_rate)), 1)


def _add_wrap(buf: np.ndarray, start: int, seg: np.ndarray) -> None:
    """Add ``seg`` into circular buffer ``buf`` starting at sample ``start``.

    Handles segments longer than the loop by wrapping multiple times.
    """
    n = len(buf)
    if len(seg) == 0 or n == 0:
        return
    idx = start % n
    remaining = len(seg)
    offset = 0
    while remaining > 0:
        chunk = min(remaining, n - idx)
        buf[idx : idx + chunk] += seg[offset : offset + chunk]
        remaining -= chunk
        offset += chunk
        idx = 0


def decompose_pattern(
    params: BassParams,
    notes: list[Note],
    bpm: float = 70.0,
    *,
    transpose: float = 0.0,
    gain: float = 1.0,
) -> PatternParts:
    """Split a tempo-locked pattern loop into its musical layers.

    Returns :class:`PatternParts` with the per-layer loops (pre-master) and
    the mastered stereo/mono output. Rests are skipped (their beats still
    advance the grid).
    """
    p = params.validated()
    fs = p.sample_rate
    beat_seconds = 60.0 / max(float(bpm), 1.0)
    n_total = _loop_length(params, notes, bpm)

    sub_buf = np.zeros(n_total)
    side_buf = np.zeros(n_total)
    snap_buf = np.zeros(n_total)
    env_buf = np.zeros(n_total)

    pos = 0.0
    phase = 0.0
    prev_freq: float | None = None
    note_midis: list[float | None] = []
    note_beats: list[float] = []

    for note in notes:
        if note.is_rest:
            pos += note.beats * beat_seconds
            continue
        note_midi = note.midi
        assert note_midi is not None
        midi = note_midi + float(transpose)
        freq = midi_to_hz(midi)
        dur = note.beats * beat_seconds
        seg_dur = dur + p.release + 0.05
        glide_time = max(0.02, p.glide * dur * 0.6) if p.glide > 0.0 else 0.25

        parts, end_phase = decompose_voice(
            p,
            freq,
            seg_dur,
            start_phase=phase,
            glide_from=prev_freq,
            glide_time=glide_time,
        )

        start = int(pos * fs)
        _add_wrap(sub_buf, start, parts.sub)
        _add_wrap(side_buf, start, parts.warmth)
        _add_wrap(snap_buf, start, parts.snap)
        _add_wrap(env_buf, start, parts.envelope)

        note_midis.append(midi)
        note_beats.append(note.beats)

        phase = end_phase
        prev_freq = freq
        pos += note.beats * beat_seconds

    if p.width > 0.0:
        w = p.width * 0.9
        audio = np.stack(
            [
                sub_buf + snap_buf + side_buf * w,
                sub_buf + snap_buf - side_buf * w,
            ],
            axis=1,
        )
    else:
        audio = sub_buf + snap_buf + side_buf

    mastered = apply_master(p, audio, periodic=True)
    if gain != 1.0:
        mastered = mastered * gain

    return PatternParts(
        sub=sub_buf,
        warmth=side_buf,
        snap=snap_buf,
        envelope=env_buf,
        mastered=mastered,
        note_midis=note_midis,
        note_beats=note_beats,
    )


def render_pattern(
    params: BassParams,
    notes: list[Note],
    bpm: float = 70.0,
    *,
    transpose: float = 0.0,
    gain: float = 1.0,
) -> np.ndarray:
    """Render a repeating pattern to a mono or stereo audio buffer.

    ``notes`` come from :func:`substratum.pattern.notation.parse_pattern`.
    ``transpose`` is a semitone shift applied to every note; ``gain`` scales
    the final output (clip the caller).

    The buffer length is exactly ``total_beats * 60 / bpm`` samples, so the
    loop is tempo-locked and tails wrap around into the next iteration.
    """
    return decompose_pattern(params, notes, bpm=bpm, transpose=transpose, gain=gain).mastered


def render_single_note(
    params: BassParams, midi: float, duration: float | None = None
) -> np.ndarray:
    """Render one note at an absolute MIDI pitch (mono unless width > 0)."""
    p = params.validated()
    d = p.duration if duration is None else float(duration)
    freq = midi_to_hz(midi)
    (center, side), _ = build_voice(p, freq, d)
    if p.width > 0.0:
        w = p.width * 0.9
        audio = np.stack([center + side * w, center - side * w], axis=1)
    else:
        audio = center + side
    return apply_master(p, audio)


def decompose_single_note(
    params: BassParams,
    *,
    freq: float | None = None,
    duration: float | None = None,
    gain: float = 1.0,
) -> PatternParts:
    """Decompose a single free-running note into :class:`PatternParts`.

    Used when there is no pattern to render, so the GUI can still show the
    waveform and decomposition plot. The note is not tempo-locked (there is
    no beat grid); ``freq`` defaults to ``params.freq`` and ``duration`` to
    ``params.duration``.
    """
    p = params.validated()
    f = p.freq if freq is None else float(freq)
    d = p.duration if duration is None else float(duration)

    parts, _ = decompose_voice(p, f, d)
    if p.width > 0.0:
        w = p.width * 0.9
        audio = np.stack([parts.center + parts.side * w, parts.center - parts.side * w], axis=1)
    else:
        audio = parts.center + parts.side

    mastered = apply_master(p, audio)
    if gain != 1.0:
        mastered = mastered * gain

    return PatternParts(
        sub=parts.sub,
        warmth=parts.warmth,
        snap=parts.snap,
        envelope=parts.envelope,
        mastered=mastered,
        note_midis=[float(hz_to_midi(f))],
        note_beats=[1.0],
    )
