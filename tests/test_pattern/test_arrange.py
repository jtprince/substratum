import numpy as np

from substratum.bass.synth import SAMPLE_RATE, BassParams
from substratum.pattern.arrange import (
    decompose_pattern,
    decompose_single_note,
    render_pattern,
    render_single_note,
)
from substratum.pattern.notation import parse_pattern


def test_render_pattern_mono_shape():
    audio = render_pattern(BassParams(), parse_pattern("C1 E1 G1"), bpm=60)
    assert audio.ndim == 1
    assert len(audio) > SAMPLE_RATE


def test_render_pattern_stereo_when_width():
    audio = render_pattern(BassParams(width=0.8), parse_pattern("C1"), bpm=60)
    assert audio.ndim == 2
    assert audio.shape[1] == 2


def test_bpm_changes_length():
    slow = render_pattern(BassParams(), parse_pattern("C1"), bpm=50)
    fast = render_pattern(BassParams(), parse_pattern("C1"), bpm=200)
    assert len(slow) > len(fast)


def test_loop_length_is_tempo_locked():
    notes = parse_pattern("C1 D1 E1 G1")  # 4 beats
    for bpm, expected in (
        (70.0, 4 * 60.0 / 70.0),
        (140.0, 4 * 60.0 / 140.0),
        (200.0, 4 * 60.0 / 200.0),
    ):
        audio = render_pattern(BassParams(), notes, bpm=bpm)
        assert abs(len(audio) / SAMPLE_RATE - expected) <= 1e-3


def test_loop_length_proportional_to_bpm():
    notes = parse_pattern("C1 E1 G1")
    slow = len(render_pattern(BassParams(), notes, bpm=70))
    fast = len(render_pattern(BassParams(), notes, bpm=140))
    assert abs(slow / fast - 2.0) < 1e-3


def test_rest_beats_still_lengthen_loop():
    with_rest = render_pattern(BassParams(), parse_pattern("C2 z2"), bpm=60)
    no_rest = render_pattern(BassParams(), parse_pattern("C2"), bpm=60)
    assert len(with_rest) == len(no_rest) * 2


def test_transpose_shifts_pitch():
    base = render_pattern(BassParams(punch=0), parse_pattern("C1"), bpm=120)
    up = render_pattern(BassParams(punch=0), parse_pattern("C1"), bpm=120, transpose=12)
    assert not np.allclose(base, up, atol=0.05)


def test_pattern_is_normalized_and_clipped():
    audio = render_pattern(BassParams(makeup_db=6.0), parse_pattern("C1 E1 G1 A1"), bpm=70)
    assert np.max(np.abs(audio)) <= 1.0
    assert np.max(np.abs(audio)) >= 0.3


def test_empty_pattern_produces_audio():
    audio = render_pattern(BassParams(), [], bpm=70)
    assert len(audio) > 0


def test_render_single_note_midi_pitch():
    audio = render_single_note(BassParams(), midi=28, duration=1.0)
    assert len(audio) == SAMPLE_RATE


def test_decompose_pattern_parts_are_tempo_locked():
    parts = decompose_pattern(BassParams(snap=0.4, warmth=0.6), parse_pattern("C1 E1 G1"), bpm=90)
    for arr in (parts.sub, parts.warmth, parts.snap, parts.envelope, parts.mastered):
        assert len(arr) == len(parts.mastered)
    assert parts.mastered.ndim == 1
    assert np.any(parts.sub != 0)
    assert np.any(parts.snap != 0)
    assert np.any(parts.envelope != 0)


def test_decompose_stereo_matches_mastered_length():
    parts = decompose_pattern(BassParams(width=0.8), parse_pattern("C1 D1"), bpm=120)
    assert parts.mastered.ndim == 2
    assert parts.mastered.shape[0] == len(parts.sub)


def test_decompose_rest_skips_silence():
    notes = parse_pattern("C1 z1 C1")
    parts = decompose_pattern(BassParams(), notes, bpm=60)
    assert len(parts.note_midis) == 2


def test_decompose_single_note_parts_align():
    parts = decompose_single_note(BassParams(snap=0.4, warmth=0.5), freq=38.0, duration=1.0)
    assert parts.mastered.ndim == 1
    assert len(parts.sub) == len(parts.mastered)
    assert np.any(parts.sub != 0)
    assert len(parts.note_midis) == 1


def test_decompose_single_note_stereo_when_width():
    parts = decompose_single_note(BassParams(width=0.8), freq=38.0, duration=1.0)
    assert parts.mastered.ndim == 2
    assert parts.mastered.shape[1] == 2


def test_decompose_single_note_gain_scales():
    low = decompose_single_note(BassParams(), freq=38.0, duration=1.0, gain=0.2)
    high = decompose_single_note(BassParams(), freq=38.0, duration=1.0, gain=1.0)
    assert np.max(np.abs(low.mastered)) < np.max(np.abs(high.mastered))


def _loop_seam(audio: np.ndarray) -> tuple[float, float]:
    """Return (loop seam, max interior sample delta) for the first channel."""
    m = audio[:, 0] if audio.ndim == 2 else audio
    d = np.abs(np.diff(np.concatenate([m, m[:1]])))
    return float(abs(m[0] - m[-1])), float(d[:-1].max())


def test_mastered_loop_seam_is_within_interior_delta():
    # The master chain must not break the loop's periodicity: the wrap-around
    # jump (last -> first sample) must be no larger than the largest ordinary
    # sample-to-sample delta, otherwise looping clicks at the seam.
    cases = [
        (BassParams(), "C1", 70),
        (BassParams(), "C1 E1 G1", 120),
        (BassParams(), "C1 E1 G1 A1", 200),
        (BassParams(width=0.6, makeup_db=4.0, snap=0.2, warmth=0.5), "C1", 70),
        (BassParams(width=0.6, makeup_db=4.0, snap=0.2, warmth=0.5), "C1 E1 G1 A1", 140),
    ]
    for params, pattern, bpm in cases:
        audio = render_pattern(params, parse_pattern(pattern), bpm=bpm)
        seam, interior = _loop_seam(audio)
        assert seam <= interior, (pattern, bpm, seam, interior)
