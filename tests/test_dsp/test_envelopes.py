import numpy as np

from substratum.dsp.envelopes import (
    adsr_envelope,
    exponential_pitch_envelope,
    pitch_envelope_from_punch,
)

SR = 48000


def test_adsr_envelope_length():
    env = adsr_envelope(1.0, SR)
    assert len(env) == SR


def test_adsr_envelope_starts_at_zero_ends_at_zero():
    env = adsr_envelope(1.0, SR)
    assert env[0] == 0.0
    assert env[-1] == 0.0


def test_adsr_envelope_reaches_unity():
    env = adsr_envelope(1.0, SR)
    assert np.max(env) == 1.0


def test_adsr_envelope_is_non_negative_and_bounded():
    env = adsr_envelope(0.5, SR, decay=0.2, release=0.05)
    assert np.all(env >= 0.0)
    assert np.all(env <= 1.0)


def test_adsr_envelope_sustain_plateau():
    env = adsr_envelope(1.0, SR, attack=0.05, decay=0.2, sustain=0.7, release=0.2)
    plateau = env[int(0.3 * SR) : int(0.6 * SR)]
    assert np.allclose(plateau, 0.7, atol=1e-2)


def test_exponential_pitch_envelope_decays_to_unity():
    env = exponential_pitch_envelope(0.5, SR, start_ratio=2.0, decay_time=0.1)
    assert np.isclose(env[0], 2.0)
    assert np.isclose(env[-1], 1.0, atol=1e-2)


def test_exponential_pitch_envelope_is_decreasing():
    env = exponential_pitch_envelope(0.5, SR, start_ratio=2.0, decay_time=0.1)
    assert np.all(np.diff(env) <= 1e-6)


def test_punch_zero_is_flat():
    env = pitch_envelope_from_punch(0.0, 0.5, SR)
    assert np.allclose(env, 1.0)


def test_punch_increases_start_ratio_and_is_decreasing():
    env = pitch_envelope_from_punch(1.0, 0.5, SR)
    assert env[0] > 1.0
    assert np.all(np.diff(env) <= 1e-6)
    assert np.isclose(env[-1], 1.0, atol=0.02)
