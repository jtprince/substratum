import numpy as np

from substratum.utils.math import db_to_linear, linear_to_db, log_freq_mapping, midi_to_hz


def test_midi_to_hz_a440():
    assert midi_to_hz(69) == 440.0
    assert np.isclose(midi_to_hz(81), 880.0)


def test_log_freq_mapping_bounds():
    assert np.isclose(log_freq_mapping(0.0), 25.0)
    assert np.isclose(log_freq_mapping(1.0), 70.0)


def test_log_freq_mapping_is_logarithmic():
    mid = log_freq_mapping(0.5)
    low, high = log_freq_mapping(0.0), log_freq_mapping(1.0)
    assert np.isclose(mid, np.sqrt(low * high))


def test_db_to_linear_and_back():
    assert np.isclose(db_to_linear(0.0), 1.0)
    assert np.isclose(db_to_linear(-6.0), 0.5, atol=0.01)
    assert np.isclose(linear_to_db(1.0), 0.0)
    assert np.isclose(db_to_linear(linear_to_db(0.3)), 0.3, atol=1e-6)
