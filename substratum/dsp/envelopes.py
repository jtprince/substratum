"""Envelope generators for amplitude and pitch."""

import numpy as np


def adsr_envelope(
    duration: float,
    sample_rate: int = 48000,
    attack: float = 0.002,
    decay: float = 0.85,
    sustain: float = 0.0,
    release: float = 0.3,
) -> np.ndarray:
    """Generate ADSR amplitude envelope.

    Args:
        duration: Total duration in seconds
        sample_rate: Sample rate
        attack: Attack time in seconds
        decay: Decay time in seconds
        sustain: Sustain level (0-1)
        release: Release time in seconds

    Returns:
        Envelope array
    """
    total_samples = int(duration * sample_rate)
    if total_samples <= 0:
        return np.zeros(0)

    attack_samples = min(int(attack * sample_rate), total_samples)
    release_samples = min(int(release * sample_rate), total_samples - attack_samples)
    decay_samples = min(int(decay * sample_rate), total_samples - attack_samples - release_samples)
    sustain_samples = total_samples - attack_samples - decay_samples - release_samples

    env = np.zeros(total_samples)

    if attack_samples > 0:
        env[:attack_samples] = np.linspace(0, 1, attack_samples)

    if decay_samples > 0:
        start = attack_samples
        end = start + decay_samples
        env[start:end] = np.linspace(1, sustain, decay_samples)

    if sustain_samples > 0:
        start = attack_samples + decay_samples
        env[start : start + sustain_samples] = sustain

    if release_samples > 0:
        start = total_samples - release_samples
        env[start:] = np.linspace(sustain, 0, release_samples)

    return env


def exponential_pitch_envelope(
    duration: float,
    sample_rate: int = 48000,
    start_ratio: float = 2.75,  # e.g., 110Hz / 40Hz
    decay_time: float = 0.1,
) -> np.ndarray:
    """Generate exponential pitch envelope (pitch drops from start_ratio to 1.0).

    Args:
        duration: Total duration in seconds
        sample_rate: Sample rate
        start_ratio: Starting pitch ratio relative to fundamental
        decay_time: Decay time constant in seconds

    Returns:
        Pitch envelope array (multiplier for frequency)
    """
    total_samples = int(duration * sample_rate)
    t = np.arange(total_samples) / sample_rate
    # Exponential decay from start_ratio to 1.0
    env = 1.0 + (start_ratio - 1.0) * np.exp(-t / decay_time)
    return env


def pitch_envelope_from_punch(
    punch: float,
    duration: float,
    sample_rate: int = 48000,
    base_freq: float = 40.0,
) -> np.ndarray:
    """Generate pitch envelope based on punch parameter (0-1).

    Args:
        punch: Punch amount (0 = no pitch sweep, 1 = maximum)
        duration: Total duration in seconds
        sample_rate: Sample rate
        base_freq: Base frequency in Hz

    Returns:
        Pitch envelope array
    """
    if punch <= 0:
        return np.ones(int(duration * sample_rate))

    # Map punch to start ratio: 0 -> 1.0, 1 -> ~2.75 (110/40)
    start_ratio = 1.0 + punch * 1.75
    # Map punch to decay time: 0 -> very fast, 1 -> ~100ms
    decay_time = 0.01 + punch * 0.09

    return exponential_pitch_envelope(duration, sample_rate, start_ratio, decay_time)
