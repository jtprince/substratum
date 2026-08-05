"""Generate all figures for docs/bass-theory.md.

Reproducible: run ``uv run python docs/generate_figures.py`` and every image
referenced by the theory document is regenerated from the actual DSP code
into ``docs/figures/`` as PNG + SVG.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import spectrogram as specgram  # noqa: E402

from substratum.bass.presets import PRESETS  # noqa: E402
from substratum.bass.synth import (  # noqa: E402
    _weight_decay,
    _weight_limiter,
    _weight_lowpass,
    _weight_sub_level,
    render,
)
from substratum.dsp.envelopes import pitch_envelope_from_punch  # noqa: E402
from substratum.dsp.saturation import atan_saturate, tanh_saturate  # noqa: E402
from substratum.utils.math import db_to_linear, log_freq_mapping  # noqa: E402

FIG_DIR = Path(__file__).parent / "figures"
SR = 48000


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150)
    fig.savefig(FIG_DIR / f"{name}.svg")
    plt.close(fig)


def fig_freq_log() -> None:
    values = np.linspace(0, 1, 100)
    hz = np.array([log_freq_mapping(v) for v in values])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(values, hz, linewidth=2)
    ax.axhline(25, color="grey", ls="--", lw=0.8)
    ax.axhline(70, color="grey", ls="--", lw=0.8)
    ax.set_title("Frequency knob: logarithmic mapping")
    ax.set_xlabel("Knob position (0-1)")
    ax.set_ylabel("Frequency (Hz)")
    ax.grid(True, alpha=0.3)
    _save(fig, "freq_log")


def fig_pitch_envelope() -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for punch in (0.25, 0.5, 0.75, 1.0):
        env = pitch_envelope_from_punch(punch, 0.5, SR)
        t = np.arange(len(env)) / SR
        ax.plot(t, 40 * env, label=f"punch={punch}")
    ax.set_title("Pitch envelope: frequency over time (base 40 Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Instantaneous frequency (Hz)")
    ax.set_ylim(35, 120)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "pitch_envelope")


def fig_saturation() -> None:
    x = np.linspace(-3, 3, 1000)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, x, color="grey", ls="--", lw=1, label="linear")
    for drive in (0.3, 0.7, 1.0):
        ax.plot(x, tanh_saturate(x, drive), label=f"tanh drive={drive}")
    ax.plot(x, atan_saturate(x, 1.0), ls=":", label="atan drive=1.0")
    ax.set_title("Saturation curves")
    ax.set_xlabel("Input")
    ax.set_ylabel("Output")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "saturation")


def fig_warmth_harmonics() -> None:
    warmth = np.linspace(0, 1, 100)
    octave = warmth * db_to_linear(-22 + 12 * warmth)
    tri = 0.08 * warmth
    h3 = warmth * db_to_linear(-26 + 10 * warmth)
    h4 = warmth * db_to_linear(-30 + 12 * warmth)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(warmth, octave, label="octave (2f)")
    ax.plot(warmth, tri, label="triangle")
    ax.plot(warmth, h3, label="3rd harmonic")
    ax.plot(warmth, h4, label="4th harmonic")
    ax.set_title("Warmth: component amplitudes vs warmth knob")
    ax.set_xlabel("Warmth")
    ax.set_ylabel("Linear amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "warmth_harmonics")


def fig_weight_macro() -> None:
    weight = np.linspace(0, 1, 100)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    axes[0, 0].plot(weight, [_weight_decay(0.85, w) for w in weight])
    axes[0, 0].set_title("Decay time (s)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(weight, [_weight_sub_level(w) for w in weight])
    axes[0, 1].set_title("Sub oscillator level")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(weight, [_weight_lowpass(0.35, w) for w in weight])
    axes[1, 0].set_title("Tone low-pass cutoff (Hz)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(weight, [_weight_limiter(w)[0] for w in weight])
    axes[1, 1].set_title("Limiter threshold")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Weight macro: what it adjusts")
    for ax in axes.flat:
        ax.set_xlabel("Weight")
    _save(fig, "weight_macro")


def fig_preset_spectrograms() -> None:
    names = list(PRESETS)
    fig, axes = plt.subplots(1, len(names), figsize=(16, 3.5))
    for ax, name in zip(axes, names, strict=False):
        audio = render(PRESETS[name])
        f, t, s = specgram(audio, fs=SR, nperseg=2048, noverlap=1024)
        ax.pcolormesh(t, f, 10 * np.log10(s + 1e-12), vmin=-80, vmax=0, cmap="magma")
        ax.set_title(name)
        ax.set_ylim(0, 500)
        ax.set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    fig.tight_layout()
    _save(fig, "preset_spectrograms")


def main() -> None:
    fig_freq_log()
    fig_pitch_envelope()
    fig_saturation()
    fig_warmth_harmonics()
    fig_weight_macro()
    fig_preset_spectrograms()
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
