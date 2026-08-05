"""Analysis and visualization.

Reads a rendered WAV and produces PNG/SVG figures: waveform, FFT spectrum,
harmonic amplitudes, amplitude envelope and spectrogram. Uses the Agg
matplotlib backend so it works headless and in scripts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import spectrogram as specgram  # noqa: E402

from substratum.io.audio import read_wav  # noqa: E402


def estimate_fundamental(audio: np.ndarray, sample_rate: int) -> float:
    """Estimate the dominant fundamental frequency in the 20-120 Hz band."""
    n = len(audio)
    if n < 2:
        return 0.0
    windowed = audio * np.hanning(n)
    spec = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    mask = (freqs >= 20.0) & (freqs <= 120.0)
    if not mask.any():
        return 0.0
    idx = int(np.argmax(spec[mask]))
    idx_abs = int(np.where(mask)[0][idx])
    return float(freqs[idx_abs])


def harmonic_series(
    audio: np.ndarray,
    sample_rate: int,
    fundamental_hz: float,
    num_harmonics: int = 12,
) -> list[float]:
    """Return per-harmonic peak amplitude normalized to the fundamental."""
    n = len(audio)
    windowed = audio * np.hanning(n)
    spec = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    if fundamental_hz <= 0.0:
        return [0.0] * num_harmonics
    band = fundamental_hz * 0.5
    amplitudes: list[float] = []
    for k in range(1, num_harmonics + 1):
        center = fundamental_hz * k
        mask = (freqs >= center - band) & (freqs <= center + band)
        if mask.any():
            idx = int(np.argmax(spec[mask]))
            amplitudes.append(float(spec[mask][idx]))
        else:
            amplitudes.append(0.0)
    peak = max(amplitudes) if amplitudes else 0.0
    if peak > 0.0:
        amplitudes = [a / peak for a in amplitudes]
    return amplitudes


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=150)
    fig.savefig(out_dir / f"{name}.svg")
    plt.close(fig)
    return out_dir / f"{name}.png"


def waveform_figure(audio: np.ndarray, sample_rate: int, out_dir: Path) -> Path:
    """Waveform over the full duration."""
    t = np.arange(len(audio)) / sample_rate
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(t, audio, linewidth=0.6)
    ax.set_title("Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, t[-1])
    return _save(fig, out_dir, "waveform")


def spectrum_figure(audio: np.ndarray, sample_rate: int, out_dir: Path) -> Path:
    """FFT magnitude spectrum, 0-1 kHz."""
    n = len(audio)
    windowed = audio * np.hanning(n)
    spec = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    limit = freqs <= 1000.0
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.semilogx(freqs[limit], 20 * np.log10(spec[limit] + 1e-10))
    ax.set_title("Spectrum (FFT)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, which="both", alpha=0.3)
    return _save(fig, out_dir, "spectrum")


def harmonics_figure(
    audio: np.ndarray,
    sample_rate: int,
    fundamental_hz: float,
    out_dir: Path,
) -> Path:
    """Bar chart of harmonic amplitudes."""
    amps = harmonic_series(audio, sample_rate, fundamental_hz)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.bar(range(1, len(amps) + 1), amps)
    ax.set_title(f"Harmonic amplitudes (fundamental {fundamental_hz:.1f} Hz)")
    ax.set_xlabel("Harmonic")
    ax.set_ylabel("Relative amplitude")
    ax.set_xlim(0, len(amps) + 1)
    return _save(fig, out_dir, "harmonics")


def envelope_figure(audio: np.ndarray, sample_rate: int, out_dir: Path) -> Path:
    """Amplitude envelope computed from a rolling RMS window."""
    window = int(0.02 * sample_rate)
    if window < 1 or window > len(audio):
        rms = np.abs(audio)
    else:
        sq = np.convolve(audio**2, np.ones(window) / window, mode="same")
        rms = np.sqrt(np.maximum(sq, 0))
    t = np.arange(len(rms)) / sample_rate
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(t, rms, linewidth=1.2)
    ax.set_title("Amplitude envelope (20 ms RMS)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS amplitude")
    ax.set_xlim(0, t[-1])
    return _save(fig, out_dir, "envelope")


def spectrogram_figure(audio: np.ndarray, sample_rate: int, out_dir: Path) -> Path:
    """Spectrogram focused on the sub-bass region."""
    f, t, s = specgram(audio, fs=sample_rate, nperseg=2048, noverlap=1024)
    fig, ax = plt.subplots(figsize=(9, 4))
    db = 10 * np.log10(s + 1e-12)
    im = ax.pcolormesh(t, f, db, shading="gouraud", vmin=-80, vmax=0, cmap="magma")
    ax.set_title("Spectrogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0, 600)
    fig.colorbar(im, ax=ax, label="dB")
    return _save(fig, out_dir, "spectrogram")


def analyze(path: str | Path, out_dir: str | Path = "output/analysis") -> list[Path]:
    """Produce all figures for a rendered WAV and return their paths."""
    audio, sample_rate = read_wav(path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fundamental = estimate_fundamental(audio, sample_rate)

    figures = [
        waveform_figure(audio, sample_rate, out),
        spectrum_figure(audio, sample_rate, out),
        harmonics_figure(audio, sample_rate, fundamental, out),
        envelope_figure(audio, sample_rate, out),
        spectrogram_figure(audio, sample_rate, out),
    ]
    return figures
