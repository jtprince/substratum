"""Exploration mode: batch rendering across the parameter space.

Sweeps each of the primary controls (punch, drive, warmth, weight) over a
grid of frequencies, writing one WAV per combination so the effect of each
knob can be heard directly.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from substratum.bass.synth import BassParams, render
from substratum.io.audio import write_wav

DEFAULT_FREQS = (33.0, 38.0, 45.0)
DEFAULT_STEP = 0.2
PARAM_NAMES = ("punch", "drive", "warmth", "weight")


def _values(start: float, stop: float, step: float) -> tuple[float, ...]:
    vals = []
    v = start
    while v <= stop + 1e-9:
        vals.append(round(v, 3))
        v += step
    return tuple(vals)


def sweep(
    out_dir: str | Path,
    freqs: tuple[float, ...] = DEFAULT_FREQS,
    step: float = DEFAULT_STEP,
    params: tuple[str, ...] = PARAM_NAMES,
    base: BassParams | None = None,
) -> list[Path]:
    """Render a WAV for every (freq, param, value) combination.

    Returns the list of written files. Naming follows the spec:
    ``freq{freq:02.0f}_{param}{value:02.0f}.wav``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = base or BassParams()
    written: list[Path] = []

    for freq in freqs:
        for param in params:
            if param not in PARAM_NAMES:
                raise ValueError(f"Unknown parameter '{param}'. Valid: {PARAM_NAMES}")
            for value in _values(0.0, 1.0, step):
                patch = replace(base, freq=freq, description="")
                setattr(patch, param, value)
                audio = render(patch)
                name = f"freq{freq:02.0f}_{param}{value * 100:02.0f}.wav"
                path = out / name
                write_wav(path, audio, patch.sample_rate)
                written.append(path)

    return written
