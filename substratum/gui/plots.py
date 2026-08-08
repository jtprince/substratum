"""Matplotlib decomposition plot, rendered to a kitty terminal.

The figure stacks the pattern's piano-roll strip, the ADSR envelope, and the
sub / warmth / snap layers plus the mastered output so each musical layer is
visible against the same time axis. It is drawn with the Agg backend (no
window) and pushed to the terminal via ``kitty +kitten icat`` from inside a
Textual ``App.suspend()`` block.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from matplotlib import use as mpl_use

mpl_use("Agg")

from matplotlib.figure import Figure  # noqa: E402  (after backend selection)

from substratum.bass.synth import BassParams
from substratum.pattern.arrange import PatternParts
from substratum.pattern.notation import note_label

#: Set by kitty for its own processes; lets us detect the terminal.
KITTY_WINDOW_ID = "KITTY_WINDOW_ID"


def decomposition_figure(
    parts: PatternParts,
    params: BassParams,
    bpm: float,
) -> Figure:
    """Build the 6-panel decomposition figure for ``parts``."""
    p = params.validated()
    fs = p.sample_rate
    n = len(parts.mastered)
    t = np.arange(n) / fs
    loop_seconds = n / fs

    fig = Figure(figsize=(11, 9), dpi=110)
    fig.subplots_adjust(hspace=0.55, left=0.06, right=0.97, top=0.96, bottom=0.07)

    # 1. Piano-roll strip.
    ax0 = fig.add_subplot(6, 1, 1)
    ax0.set_title(
        f"Pattern decomposition — {bpm:.0f} bpm, {loop_seconds:.2f}s loop",
        fontsize=11,
    )
    ax0.set_yticks([])
    ax0.set_xlim(0, loop_seconds)
    pos = 0.0
    for midi, beats in zip(parts.note_midis, parts.note_beats, strict=True):
        if midi is None:
            pos += beats * 60.0 / bpm
            continue
        ax0.barh(0.5, beats * 60.0 / bpm, left=pos, height=0.7, color="#7aa2f7")
        ax0.text(
            pos + beats * 60.0 / (2.0 * bpm),
            0.5,
            note_label(midi),
            ha="center",
            va="center",
            fontsize=7,
        )
        pos += beats * 60.0 / bpm
    ax0.set_ylim(0, 1)
    ax0.set_yticks([])

    layers = [
        ("ADSR envelope", parts.envelope, "#e0af68"),
        ("Sub fundamental", parts.sub, "#9ece6a"),
        ("Warmth / harmonics", parts.warmth, "#bb9af7"),
        ("Snap transient", parts.snap, "#f7768e"),
    ]
    for i, (name, signal, color) in enumerate(layers):
        ax = fig.add_subplot(6, 1, i + 2, sharex=ax0)
        ax.fill_between(t, signal, color=color, alpha=0.45, linewidth=0)
        ax.plot(t, signal, color=color, linewidth=0.7)
        ax.set_title(name, fontsize=9, loc="left")
        ax.set_yticks([])
        ax.set_xlim(0, loop_seconds)

    # 6. Mastered output.
    ax_mix = fig.add_subplot(6, 1, 6, sharex=ax0)
    mono = parts.mastered[:, 0] if parts.mastered.ndim == 2 else parts.mastered
    ax_mix.fill_between(t, mono, color="#c0caf5", alpha=0.4, linewidth=0)
    ax_mix.plot(t, mono, color="#a9b1d6", linewidth=0.7)
    ax_mix.set_title("Mastered output", fontsize=9, loc="left")
    ax_mix.set_yticks([])
    ax_mix.set_xlabel("time (s)")
    ax_mix.set_xlim(0, loop_seconds)

    return fig


def figure_to_png(fig: Figure) -> bytes:
    """Render a figure to PNG bytes."""
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return buf.getvalue()


def show_in_kitty(png: bytes, *, kitty: str = "kitty") -> None:
    """Display ``png`` in the terminal via ``kitty +kitten icat``.

    Blocks until the user presses a key to dismiss the image. Called from
    inside a Textual ``App.suspend()`` block.
    """
    if not os.environ.get(KITTY_WINDOW_ID):
        raise RuntimeError("not running under kitty (KITTY_WINDOW_ID unset)")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bass.png"
        path.write_bytes(png)
        subprocess.run(
            [kitty, "+kitten", "icat", "--hold", str(path)],
            check=True,
        )


def plot_pattern(
    parts: PatternParts,
    params: BassParams,
    bpm: float,
    *,
    kitty: str = "kitty",
) -> None:
    """Render the decomposition and push it to the kitty terminal."""
    show_in_kitty(figure_to_png(decomposition_figure(parts, params, bpm)), kitty=kitty)
