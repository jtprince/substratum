"""Tests for the matplotlib decomposition plot + kitty rendering."""

import numpy as np
import pytest

from substratum.bass.synth import BassParams
from substratum.gui import plots
from substratum.pattern.arrange import decompose_pattern
from substratum.pattern.notation import parse_pattern


def test_decomposition_figure_is_png():
    parts = decompose_pattern(BassParams(snap=0.3, warmth=0.5), parse_pattern("C1 E1 G1"), bpm=90)
    fig = plots.decomposition_figure(parts, BassParams(), bpm=90)
    png = plots.figure_to_png(fig)
    assert png.startswith(b"\x89PNG")
    assert len(png) > 1000


def test_show_in_kitty_requires_kitty_env(monkeypatch):
    monkeypatch.delenv(plots.KITTY_WINDOW_ID, raising=False)
    with pytest.raises(RuntimeError, match="kitty"):
        plots.show_in_kitty(b"\x89PNGtest")


def test_show_in_kitty_runs_icat(monkeypatch):
    monkeypatch.setenv(plots.KITTY_WINDOW_ID, "1")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        assert cmd[:4] == ["kitty", "+kitten", "icat", "--hold"]
        return None

    monkeypatch.setattr(plots.subprocess, "run", fake_run)
    plots.show_in_kitty(b"\x89PNGtest")
    assert calls


def test_plot_pattern_end_to_end(monkeypatch):
    parts = decompose_pattern(BassParams(), parse_pattern("C1"), bpm=70)
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["png"] = __import__("pathlib").Path(cmd[-1]).read_bytes()

    monkeypatch.setattr(plots.subprocess, "run", fake_run)
    monkeypatch.setenv(plots.KITTY_WINDOW_ID, "1")
    plots.plot_pattern(parts, BassParams(), bpm=70)
    assert seen["png"].startswith(b"\x89PNG")


def test_decomposition_figure_matches_loop_length():
    parts = decompose_pattern(BassParams(), parse_pattern("C1 E1 G1"), bpm=120)
    fig = plots.decomposition_figure(parts, BassParams(), bpm=120)
    axis = fig.axes[0]
    assert axis.get_xlim() == (0.0, pytest.approx(len(parts.mastered) / BassParams().sample_rate))
    assert len(fig.axes) == 6
    _ = np.asarray(fig.canvas.get_width_height())
