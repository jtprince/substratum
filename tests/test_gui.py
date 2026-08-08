"""Headless smoke tests for the Textual GUI (mount + basic interactions)."""

import asyncio

import pytest

from substratum.gui.app import BassGui
from substratum.gui.widgets import HoverSlider, MiniWaveform, PianoRoll


def _run(app: BassGui, body):
    async def main() -> None:
        async with app.run_test(size=(150, 50)) as pilot:
            await pilot.pause()
            await body(app, pilot)

    asyncio.run(main())


def test_gui_mounts_and_renders_default(tmp_path):
    app = BassGui(pattern="C1", samples_dir=tmp_path)

    async def body(app, pilot) -> None:  # noqa: F811
        assert app.query_one("#pattern").text == "C1"
        assert app.audio is not None
        assert len(app.audio) > 0
        assert app.query_one("#roll", PianoRoll).notes
        assert app.parts is not None
        assert app.query_one("#wave", MiniWaveform).parts is not None

    _run(app, body)


def test_slider_change_schedules_rerender(tmp_path):
    app = BassGui(pattern="C1 E1 G1", samples_dir=tmp_path)

    async def body(app, pilot) -> None:  # noqa: F811
        slider = app.query_one("#sl-punch", HoverSlider)
        before = app.audio
        slider.set_value(0.95, notify=True)
        await pilot.pause(0.4)
        assert app.audio is not None
        assert before is None or len(app.audio) > 0

    _run(app, body)


def test_invalid_pattern_shows_error_and_falls_back(tmp_path):
    app = BassGui(pattern="C1 Q9", samples_dir=tmp_path)

    async def body(app, pilot) -> None:  # noqa: F811
        assert "bad note" in str(app.query_one("#noterr").render())
        assert app.audio is not None

    _run(app, body)


def test_notation_help_text_is_shown(tmp_path):
    app = BassGui(pattern="C1", samples_dir=tmp_path)

    async def body(app, pilot) -> None:  # noqa: F811
        help_text = str(app.query_one("#nothelp").render())
        assert "z/r" in help_text
        assert "C2 z1 E1" in help_text

    _run(app, body)


def test_plot_without_kitty_does_not_invoke_icat(tmp_path, monkeypatch):
    app = BassGui(pattern="C1", samples_dir=tmp_path)
    calls = []
    monkeypatch.setattr("substratum.gui.plots.plot_pattern", lambda *a, **k: calls.append(1))
    monkeypatch.delenv("KITTY_WINDOW_ID", raising=False)

    async def body(app, pilot) -> None:  # noqa: F811
        assert app.parts is not None
        app.action_plot()
        assert not calls
        assert "kitty" in str(app.query_one("#status").render())

    _run(app, body)


def test_plot_with_kitty_invokes_plot_pattern(tmp_path, monkeypatch):
    app = BassGui(pattern="C1", samples_dir=tmp_path)
    calls = []
    monkeypatch.setenv("KITTY_WINDOW_ID", "1")
    monkeypatch.setattr("substratum.gui.plots.plot_pattern", lambda *a, **k: calls.append(1))

    async def body(app, pilot) -> None:  # noqa: F811
        app.action_plot()
        assert calls == [1]
        assert "plot pushed" in str(app.query_one("#status").render())

    _run(app, body)


def test_empty_pattern_still_shows_waveform_and_plot(tmp_path, monkeypatch):
    app = BassGui(pattern="", samples_dir=tmp_path)
    calls = []
    monkeypatch.setenv("KITTY_WINDOW_ID", "1")
    monkeypatch.setattr("substratum.gui.plots.plot_pattern", lambda *a, **k: calls.append(1))

    async def body(app, pilot) -> None:  # noqa: F811
        assert app.parts is not None
        assert app.audio is not None
        wave = app.query_one("#wave", MiniWaveform)
        lines = wave.render().plain.split("\n")
        assert len(lines) > 5
        assert any("█" in line for line in lines)
        app.action_plot()
        assert calls == [1]

    _run(app, body)


def test_invalid_pattern_still_shows_waveform(tmp_path):
    app = BassGui(pattern="C1 Q9", samples_dir=tmp_path)

    async def body(app, pilot) -> None:  # noqa: F811
        assert app.parts is not None
        wave = app.query_one("#wave", MiniWaveform)
        assert any("█" in line for line in wave.render().plain.split("\n"))

    _run(app, body)


def test_save_and_load_via_app(tmp_path):
    app = BassGui(pattern="C1 D1", samples_dir=tmp_path)

    async def body(app, pilot) -> None:  # noqa: F811
        app.query_one("#savename").value = "smoke_test"
        app.action_save()
        await pilot.pause()
        assert (tmp_path / "smoke_test.mp3").exists()
        # restore default pattern, then load back
        app.query_one("#pattern").text = ""
        app._load_saved()
        await pilot.pause()
        assert app.query_one("#pattern").text == "C1 D1"

    _run(app, body)


@pytest.mark.skip(reason="needs an audio device")
def test_play_toggle_does_not_crash(tmp_path):
    app = BassGui(pattern="C1", samples_dir=tmp_path)

    async def body(app, pilot) -> None:  # noqa: F811
        app.action_toggle_play()
        await pilot.pause()
        app.action_toggle_play()
        await pilot.pause()

    _run(app, body)


def test_fill_wave_half_block_resolution():
    import numpy as np

    from substratum.gui.widgets import _fill_wave

    n = 96000
    t = np.arange(n) / 48000
    sig = np.sin(2 * np.pi * 40 * t) * np.exp(-t / 0.9)
    lines = _fill_wave(sig, cols=100, rows=12).split("\n")
    assert len(lines) == 12
    assert all(len(line) == 100 for line in lines)
    # A full-scale sine crosses both halves of the vertical range.
    assert "█" in lines[0] and "█" in lines[-1]
    # Half-block chars appear (vertical resolution beyond 8 bands).
    assert any(c in "▀▄" for line in lines for c in line)


def test_loop_player_loops_and_crossfades():
    import numpy as np

    from substratum.gui.player import LoopPlayer

    sr = 48000
    n = sr // 4
    player = LoopPlayer(sr)
    old = player._as_stereo(np.sin(2 * np.pi * 40 * np.arange(n) / sr))
    player._buffer = np.ascontiguousarray(old, np.float32)

    # Loop wraps: consecutive blocks reproduce the buffer, then sample 0 again.
    out = np.zeros((n // 2, 2), np.float32)
    player._callback(out, n // 2, None, None)
    player._callback(out, n // 2, None, None)
    assert np.allclose(out[:, 0], old[: n // 2, 0], atol=1e-6)
    next4 = np.zeros((4, 2), np.float32)
    player._callback(next4, 4, None, None)
    assert np.allclose(next4[:, 0], old[:4, 0], atol=1e-6)

    # Buffer swap crossfades exactly over FADE_SECONDS.
    new = player._as_stereo(np.sin(2 * np.pi * 55 * np.arange(n) / sr))
    new = np.ascontiguousarray(new, np.float32)
    frame_at_swap = player._frame
    player._running = True
    player.update(new)
    fade_total = player._fade_total
    steps = 32
    blended = []
    while player._old is not None:
        b = np.zeros((steps, 2), np.float32)
        player._callback(b, steps, None, None)
        blended.append(b[:, 0].copy())
    blended = np.concatenate(blended)
    ideal = np.zeros_like(blended)
    for i in range(fade_total):
        g = 1.0 - i / fade_total
        ideal[i] = old[(frame_at_swap + i) % n, 0] * g + new[(frame_at_swap + i) % n, 0] * (1.0 - g)
    assert len(blended) == fade_total
    assert np.allclose(blended, ideal, atol=1e-6)

    player.close()
