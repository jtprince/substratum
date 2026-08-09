"""Headless smoke tests for the Textual GUI (mount + basic interactions)."""

import asyncio

import pytest

from substratum.gui.app import BassGui
from substratum.gui.widgets import HoverSlider, MiniWaveform, PianoRoll
from substratum.pattern.notation import parse_pattern


def _has_output_device() -> bool:
    """True when sounddevice can open a default output stream."""
    try:
        import sounddevice as sd

        sd.query_devices(kind="output")
        return True
    except Exception:  # noqa: BLE001
        return False


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


def test_slider_click_jumps_to_position(tmp_path):
    from textual import events

    app = BassGui(pattern="C1", samples_dir=tmp_path)

    async def body(app, pilot) -> None:  # noqa: F811
        slider = app.query_one("#sl-punch", HoverSlider)
        x0, x1 = slider._bar_x0(), slider._bar_x1()
        span = x1 - x0
        assert slider.min <= 0 and slider.max >= 1
        mid = x0 + span / 2
        slider.on_mouse_down(events.MouseDown(slider, mid, 0, 0, 0, 1, False, False, False))
        assert abs(slider.value - 0.5) < 1e-6
        # A click past the bar clamps to the maximum.
        slider.on_mouse_down(events.MouseDown(slider, x1 + 20, 0, 0, 0, 1, False, False, False))
        assert slider.value == slider.max
        # A click on the label leaves the value untouched.
        before = slider.value
        slider.on_mouse_down(events.MouseDown(slider, x0 - 5, 0, 0, 0, 1, False, False, False))
        assert slider.value == before

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


def test_waveform_and_roll_grow_to_content_height(tmp_path):
    # The wave/roll mount with placeholder content (1 line); after the first
    # render their multi-line bodies must trigger a re-layout, not stay 1 tall.
    app = BassGui(pattern="C1 E1 G1", samples_dir=tmp_path)

    async def body(app, pilot) -> None:  # noqa: F811
        await pilot.pause(0.5)
        wave = app.query_one("#wave")
        roll = app.query_one("#roll")
        assert wave.size.height >= 16
        assert roll.size.height >= 9

    _run(app, body)


def test_waveform_does_not_wrap_at_narrow_width(tmp_path):
    # At a very narrow window the waveform must keep its 16-row shape (clipped
    # at the panel edge), not wrap into a tall mangled block.
    app = BassGui(pattern="C1 E1 G1", samples_dir=tmp_path)

    async def body(app, pilot) -> None:  # noqa: F811
        await pilot.pause(0.5)
        wave = app.query_one("#wave")
        assert wave.size.height == 16

    _run_sized(app, body, size=(60, 50))


def _run_sized(app: BassGui, body, size):
    async def main() -> None:
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            await body(app, pilot)

    asyncio.run(main())


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


@pytest.mark.skipif(not _has_output_device(), reason="needs an audio device")
def test_play_toggle_does_not_crash(tmp_path):
    app = BassGui(pattern="C1", samples_dir=tmp_path)

    async def body(app, pilot) -> None:  # noqa: F811
        app.action_toggle_play()
        await pilot.pause(0.2)
        assert app.playing
        app.action_toggle_play()
        await pilot.pause()
        assert not app.playing
        app.action_toggle_play()
        await pilot.pause(0.2)
        app.exit()  # must not deadlock in on_unmount while the stream is live

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


def test_piano_roll_marks_detuned_notes():
    roll = PianoRoll()
    roll.set_pattern(parse_pattern("C1"), bpm=70.0, transpose=0.0)
    plain = roll.render().plain
    assert "■" in plain and "▲" not in plain and "▼" not in plain

    roll.set_pattern(parse_pattern("C1"), bpm=70.0, transpose=0.5)
    plain = roll.render().plain
    assert "▲" in plain and "■" not in plain

    roll.set_pattern(parse_pattern("C1"), bpm=70.0, transpose=-0.5)
    plain = roll.render().plain
    assert "▼" in plain and "■" not in plain

    roll.set_pattern(parse_pattern("C1 C2"), bpm=70.0, transpose=0.25)
    assert "▲" in roll.render().plain


def test_fractional_pitch_slider_renders_and_plays(tmp_path):
    app = BassGui(pattern="C1 E1 G1", samples_dir=tmp_path)

    async def body(app, pilot) -> None:  # noqa: F811
        slider = app.query_one("#sl-pitch", HoverSlider)
        slider.set_value(0.5, notify=True)
        await pilot.pause(0.4)
        assert slider.value == 0.5
        assert app.audio is not None and len(app.audio) > 0
        assert "▲" in app.query_one("#roll", PianoRoll).render().plain

    _run(app, body)


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


def _run_player_subprocess(body: str) -> str:
    """Run a player exercise in a subprocess (a close/stop deadlock would
    hang the subprocess and be killed by the timeout instead of pytest)."""
    import subprocess
    import sys

    code = (
        "import time\n"
        "import numpy as np\n"
        "from substratum.gui.player import LoopPlayer\n"
        "p = LoopPlayer(48000)\n"
        "sr = 48000\n"
        f"a = np.sin(2 * np.pi * 40 * np.arange(sr) / sr)\n"
        f"{body}\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
    return result.stdout


@pytest.mark.skipif(not _has_output_device(), reason="needs an audio device")
def test_loop_player_close_while_playing_does_not_deadlock():
    _run_player_subprocess("p.update(a); p.start(); time.sleep(0.2); p.close()")


@pytest.mark.skipif(not _has_output_device(), reason="needs an audio device")
def test_loop_player_stop_and_restart_do_not_deadlock():
    _run_player_subprocess(
        "p.update(a); p.start(); time.sleep(0.15); p.stop(); "
        "time.sleep(0.05); p.start(); time.sleep(0.15); p.close()"
    )
