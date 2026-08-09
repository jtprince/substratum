"""Reusable widgets for the bass GUI.

- :class:`HoverSlider`: a labeled slider changed by hover + mouse wheel,
  click-drag, or keyboard. Posts a ``Changed`` message on user input.
- :class:`PianoRoll`: renders the parsed pattern as an ASCII piano-roll bar.
- :class:`MiniWaveform`: a compact ASCII waveform of the rendered audio.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from rich.text import Text
from textual import events
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static

from substratum.pattern.arrange import PatternParts
from substratum.pattern.notation import Note, note_label

_BLOCK_CHARS = " ▁▂▃▄▅▆▇█"

#: Width of the label column and the filled bar (kept in sync with render()).
_LABEL_WIDTH = 14
_BAR_WIDTH = 14


class HoverSlider(Static, can_focus=True):
    """A labeled 0-1-style slider driven by hover + wheel / drag / keys."""

    class Changed(Message):
        def __init__(self, key: str, value: float) -> None:
            self.key = key
            self.value = value
            super().__init__()

    def __init__(
        self,
        key: str,
        label: str,
        *,
        min: float,
        max: float,
        value: float,
        step: float,
        fmt: str = "{:.2f}",
        units: str = "",
        zero_label: str = "",
        id: str | None = None,
        name: str | None = None,
        classes: str | None = None,
    ) -> None:
        self.key = key
        self.label_text = label
        self.min = min
        self.max = max
        self.step = step
        self.fmt = fmt
        self.units = units
        self.zero_label = zero_label
        self._dragging = False
        self._last_y = 0
        super().__init__(id=id, name=name, classes=classes)
        self.value = self._clamp(value)

    def _clamp(self, value: float) -> float:
        return float(np.clip(value, self.min, self.max))

    def set_value(self, value: float, notify: bool = False) -> None:
        self.value = self._clamp(value)
        self.refresh()
        if notify:
            self.post_message(self.Changed(self.key, self.value))

    def _step_amount(self, shift: bool) -> float:
        return self.step / 5.0 if shift else self.step

    def _change(self, delta: float, shift: bool) -> None:
        self.set_value(self.value + delta * self._step_amount(shift), notify=True)

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        self._change(+1.0, event.shift)
        event.stop()

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        self._change(-1.0, event.shift)
        event.stop()

    def _bar_x0(self) -> int:
        """Column of the bar's opening bracket."""
        return _LABEL_WIDTH

    def _bar_x1(self) -> int:
        """Column just past the bar's closing bracket."""
        return _LABEL_WIDTH + _BAR_WIDTH + 2

    def on_mouse_down(self, event: events.MouseDown) -> None:
        self._dragging = True
        self._last_y = event.y
        # Clicking on (or right of) the bar jumps the value there; clicks on
        # the label leave it untouched.
        if event.x >= self._bar_x0():
            frac = float(np.clip((event.x - self._bar_x0()) / (_BAR_WIDTH + 2), 0.0, 1.0))
            self.set_value(self.min + frac * (self.max - self.min), notify=True)
        self.focus()
        self.capture_mouse()
        event.stop()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if not self._dragging:
            return
        delta = self._last_y - event.y
        self._last_y = event.y
        if delta != 0:
            self.set_value(self.value + delta * self.step * 6.0, notify=True)
        event.stop()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if self._dragging:
            self._dragging = False
            self.release_mouse()
        event.stop()

    def on_key(self, event: events.Key) -> None:
        if event.key == "up" or event.key == "right":
            self._change(+1.0, False)
            event.stop()
        elif event.key == "down" or event.key == "left":
            self._change(-1.0, False)
            event.stop()
        elif event.key == "home":
            self.set_value(self.min, notify=True)
            event.stop()
        elif event.key == "end":
            self.set_value(self.max, notify=True)
            event.stop()

    def render(self) -> Text:
        bar = _bar(self.value, self.min, self.max, width=_BAR_WIDTH)
        value = (
            self.zero_label
            if self.zero_label and self.value == self.min
            else self.fmt.format(self.value)
        )
        if self.units and value != self.zero_label:
            value = f"{value} {self.units}"
        return Text(f"{self.label_text:<{_LABEL_WIDTH}}{bar} {value}")


def _bar(value: float, min: float, max: float, width: int = 14) -> str:
    frac = (value - min) / (max - min) if max > min else 0.0
    frac = float(np.clip(frac, 0.0, 1.0))
    filled = int(round(frac * width))
    return f"[{'█' * filled}{'░' * (width - filled)}]"


class PianoRoll(Static):
    """ASCII piano-roll of the current pattern, with a playhead."""

    class Audition(Message):
        def __init__(self, midi: float) -> None:
            self.midi = midi
            super().__init__()

    MAX_ROWS = 16
    MAX_COLS = 48

    def __init__(
        self,
        id: str | None = None,
        name: str | None = None,
        classes: str | None = None,
    ) -> None:
        self.notes: list[Note] = []
        self.bpm: float = 70.0
        self.transpose: float = 0.0
        self.playhead: float | None = None
        self._row_midi: list[float] = []
        super().__init__(id=id, name=name, classes=classes)

    def set_pattern(self, notes: list[Note], bpm: float, transpose: float = 0.0) -> None:
        self.notes = notes
        self.bpm = bpm
        self.transpose = transpose
        self.playhead = None
        self.refresh(layout=True)

    def set_playhead(self, beat: float | None) -> None:
        self.playhead = beat
        self.refresh()

    def render(self) -> Text:
        if not self.notes:
            return Text("— pattern —")
        midis = []
        for n in self.notes:
            if n.midi is not None:
                midis.append(n.midi + self.transpose)
        if not midis:
            return Text("— pattern —")
        lo = int(np.floor(min(midis)))
        hi = int(np.ceil(max(midis)))
        self._row_midi = [float(m) for m in range(lo, hi + 1)][-self.MAX_ROWS :]

        total = sum(n.beats for n in self.notes)
        cols = min(int(np.ceil(total)), self.MAX_COLS)

        lines: list[str] = []
        header = "   " + "".join(str(i % 10) for i in range(cols))
        lines.append(header)
        for midi in self._row_midi:
            cells = [" "] * cols
            pos = 0.0
            for note in self.notes:
                note_midi = note.midi + self.transpose if note.midi is not None else None
                start = int(pos)
                if note_midi is not None and start < cols:
                    nearest = int(round(note_midi))
                    if nearest == midi:
                        if abs(note_midi - nearest) < 1e-9:
                            cells[start] = "■"
                        elif note_midi > nearest:
                            cells[start] = "▲"
                        else:
                            cells[start] = "▼"
                pos += note.beats
            if self.playhead is not None and 0 <= int(self.playhead) < cols:
                col = int(self.playhead)
                if cells[col] != "■":
                    cells[col] = "#"
            lines.append(f"{note_label(float(midi)):<4}{''.join(cells)}")
        return Text("\n".join(lines))

    def on_mouse_down(self, event: events.MouseDown) -> None:
        # Row 0 is the beat header; rows below map to notes.
        row = event.y - self.region.y - 1
        if 0 <= row < len(self._row_midi):
            self.post_message(self.Audition(float(self._row_midi[row])))
        event.stop()


class MiniWaveform(Static):
    """Multi-line ASCII decomposition of the rendered pattern loop.

    Renders the mastered waveform as a tall filled oscilloscope using
    half-block characters (so each terminal line carries two vertical
    sub-bands), plus one-line sparklines for the sub, warmth and snap
    layers. The ``parts`` attribute may be a
    :class:`~substratum.pattern.arrange.PatternParts`; for a plain mono
    audio buffer only the waveform is shown.
    """

    #: Lines the tall oscilloscope occupies (2x vertical resolution each).
    WAVE_ROWS = 12
    #: Columns shown for the waveform / sparklines when the widget is unsized.
    MIN_COLS = 40

    def __init__(
        self,
        id: str | None = None,
        name: str | None = None,
        classes: str | None = None,
    ) -> None:
        self.audio: np.ndarray | None = None
        self.parts: PatternParts | None = None
        self.bpm: float = 70.0
        super().__init__(id=id, name=name, classes=classes)

    def set_audio(self, audio: np.ndarray | None) -> None:
        self.audio = audio
        self.parts = None
        self.refresh(layout=True)

    def set_parts(self, parts: PatternParts | None, bpm: float) -> None:
        self.parts = parts
        self.bpm = bpm
        if parts is not None:
            self.audio = None
        self.refresh(layout=True)

    def _cols(self) -> int:
        # Size to the container so `width: auto` tracks the panel instead of
        # the widget's own (circular) laid-out width; clamp to a readable
        # range so very narrow panels clip rather than wrap.
        parent = cast(Widget | None, self.parent)
        width = parent.size.width if parent is not None else self.size.width
        return int(max(self.MIN_COLS, min(width - 8, 140)))

    def render(self) -> Text:
        if self.parts is not None and len(self.parts.mastered) > 0:
            return self._render_parts()
        return self._render_audio()

    def _render_audio(self) -> Text:
        if self.audio is None or len(self.audio) == 0:
            return Text("— waveform —")
        mono = self.audio[:, 0] if self.audio.ndim == 2 else self.audio
        if len(mono) < 64:
            return Text("— waveform too short —")
        cols = self._cols()
        rows: list[Text] = [Text(f"wave ({self.bpm:.0f} bpm)", style="bold")]
        rows.append(Text(_fill_wave(mono, cols=cols, rows=self.WAVE_ROWS), style="dim"))
        return Text("\n").join(rows)

    def _render_parts(self) -> Text:
        parts = self.parts
        assert parts is not None
        mono = parts.mastered[:, 0] if parts.mastered.ndim == 2 else parts.mastered
        cols = self._cols()
        rows: list[Text] = [Text(f"wave ({self.bpm:.0f} bpm)", style="bold")]
        rows.append(Text(_fill_wave(mono, cols=cols, rows=self.WAVE_ROWS)))
        for label, signal in (
            ("sub", parts.sub),
            ("warm ", parts.warmth),
            ("snap ", parts.snap),
        ):
            rows.append(Text(f"{label:<6}", style="bold") + Text(_sparkline(signal, cols)))
        return Text("\n").join(rows)


def _sparkline(mono: np.ndarray, cols: int) -> str:
    """One-line block-character sparkline of ``mono``'s absolute envelope."""
    n = len(mono)
    if n < cols:
        return _BLOCK_CHARS[0] * cols
    usable = n - (n % cols)
    buckets = np.abs(mono[:usable]).reshape(cols, -1).max(axis=1)
    peak = float(np.max(buckets)) or 1.0
    levels = (buckets / peak * 8.0).astype(int)
    return "".join(_BLOCK_CHARS[min(level, 8)] for level in levels)


def _fill_wave(mono: np.ndarray, cols: int, rows: int) -> str:
    """A filled oscilloscope with half-block vertical resolution.

    Each terminal line encodes two vertical sub-bands: the top half uses the
    lower half of a cell (``▄``) and the bottom half the upper (``▀``), so
    ``rows`` lines give ``2 * rows`` vertical steps. The signal is split into
    ``cols`` time buckets; every cell is filled wherever a bucket's min/max
    span crosses a sub-band, producing a solid silhouette that tracks the
    envelope.
    """
    n = len(mono)
    usable = n - (n % cols)
    if usable < cols:
        return "\n".join([" " * cols] * rows)
    buckets = mono[:usable].reshape(cols, -1)
    peak = float(np.max(np.abs(buckets))) or 1.0
    bands = rows * 2
    lines: list[str] = []
    for row in range(rows):
        cells: list[str] = []
        for b in range(cols):
            mn = float(buckets[b].min()) / peak
            mx = float(buckets[b].max()) / peak
            # Sub-band y ranges for this line: band 2*row is the top half,
            # band 2*row+1 the bottom half (both span 2/bands of amplitude).
            top_lo = 1.0 - 2.0 * (2 * row + 2) / bands
            top_hi = 1.0 - 2.0 * (2 * row) / bands
            bot_lo = 1.0 - 2.0 * (2 * row + 3) / bands
            bot_hi = 1.0 - 2.0 * (2 * row + 1) / bands
            top = mx >= top_lo and mn <= top_hi
            bot = mx >= bot_lo and mn <= bot_hi
            if top and bot:
                cells.append("█")
            elif top:
                cells.append("▀")
            elif bot:
                cells.append("▄")
            else:
                cells.append(" ")
        lines.append("".join(cells))
    return "\n".join(lines)
