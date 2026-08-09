"""Textual GUI for the bass.

Launch with ``bass gui``. Left panel holds the 15 control sliders (5 core +
10 advanced); the right panel edits the ABC pattern, shows the piano roll and
waveform, plays the looping pattern, and saves/loads sounds as
``name.mp3`` + ``name.json`` in ``~/Music/substratum/bass/samples``.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from random import random

import numpy as np
from textual.app import App, ComposeResult, SuspendNotSupported
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.css.query import NoMatches
from textual.timer import Timer
from textual.widgets import (
    Button,
    Collapsible,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
    TextArea,
)

from substratum.bass.presets import get_preset, list_presets
from substratum.bass.synth import BassParams
from substratum.gui import plots
from substratum.gui.player import LoopPlayer
from substratum.gui.widgets import HoverSlider, MiniWaveform, PianoRoll
from substratum.io import samples as sample_io
from substratum.pattern.arrange import (
    PatternParts,
    decompose_pattern,
    decompose_single_note,
    render_single_note,
)
from substratum.pattern.notation import NotationError, Note, parse_pattern, total_beats

#: (key, field, label, units, min, max, step, fmt)
CORE_SLIDERS = [
    ("pitch", "transpose", "Pitch", "st", -12.0, 12.0, 0.1, "{:.1f}"),
    ("punch", "punch", "Punch", "", 0.0, 1.0, 0.01, "{:.2f}"),
    ("drive", "drive", "Drive", "", 0.0, 1.0, 0.01, "{:.2f}"),
    ("warmth", "warmth", "Warmth", "", 0.0, 1.0, 0.01, "{:.2f}"),
    ("weight", "weight", "Weight", "", 0.0, 1.0, 0.01, "{:.2f}"),
]

ADVANCED_SLIDERS = [
    ("tail", "decay", "Tail", "s", 0.05, 3.0, 0.01, "{:.2f}"),
    ("attack", "attack", "Attack", "s", 0.001, 0.2, 0.001, "{:.3f}"),
    ("tone", "tone_hz", "Tone", "Hz", 0.0, 600.0, 5.0, "{:.0f}"),
    ("sub", "sub_level", "Sub Level", "", 0.0, 1.0, 0.01, "{:.2f}"),
    ("glide", "glide", "Glide", "", 0.0, 1.0, 0.01, "{:.2f}"),
    ("snap", "snap", "Snap", "", 0.0, 1.0, 0.01, "{:.2f}"),
    ("sustain", "sustain", "Sustain", "", 0.0, 1.0, 0.01, "{:.2f}"),
    ("width", "width", "Width", "", 0.0, 1.0, 0.01, "{:.2f}"),
    ("makeup", "makeup_db", "Limiter", "dB", 0.0, 6.0, 0.1, "{:.1f}"),
    ("curve", "curve", "Curve", "", 0.0, 1.0, 0.01, "{:.2f}"),
    ("dist", "distortion", "Distortion", "", 0.0, 1.0, 0.01, "{:.2f}"),
    ("crush", "crush", "Bitcrush", "", 0.0, 1.0, 0.01, "{:.2f}"),
]

FIELD_TO_KEY = {field: key for key, field, *_ in CORE_SLIDERS + ADVANCED_SLIDERS}

#: Shown under the pattern box so valid notation is discoverable.
NOTATION_HELP = (
    "C1 · F#1 D1/2 E2 · C' C, C,2 · C2 z1 E1   "
    "(z/r = rest, #/b = accidental, ' , = octave, number = beats)"
)

CSS = """
Screen { overflow: hidden; }
#main { height: 1fr; }
#left { width: 44; border: round $primary; padding: 0 1; }
#right { width: 1fr; border: round $primary; padding: 0 1; }
#right ScrollableContainer { height: 1fr; overflow-x: hidden; }
.banner { text-style: bold; color: $accent; margin-top: 1; }
#noterr { color: $error; }
#status { color: $text-muted; }
#nothelp { color: $text-muted; }
HoverSlider { height: 1; }
TextArea { height: 4; border: round $primary; }
PianoRoll { height: auto; width: auto; }
MiniWaveform { height: auto; width: auto; color: $text; }
#savename { width: 1fr; max-width: 24; }
"""


class BassGui(App[None]):
    """Interactive 808-style bass pattern synthesizer."""

    CSS = CSS
    BINDINGS = [
        ("space", "toggle_play", "Play/Stop"),
        ("v", "plot", "Plot"),
        ("r", "randomize", "Random"),
        ("ctrl+s", "save", "Save"),
        ("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        params: BassParams | None = None,
        pattern: str = "C1",
        bpm: float = 70.0,
        gain: float = 0.9,
        samples_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.base_params = BassParams() if params is None else params
        self.pattern_text = pattern
        self.bpm = bpm
        self.gain = gain
        self.samples_dir = Path(samples_dir) if samples_dir else sample_io.default_dir()

        self.audio: np.ndarray | None = None
        self.parsed_notes: list[Note] = []
        self.parts: PatternParts | None = None
        self.playing = False
        self.playhead: float | None = None
        self._playhead_timer: Timer | None = None
        self._render_gen = 0
        self._sample_rate = self.base_params.sample_rate
        self._player = LoopPlayer(self._sample_rate)

    # ---- layout -----------------------------------------------------------

    def _slider(
        self,
        key: str,
        field: str,
        label: str,
        units: str,
        min: float,
        max: float,
        step: float,
        fmt: str,
    ) -> HoverSlider:
        value = getattr(self.base_params, field)
        value = (0.0 if value is None else float(value)) if field == "tone_hz" else float(value)
        return HoverSlider(
            key,
            label,
            min=min,
            max=max,
            value=value,
            step=step,
            fmt=fmt,
            units=units,
            zero_label="Auto" if field == "tone_hz" else "",
            id=f"sl-{key}",
        )

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            with Vertical(id="left"), ScrollableContainer():
                with Collapsible(title="Core", collapsed=False):
                    for spec in CORE_SLIDERS:
                        yield self._slider(*spec)
                with Collapsible(title="Advanced", collapsed=False):
                    for spec in ADVANCED_SLIDERS:
                        yield self._slider(*spec)
            with Vertical(id="right"), ScrollableContainer():
                yield Label("Pattern (ABC)", classes="banner")
                yield TextArea(self.pattern_text, id="pattern")
                yield Static("", id="noterr")
                yield Static(NOTATION_HELP, id="nothelp")
                yield Label("Music bar  (click a note to audition)", classes="banner")
                yield PianoRoll(id="roll")
                yield Label("Waveform", classes="banner")
                yield MiniWaveform(id="wave")
                yield Label("Transport", classes="banner")
                with Horizontal():
                    yield Button("Play", id="play", variant="primary")
                    yield Button("Plot", id="plot", variant="default")
                    yield Static("", id="status")
                yield HoverSlider(
                    "bpm",
                    "BPM",
                    min=30.0,
                    max=200.0,
                    value=self.bpm,
                    step=1.0,
                    fmt="{:.0f}",
                    id="sl-bpm",
                )
                yield HoverSlider(
                    "gain",
                    "Gain",
                    min=0.0,
                    max=1.0,
                    value=self.gain,
                    step=0.01,
                    fmt="{:.2f}",
                    id="sl-gain",
                )
                yield Label("Sounds", classes="banner")
                with Horizontal():
                    yield Select(
                        [(name, name) for name in list_presets()],
                        value="velvet",
                        prompt="Preset",
                        allow_blank=True,
                        id="presets",
                    )
                    yield Button("Random", id="random", variant="default")
                with Horizontal():
                    yield Input(placeholder="name", id="savename")
                    yield Button("Save", id="save", variant="success")
                with Horizontal():
                    yield Select([], prompt="Saved…", allow_blank=True, id="saved")
                    yield Button("Load", id="load", variant="primary")
                    yield Button("Delete", id="delete", variant="error")
                    yield Button("Refresh", id="refresh")
        yield Footer()

    # ---- mount / state ----------------------------------------------------

    def on_mount(self) -> None:
        self._refresh_saved()
        self.call_after_refresh(self._render)

    def on_unmount(self) -> None:
        with contextlib.suppress(Exception):
            self._player.close()

    def _current_params(self) -> BassParams:
        values = {
            field: float(self.query_one(f"#sl-{FIELD_TO_KEY[field]}", HoverSlider).value)
            for field in FIELD_TO_KEY
        }
        return BassParams(
            transpose=values["transpose"],
            punch=values["punch"],
            drive=values["drive"],
            warmth=values["warmth"],
            weight=values["weight"],
            decay=values["decay"],
            attack=values["attack"],
            tone_hz=None if values["tone_hz"] <= 0 else values["tone_hz"],
            sub_level=values["sub_level"],
            glide=values["glide"],
            snap=values["snap"],
            sustain=values["sustain"],
            width=values["width"],
            makeup_db=values["makeup_db"],
            curve=values["curve"],
            distortion=values["distortion"],
            crush=values["crush"],
        )

    def _set_params(self, p: BassParams) -> None:
        mapping = dict(FIELD_TO_KEY)
        mapping["tone_hz"] = "tone"
        mapping["transpose"] = "pitch"
        for field, value in _as_dict(p).items():
            key = mapping.get(field)
            if key:
                slider = self.query_one(f"#sl-{key}", HoverSlider)
                slider.set_value(value if field != "tone_hz" or value is not None else 0.0)

    def _pattern_text(self) -> str:
        return self.query_one("#pattern", TextArea).text or ""

    def _bpm(self) -> float:
        return float(self.query_one("#sl-bpm", HoverSlider).value)

    def _gain(self) -> float:
        return float(self.query_one("#sl-gain", HoverSlider).value)

    def _refresh_saved(self) -> None:
        items = sample_io.list_saves(self.samples_dir)
        select = self.query_one("#saved", Select)
        options = [(item["name"], item["name"]) for item in items]
        select.set_options(options)

    # ---- messages ---------------------------------------------------------

    def on_hover_slider_changed(self, event: HoverSlider.Changed) -> None:
        self._schedule_render()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        self._schedule_render()

    def on_piano_roll_audition(self, event: PianoRoll.Audition) -> None:
        try:
            import sounddevice as sd

            audio = render_single_note(self._current_params(), event.midi, duration=1.2)
            self._stop()
            sd.stop()
            sd.play(audio, self._sample_rate, loop=False)
        except Exception as exc:  # noqa: BLE001
            self._status(f"audition failed: {exc}")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "presets" and event.value:
            self._set_params(get_preset(str(event.value)))
            self._schedule_render()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "play":
            self.action_toggle_play()
        elif button_id == "plot":
            self.action_plot()
        elif button_id == "random":
            self.action_randomize()
        elif button_id == "save":
            self.action_save()
        elif button_id == "load":
            self._load_saved()
        elif button_id == "delete":
            self._delete_saved()
        elif button_id == "refresh":
            self._refresh_saved()

    # ---- actions ----------------------------------------------------------

    def action_toggle_play(self) -> None:
        if self.playing:
            self._stop()
        else:
            self._play()

    def action_plot(self) -> None:
        """Render the decomposition and push it to the terminal via kitty."""
        if not self.parts or len(self.parts.mastered) == 0:
            self._render()
        if not self.parts or len(self.parts.mastered) == 0:
            self._status("nothing to plot")
            return
        try:
            import os

            if not os.environ.get(plots.KITTY_WINDOW_ID):
                self._status("kitty terminal not detected (KITTY_WINDOW_ID unset)")
                return
            try:
                with self.suspend():
                    plots.plot_pattern(self.parts, self._current_params(), self._bpm())
            except SuspendNotSupported:
                plots.plot_pattern(self.parts, self._current_params(), self._bpm())
        except Exception as exc:  # noqa: BLE001
            self._status(f"plot failed: {exc}")
            return
        self._status("plot pushed to terminal")

    def action_randomize(self) -> None:
        p = self._current_params()
        p.punch = random()
        p.drive = random() * 0.8
        p.warmth = random()
        p.weight = random()
        p.transpose = round(float(np.random.uniform(-7, 6)), 1)
        p.glide = random() * 0.6
        p.width = random()
        p.curve = random()
        p.snap = random() * 0.4
        p.makeup_db = random() * 4.0
        p.decay = 0.2 + random() * 1.8
        p.attack = random() * 0.05
        p.sustain = random() * 0.3
        self._set_params(p)
        self._schedule_render()

    def action_save(self) -> None:
        name = (self.query_one("#savename", Input).value or "").strip() or "untitled"
        try:
            sample_io.save(
                name,
                self._current_params(),
                self._pattern_text(),
                self._bpm(),
                self._gain(),
                dir_path=self.samples_dir,
            )
        except (sample_io.SaveError, NotationError) as exc:
            self._status(f"save failed: {exc}")
            return
        self._status(f"saved {name}.mp3 + {name}.json")
        self._refresh_saved()
        self.query_one("#saved", Select).value = name

    def _load_saved(self) -> None:
        select = self.query_one("#saved", Select)
        name = select.value
        if not name:
            self._status("no saved sound selected")
            return
        try:
            data = sample_io.load(str(name), self.samples_dir)
        except sample_io.SaveError as exc:
            self._status(str(exc))
            return
        self._set_params(data["params"])
        self.query_one("#pattern", TextArea).text = data.get("pattern", "")
        self.query_one("#sl-bpm", HoverSlider).set_value(float(data.get("bpm", 70.0)))
        self.query_one("#sl-gain", HoverSlider).set_value(float(data.get("gain", 1.0)))
        self.query_one("#savename", Input).value = data.get("name", "")
        self._status(f"loaded {name}")
        self._schedule_render()

    def _delete_saved(self) -> None:
        select = self.query_one("#saved", Select)
        name = select.value
        if not name:
            self._status("no saved sound selected")
            return
        sample_io.delete(str(name), self.samples_dir)
        self._refresh_saved()
        self._status(f"deleted {name}")

    # ---- render / playback ------------------------------------------------

    def _schedule_render(self) -> None:
        self._render_gen += 1
        gen = self._render_gen

        def render_now() -> None:
            if gen == self._render_gen:
                self._render()

        self.set_timer(0.15, render_now)

    def _render(self) -> None:
        if not self._widgets_ready():
            return
        params = self._current_params()
        bpm = self._bpm()
        gain = self._gain()
        pattern = self._pattern_text()

        noterr = self.query_one("#noterr", Static)
        try:
            self.parsed_notes = parse_pattern(pattern)
            noterr.update("")
        except NotationError as exc:
            self.parsed_notes = []
            noterr.update(str(exc))
            self.parts = decompose_single_note(params, gain=gain)
            self.audio = self.parts.mastered
            self.query_one("#roll", PianoRoll).set_pattern([], bpm, params.transpose)
            self.query_one("#wave", MiniWaveform).set_parts(self.parts, bpm)
            self._status(f"bad notation: {exc.token}")
            self._restart_playback()
            return

        roll = self.query_one("#roll", PianoRoll)
        if self.parsed_notes:
            self.parts = decompose_pattern(
                params, self.parsed_notes, bpm, transpose=params.transpose, gain=gain
            )
            self.audio = self.parts.mastered
            beats = total_beats(self.parsed_notes)
            seconds = beats * 60.0 / bpm
            self._status(
                f"{len(self.parsed_notes)} notes, {beats:g} beats @ {bpm:.0f} bpm "
                f"({seconds:.2f}s loop)"
            )
            roll.set_pattern(self.parsed_notes, bpm, params.transpose)
        else:
            self.parts = decompose_single_note(params, gain=gain)
            self.audio = self.parts.mastered
            self._status("empty pattern — playing single note")
            roll.set_pattern([], bpm, params.transpose)
        self.query_one("#wave", MiniWaveform).set_parts(self.parts, bpm)
        self._restart_playback()

    def _restart_playback(self) -> None:
        if not self.playing:
            return
        self._play()

    def _play(self) -> None:
        try:
            audio = self.audio
            if audio is None or len(audio) == 0:
                self._render()
                audio = self.audio
            if audio is None or len(audio) == 0:
                raise RuntimeError("no audio rendered")
            self._player.update(audio)
            self._player.start()
        except Exception as exc:  # noqa: BLE001
            self._status(f"playback unavailable: {exc}")
            return
        self.playing = True
        self.playhead = 0.0
        self.query_one("#play", Button).label = "Stop"
        self._start_playhead_timer()

    def _stop(self) -> None:
        with contextlib.suppress(Exception):
            self._player.stop()
        self.playing = False
        self._stop_playhead_timer()
        self.query_one("#play", Button).label = "Play"
        self.query_one("#roll", PianoRoll).set_playhead(None)

    def _start_playhead_timer(self) -> None:
        self._stop_playhead_timer()
        beat_seconds = 60.0 / self._bpm()

        def tick() -> None:
            if not self.playing:
                return
            total = max(total_beats(self.parsed_notes), 1.0)
            self.playhead = ((self.playhead or 0.0) + 1.0) % total
            self.query_one("#roll", PianoRoll).set_playhead(self.playhead)

        self._playhead_timer = self.set_interval(beat_seconds, tick)

    def _stop_playhead_timer(self) -> None:
        if self._playhead_timer is not None:
            self._playhead_timer.stop()
            self._playhead_timer = None

    def _status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    def _widgets_ready(self) -> bool:
        try:
            self.query_one("#sl-pitch")
            return True
        except NoMatches:
            return False


def _as_dict(params: BassParams) -> dict:
    return {
        "transpose": params.transpose,
        "punch": params.punch,
        "drive": params.drive,
        "warmth": params.warmth,
        "weight": params.weight,
        "decay": params.decay,
        "attack": params.attack,
        "tone_hz": params.tone_hz,
        "sub_level": params.sub_level,
        "glide": params.glide,
        "snap": params.snap,
        "sustain": params.sustain,
        "width": params.width,
        "makeup_db": params.makeup_db,
        "curve": params.curve,
        "distortion": params.distortion,
        "crush": params.crush,
    }


def run() -> None:
    BassGui().run()


if __name__ == "__main__":
    run()
