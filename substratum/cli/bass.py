"""Deep808Lab CLI. Very thin: argument parsing only, all logic in libraries."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from substratum.bass.analyze import analyze as analyze_audio
from substratum.bass.explore import DEFAULT_FREQS, DEFAULT_STEP, sweep
from substratum.bass.presets import PRESETS, get_preset, list_presets
from substratum.bass.synth import BassParams, render
from substratum.gallery.generate import generate as generate_gallery
from substratum.io.audio import write_wav

app = typer.Typer(
    help="Deep808Lab - explore modern 808-style bass timbres.",
    no_args_is_help=False,
)
console = Console()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    freq: float = typer.Option(38.0, min=25.0, max=70.0, help="Fundamental pitch (Hz)."),
    punch: float = typer.Option(0.5, min=0.0, max=1.0, help="Pitch-envelope punch."),
    drive: float = typer.Option(0.35, min=0.0, max=1.0, help="Analog saturation amount."),
    warmth: float = typer.Option(0.25, min=0.0, max=1.0, help="Harmonic warmth."),
    weight: float = typer.Option(0.5, min=0.0, max=1.0, help="Macro knob: physical size."),
    preset: str | None = typer.Option(None, help="Preset name, overrides the five knobs."),
    duration: float = typer.Option(2.0, help="Note duration in seconds."),
    output: Path = typer.Option("demo.wav", help="Output WAV path."),
) -> None:
    """Render a bass note (default command when no subcommand is given)."""
    if ctx.invoked_subcommand is not None:
        return

    if preset:
        params = get_preset(preset)
    else:
        params = BassParams(freq=freq, punch=punch, drive=drive, warmth=warmth, weight=weight)
    params.duration = duration

    audio = render(params)
    write_wav(output, audio, params.sample_rate)
    kbps = params.sample_rate / 1000
    console.print(f"Wrote [bold]{output}[/bold] ({kbps:.0f} kHz, 24-bit, mono)")


@app.command("explore")
def explore(
    out_dir: Path = typer.Option("output/explore", help="Output directory."),
    freqs: str = typer.Option(
        ",".join(f"{f:g}" for f in DEFAULT_FREQS), help="Comma-separated frequencies (Hz)."
    ),
    step: float = typer.Option(DEFAULT_STEP, min=0.05, max=0.5, help="Sweep step in [0,1]."),
) -> None:
    """Render WAVs across the parameter space (punch/drive/warmth/weight)."""
    freq_list = tuple(float(f) for f in freqs.split(",") if f.strip())
    files = sweep(out_dir, freqs=freq_list, step=step)
    console.print(f"Wrote [bold]{len(files)}[/bold] files to [bold]{out_dir}[/bold]")


@app.command("analyze")
def analyze(
    path: Path = typer.Argument(..., help="WAV file to analyze."),
    out_dir: Path = typer.Option("output/analysis", help="Output directory."),
) -> None:
    """Produce waveform/spectrum/harmonics/envelope/spectrogram figures."""
    figures = analyze_audio(path, out_dir)
    for fig in figures:
        console.print(f"Wrote [bold]{fig}[/bold]")


@app.command("gallery")
def gallery(
    out_dir: Path = typer.Option("output/gallery", help="Output directory."),
    duration: float = typer.Option(2.0, help="Per-sound duration in seconds."),
) -> None:
    """Build the playable sound gallery (HTML + MP3 snippets)."""
    index = generate_gallery(out_dir, duration=duration)
    console.print(f"Gallery at [bold]{index}[/bold]")


@app.command("presets")
def presets() -> None:
    """List available presets and their settings."""
    table = Table(title="Presets")
    table.add_column("Name", style="cyan", no_wrap=True)
    for col in ("freq", "punch", "drive", "warmth", "weight"):
        table.add_column(col, justify="right")
    table.add_column("Description")
    for name in list_presets():
        p = PRESETS[name]
        table.add_row(
            name,
            f"{p.freq:g}",
            f"{p.punch:g}",
            f"{p.drive:g}",
            f"{p.warmth:g}",
            f"{p.weight:g}",
            p.description,
        )
    console.print(table)
