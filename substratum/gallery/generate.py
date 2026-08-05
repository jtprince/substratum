"""Generate the sound gallery.

Renders every curated sound to a 24-bit WAV, converts it to an MP3 snippet
with pydub (ffmpeg), and writes a self-contained ``index.html`` grid of
playable cards, each showing the exact CLI invocation that produced it.
"""

from __future__ import annotations

import html
from dataclasses import replace
from pathlib import Path

from substratum.bass.presets import get_preset
from substratum.bass.synth import BassParams, render
from substratum.gallery.sounds import GALLERY_SOUNDS
from substratum.io.audio import write_wav

MP3_BITRATE = "192k"

CARD_TEMPLATE = """\
<div class="card">
  <h3>{title}</h3>
  <audio controls preload="none" src="sounds/{name}.mp3"></audio>
  <p>{description}</p>
  <pre><code>{command}</code></pre>
</div>
"""

PAGE_TEMPLATE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Substratum - Bass Gallery</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
  h1 {{ margin-bottom: 0.25rem; }}
  .sub {{ color: #666; margin-top: 0; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 1.25rem; }}
  .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 1rem; background: #fafafa; }}
  .card h3 {{ margin-top: 0; }}
  audio {{ width: 100%; }}
  p {{ font-size: 0.9rem; line-height: 1.45; }}
  pre {{ background: #222; color: #eee; padding: 0.6rem 0.8rem; border-radius: 6px; overflow-x: auto; font-size: 0.78rem; }}
</style>
</head>
<body>
<h1>Substratum &mdash; Bass Sounds</h1>
<p class="sub">Regenerate with <code>bass gallery</code>. The CLI command under each
sound reproduces it exactly.</p>
<div class="grid">
{cards}
</div>
</body>
</html>
"""


def _cli_command(sound: dict) -> str:
    if sound.get("preset"):
        return f"bass --preset {sound['preset']} --output {sound['name']}.wav"
    p = sound["params"]
    return (
        f"bass --freq {p['freq']:g} --punch {p['punch']:g} "
        f"--drive {p['drive']:g} --warmth {p['warmth']:g} "
        f"--weight {p['weight']:g} --output {sound['name']}.wav"
    )


def _resolve_params(sound: dict) -> BassParams:
    if sound.get("preset"):
        return get_preset(sound["preset"])
    params = sound["params"]
    base = BassParams()
    overrides = {k: v for k, v in params.items() if k != "description"}
    return replace(base, **overrides)


def _to_mp3(wav_path: Path, mp3_path: Path) -> None:
    from pydub import AudioSegment

    segment = AudioSegment.from_wav(str(wav_path))
    segment.export(str(mp3_path), format="mp3", bitrate=MP3_BITRATE)


def generate(out_dir: str | Path = "output/gallery", duration: float = 2.0) -> Path:
    """Build the full gallery (WAVs + MP3s + index.html) and return its path."""
    out = Path(out_dir)
    sounds_dir = out / "sounds"
    sounds_dir.mkdir(parents=True, exist_ok=True)

    cards: list[str] = []
    for sound in GALLERY_SOUNDS:
        params = _resolve_params(sound)
        params.duration = duration
        audio = render(params)

        wav_path = sounds_dir / f"{sound['name']}.wav"
        mp3_path = sounds_dir / f"{sound['name']}.mp3"
        write_wav(wav_path, audio, params.sample_rate)
        _to_mp3(wav_path, mp3_path)

        cards.append(
            CARD_TEMPLATE.format(
                title=html.escape(sound["title"]),
                name=html.escape(sound["name"]),
                description=html.escape(sound["description"]),
                command=html.escape(_cli_command(sound)),
            )
        )

    index = out / "index.html"
    index.write_text(PAGE_TEMPLATE.format(cards="\n".join(cards)))
    return index
