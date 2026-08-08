"""Save and load bass sounds as ``name.mp3`` + ``name.json``.

A save bundles the rendered audio (MP3 by default) with the exact parameters
that produced it, so a sound can be reloaded into the GUI or recreated from
the CLI. Files live in ``~/Music/substratum/bass/samples``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from substratum.bass.synth import BassParams, render
from substratum.pattern.arrange import render_pattern
from substratum.pattern.notation import parse_pattern

FORMAT = "substratum-bass"
VERSION = 1
DEFAULT_EXT = "mp3"


class SaveError(RuntimeError):
    """Raised when a save/load operation fails."""


def default_dir() -> Path:
    """The per-user samples directory."""
    return Path.home() / "Music" / "substratum" / "bass" / "samples"


def ensure_dir(dir_path: str | Path | None = None) -> Path:
    """Return (and create if needed) the samples directory."""
    out = Path(dir_path) if dir_path else default_dir()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _json_path(name: str, dir_path: str | Path | None) -> Path:
    return ensure_dir(dir_path) / f"{name}.json"


def _audio_path(name: str, ext: str, dir_path: str | Path | None) -> Path:
    return ensure_dir(dir_path) / f"{name}.{ext}"


def _to_mp3(audio: np.ndarray, sample_rate: int, out_path: Path) -> None:
    """Write ``audio`` to an MP3 file via pydub (requires ffmpeg)."""
    import io

    import soundfile as sf
    from pydub import AudioSegment

    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    buf = io.BytesIO()
    sf.write(buf, data, sample_rate, format="wav", subtype="PCM_24")
    buf.seek(0)
    segment = AudioSegment.from_file(buf, format="wav")
    segment.export(str(out_path), format="mp3", bitrate="192k")


def _json_data(
    params: BassParams,
    pattern: str,
    bpm: float,
    gain: float,
    name: str,
) -> dict:
    return {
        "format": FORMAT,
        "version": VERSION,
        "name": name,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "params": asdict(params),
        "pattern": pattern,
        "bpm": bpm,
        "gain": gain,
    }


def save(
    name: str,
    params: BassParams,
    pattern: str,
    bpm: float,
    gain: float = 1.0,
    *,
    dir_path: str | Path | None = None,
    ext: str = DEFAULT_EXT,
) -> Path:
    """Render and save a sound as ``name.<ext>`` plus ``name.json``.

    Returns the path to the audio file. MP3 is the default; any extension
    supported by soundfile (e.g. ``wav``) is accepted.
    """
    name = name.strip()
    if not name or name in (".", ".."):
        raise SaveError("empty or invalid name")
    if "/" in name or "\\" in name:
        raise SaveError(f"invalid name: {name!r}")

    notes = parse_pattern(pattern)
    audio = (
        render_pattern(params, notes, bpm, transpose=params.transpose, gain=gain)
        if notes
        else render(params) * gain
    )
    out_dir = ensure_dir(dir_path)

    json_path = out_dir / f"{name}.json"
    json_path.write_text(json.dumps(_json_data(params, pattern, bpm, gain, name), indent=2))

    audio_path = out_dir / f"{name}.{ext}"
    if ext == "mp3":
        try:
            _to_mp3(audio, params.sample_rate, audio_path)
        except Exception as exc:  # noqa: BLE001 - surface a helpful message
            json_path.unlink(missing_ok=True)
            raise SaveError(
                "MP3 export failed (is ffmpeg installed?). "
                "Install ffmpeg or pass ext='wav' to save losslessly."
            ) from exc
    else:
        import soundfile as sf

        data = np.asarray(audio, dtype=np.float32)
        sf.write(str(audio_path), data, params.sample_rate, subtype="PCM_24")

    return audio_path


def load(name: str, dir_path: str | Path | None = None) -> dict:
    """Load a saved sound's JSON metadata."""
    path = _json_path(name, dir_path)
    if not path.exists():
        raise SaveError(f"no saved sound named {name!r} in {path.parent}")
    data = json.loads(path.read_text())
    if data.get("format") != FORMAT:
        raise SaveError(f"{path} is not a substratum save file")
    data["params"] = BassParams(**data["params"])
    return data


def list_saves(dir_path: str | Path | None = None) -> list[dict]:
    """List saved sounds (name, created, bpm, pattern) sorted by name."""
    out = ensure_dir(dir_path)
    items: list[dict] = []
    for json_path in sorted(out.glob("*.json")):
        try:
            data = json.loads(json_path.read_text())
        except json.JSONDecodeError:
            continue
        items.append(
            {
                "name": data.get("name", json_path.stem),
                "created": data.get("created", ""),
                "bpm": data.get("bpm"),
                "pattern": data.get("pattern", ""),
            }
        )
    return items


def delete(name: str, dir_path: str | Path | None = None) -> bool:
    """Delete a saved sound's audio + JSON files. Returns True if removed."""
    out = ensure_dir(dir_path)
    removed = False
    for path in out.glob(f"{name}.*"):
        if path.suffix in (".json", ".wav", ".mp3"):
            path.unlink(missing_ok=True)
            removed = True
    return removed


def decode_audio(name: str, dir_path: str | Path | None = None) -> tuple[np.ndarray, int]:
    """Decode a saved sound's audio file into a float array for playback."""
    out = ensure_dir(dir_path)
    for path in sorted(out.glob(f"{name}.*")):
        if path.suffix in (".wav", ".mp3"):
            if path.suffix == ".wav":
                import soundfile as sf

                audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
                return audio, sr
            from pydub import AudioSegment

            segment = AudioSegment.from_file(str(path))
            samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
            if segment.channels > 1:
                samples = samples.reshape(-1, segment.channels)
            return samples / (2 ** (segment.sample_width * 8 - 1)), segment.frame_rate
    raise SaveError(f"no audio found for saved sound {name!r}")
