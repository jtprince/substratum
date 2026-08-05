"""Preset bass patches."""

from __future__ import annotations

from dataclasses import replace

from substratum.bass.synth import BassParams

PRESETS: dict[str, BassParams] = {
    "clean-sub": BassParams(
        freq=40.0,
        punch=0.0,
        drive=0.0,
        warmth=0.0,
        weight=0.2,
        description="Minimal harmonics. Almost a pure sine.",
    ),
    "warm": BassParams(
        freq=42.0,
        punch=0.15,
        drive=0.2,
        warmth=0.35,
        weight=0.3,
        description="Slight saturation with a small octave layer.",
    ),
    "velvet": BassParams(
        freq=38.0,
        punch=0.25,
        drive=0.3,
        warmth=0.5,
        weight=0.4,
        description="Smooth, warm and rich. The nicest all-rounder.",
    ),
    "cinematic": BassParams(
        freq=32.0,
        punch=0.75,
        drive=0.35,
        warmth=0.4,
        weight=0.7,
        description="Large pitch sweep and a long decaying tail.",
    ),
    "earthquake": BassParams(
        freq=28.0,
        punch=1.0,
        drive=0.55,
        warmth=0.6,
        weight=1.0,
        description="Everything turned up. Designed for subs and headphones.",
    ),
}


def list_presets() -> list[str]:
    """Return preset names in definition order."""
    return list(PRESETS)


def get_preset(name: str) -> BassParams:
    """Return a copy of a named preset."""
    if name not in PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Available: {list_presets()}")
    return replace(PRESETS[name])
