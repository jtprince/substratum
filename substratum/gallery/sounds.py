"""Curated gallery sounds.

Each entry describes one playable example. ``preset`` names a preset from
``substratum.bass.presets``; alternatively ``params`` is a dict of override
BassParams. The CLI command shown on the gallery page is built from these.
"""

from __future__ import annotations

from typing import Any

GALLERY_SOUNDS: list[dict[str, Any]] = [
    {
        "name": "clean-sub",
        "title": "Clean Sub",
        "preset": "clean-sub",
        "description": (
            "A nearly pure sine sub. Almost no harmonics, minimal saturation. "
            "This is the raw material everything else is built from."
        ),
    },
    {
        "name": "subtle-sub",
        "title": "Subtle Sub",
        "params": {"freq": 35.0, "punch": 0.05, "drive": 0.05, "warmth": 0.1, "weight": 0.15},
        "description": (
            "Barely-there bass: felt more than heard. Good for low-key fills "
            "that support a mix without grabbing attention."
        ),
    },
    {
        "name": "warm-velvet",
        "title": "Warm Velvet",
        "preset": "velvet",
        "description": (
            "The all-rounder preset. Smooth saturation, a touch of warmth and "
            "a soft pitch drop. Audible on laptop speakers without getting buzzy."
        ),
    },
    {
        "name": "punchy-trap",
        "title": "Punchy Trap",
        "params": {"freq": 45.0, "punch": 0.7, "drive": 0.35, "warmth": 0.2, "weight": 0.4},
        "description": (
            "A tighter, brighter 808 with a pronounced pitch drop at the front. "
            "Chest impact without a long tail."
        ),
    },
    {
        "name": "cinematic-drop",
        "title": "Cinematic Drop",
        "preset": "cinematic",
        "description": (
            "Big exponential pitch sweep over a long decay. Heavy low end meant to land at a drop."
        ),
    },
    {
        "name": "earthquake",
        "title": "Earthquake",
        "preset": "earthquake",
        "description": (
            "Everything maxed: deepest frequency, maximum punch, heavy warmth. "
            "Designed for headphones or a subwoofer."
        ),
    },
]
