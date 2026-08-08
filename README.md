# substratum

Explore waveforms with programming.

## Install

```bash
uv sync
```

Requires `ffmpeg` for the gallery (MP3 export).

## Render a bass

```bash
bass --freq 38 --punch 0.65 --drive 0.35 --warmth 0.25 --weight 0.55 --output demo.wav
bass --preset velvet --output velvet.wav
bass presets   # list presets and their settings
```

## GUI

```bash
bass gui   # interactive Textual UI (15 sliders, ABC pattern, piano roll,
           # looping playback, save/load as name.mp3 + name.json in
           # ~/Music/substratum/bass/samples)
```

Requires an audio device for playback (sounddevice). MP3 save needs `ffmpeg`.

The pattern box takes simplified ABC: `C1` is a one-beat sub C (MIDI 24),
`C'`/`C,` raise/lower an octave, `#`/`b` add accidentals, a trailing number
sets the length in beats (`C2` = half note, `E1.5` = dotted), `z`/`r` are
rests, and `|` marks bar lines. Examples: `C1 E1 G1`, `F#1 D1/2 E2`,
`C2 z1 E1`.

The loop is tempo-locked to the BPM slider, so every hit lands on the beat
grid. The waveform panel shows a 12-line oscilloscope with half-block
vertical resolution (24 amplitude steps) plus sub/warmth/snap sparklines;
with an empty or invalid pattern it falls back to a single note so a
waveform is always visible. Press `v` (or the Plot button) to push a
matplotlib decomposition of the loop — ADSR envelope, sub, warmth, snap and
the mastered mix — into the terminal via `kitty +kitten icat` (kitty only).

## Explore

```bash
bass explore                          # batch-render parameter sweeps to output/explore
bass analyze velvet.wav               # figures to output/analysis
bass gallery                          # playable HTML gallery to output/gallery
```

See `docs/bass-theory.md` for the theory behind the controls.
