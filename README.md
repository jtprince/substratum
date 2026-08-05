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

## Explore

```bash
bass explore                          # batch-render parameter sweeps to output/explore
bass analyze velvet.wav               # figures to output/analysis
bass gallery                          # playable HTML gallery to output/gallery
```

See `docs/bass-theory.md` for the theory behind the controls.
