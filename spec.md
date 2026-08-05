# Deep808Lab
### A Small, Focused Synthesizer for Exploring Modern 808-Style Bass Timbres

Version: 1.0

---

# Goal

Build a small Python command-line synthesizer that reproduces the *design principles* behind modern, cinematic, trap-inspired sub basses (similar in character to artists like Sebastian Paul), emphasizing:

- Extremely deep sub frequencies
- Rich harmonic content
- Powerful transient punch
- Smooth analog-like saturation
- Simple controls that meaningfully affect timbre

This is **not** intended to clone any copyrighted sound. Instead, it should be an educational synthesizer that explores the same DSP techniques.

---

# Philosophy

Most "808 generators" are just:

- sine wave
- distortion

That misses much of what makes modern bass feel expensive.

Instead, model the sound as several interacting components:

```
Oscillator
    ↓
Pitch Envelope
    ↓
Amplitude Envelope
    ↓
Harmonic Layer
    ↓
Saturation
    ↓
Tone Filter
    ↓
Soft Clipper
    ↓
Limiter
```

Every stage should be individually understandable.

---

# External Dependencies

Preferred:

- numpy
- scipy
- soundfile

Optional:

- matplotlib
- rich
- typer

No heavyweight audio frameworks.

No VSTs.

No DAWs.

---

# CLI

Example:

```bash
python bass.py \
    --freq 38 \
    --drive 0.35 \
    --punch 0.65 \
    --warmth 0.25 \
    --weight 0.55 \
    --output demo.wav
```

---

# Primary Controls

## 1. Frequency

Range:

```
25–70 Hz
```

Controls the fundamental pitch.

This should be logarithmic rather than linear.

---

## 2. Punch

Controls the pitch envelope.

This is one of the defining characteristics of modern 808s.

Internally controls:

- starting pitch
- envelope decay
- exponential curve

Example:

```
0.0

40 Hz
────────────

0.5

65 Hz
 \
  \
   \
    40 Hz

1.0

110 Hz
 \
  \
   \
    \
      40 Hz
```

Without punch:

Soft.

With punch:

Chest impact.

---

## 3. Drive

Analog-style saturation amount.

NOT hard clipping.

Preferred:

```
tanh()
```

or

```
atan()
```

Oversample before saturation.

Downsample afterward.

Higher drive creates harmonics that make the bass audible on small speakers.

---

## 4. Warmth

Adds harmonic complexity.

Should simultaneously increase:

- octave-up sine
- subtle triangle component
- low-order harmonics

Should never sound buzzy.

Goal:

From

```
Pure sine
```

to

```
Rich velvet bass
```

---

## 5. Weight (Recommended Fifth Knob)

This is the missing "magic" parameter.

Instead of controlling one DSP block, Weight changes several together.

Internally adjusts:

- sub oscillator level
- envelope decay
- compressor threshold
- low-pass cutoff
- limiter makeup gain

Subjectively:

```
Lean
↓

Focused

↓

Huge

↓

Earthquake
```

This single control changes how physically large the bass feels.

It is probably the second-most important control after Punch.

---

# Hidden DSP Parameters

These are fixed in Normal Mode.

---

Attack

Default:

```
2 ms
```

---

Decay

Default:

```
850 ms
```

---

Release

Default:

```
300 ms
```

---

Pitch Envelope

Default:

```
Exponential
```

Not linear.

---

Oscillator

Primary:

Pure sine.

Optional reinforcement:

Triangle at

```
-18 dB
```

---

Upper Harmonic Oscillator

Optional sine

```
+1 octave
```

Very quiet.

Typically

```
-20 dB
```

---

# Saturation

Preferred algorithm:

```
tanh()
```

Alternative:

```
atan()
```

Never use hard clipping.

---

# Tone Filter

Simple one-pole low-pass.

Purpose:

Prevent excessive fizz.

Suggested range:

```
100–600 Hz
```

Automatically linked to Drive.

---

# Soft Clipper

Final safety stage.

Very gentle.

Just enough to catch peaks.

---

# Limiter

Simple look-ahead limiter.

Goal:

Prevent clipping.

Not loudness maximization.

---

# Output

Default:

```
48 kHz

24-bit WAV

Mono
```

Bass below ~120 Hz should remain mono.

---

# Presets

## Clean Sub

Minimal harmonics.

Almost pure sine.

---

## Warm

Slight saturation.

Small octave layer.

---

## Velvet

Smooth.

Warm.

Rich.

Probably the nicest preset.

---

## Cinematic

Large pitch sweep.

Long decay.

Heavy low end.

---

## Earthquake

Everything turned up.

Massive punch.

Large harmonics.

Designed for headphones or subwoofers.

---

# Exploration Mode

```
python bass.py explore
```

Automatically generates dozens of WAV files across the parameter space.

Example:

```
freq38_drive10.wav

freq38_drive20.wav

freq38_drive30.wav

...
```

Useful for learning.

---

# Visualization Mode

```
python bass.py analyze demo.wav
```

Produces:

- waveform
- FFT spectrum
- harmonic amplitudes
- envelope
- spectrogram

---

# Advanced Mode

Hidden unless requested.

Additional controls:

- attack
- decay
- release
- saturation curve
- harmonic mix
- triangle amount
- octave level
- compressor ratio
- limiter threshold
- filter cutoff
- glide
- transient click
- saturation asymmetry
- stereo harmonic width (keep sub mono)
- oversampling factor

---

# Internal Signal Flow

```
Sine Oscillator
        │
        ▼
Pitch Envelope
        │
        ▼
Amplitude Envelope
        │
        ▼
Triangle Reinforcement
        │
        ▼
Octave Harmonics
        │
        ▼
Oversample (4×)
        │
        ▼
Analog Saturation
        │
        ▼
Low-pass Filter
        │
        ▼
Soft Clipper
        │
        ▼
Limiter
        │
        ▼
Normalize
        │
        ▼
24-bit WAV
```

---

# Stretch Goals

- MIDI note input
- Live keyboard mode
- ADSR editor
- Preset save/load (JSON)
- Random patch generator
- Real-time parameter automation
- A/B comparison mode
- Batch rendering of parameter sweeps
- Optional GUI using Dear PyGui or Textual

---

# Success Criteria

A successful implementation should satisfy the following:

- Produces a bass that feels physically deep on quality headphones or a subwoofer.
- Remains audible on small speakers due to tasteful harmonic generation.
- Avoids harsh digital distortion or aliasing.
- Demonstrates clear, audible differences as each primary control is adjusted.
- Includes educational visualization tools so users can connect waveform, spectrum, and perceived timbre.
- Is compact, readable, and suitable as both a practical synthesizer and a DSP learning project.
