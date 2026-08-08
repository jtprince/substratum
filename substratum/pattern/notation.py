"""Parse a simplified ABC-style notation for bass note patterns.

Each token is a note name followed by optional accidentals, octave markers
and a length in beats. Examples::

    C1              single note, 1 beat (C1 ~ 32.7 Hz)
    E1 G1 A1        three one-beat notes
    F#1 D1/2 E2     accidentals and dotted/half lengths
    C' C, C,2       octave up, octave down, down with 2 beats
    C2 z1 E1        half note, rest, quarter note (z/r = rest)

Rules:

- Note letters ``A-G`` (case-insensitive) map to a fixed pitch: bare ``C``
  is C1 (MIDI 24, ~32.7 Hz) so a plain token lands in the sub-bass range.
- ``z`` / ``r`` denote a rest (silence) of the given length.
- ``#`` / ``b`` shift a semitone.
- Trailing ``'`` raises one octave, ``,`` lowers one octave (repeatable).
- A trailing number sets the length in beats (decimal allowed, default 1).
- ``|`` marks bar lines (ignored for timing); whitespace separates tokens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from substratum.utils.math import midi_to_hz

#: Semitone offset of each natural note letter from C.
SEMITONES: dict[str, int] = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}

#: MIDI number of bare ``C`` (C1, ~32.7 Hz, sub-bass octave).
BASE_MIDI = 24

_TOKEN_RE = re.compile(
    r"(?i)^(?P<note>[a-gzr])(?P<acc>[#b]?)(?P<oct>[,'']*)(?P<len>\d*(?:\.\d+)?)?$"
)


class NotationError(ValueError):
    """Raised when a pattern token cannot be parsed."""

    def __init__(self, token: str, index: int, reason: str) -> None:
        self.token = token
        self.index = index
        self.reason = reason
        super().__init__(f"bad note '{token}' at position {index}: {reason}")


@dataclass(frozen=True)
class Note:
    """A single note (or rest) event in a pattern.

    ``midi`` is ``None`` for a rest; ``beats`` is always positive.
    """

    midi: float | None
    beats: float

    @property
    def is_rest(self) -> bool:
        """True for a rest (``midi`` is ``None``)."""
        return self.midi is None

    @property
    def frequency(self) -> float:
        """The note's frequency in Hz. Raises for rests."""
        if self.midi is None:
            raise ValueError("a rest has no frequency")
        return midi_to_hz(self.midi)


def parse_note(token: str) -> Note:
    """Parse a single note token into a ``Note``."""
    match = _TOKEN_RE.match(token.strip())
    if not match:
        raise NotationError(
            token, 0, "expected a note letter followed by #/b, octave marks and length"
        )

    letter = match.group("note").upper()
    accidental = match.group("acc")
    octave_shift = match.group("oct").count("'") - match.group("oct").count(",")
    length_text = match.group("len")

    beats = 1.0 if length_text in ("", ".", "-") else float(length_text)
    if beats <= 0.0:
        raise NotationError(token, 0, "length must be positive")

    if letter in ("Z", "R"):
        if accidental or octave_shift:
            raise NotationError(token, 0, "rests cannot carry accidentals or octave marks")
        return Note(midi=None, beats=beats)

    midi = (
        BASE_MIDI + SEMITONES[letter] + (1 if accidental == "#" else -1 if accidental == "b" else 0)
    )
    midi += 12 * octave_shift
    return Note(midi=float(midi), beats=beats)


def parse_pattern(text: str) -> list[Note]:
    """Parse a pattern string into a list of ``Note`` events.

    Raises :class:`NotationError` on the first bad token.
    """
    notes: list[Note] = []
    position = 0
    for token in text.replace("|", " ").split():
        notes.append(parse_note(token))
        position += len(token) + 1
    return notes


def total_beats(notes: list[Note]) -> float:
    """Total number of beats in a parsed pattern."""
    return sum(note.beats for note in notes)


def note_label(midi: float) -> str:
    """Human-readable label for a MIDI note, e.g. ``C1``."""
    octave = int(midi) // 12 - 1
    letter = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][int(midi) % 12]
    return f"{letter}{octave}"
