"""Seamless looping playback via a sounddevice ``OutputStream``.

``sounddevice.play(loop=True)`` restarts the stream from the buffer head
whenever the loop content changes (every slider drag), cutting the output at
an arbitrary sample and producing a click. This player drives an
``OutputStream`` directly and crossfades between the old and new loop buffers
so the pattern can change without an audible pop. Both mono and stereo loops
play seamlessly and re-loop at their own buffer length.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

#: Crossfade length between loop buffer swaps (seconds).
FADE_SECONDS = 0.012


class LoopPlayer:
    """Callback-driven sounddevice player with crossfading buffer swaps.

    ``update`` swaps the loop buffer (crossfading while playing), ``start`` /
    ``stop`` control playback and ``close`` releases the stream. Buffers are
    played in a loop, so playback is seamless as long as each buffer itself
    wraps cleanly.
    """

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self._stream: Any | None = None
        self._buffer: np.ndarray | None = None
        self._old: np.ndarray | None = None
        self._frame = 0
        self._running = False
        self._fade_total = int(FADE_SECONDS * sample_rate)
        self._fade_elapsed = 0

    # -- public -------------------------------------------------------------

    def update(self, audio: np.ndarray) -> None:
        """Set the loop buffer, crossfading from the current one if playing."""
        buf = self._as_stereo(audio)
        with self._lock:
            if self._running and self._buffer is not None:
                self._old = self._buffer
                self._fade_elapsed = 0
            self._buffer = buf

    def start(self) -> None:
        """Create (if needed) and start the output stream."""
        with self._lock:
            if self._buffer is None:
                return
            if self._stream is None:
                import sounddevice as sd

                self._stream = sd.OutputStream(
                    samplerate=self.sample_rate,
                    channels=2,
                    dtype="float32",
                    callback=self._callback,
                )
            stream = self._stream
            should_start = not self._running
            self._running = True
        # Stream calls are blocking (they wait for the callback to return), so
        # they must run outside the lock or they deadlock with the callback,
        # which needs the same lock to read the buffer.
        if should_start:
            stream.start()

    def stop(self) -> None:
        """Pause the stream, keeping the current buffer and position."""
        with self._lock:
            stream = self._stream
            was_running = self._running
            self._running = False
            self._old = None
            self._fade_elapsed = 0
        if stream is not None and was_running:
            stream.stop()

    def close(self) -> None:
        """Release the stream and free the buffer."""
        with self._lock:
            stream = self._stream
            self._stream = None
            self._buffer = None
            self._old = None
            self._running = False
        if stream is not None:
            stream.close()

    # -- internals ----------------------------------------------------------

    def _as_stereo(self, audio: np.ndarray) -> np.ndarray:
        a = np.asarray(audio)
        if a.ndim == 1:
            a = np.stack([a, a], axis=1)
        return np.ascontiguousarray(a, dtype=np.float32)

    def _callback(self, outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        with self._lock:
            buf = self._buffer
            if buf is None:
                outdata.fill(0.0)
                return
            n = len(buf)
            if n == 0:
                outdata.fill(0.0)
                return
            frame = self._frame
            old = self._old

            if old is not None and self._fade_elapsed < self._fade_total:
                k = min(frames, self._fade_total - self._fade_elapsed)
                pos = self._fade_elapsed + np.arange(k)
                g = 1.0 - pos / self._fade_total
                idx_new = (frame + np.arange(k)) % n
                idx_old = (frame + np.arange(k)) % len(old)
                outdata[:k] = old[idx_old] * g[:, None] + buf[idx_new] * (1.0 - g[:, None])
                if k < frames:
                    idx = (frame + np.arange(k, frames)) % n
                    outdata[k:] = buf[idx]
                self._fade_elapsed += k
                if self._fade_elapsed >= self._fade_total:
                    self._old = None
            else:
                idx = (frame + np.arange(frames)) % n
                outdata[:] = buf[idx]

            self._frame = frame + frames
