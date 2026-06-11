"""
app/player.py - Audio playback engine (sounddevice + soundfile).

Key design points
- Separate ``_reached_end`` flag so pause near end-of-song doesn't
  falsely emit "finished" (original bug: checked ``self._playing``
  which is also False after pause).
- Volume applied as a float32 multiply before writing to the output
  buffer - no allocation when volume == 1.0.
- ``seek()`` is lock-safe and usable while playing.
- ``WaveformData.from_file`` builds a 2 000-bucket peak envelope
  without loading the whole file into RAM twice (reuses soundfile read).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

try:
    import sounddevice as sd
    import soundfile as sf
    _AUDIO_OK = True
except ImportError:
    _AUDIO_OK = False


# waveform data

@dataclass
class WaveformData:
    """Down-sampled peak envelope, ready for drawing."""
    peaks_pos: np.ndarray   # [N]  positive peaks in [0, 1]
    peaks_neg: np.ndarray   # [N]  negative peaks in [-1, 0]
    duration:  float        # total seconds
    n_buckets: int          # N

    @classmethod
    def from_file(cls, path: str, n_buckets: int = 2000) -> "WaveformData":
        """Load audio and build a peak envelope. Supports MP3/WAV/FLAC/OGG via librosa."""
        try:
            # soundfile is faster but doesn't support MP3
            if not path.lower().endswith(".mp3"):
                data, sr = sf.read(path, dtype="float32", always_2d=True)
                mono = data.mean(axis=1)
            else:
                raise ValueError("MP3 - use librosa")
        except Exception:
            # librosa handles MP3, M4A, etc. via audioread
            import librosa as _lr
            mono, sr = _lr.load(path, sr=None, mono=True)

        total    = len(mono)
        duration = total / sr
        bucket   = max(1, total // n_buckets)
        n        = total // bucket
        trimmed  = mono[: n * bucket].reshape(n, bucket)
        pos      = trimmed.clip(min=0).max(axis=1).astype(np.float32)
        neg      = trimmed.clip(max=0).min(axis=1).astype(np.float32)
        return cls(peaks_pos=pos, peaks_neg=neg, duration=duration, n_buckets=n)

    @classmethod
    def synthetic(cls, duration: float, n_buckets: int = 2000) -> "WaveformData":
        """No-audio fallback - generates a plausible fake envelope."""
        x        = np.linspace(0, duration, n_buckets)
        envelope = (0.3 + 0.55 * np.abs(np.sin(x * 0.18)) +
                    0.15 * np.random.rand(n_buckets)).clip(0, 1).astype(np.float32)
        pos = envelope
        neg = -(envelope * np.random.uniform(0.7, 1.0, n_buckets)).astype(np.float32)
        return cls(peaks_pos=pos, peaks_neg=neg, duration=duration, n_buckets=n_buckets)


# audio player

class AudioPlayer:
    """
    Callback-based audio playback with precise seek and volume control.

    Thread model
    All public methods may be called from the Qt main thread.
    ``_callback`` runs on the sounddevice real-time thread.
    ``_tick_loop`` runs on a daemon thread.
    Shared state is protected by ``_lock``.

    Callbacks (set these from outside)
    on_position_changed(float)  - ~50 ms position updates
    on_state_changed(str)       - "playing" | "paused" | "stopped" | "finished"
    """

    def __init__(self):
        self._data:      Optional[np.ndarray] = None
        self._sr:        int             = 44100
        self._pos:       int             = 0       # sample index (int, lock-protected)
        self._playing:   bool            = False
        self._reached_end: bool          = False   # set by callback, consumed by _on_finished
        self._stream:    Optional[object] = None
        self._lock       = threading.Lock()
        self._tick_thread: Optional[threading.Thread] = None
        self._stop_tick  = threading.Event()

        self.volume: float = 1.0   # linear amplitude scalar [0..1]

        self.on_position_changed: Optional[Callable[[float], None]] = None
        self.on_state_changed:    Optional[Callable[[str],   None]] = None

    def load(self, path: str) -> None:
        """Load audio file. Supports MP3/WAV/FLAC/OGG/M4A. Stops current playback first."""
        self.stop()
        try:
            # soundfile is faster; doesn't support MP3
            if path.lower().endswith(".mp3"):
                raise ValueError("MP3 - use librosa")
            data, sr = sf.read(path, dtype="float32", always_2d=True)
        except Exception:
            # librosa fallback - handles MP3, M4A, etc.
            import librosa as _lr
            mono, sr = _lr.load(path, sr=None, mono=True)
            data = mono[:, np.newaxis]   # [N, 1]

        with self._lock:
            self._data        = data.astype(np.float32)
            self._sr          = int(sr)
            self._pos         = 0
            self._reached_end = False

    @property
    def duration(self) -> float:
        d = self._data
        return (len(d) / self._sr) if d is not None else 0.0

    @property
    def position(self) -> float:
        return self._pos / max(self._sr, 1)

    @property
    def is_playing(self) -> bool:
        return self._playing

    def play(self) -> None:
        if not _AUDIO_OK or self._data is None or self._playing:
            return
        self._reached_end = False
        self._open_stream()
        self._playing = True
        self._stream.start()
        self._start_tick()
        self._emit("playing")

    def pause(self) -> None:
        if not self._playing:
            return
        self._playing = False
        self._stop_tick_thread()
        if self._stream:
            try:
                self._stream.stop()
            except Exception:
                pass
        self._emit("paused")

    def toggle(self) -> None:
        if self._playing:
            self.pause()
        else:
            self.play()

    def seek(self, seconds: float) -> None:
        frame = int(max(0.0, min(seconds, self.duration)) * self._sr)
        with self._lock:
            self._pos = frame
        if self.on_position_changed:
            self.on_position_changed(self.position)

    def seek_relative(self, delta: float) -> None:
        self.seek(self.position + delta)

    def stop(self) -> None:
        self._playing = False
        self._stop_tick_thread()
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        with self._lock:
            self._pos = 0
        self._emit("stopped")

    #stream

    def _open_stream(self) -> None:
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        channels = self._data.shape[1]
        self._stream = sd.RawOutputStream(
            samplerate=self._sr,
            channels=channels,
            dtype="float32",
            callback=self._callback,
            finished_callback=self._on_finished,
        )

    def _callback(self, outdata, frames, time_info, status):
        """Real-time audio callback - must not block."""
        with self._lock:
            if self._data is None or not self._playing:
                outdata[:] = b"\x00" * len(outdata)
                return

            start  = self._pos
            end    = start + frames
            chunk  = self._data[start:end]          # [frames or less, channels]
            actual = len(chunk)

            if actual < frames:
                # Reached end of audio inside this buffer
                tail = np.zeros((frames - actual, self._data.shape[1]), dtype=np.float32)
                chunk = np.concatenate([chunk, tail])
                self._playing     = False
                self._reached_end = True

            if self.volume != 1.0:
                chunk = chunk * np.float32(self.volume)

            outdata[:] = chunk.tobytes()
            self._pos  = min(end, len(self._data))

    def _on_finished(self) -> None:
        """Called by sounddevice when the stream stops (after callback returns False)."""
        if self._reached_end:
            self._reached_end = False
            self._stop_tick_thread()
            self._emit("finished")

    # tick thread

    def _start_tick(self) -> None:
        self._stop_tick.clear()
        t = threading.Thread(target=self._tick_loop, daemon=True, name="parc-tick")
        t.start()
        self._tick_thread = t

    def _stop_tick_thread(self) -> None:
        self._stop_tick.set()
        t = self._tick_thread
        if t and t.is_alive():
            t.join(timeout=0.25)
        self._tick_thread = None

    def _tick_loop(self) -> None:
        while not self._stop_tick.is_set():
            cb = self.on_position_changed
            if cb:
                cb(self.position)
            self._stop_tick.wait(timeout=0.05)

    def _emit(self, state: str) -> None:
        cb = self.on_state_changed
        if cb:
            cb(state)