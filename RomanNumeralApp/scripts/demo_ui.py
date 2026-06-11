#!/usr/bin/env python3
"""
scripts/demo_ui.py - Smoke-test the PARC UI with synthetic data.

Generates a fake chord timeline and a fake waveform so you can verify
the entire UI without a model checkpoint or an audio file.

Usage:
    python scripts/demo_ui.py
"""
from __future__ import annotations

import math
import sys
import os
import random

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np



DEMO_LABEL_SIZES = {
    "global_key":           24,
    "tonicization":         8,
    "root_scale_degree":    7,
    "quality":              13,
    "inversion":            3,
    "root_pitch_class":     12,
    "bass_pitch_class":     12,
    "tonicized_pitch_class":12,
    "roman_numeral":        56,
}

DEMO_LABEL_DOMAINS = {
    "global_key": [f"{n} {'major' if i % 2 == 0 else 'minor'}"
                   for i, n in enumerate(
                       ["C","C","G","G","D","D","A","A",
                        "E","E","B","B","F#","F#","F","F",
                        "Bb","Bb","Eb","Eb","Ab","Ab","Db","Db"])],
    "tonicization": ["", "V/II", "V/III", "V/IV", "V/V", "V/VI", "V/VII", "V/I"],
    "root_scale_degree": ["I", "II", "III", "IV", "V", "VI", "VII"],
    "quality": ["M", "m", "d", "h7", "D7", "M7", "m7", "d7", "a", "a7", "aM7", "mM7", "oM7"],
    "inversion": ["0", "1", "2"],
    "root_pitch_class":     ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"],
    "bass_pitch_class":     ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"],
    "tonicized_pitch_class":["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"],
    "roman_numeral": (
        [f"{d}{q}"
         for d in ["I","II","III","IV","V","VI","VII"]
         for q in ["","m","°","ø⁷","⁷","m⁷","°⁷","M⁷"]]
    ),
}


class _DummyModel:
    label_sizes = DEMO_LABEL_SIZES
    num_parameters = 0

    def eval(self): return self

    def __call__(self, features):
        import torch
        T = features[0].shape[-1]
        return {name: torch.randn(1, T, n)
                for name, n in self.label_sizes.items()}

    def to(self, device): return self


def make_demo_timeline(duration: float = 120.0, bpm: float = 100.0):
    """Build a plausible I–V–vi–IV progression timeline."""
    from app.inference import chord_tones, format_chord, CHROMATIC_SCALE

    bar = 60.0 / bpm * 4          # seconds per bar
    progressions = [
        # (degree_idx, quality, root_pc, function)
        (0, "M",  0, "tonic"),       # I
        (4, "D7", 7, "dominant"),    # V7
        (5, "m",  9, "tonic"),       # vi
        (3, "M",  5, "subdominant"), # IV
    ]

    events = []
    t = 0.0
    while t < duration:
        for deg_idx, qual, root_pc, func in progressions:
            if t >= duration:
                break
            end = min(t + bar, duration)
            degree = DEMO_LABEL_DOMAINS["root_scale_degree"][deg_idx]
            ev = {
                "time": round(t, 4),
                "end":  round(end, 4),
                "roman_numeral":     degree,
                "root_scale_degree": degree,
                "quality":           qual,
                "inversion":         0,
                "global_key":        "C major",
                "root_pitch_class":  root_pc,
                "bass_pitch_class":  root_pc,
                "tonicization":      "",
                "tonicized_pitch_class": 0,
                "confidence":        random.uniform(0.72, 0.97),
                "function":          func,
            }
            ev["chord_label"] = format_chord(ev)
            ev["chord_tones"] = chord_tones(root_pc, qual)
            events.append(ev)
            t = end
    return events



def make_demo_waveform(duration: float = 120.0, n_buckets: int = 2000):
    from app.player import WaveformData
    x = np.linspace(0, duration, n_buckets)
    # Simulate a waveform: sinusoidal peaks with random amplitude variation
    envelope = 0.3 + 0.5 * np.abs(np.sin(x * 0.12))
    noise = np.random.uniform(0.05, 0.15, n_buckets)
    peaks_pos = (envelope + noise).clip(0, 1).astype(np.float32)
    peaks_neg = -(peaks_pos * np.random.uniform(0.7, 1.0, n_buckets)).astype(np.float32)
    return WaveformData(
        peaks_pos=peaks_pos,
        peaks_neg=peaks_neg,
        duration=duration,
        n_buckets=n_buckets,
    )


def main():
    import torch
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QTimer

    app = QApplication(sys.argv)
    app.setApplicationName("PARC Demo")

    from app.ui.main_window import MainWindow

    device = torch.device("cpu")
    model  = _DummyModel()

    win = MainWindow(model, DEMO_LABEL_DOMAINS, DEMO_LABEL_SIZES, device)
    win.setWindowTitle("PARC - Demo Mode (synthetic data)")
    win.show()

    # After 400ms auto-populate with synthetic data (skips file I/O)
    def _inject_demo():
        duration = 120.0
        timeline = make_demo_timeline(duration)
        waveform = make_demo_waveform(duration)

        win.lbl_song.setText("demo_progression.wav  [synthetic]")
        win._timeline = timeline
        win._stack.setCurrentIndex(1)

        win.waveform.set_waveform(waveform)
        win.waveform.set_timeline(timeline)
        win.timeline.set_timeline(timeline, duration)
        win.controls.set_duration(duration)
        win.controls.set_enabled_controls(True)

        if timeline:
            win.chord_display.set_chord(timeline[0])
            win.cof_widget.set_chord(timeline[0])
            win.detail.set_chord(timeline[0])

        # Simulate playback by advancing position every 100ms
        _pos = [0.0]
        _playing = [False]

        def _tick():
            if _playing[0]:
                _pos[0] = min(_pos[0] + 0.1, duration)
                win._update_position(_pos[0])
                if _pos[0] >= duration:
                    _playing[0] = False
                    win._update_state("finished")
                    _pos[0] = 0.0

        _timer = QTimer()
        _timer.setInterval(100)
        _timer.timeout.connect(_tick)
        _timer.start()

        def _toggle():
            _playing[0] = not _playing[0]
            win.controls.set_playing(_playing[0])

        def _restart():
            _pos[0] = 0.0

        def _seek(frac):
            _pos[0] = frac * duration

        win.controls.play_pause_clicked.connect(_toggle)
        win.controls.restart_clicked.connect(_restart)
        win.controls.seek_changed.connect(_seek)
        win.waveform.seek_requested.connect(lambda s: _pos.__setitem__(0, s))
        win.timeline.seek_requested.connect(lambda s: _pos.__setitem__(0, s))

        # Keep reference so timer isn't GC'd
        win._demo_timer = _timer

    QTimer.singleShot(400, _inject_demo)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()