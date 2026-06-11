#!/usr/bin/env python3
"""
scripts/test_vamp_pitch.py - Check whether your Vamp installation has the
3-semitone offset that PARC's training pipeline corrects for.

Usage:
    python scripts/test_vamp_pitch.py

Expected output if Vamp is CORRECT (no offset, set VAMP_SEMITONE_ROLL = 0):
    Peak bin: 0 (C)

Expected output if Vamp has PARC's offset (set VAMP_SEMITONE_ROLL = -3):
    Peak bin: 3 (D#)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    import vamp
except ImportError:
    print("ERROR: vamp not installed. Run: pip install --no-build-isolation vamp")
    sys.exit(1)

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
SR = 44100

def test_note(freq_hz: float, name: str):
    t    = np.linspace(0, 3.0, int(SR * 3.0), endpoint=False)
    tone = (np.sin(2 * np.pi * freq_hz * t) * 0.5).astype(np.float32)

    result = vamp.collect(
        tone, sample_rate=SR,
        plugin_key="nnls-chroma:nnls-chroma",
        output="chroma",
        parameters={"chromanormalize": 1},
    )
    if "matrix" in result:
        _, matrix = result["matrix"]
        chroma = np.array(matrix).T
    else:
        chroma = np.array([f.values for f in result["list"]]).T

    avg     = chroma.mean(axis=1)
    peak    = int(avg.argmax())
    print(f"\n{name} ({freq_hz:.2f} Hz)")
    print("  Bin  Note   Energy")
    for i, (n, v) in enumerate(zip(NOTES, avg)):
        bar    = "█" * int(v * 30)
        marker = " ← PEAK" if i == peak else ""
        print(f"  {i:2d}   {n:3s}   {bar}{marker}")
    return peak

print("=" * 50)
print("PARC Vamp Pitch Calibration Test")
print("=" * 50)

c4_peak = test_note(261.63, "C4 (middle C)")
g4_peak = test_note(392.00, "G4")
a4_peak = test_note(440.00, "A4")

print("\n" + "=" * 50)
print("RESULT")
print("=" * 50)
print(f"  C4 peak at bin {c4_peak} ({NOTES[c4_peak]})")
print(f"  G4 peak at bin {g4_peak} ({NOTES[g4_peak]})")
print(f"  A4 peak at bin {a4_peak} ({NOTES[a4_peak]})")

expected_correct = (c4_peak == 0 and g4_peak == 7 and a4_peak == 9)
expected_parc    = (c4_peak == 3 and g4_peak == 10 and a4_peak == 0)

print()
if expected_correct:
    print("Vamp is PITCH CORRECT - set VAMP_SEMITONE_ROLL = 0 in inference.py")
elif expected_parc:
    print("Vamp has PARC's 3-semitone offset - keep VAMP_SEMITONE_ROLL = -3")
else:
    roll = (0 - c4_peak) % 12
    if roll > 6:
        roll -= 12
    print(f"Unexpected offset. C4 is at bin {c4_peak}.")
    print(f"Try setting VAMP_SEMITONE_ROLL = {roll} in inference.py")