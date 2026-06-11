"""
tests/test_inference.py
═══════════════════════════════════════════════════════════════════════════════

Unit tests for app/inference.py.

Run:
    pytest tests/test_inference.py -v
    pytest tests/test_inference.py -v --tb=short   # compact tracebacks
    pytest tests/test_inference.py -v -k "minmax"  # run one group

No real audio, no real model, no GPU required - all external I/O is mocked.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import types

import numpy as np
import pytest
import torch
import torch.nn as nn

# Make the project root importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.inference import (
    CHROMATIC_SCALE,
    QUALITY_INTERVALS,
    VAMP_FEATURE_STEP,
    VAMP_SEMITONE_ROLL,
    WINDOW_SIZE,
    STEP_SIZE,
    _build_beats_to_frames,
    _detect_arch,
    _harmonic_function,
    _minmax,
    _resample_to_beats,
    _softmax,
    _standardize,
    build_chord_timeline,
    chord_tones,
    export_csv,
    export_json,
    format_chord,
    get_current_chord,
    sliding_window_inference,
)


# Fixtures

LABEL_DOMAINS = {
    "global_key": [
        "C major", "C# major", "D major", "D# major", "E major", "F major",
        "F# major", "G major", "G# major", "A major", "A# major", "B major",
        "C minor", "C# minor", "D minor", "D# minor", "E minor", "F minor",
        "F# minor", "G minor", "G# minor", "A minor", "A# minor", "B minor",
    ],
    "roman_numeral": [
        "I", "ii", "iii", "IV", "V", "vi", "vii", "I7", "ii7", "V7",
        "Imaj7", "IVmaj7", "VImaj7", "bVII", "V/vi", "iv",
    ],
    "quality": ["M", "m", "D7", "M7", "m7", "d", "h7", "a", "d7"],
    "inversion": ["0", "1", "2"],
    "root_scale_degree": ["1", "2", "3", "4", "5", "6", "7"],
    "root_pitch_class": [str(i) for i in range(12)],
    "bass_pitch_class": [str(i) for i in range(12)],
    "tonicization": ["0", "vi", "IV", "ii"],
    "tonicized_pitch_class": [str(i) for i in range(12)],
}

LABEL_SIZES = {k: len(v) for k, v in LABEL_DOMAINS.items()}


def make_fake_model(label_sizes: dict, window_size: int = 4) -> nn.Module:
    """
    Minimal nn.Module that mimics the ImprovedRNATransformer output contract.
    Returns constant logits (all-ones) for every task at every frame position.
    """
    class FakeModel(nn.Module):
        def __init__(self, lsizes, ws):
            super().__init__()
            self.label_sizes = lsizes
            self._ws = ws

        def forward(self, features):
            B = features[0].shape[0]
            T = self._ws
            return {
                name: torch.ones(B, T, n_cls)
                for name, n_cls in self.label_sizes.items()
            }

    return FakeModel(label_sizes, window_size)


def make_features(T: int = 16, bins: int = 12) -> list[np.ndarray]:
    """Two random [bins, T] feature arrays (chroma + bass-chroma)."""
    rng = np.random.default_rng(0)
    return [rng.random((bins, T)).astype(np.float32),
            rng.random((bins, T)).astype(np.float32)]


def make_predictions(T: int, label_sizes: dict,
                     rng_seed: int = 42) -> tuple[dict, dict]:
    """Random per-frame predictions + probabilities for T frames."""
    rng = np.random.default_rng(rng_seed)
    preds = {}
    probs = {}
    for name, n_cls in label_sizes.items():
        preds[name] = rng.integers(0, n_cls, size=T)
        raw = rng.random((T, n_cls)).astype(np.float32)
        probs[name] = raw / raw.sum(axis=-1, keepdims=True)
    return preds, probs


# 1. Feature normalisation

class TestMinmax:
    def test_output_range(self):
        x = np.random.default_rng(0).random((12, 50)).astype(np.float32)
        y = _minmax(x)
        assert y.min() >= -1e-6
        assert y.max() <= 1.0 + 1e-6

    def test_max_per_frame_is_one(self):
        x = np.random.default_rng(1).random((12, 30)).astype(np.float32) + 0.1
        y = _minmax(x)
        np.testing.assert_allclose(y.max(axis=0), np.ones(30), atol=1e-5)

    def test_min_per_frame_is_zero(self):
        x = np.random.default_rng(2).random((12, 30)).astype(np.float32) + 0.1
        y = _minmax(x)
        np.testing.assert_allclose(y.min(axis=0), np.zeros(30), atol=1e-5)

    def test_constant_frame_no_nan(self):
        """A frame with all identical values should not produce NaN."""
        x = np.ones((12, 10), dtype=np.float32)
        y = _minmax(x)
        assert not np.isnan(y).any()

    def test_shape_preserved(self):
        x = np.random.default_rng(3).random((12, 64)).astype(np.float32)
        assert _minmax(x).shape == (12, 64)

    def test_column_wise_not_row_wise(self):
        """Each column (time frame) is independently normalised."""
        x = np.zeros((12, 4), dtype=np.float32)
        x[0, :] = [1, 2, 3, 4]   # row 0: increasing
        x[1, :] = [4, 3, 2, 1]   # row 1: decreasing
        y = _minmax(x)
        # In each column the max across all 12 rows should be 1
        np.testing.assert_allclose(y.max(axis=0), [1, 1, 1, 1], atol=1e-5)


class TestStandardize:
    def test_zero_mean_per_frame(self):
        x = np.random.default_rng(4).random((12, 40)).astype(np.float32)
        y = _standardize(x)
        np.testing.assert_allclose(y.mean(axis=0), np.zeros(40), atol=1e-4)

    def test_shape_preserved(self):
        x = np.random.default_rng(5).random((12, 64)).astype(np.float32)
        assert _standardize(x).shape == (12, 64)

    def test_constant_frame_no_nan(self):
        x = np.ones((12, 5), dtype=np.float32)
        y = _standardize(x)
        assert not np.isnan(y).any()


class TestSoftmax:
    def test_sums_to_one(self):
        x = np.random.default_rng(6).random((20, 10)).astype(np.float32)
        s = _softmax(x)
        np.testing.assert_allclose(s.sum(axis=-1), np.ones(20), atol=1e-5)

    def test_all_positive(self):
        x = np.random.default_rng(7).random((10, 5)).astype(np.float32)
        assert (_softmax(x) > 0).all()

    def test_argmax_preserved(self):
        """argmax of logits == argmax of softmax probabilities."""
        x = np.random.default_rng(8).random((15, 8)).astype(np.float32)
        np.testing.assert_array_equal(np.argmax(x, axis=-1),
                                      np.argmax(_softmax(x), axis=-1))

    def test_numerical_stability_large_values(self):
        x = np.array([[1000.0, 1001.0, 999.0]], dtype=np.float32)
        s = _softmax(x)
        assert not np.isnan(s).any()
        np.testing.assert_allclose(s.sum(axis=-1), [1.0], atol=1e-5)


# 2. Pitch correction (VAMP_SEMITONE_ROLL)

class TestVampPitchRoll:
    def test_roll_value_is_minus_three(self):
        """Project-wide constant: Vamp places C at bin 3; roll by -3 → bin 0."""
        assert VAMP_SEMITONE_ROLL == -3

    def test_c_moves_from_bin3_to_bin0(self):
        x = np.zeros((12, 1), dtype=np.float32)
        x[3, 0] = 1.0   # C at Vamp bin 3
        y = np.roll(x, shift=VAMP_SEMITONE_ROLL, axis=0)
        assert y[0, 0] == 1.0, "C should be at bin 0 after roll(-3)"

    def test_g_moves_to_correct_bin(self):
        """G is 7 semitones above C → Vamp bin 10 → roll(-3) → bin 7."""
        x = np.zeros((12, 1), dtype=np.float32)
        x[10, 0] = 1.0
        y = np.roll(x, shift=VAMP_SEMITONE_ROLL, axis=0)
        assert y[7, 0] == 1.0

    def test_roll_is_invertible(self):
        x = np.random.default_rng(9).random((12, 20)).astype(np.float32)
        y = np.roll(np.roll(x, VAMP_SEMITONE_ROLL, axis=0),
                    -VAMP_SEMITONE_ROLL, axis=0)
        np.testing.assert_array_equal(x, y)

    @pytest.mark.parametrize("pitch,vamp_bin,expected_bin", [
        ("C",  3,  0),
        ("D",  5,  2),
        ("E",  7,  4),
        ("F",  8,  5),
        ("G",  10, 7),
        ("A",  0,  9),
        ("B",  2,  11),
    ])
    def test_all_natural_notes(self, pitch, vamp_bin, expected_bin):
        x = np.zeros((12, 1), dtype=np.float32)
        x[vamp_bin, 0] = 1.0
        y = np.roll(x, shift=VAMP_SEMITONE_ROLL, axis=0)
        assert y[expected_bin, 0] == 1.0, (
            f"{pitch}: Vamp bin {vamp_bin} should map to bin {expected_bin}")


# 3. Chord tones

class TestChordTones:
    def test_c_major(self):
        assert chord_tones(0, "M") == ["C", "E", "G"]

    def test_g_major(self):
        assert chord_tones(7, "M") == ["G", "B", "D"]

    def test_d_minor(self):
        assert chord_tones(2, "m") == ["D", "F", "A"]

    def test_g_dominant_seventh(self):
        assert chord_tones(7, "D7") == ["G", "B", "D", "F"]

    def test_c_major_seventh(self):
        assert chord_tones(0, "M7") == ["C", "E", "G", "B"]

    def test_b_minor_seventh(self):
        assert chord_tones(11, "m7") == ["B", "D", "F#", "A"]

    def test_unknown_quality_returns_root_only(self):
        tones = chord_tones(0, "UNKNOWN")
        assert tones == ["C"]

    def test_root_wraps_octave(self):
        """root_pc > 11 should still work via modulo."""
        tones = chord_tones(12, "M")   # 12 % 12 = 0 = C
        assert tones == ["C", "E", "G"]

    def test_all_roots_produce_3_tones_for_triad(self):
        for root in range(12):
            tones = chord_tones(root, "M")
            assert len(tones) == 3, f"root={root} expected 3 tones"

    def test_output_names_are_valid_pitch_classes(self):
        for root in range(12):
            for tones in [chord_tones(root, q) for q in QUALITY_INTERVALS]:
                for t in tones:
                    assert t in CHROMATIC_SCALE, f"'{t}' is not a valid pitch class"


# 4. format_chord

class TestFormatChord:
    def make(self, rn, inv=0, tonic="0"):
        return {"roman_numeral": rn, "inversion": inv, "tonicization": tonic}

    def test_root_position_no_suffix(self):
        assert format_chord(self.make("I")) == "I"

    def test_first_inversion(self):
        assert format_chord(self.make("IV", inv=1)) == "IV⁶"

    def test_second_inversion(self):
        assert format_chord(self.make("V", inv=2)) == "V⁶₄"

    def test_no_double_quality(self):
        """roman_numeral already includes quality - must not be appended again."""
        label = format_chord(self.make("Imaj7"))
        assert label == "Imaj7"
        assert label.count("maj") == 1

    def test_secondary_chord_prefix(self):
        label = format_chord(self.make("V7", tonic="vi"))
        assert label == "vi/V7"

    def test_tonicization_zero_string_omitted(self):
        label = format_chord(self.make("I", tonic="0"))
        assert "/" not in label

    def test_tonicization_none_string_omitted(self):
        label = format_chord(self.make("I", tonic="None"))
        assert "/" not in label

    def test_tonicization_empty_omitted(self):
        label = format_chord(self.make("I", tonic=""))
        assert "/" not in label

    def test_invalid_inversion_treated_as_root(self):
        label = format_chord({"roman_numeral": "V", "inversion": "bad"})
        assert label == "V"

    def test_missing_keys_use_defaults(self):
        label = format_chord({})
        assert isinstance(label, str)


# 5. Harmonic function

class TestHarmonicFunction:
    @pytest.mark.parametrize("rn,expected", [
        ("I",      "tonic"),
        ("i",      "tonic"),
        ("III",    "tonic"),
        ("iii",    "tonic"),
        ("VI",     "tonic"),
        ("vi",     "tonic"),
        ("VImaj7", "tonic"),
        ("ii",     "subdominant"),
        ("II",     "subdominant"),
        ("IV",     "subdominant"),
        ("iv",     "subdominant"),
        ("IVmaj7", "subdominant"),
        ("ii7",    "subdominant"),
        ("V",      "dominant"),
        ("V7",     "dominant"),
        ("vii",    "dominant"),
        ("VII",    "dominant"),
        ("bVII",   "dominant"),
        ("",       "other"),
        ("N",      "other"),
    ])
    def test_basic_degrees(self, rn, expected):
        assert _harmonic_function(rn) == expected, f"'{rn}' → expected '{expected}'"

    def test_secondary_dominant(self):
        """V7/vi should be classified as dominant (the chord before '/')."""
        assert _harmonic_function("V7/vi") == "dominant"

    def test_secondary_subdominant(self):
        assert _harmonic_function("IV/V") == "subdominant"

    def test_accidental_prefix_ignored(self):
        assert _harmonic_function("#IV") == "subdominant"
        assert _harmonic_function("bII") == "subdominant"


# 6. get_current_chord (binary search)

class TestGetCurrentChord:
    @pytest.fixture
    def timeline(self):
        return [
            {"time": 0.0,  "end": 5.0,  "roman_numeral": "I"},
            {"time": 5.0,  "end": 10.0, "roman_numeral": "IV"},
            {"time": 10.0, "end": 15.0, "roman_numeral": "V"},
            {"time": 15.0, "end": 20.0, "roman_numeral": "I"},
        ]

    def test_first_chord_at_zero(self, timeline):
        assert get_current_chord(0.0, timeline)["roman_numeral"] == "I"

    def test_boundary_exact(self, timeline):
        assert get_current_chord(5.0, timeline)["roman_numeral"] == "IV"

    def test_mid_chord(self, timeline):
        assert get_current_chord(7.5, timeline)["roman_numeral"] == "IV"

    def test_last_chord(self, timeline):
        assert get_current_chord(16.0, timeline)["roman_numeral"] == "I"

    def test_before_first_chord(self, timeline):
        """Any time before the first chord → return first chord."""
        result = get_current_chord(-1.0, timeline)
        assert result["roman_numeral"] == "I"

    def test_beyond_last_chord(self, timeline):
        """Time beyond last chord end → return last chord."""
        result = get_current_chord(999.0, timeline)
        assert result["roman_numeral"] == "I"

    def test_empty_timeline(self):
        assert get_current_chord(5.0, []) is None

    def test_single_chord_timeline(self):
        tl = [{"time": 0.0, "end": 10.0, "roman_numeral": "I"}]
        assert get_current_chord(3.0, tl)["roman_numeral"] == "I"

    def test_returns_correct_event_object(self, timeline):
        result = get_current_chord(11.0, timeline)
        assert result["time"] == 10.0
        assert result["end"] == 15.0


# 7. build_chord_timeline

class TestBuildChordTimeline:
    def _make_timeline(self, rn_seq: list[int], T: int | None = None):
        """Helper: build predictions with a specific roman_numeral sequence."""
        if T is None:
            T = len(rn_seq)
        rng = np.random.default_rng(0)
        preds = {}
        probs = {}
        for name, n_cls in LABEL_SIZES.items():
            if name == "roman_numeral":
                arr = np.array(rn_seq[:T], dtype=np.int64)
            else:
                arr = rng.integers(0, n_cls, size=T)
            preds[name] = arr
            raw = rng.random((T, n_cls)).astype(np.float32)
            probs[name] = raw / raw.sum(axis=-1, keepdims=True)
        frame_times = np.arange(T, dtype=np.float32) * VAMP_FEATURE_STEP
        return build_chord_timeline(preds, probs, LABEL_DOMAINS, frame_times)

    def test_all_same_rn_gives_one_event(self):
        events = self._make_timeline([0] * 20)
        assert len(events) == 1

    def test_alternating_rn_gives_correct_count(self):
        seq = [0, 1, 0, 1, 0, 1]
        events = self._make_timeline(seq)
        assert len(events) == len(seq)

    def test_first_event_starts_at_zero(self):
        events = self._make_timeline([0] * 10)
        assert events[0]["time"] == pytest.approx(0.0)

    def test_last_event_ends_at_final_frame(self):
        T = 10
        events = self._make_timeline([0] * T, T)
        expected_end = float((T - 1) * VAMP_FEATURE_STEP)
        assert events[-1]["end"] == pytest.approx(expected_end, abs=1e-4)

    def test_events_are_contiguous(self):
        """End of event n == start of event n+1."""
        events = self._make_timeline([0, 0, 1, 1, 2, 2, 0])
        for a, b in zip(events, events[1:]):
            assert a["end"] == pytest.approx(b["time"], abs=1e-5)

    def test_roman_numeral_resolves_to_domain_string(self):
        events = self._make_timeline([0])
        assert events[0]["roman_numeral"] == LABEL_DOMAINS["roman_numeral"][0]

    def test_event_has_required_keys(self):
        required = {"time", "end", "roman_numeral", "quality", "inversion",
                    "global_key", "root_pitch_class", "bass_pitch_class",
                    "tonicization", "confidence", "function", "chord_label",
                    "chord_tones"}
        events = self._make_timeline([0, 1])
        for ev in events:
            assert required.issubset(ev.keys()), \
                f"Missing keys: {required - ev.keys()}"

    def test_confidence_between_zero_and_one(self):
        events = self._make_timeline([0, 1, 0])
        for ev in events:
            assert 0.0 <= ev["confidence"] <= 1.0

    def test_function_is_valid_category(self):
        valid = {"tonic", "subdominant", "dominant", "other"}
        events = self._make_timeline(list(range(len(LABEL_DOMAINS["roman_numeral"]))))
        for ev in events:
            assert ev["function"] in valid

    def test_chord_tones_are_pitch_class_names(self):
        events = self._make_timeline([0, 3])
        for ev in events:
            for tone in ev["chord_tones"]:
                assert tone in CHROMATIC_SCALE

    def test_single_frame_audio(self):
        events = self._make_timeline([0], T=1)
        assert len(events) == 1
        assert events[0]["time"] == pytest.approx(0.0)


# 8. _resample_to_beats / _build_beats_to_frames

class TestBeatResampling:
    def test_output_shape(self):
        feature = np.random.default_rng(0).random((12, 100)).astype(np.float32)
        b2f = np.array([0, 10, 20, 30, 40])
        out = _resample_to_beats(feature, b2f)
        assert out.shape == (12, 4)

    def test_averages_within_interval(self):
        feature = np.zeros((12, 10), dtype=np.float32)
        feature[:, 0:5] = 1.0   # first 5 frames all ones
        b2f = np.array([0, 5, 10])
        out = _resample_to_beats(feature, b2f)
        np.testing.assert_allclose(out[:, 0], np.ones(12), atol=1e-5)
        np.testing.assert_allclose(out[:, 1], np.zeros(12), atol=1e-5)

    def test_empty_interval_uses_single_frame(self):
        """When lo == hi the function falls back to a single-frame slice."""
        feature = np.arange(12, dtype=np.float32).reshape(12, 1)
        b2f = np.array([0, 0, 1])
        out = _resample_to_beats(feature, b2f)
        assert out.shape == (12, 2)

    def test_build_beats_to_frames_length(self):
        beat_times = np.array([0.5, 1.0, 1.5, 2.0])
        b2f = _build_beats_to_frames(beat_times, num_vamp_frames=100)
        assert len(b2f) == len(beat_times) + 1

    def test_build_beats_to_frames_starts_at_zero(self):
        beat_times = np.array([0.1, 0.2, 0.3])
        b2f = _build_beats_to_frames(beat_times, num_vamp_frames=50)
        assert b2f[0] == 0

    def test_build_beats_to_frames_does_not_exceed_num_frames(self):
        beat_times = np.array([1.0, 2.0, 100.0])   # last beat beyond audio
        b2f = _build_beats_to_frames(beat_times, num_vamp_frames=50)
        assert b2f.max() <= 50


# 9. sliding_window_inference

class TestSlidingWindowInference:
    """
    Uses a FakeModel that returns all-ones logits so we can predict shapes
    and accumulation behaviour without a real checkpoint.
    """

    WINDOW = 4
    STEP   = 2

    def _run(self, T: int, label_sizes: dict | None = None):
        ls = label_sizes or {"global_key": 24, "roman_numeral": 16,
                             "quality": 9, "inversion": 3}
        model  = make_fake_model(ls, self.WINDOW)
        feats  = make_features(T)
        device = torch.device("cpu")
        preds, probs = sliding_window_inference(
            model, feats, device,
            window_size=self.WINDOW, step_size=self.STEP)
        return preds, probs, ls

    def test_output_keys_match_label_sizes(self):
        preds, probs, ls = self._run(T=10)
        assert set(preds.keys()) == set(ls.keys())
        assert set(probs.keys()) == set(ls.keys())

    def test_prediction_shape_matches_T(self):
        T = 10
        preds, probs, ls = self._run(T=T)
        for name in ls:
            assert preds[name].shape == (T,), f"preds[{name}] wrong shape"
            assert probs[name].shape  == (T, ls[name]), f"probs[{name}] wrong shape"

    def test_probabilities_sum_to_one(self):
        _, probs, ls = self._run(T=8)
        for name in ls:
            sums = probs[name].sum(axis=-1)
            np.testing.assert_allclose(sums, np.ones(8), atol=1e-5,
                                       err_msg=f"probs[{name}] does not sum to 1")

    def test_predictions_are_valid_class_indices(self):
        preds, _, ls = self._run(T=12)
        for name, n_cls in ls.items():
            assert preds[name].min() >= 0
            assert preds[name].max() < n_cls

    def test_short_audio_shorter_than_window(self):
        """T < WINDOW must not crash and must return T predictions."""
        T = 2
        preds, probs, ls = self._run(T=T)
        for name in ls:
            assert preds[name].shape == (T,)

    def test_exact_window_multiple(self):
        """T divisible by STEP → no fractional last window."""
        preds, _, ls = self._run(T=self.WINDOW * 3)
        for name in ls:
            assert preds[name].shape == (self.WINDOW * 3,)

    def test_progress_callback_is_called(self):
        calls = []
        model  = make_fake_model({"roman_numeral": 16}, self.WINDOW)
        feats  = make_features(12)
        device = torch.device("cpu")
        sliding_window_inference(
            model, feats, device,
            window_size=self.WINDOW, step_size=self.STEP,
            progress_callback=lambda p, m: calls.append(p))
        assert len(calls) > 0

    def test_all_ones_logits_give_uniform_probs(self):
        """All-ones logits → softmax → all equal probs (1/n_cls each)."""
        _, probs, ls = self._run(T=8)
        for name, n_cls in ls.items():
            expected = np.full((8, n_cls), 1.0 / n_cls, dtype=np.float32)
            np.testing.assert_allclose(probs[name], expected, atol=1e-5)


# 10. _detect_arch

class TestDetectArch:
    def _make_state(self, d_model=128, num_layers=4, dim_feedforward=512,
                    pcp_channels=24, has_chord_change=True):
        """Construct a minimal fake state dict with the shapes _detect_arch reads."""
        state = {
            "encoder_layers.0.norm1.weight": torch.zeros(d_model),
            "encoder_layers.0.ffn.0.weight": torch.zeros(dim_feedforward, d_model),
            "task_heads.global_key.net.0.weight": torch.zeros(
                64, d_model + pcp_channels),
        }
        # Add num_layers encoder layer keys
        for i in range(num_layers):
            state[f"encoder_layers.{i}.norm1.weight"] = torch.zeros(d_model)
        if has_chord_change:
            state["task_heads.chord_change.net.0.weight"] = torch.zeros(16, d_model)
        return state

    def test_d_model_detected(self):
        state = self._make_state(d_model=128)
        assert _detect_arch(state)["d_model"] == 128

    def test_num_layers_detected(self):
        state = self._make_state(num_layers=4)
        assert _detect_arch(state)["num_layers"] == 4

    def test_dim_feedforward_detected(self):
        state = self._make_state(dim_feedforward=512)
        assert _detect_arch(state)["dim_feedforward"] == 512

    def test_pcp_channels_detected(self):
        state = self._make_state(d_model=128, pcp_channels=24)
        assert _detect_arch(state)["pcp_channels"] == 24

    def test_pcp_channels_zero_for_old_model(self):
        """Old model: key head input == d_model, so pcp_channels = 0."""
        state = self._make_state(d_model=128, pcp_channels=0)
        result = _detect_arch(state)
        assert result.get("pcp_channels", 0) == 0

    def test_chord_change_head_detected(self):
        state = self._make_state(has_chord_change=True)
        assert _detect_arch(state).get("add_chord_change_head") is True

    def test_chord_change_head_absent(self):
        state = self._make_state(has_chord_change=False)
        assert "add_chord_change_head" not in _detect_arch(state)

    def test_tome_r_always_zero(self):
        state = self._make_state()
        assert _detect_arch(state)["tome_r"] == 0

    def test_nhead_defaults_to_four(self):
        state = self._make_state()
        assert _detect_arch(state)["nhead"] == 4

    def test_larger_model_sizes(self):
        state = self._make_state(d_model=256, num_layers=6, dim_feedforward=1024)
        result = _detect_arch(state)
        assert result["d_model"]         == 256
        assert result["num_layers"]      == 6
        assert result["dim_feedforward"] == 1024


# 11. Export functions

@pytest.fixture
def sample_timeline():
    return [
        {"time": 0.0, "end": 2.4, "chord_label": "I",  "roman_numeral": "I",
         "global_key": "G major", "quality": "M", "inversion": 0,
         "root_pitch_class": 7, "bass_pitch_class": 7,
         "tonicization": "0", "confidence": 0.72,
         "function": "tonic", "chord_tones": ["G", "B", "D"]},
        {"time": 2.4, "end": 4.8, "chord_label": "iii", "roman_numeral": "iii",
         "global_key": "G major", "quality": "m", "inversion": 0,
         "root_pitch_class": 11, "bass_pitch_class": 11,
         "tonicization": "0", "confidence": 0.45,
         "function": "tonic", "chord_tones": ["B", "D", "F#"]},
    ]


class TestExportJSON:
    def test_creates_file(self, sample_timeline, tmp_path):
        path = str(tmp_path / "out.json")
        export_json(sample_timeline, path)
        assert os.path.exists(path)

    def test_roundtrip(self, sample_timeline, tmp_path):
        path = str(tmp_path / "out.json")
        export_json(sample_timeline, path)
        with open(path) as f:
            loaded = json.load(f)
        assert len(loaded) == len(sample_timeline)
        assert loaded[0]["roman_numeral"] == "I"
        assert loaded[1]["roman_numeral"] == "iii"

    def test_all_fields_preserved(self, sample_timeline, tmp_path):
        path = str(tmp_path / "out.json")
        export_json(sample_timeline, path)
        with open(path) as f:
            loaded = json.load(f)
        for key in sample_timeline[0]:
            assert key in loaded[0], f"Field '{key}' missing from exported JSON"

    def test_timestamps_are_floats(self, sample_timeline, tmp_path):
        path = str(tmp_path / "out.json")
        export_json(sample_timeline, path)
        with open(path) as f:
            loaded = json.load(f)
        assert isinstance(loaded[0]["time"], float)
        assert isinstance(loaded[0]["end"], float)

    def test_empty_timeline(self, tmp_path):
        path = str(tmp_path / "empty.json")
        export_json([], path)
        with open(path) as f:
            assert json.load(f) == []


class TestExportCSV:
    def test_creates_file(self, sample_timeline, tmp_path):
        path = str(tmp_path / "out.csv")
        export_csv(sample_timeline, path)
        assert os.path.exists(path)

    def test_header_row(self, sample_timeline, tmp_path):
        path = str(tmp_path / "out.csv")
        export_csv(sample_timeline, path)
        with open(path) as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
        assert "time" in headers
        assert "roman_numeral" in headers
        assert "global_key" in headers
        assert "confidence" in headers

    def test_row_count(self, sample_timeline, tmp_path):
        path = str(tmp_path / "out.csv")
        export_csv(sample_timeline, path)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len(sample_timeline)

    def test_values_correct(self, sample_timeline, tmp_path):
        path = str(tmp_path / "out.csv")
        export_csv(sample_timeline, path)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["roman_numeral"] == "I"
        assert rows[1]["roman_numeral"] == "iii"
        assert rows[0]["global_key"] == "G major"

    def test_empty_timeline_no_crash(self, tmp_path):
        path = str(tmp_path / "empty.csv")
        export_csv([], path)
        # File should not be created or should be empty - either is acceptable
        if os.path.exists(path):
            with open(path) as f:
                assert f.read() == ""

    def test_confidence_value_survives_roundtrip(self, sample_timeline, tmp_path):
        path = str(tmp_path / "out.csv")
        export_csv(sample_timeline, path)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert float(rows[0]["confidence"]) == pytest.approx(0.72, abs=1e-6)


# 12. Integration: sliding_window → build_chord_timeline

class TestIntegration:
    """
    End-to-end pipeline test using a FakeModel with deterministic outputs.
    Verifies that sliding_window_inference → build_chord_timeline produces
    a coherent timeline without crashing.
    """

    WINDOW = 4
    STEP   = 2

    def test_pipeline_produces_at_least_one_event(self):
        model  = make_fake_model(LABEL_SIZES, self.WINDOW)
        feats  = make_features(T=20)
        device = torch.device("cpu")
        preds, probs = sliding_window_inference(
            model, feats, device,
            window_size=self.WINDOW, step_size=self.STEP)
        T           = feats[0].shape[1]
        frame_times = np.arange(T) * VAMP_FEATURE_STEP
        timeline    = build_chord_timeline(preds, probs, LABEL_DOMAINS, frame_times)
        assert len(timeline) >= 1

    def test_timeline_covers_full_duration(self):
        T      = 30
        model  = make_fake_model(LABEL_SIZES, self.WINDOW)
        feats  = make_features(T=T)
        device = torch.device("cpu")
        preds, probs = sliding_window_inference(
            model, feats, device,
            window_size=self.WINDOW, step_size=self.STEP)
        frame_times = np.arange(T) * VAMP_FEATURE_STEP
        timeline    = build_chord_timeline(preds, probs, LABEL_DOMAINS, frame_times)
        # First event starts at or near t=0
        assert timeline[0]["time"] == pytest.approx(0.0, abs=VAMP_FEATURE_STEP)
        # Last event ends at or near final frame
        assert timeline[-1]["end"] == pytest.approx(
            float((T - 1) * VAMP_FEATURE_STEP), abs=VAMP_FEATURE_STEP)

    def test_export_json_after_pipeline(self, tmp_path):
        T      = 10
        model  = make_fake_model(LABEL_SIZES, self.WINDOW)
        feats  = make_features(T=T)
        device = torch.device("cpu")
        preds, probs = sliding_window_inference(
            model, feats, device,
            window_size=self.WINDOW, step_size=self.STEP)
        frame_times = np.arange(T) * VAMP_FEATURE_STEP
        timeline    = build_chord_timeline(preds, probs, LABEL_DOMAINS, frame_times)
        path = str(tmp_path / "result.json")
        export_json(timeline, path)
        with open(path) as f:
            loaded = json.load(f)
        assert len(loaded) == len(timeline)