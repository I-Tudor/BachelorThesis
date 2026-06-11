"""
app/inference.py - Feature extraction, sliding-window inference, chord timeline builder.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# constants

WINDOW_SIZE       = 256
STEP_SIZE         = 32
SAMPLING_RATE     = 44100
VAMP_FEATURE_STEP = 2048 / 44100          # ≈ 0.04644 s / frame
VAMP_BLOCK_SIZE   = 8192
VAMP_STEP_SIZE    = 2048

CHROMATIC_SCALE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# PARC pitch correction.
# Confirmed by scripts/test_vamp_pitch.py: on this system, Vamp places C at
# chroma bin 3 (3 semitones too high). Rolling by -3 moves C to bin 0.
# np.roll(arr, shift=-3): new[i] = arr[(i+3)%12], so bin 3 -> bin 0.

VAMP_SEMITONE_ROLL = -3

QUALITY_INTERVALS: Dict[str, List[int]] = {
    "D7":  [4, 3, 3], "M":   [4, 3],    "M7":  [4, 3, 4],
    "a":   [4, 4],    "a7":  [4, 4, 2], "aM7": [4, 4, 3],
    "d":   [3, 3],    "d7":  [3, 3, 3], "h7":  [3, 3, 4],
    "m":   [3, 4],    "m7":  [3, 4, 3], "mM7": [3, 4, 4],
    "oM7": [3, 3, 5],
}

QUALITY_DISPLAY: Dict[str, str] = {
    "M":   "",      "M7":  "maj⁷",  "m":   "m",     "m7":  "m⁷",
    "d":   "°",     "d7":  "°⁷",    "h7":  "ø⁷",    "D7":  "⁷",
    "a":   "+",     "a7":  "+⁷",    "aM7": "+M⁷",   "mM7": "mM⁷",
    "oM7": "°M⁷",
}


# feature extraction

try:
    import vamp as _vamp_lib
    _vamp_available = True
except ImportError:
    _vamp_lib = None
    _vamp_available = False


def _vamp_collect_raw(audio: np.ndarray, sr: int, output: str,
                      chromanormalize: int = 1) -> np.ndarray:
    """
    Call vamp.collect exactly as PARC does (no step_size/block_size override,
    chromanormalize parameter passed through) and return a [bins, T] float32 array.
    Handles both 'list' and 'matrix' return formats.
    """
    result = _vamp_lib.collect(
        audio,
        sample_rate=sr,
        plugin_key="nnls-chroma:nnls-chroma",
        output=output,
        parameters={"chromanormalize": chromanormalize},
    )
    if "list" in result:
        arr = np.array([f.values for f in result["list"]]).T
    elif "matrix" in result:
        _, matrix = result["matrix"]
        arr = np.array(matrix).T
    else:
        raise ValueError(f"Unexpected vamp.collect keys: {list(result.keys())}")
    return arr.astype(np.float32)


def _minmax(x: np.ndarray) -> np.ndarray:
    """
    Per-beat (column-wise) min-max normalisation. PARC calls minmax(x, axis=0).
    For shape [bins, T_beats]: axis=0 normalises across bins for each beat,
    so the most prominent pitch class per beat = 1, least = 0.
    """
    mn = x.min(axis=0, keepdims=True)   # shape [1, T_beats]
    mx = x.max(axis=0, keepdims=True)   # shape [1, T_beats]
    return (x - mn) / (mx - mn + 1e-8)


def _standardize(x: np.ndarray) -> np.ndarray:
    """
    Per-beat z-score. PARC calls standardize(x, axis=0).
    For shape [bins, T_beats]: axis=0 standardises across bins for each beat.
    """
    mu  = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0,  keepdims=True)
    return (x - mu) / (std + 1e-8)


def _resample_to_beats(feature: np.ndarray,
                       beats_to_frames: np.ndarray) -> np.ndarray:
    """
    Average Vamp frames within each beat interval.
    Replicates PARC's resample_feature() exactly.
    feature        : [bins, T_frames]
    beats_to_frames: [num_beats+1]  frame indices of beat boundaries
    Returns        : [bins, num_beats]
    """
    num_beats = len(beats_to_frames) - 1
    T = feature.shape[1]
    cols = []
    for i in range(num_beats):
        lo = int(np.clip(beats_to_frames[i],     0, T - 1))
        hi = int(np.clip(beats_to_frames[i + 1], 0, T))
        if lo >= hi:
            cols.append(feature[:, lo])
        else:
            cols.append(feature[:, lo:hi].mean(axis=1))
    return np.stack(cols, axis=1).astype(np.float32)


def _build_beats_to_frames(beat_times: np.ndarray,
                            num_vamp_frames: int) -> np.ndarray:
    """
    Build beat boundary frame indices for resample_to_beats().

    beat_times      : [K]   beat timestamps in seconds from librosa
    num_vamp_frames : int   total Vamp frames in the audio

    Returns [K+1] frame indices so that resample_to_beats produces exactly K
    beat vectors - one per entry in beat_times.

    Boundary layout:
      index 0   -> frame 0 (start of audio, before first detected beat)
      index 1…K -> frame of beat_times[0…K-1]
    Each interval [i, i+1) maps to one beat vector.
    """
    # Convert beat times (seconds) to Vamp frame indices directly.
    # VAMP_FEATURE_STEP is seconds-per-frame, so frame = time / step.
    beat_frame_indices = np.round(beat_times / VAMP_FEATURE_STEP).astype(int)
    beat_frame_indices = np.clip(beat_frame_indices, 0, num_vamp_frames)
    # Prepend frame 0 as the left boundary of the first beat interval.
    # Result: K+1 values -> K intervals -> K beat vectors matching beat_times.
    return np.concatenate([[0], beat_frame_indices]).astype(int)


def extract_features_vamp(
    audio_path: str,
    use_semitone_spectrum: bool = False,
    progress_callback=None,
    bpm: Optional[float] = None,   # kept for API compatibility, unused
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Extract chroma + basschroma features using Vamp NNLS-Chroma.

    Uses raw Vamp frames (46ms each) - no beat resampling.
    Beat resampling requires ground-truth beat counts from TheoryTab
    annotations unavailable at inference time, and beat tracking on
    piano/vocal audio is unreliable. Raw frames at 46ms resolution
    give the model sufficient temporal detail.

    Steps:
      1. Load audio at 44100 Hz
      2. Run Vamp NNLS-Chroma (chromanormalize=1)
      3. Roll by VAMP_SEMITONE_ROLL semitones (pitch correction)
      4. Per-frame minmax normalisation (matches training distribution)
      5. Return features + frame timestamps
    """
    import librosa

    if not _vamp_available:
        logger.warning("vamp not available - falling back to librosa chroma_cqt "
                       "(accuracy will be significantly lower)")

    if progress_callback:
        progress_callback(5, "Loading audio…")

    audio, sr = librosa.load(audio_path, sr=SAMPLING_RATE, mono=True)
    duration_s = len(audio) / sr
    logger.info(f"Loaded audio: {duration_s:.1f}s at {sr}Hz")

    if progress_callback:
        progress_callback(20, "Extracting Vamp features…")

    if _vamp_available:
        chroma_raw     = _vamp_collect_raw(audio, sr, "chroma",     chromanormalize=1)
        basschroma_raw = _vamp_collect_raw(audio, sr, "basschroma", chromanormalize=1)

        if VAMP_SEMITONE_ROLL != 0:
            chroma_raw     = np.roll(chroma_raw,     shift=VAMP_SEMITONE_ROLL, axis=0)
            basschroma_raw = np.roll(basschroma_raw, shift=VAMP_SEMITONE_ROLL, axis=0)

        if use_semitone_spectrum:
            if progress_callback:
                progress_callback(40, "Extracting semitone spectrum…")
            spec_raw = _vamp_collect_raw(audio, sr, "semitonespectrum", chromanormalize=0)
            if VAMP_SEMITONE_ROLL != 0:
                spec_raw = np.roll(spec_raw, shift=VAMP_SEMITONE_ROLL, axis=0)
            spectrum = _minmax(spec_raw)
            T = spectrum.shape[1]
            frame_times = np.arange(T) * VAMP_FEATURE_STEP
            if progress_callback:
                progress_callback(55, f"Features ready ({T} frames).")
            return [spectrum], frame_times

        chroma     = _minmax(chroma_raw)
        basschroma = _minmax(basschroma_raw)

    else:
        hop = 2048
        if progress_callback:
            progress_callback(25, "Computing chroma (librosa fallback)…")
        chroma_raw = librosa.feature.chroma_cqt(
            y=audio, sr=sr, hop_length=hop).astype(np.float32)
        if progress_callback:
            progress_callback(42, "Computing bass chroma…")
        basschroma_raw = librosa.feature.chroma_cqt(
            y=librosa.effects.harmonic(audio), sr=sr, hop_length=hop
        ).astype(np.float32)
        chroma     = _minmax(chroma_raw)
        basschroma = _minmax(basschroma_raw)

    T = chroma.shape[1]
    frame_times = np.arange(T) * VAMP_FEATURE_STEP
    logger.info(f"Features: {T} frames, {VAMP_FEATURE_STEP*1000:.1f}ms/frame, "
                f"{duration_s:.1f}s total")

    if progress_callback:
        progress_callback(55, f"Features ready ({T} frames).")

    return [chroma, basschroma], frame_times




# sliding-window inference

def sliding_window_inference(
    model: torch.nn.Module,
    features: List[np.ndarray],
    device: torch.device,
    window_size: int = WINDOW_SIZE,
    step_size: int   = STEP_SIZE,
    progress_callback=None,
) -> Dict[str, np.ndarray]:
    """
    Sliding-window inference over beat-level features.

    Matches PARC's chunkify_feature behaviour:
      - Right-side zero padding only (not edge padding on both sides)
      - Accumulates overlapping window predictions and averages

    Returns
    -------
    dict[task_name -> np.ndarray shape [T]]     per-beat argmax predictions
    dict[task_name -> np.ndarray shape [T, C]]  per-beat softmax probabilities
    """
    model.eval()
    T = features[0].shape[1]

    # Zero-pad right side only to ensure at least one full window - matches
    # PARC's chunkify_feature which pads short songs with zeros on the right.
    pad_right = max(0, window_size - T) + step_size
    padded = [np.pad(f, ((0, 0), (0, pad_right))) for f in features]

    task_names = list(model.label_sizes.keys())
    n_classes  = {n: model.label_sizes[n] for n in task_names}
    accum  = {n: np.zeros((T, n_classes[n]), dtype=np.float32) for n in task_names}
    counts = {n: np.zeros(T, dtype=np.float32) for n in task_names}

    starts = list(range(0, T, step_size))
    total  = len(starts)

    with torch.no_grad():
        for idx, start in enumerate(starts):
            end   = start + window_size
            chunk = [
                torch.tensor(f[:, start:end], dtype=torch.float32).unsqueeze(0).to(device)
                for f in padded
            ]
            if chunk[0].shape[-1] < window_size:
                chunk = [
                    torch.nn.functional.pad(c, (0, window_size - c.shape[-1]))
                    for c in chunk
                ]

            outputs = model(chunk)

            for name in task_names:
                logits     = outputs[name][0].cpu().numpy()   # [window_size, C]
                actual_len = min(window_size, T - start)
                accum[name][start:start + actual_len]  += logits[:actual_len]
                counts[name][start:start + actual_len] += 1

            if progress_callback and idx % 10 == 0:
                pct = 60 + int(35 * idx / total)
                progress_callback(pct, f"Running model… ({idx}/{total} windows)")

    predictions = {}
    probabilities = {}
    for name in task_names:
        avg = accum[name] / np.maximum(counts[name], 1)[:, None]
        probabilities[name] = _softmax(avg)
        predictions[name]   = np.argmax(avg, axis=-1)

    if progress_callback:
        progress_callback(95, "Building chord timeline…")

    return predictions, probabilities


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# chord timeline

def build_chord_timeline(
    predictions: Dict[str, np.ndarray],
    probabilities: Dict[str, np.ndarray],
    label_domains: Dict[str, List[str]],
    frame_times: np.ndarray,
) -> List[dict]:
    """
    Collapse per-frame predictions into chord-change events.

    Each event:
    {
        "time": float,         # start (seconds)
        "end":  float,         # end   (seconds)
        "roman_numeral": str,
        "root_scale_degree": str,
        "quality": str,
        "inversion": int,
        "global_key": str,
        "root_pitch_class": int,
        "bass_pitch_class": int,
        "tonicization": str,
        "tonicized_pitch_class": int,
        "confidence": float,   # softmax prob of top roman_numeral prediction
        "function": str,       # "tonic" | "subdominant" | "dominant" | "other"
    }
    """
    T = len(frame_times)
    events: List[dict] = []
    prev_rn  = None

    def _resolve(task: str, idx: int) -> str | int:
        domain = label_domains.get(task, [])
        if domain and idx < len(domain):
            return domain[idx]
        return idx

    for t in range(T):
        rn_idx = int(predictions["roman_numeral"][t])
        rn     = _resolve("roman_numeral", rn_idx)

        if rn != prev_rn:
            if events:
                events[-1]["end"] = float(frame_times[t])

            event: dict = {
                "time": float(frame_times[t]),
                "end":  float(frame_times[min(t + 1, T - 1)]),
                "roman_numeral":     str(rn),
                "root_scale_degree": str(_resolve("root_scale_degree", int(predictions["root_scale_degree"][t]))),
                "quality":           str(_resolve("quality", int(predictions["quality"][t]))),
                "inversion":         int(predictions["inversion"][t]),
                "global_key":        str(_resolve("global_key", int(predictions["global_key"][t]))),
                "root_pitch_class":  int(predictions["root_pitch_class"][t]),
                "bass_pitch_class":  int(predictions["bass_pitch_class"][t]),
                "tonicization":      str(_resolve("tonicization", int(predictions["tonicization"][t]))),
                "tonicized_pitch_class": int(predictions["tonicized_pitch_class"][t]),
                "confidence":        float(probabilities["roman_numeral"][t, rn_idx]),
            }
            event["function"]    = _harmonic_function(str(rn))  # use roman_numeral, not numeric root_scale_degree
            event["chord_label"] = format_chord(event)
            event["chord_tones"] = chord_tones(event["root_pitch_class"], event["quality"])

            events.append(event)
            prev_rn = rn

    if events:
        events[-1]["end"] = float(frame_times[-1])

    return events


def get_current_chord(playback_time: float, timeline: List[dict]) -> Optional[dict]:
    """Binary-search the chord timeline for the current playback time."""
    if not timeline:
        return None
    lo, hi = 0, len(timeline) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if timeline[mid]["time"] <= playback_time:
            if mid + 1 >= len(timeline) or timeline[mid + 1]["time"] > playback_time:
                return timeline[mid]
            lo = mid + 1
        else:
            hi = mid - 1
    return timeline[0]


# music helpers

def format_chord(event: dict) -> str:
    """
    Build the display chord label.

    The roman_numeral label domain already encodes quality in the string
    (e.g. "V7", "ii7", "VImaj7", "Imaj7") so we don't append quality
    separately - that was causing double-labels like "Imaj7maj⁷".
    We only append the inversion symbol and any tonicization prefix.
    """
    rn    = event.get("roman_numeral", "?")
    inv   = event.get("inversion", 0)
    try:
        inv = int(inv)
    except (ValueError, TypeError):
        inv = 0
    inv_str = {0: "", 1: "⁶", 2: "⁶₄"}.get(inv, "")
    tonic   = str(event.get("tonicization", ""))
    prefix  = f"{tonic}/" if tonic and tonic not in ("", "None", "0") else ""
    return f"{prefix}{rn}{inv_str}"


def chord_tones(root_pc: int, quality: str) -> List[str]:
    intervals = QUALITY_INTERVALS.get(quality, [])
    pcs = [root_pc % 12]
    cur = root_pc % 12
    for iv in intervals:
        cur = (cur + iv) % 12
        pcs.append(cur)
    return [CHROMATIC_SCALE[pc] for pc in pcs]


def _harmonic_function(rn: str) -> str:
    """
    Map a roman numeral string (from the real label domain) to a harmonic function.
    Handles quality suffixes (7, maj7, o, ø), accidentals (b, #), and
    secondary chords (V/vi -> dominant).

    Examples: "V7" -> dominant, "vi7" -> tonic, "bVII" -> dominant,
              "V7/vi" -> dominant, "iiø7" -> subdominant
    """
    if not rn:
        return "other"
    import re
    # Secondary chord: take the chord before "/" (e.g. "V7/vi" -> "V7")
    base = rn.split("/")[0]
    # Extract leading Roman numeral, ignoring accidentals and quality suffixes
    m = re.match(r'^[#b]?([IViv]+)', base)
    if not m:
        return "other"
    degree = m.group(1).upper()
    if degree in ("I", "III", "VI"):
        return "tonic"
    if degree in ("II", "IV"):
        return "subdominant"
    if degree in ("V", "VII"):
        return "dominant"
    return "other"


# model loading

def load_model(
    checkpoint_path: str,
    label_sizes: Dict[str, int],
    device: torch.device,
    model_kwargs: Optional[dict] = None,
) -> torch.nn.Module:
    """
    Load a trained ImprovedRNATransformer checkpoint.

    Architecture hyperparameters (d_model, num_layers, dim_feedforward) are
    inferred automatically from the checkpoint weights so the model always
    matches what was actually trained, regardless of CLI defaults.
    """
    import sys, os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from source.models.rna_transformer import ImprovedRNATransformer

    kwargs = dict(model_kwargs or {})
    in_channels = kwargs.pop("in_channels", (12, 12))

    ckpt  = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model", ckpt)

    # Auto-detect architecture from state dict
    detected = _detect_arch(state)
    logger.info(f"Detected architecture from checkpoint: {detected}")

    # Explicit kwargs override auto-detected values
    for k, v in detected.items():
        kwargs.setdefault(k, v)

    logger.info(f"Building model with: {kwargs}")

    model = ImprovedRNATransformer(in_channels, label_sizes, **kwargs).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    logger.info(f"Loaded model ({model.num_parameters:,} params) from {checkpoint_path}")
    return model


def _detect_arch(state: dict) -> dict:
    """Infer architecture hyperparameters from checkpoint state dict."""
    d_model = state["encoder_layers.0.norm1.weight"].shape[0]

    layer_indices = {int(k.split(".")[1])
                     for k in state if k.startswith("encoder_layers.")
                     and k.split(".")[1].isdigit()}
    num_layers = max(layer_indices) + 1 if layer_indices else 6

    dim_feedforward = state["encoder_layers.0.ffn.0.weight"].shape[0]

    # Key head input is [d_model + pcp_channels] in the new CLS-based model,
    # or [d_model] in the old model. Detect which we have.
    key_head_in = state["task_heads.global_key.net.0.weight"].shape[1]
    pcp_channels = key_head_in - d_model   # 0 for old model, 24 for new
    if pcp_channels < 0:
        pcp_channels = 0

    nhead = 4   # cannot be inferred from weights - must match training config
    logger.warning(
        "nhead cannot be auto-detected from checkpoint weights. "
        "Defaulting to nhead=4. Pass --nhead N if your config differs."
    )

    result = {
        "d_model":         d_model,
        "num_layers":      num_layers,
        "dim_feedforward": dim_feedforward,
        "nhead":           nhead,
        "tome_r":          0,
    }
    if pcp_channels > 0:
        result["pcp_channels"] = pcp_channels
        logger.info(f"Detected CLS-based model with pcp_channels={pcp_channels}")

    has_chord_change = any(k.startswith("task_heads.chord_change") for k in state)
    if has_chord_change:
        result["add_chord_change_head"] = True
        logger.info("Detected chord_change head in checkpoint")

    return result


def load_metadata(label_domains_path: str, label_sizes_path: str):
    with open(label_domains_path) as f:
        label_domains = json.load(f)
    with open(label_sizes_path) as f:
        label_sizes = json.load(f)
    return label_domains, label_sizes


# export helpers

def export_json(timeline: List[dict], path: str):
    with open(path, "w") as f:
        json.dump(timeline, f, indent=2)


def export_csv(timeline: List[dict], path: str):
    import csv
    if not timeline:
        return
    fields = ["time", "end", "chord_label", "roman_numeral", "global_key",
              "quality", "inversion", "root_pitch_class", "bass_pitch_class",
              "tonicization", "confidence"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(timeline)