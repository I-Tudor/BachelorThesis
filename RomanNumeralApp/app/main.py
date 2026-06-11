"""
app/main.py - Entry point.

Usage:
    python app/main.py \\
        --checkpoint experiments/transformer_artist_chroma/best_model.ckpt \\
        --label-domains dataset/metadata/label_domains.json \\
        --label-sizes   dataset/metadata/label_sizes.json \\
        [--device cpu|cuda:0|mps] \\
        [--use-semitone-spectrum]

Demo mode (no model): omit --checkpoint to run with a dummy model that
outputs random predictions (for UI development).
"""
from __future__ import annotations

import os
import sys

# Qt platform plugin fix for PyInstaller bundles
if getattr(sys, "frozen", False):
    _exe = os.path.dirname(os.path.abspath(sys.executable))
    _plugins = os.path.abspath(
        os.path.join(_exe, "..", "Frameworks", "PyQt6", "Qt6", "plugins")
    )
    os.environ["QT_PLUGIN_PATH"] = _plugins
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(_plugins, "platforms")

import argparse
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("parc")


def _find_bundled_defaults():
    """
    When running as a PyInstaller .app, find checkpoint and metadata
    automatically. Searches relative to executable and Resources folder.
    """
    if getattr(sys, "frozen", False):
        exe_dir    = os.path.dirname(sys.executable)
        meipass    = getattr(sys, "_MEIPASS", exe_dir)
        candidates = [
            meipass,
            exe_dir,
            os.path.join(exe_dir, "..", "Resources"),
            os.path.abspath(os.path.join(exe_dir, "..", "Resources")),
        ]
    else:
        # Normal run - search from project root
        candidates = [
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ]

    defaults = {}

    for base in candidates:
        base = os.path.abspath(base)

        # Search for any .ckpt file inside experiments/
        if not defaults.get("checkpoint"):
            exp_dir = os.path.join(base, "experiments")
            if os.path.isdir(exp_dir):
                for root, _, files in os.walk(exp_dir):
                    for f in files:
                        if f.endswith(".ckpt"):
                            defaults["checkpoint"] = os.path.join(root, f)
                            break

        # Search for metadata JSONs
        meta_dir = os.path.join(base, "dataset", "metadata")
        if not defaults.get("label_domains"):
            p = os.path.join(meta_dir, "label_domains.json")
            if os.path.exists(p):
                defaults["label_domains"] = p

        if not defaults.get("label_sizes"):
            p = os.path.join(meta_dir, "label_sizes.json")
            if os.path.exists(p):
                defaults["label_sizes"] = p

    return defaults


def parse_args():
    bundled = _find_bundled_defaults()

    p = argparse.ArgumentParser(description="PARC Roman Numeral Chord Analyzer")
    p.add_argument("--checkpoint",    default=bundled.get("checkpoint"),
                   help="Path to trained model checkpoint (.ckpt)")
    p.add_argument("--label-domains", default=bundled.get("label_domains"),
                   help="Path to label_domains.json")
    p.add_argument("--label-sizes",   default=bundled.get("label_sizes"),
                   help="Path to label_sizes.json")
    p.add_argument("--device",        default=None,
                   help="Compute device: cpu | cuda:N | mps (auto-detects)")
    p.add_argument("--nhead",         default=4, type=int,
                   help="Number of attention heads (default: 4)")
    p.add_argument("--bpm",           default=None, type=float,
                   help="Override beat detection with a fixed BPM")
    p.add_argument("--use-semitone-spectrum", action="store_true",
                   help="Use semitone spectrum features instead of chroma")
    return p.parse_args()


def auto_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dummy_model(label_sizes: dict):
    """
    Thin wrapper that returns random logits - lets you test the UI
    without a trained checkpoint.
    """
    import torch
    import torch.nn as nn

    class DummyModel(nn.Module):
        def __init__(self, ls):
            super().__init__()
            self.label_sizes = ls
            self._p = nn.Parameter(torch.zeros(1))  # needs ≥1 param

        def forward(self, features):
            T = features[0].shape[-1]
            return {name: torch.randn(1, T, n)
                    for name, n in self.label_sizes.items()}

        @property
        def num_parameters(self): return 1

    return DummyModel(label_sizes)


def main():
    args = parse_args()

    import torch

    # device
    if args.device:
        device = torch.device(args.device)
    else:
        device = auto_device()
    logger.info(f"Using device: {device}")

    # metadata
    if args.label_domains and args.label_sizes:
        from app.inference import load_metadata
        label_domains, label_sizes = load_metadata(
            args.label_domains, args.label_sizes)
    else:
        logger.warning("No label metadata provided - using minimal defaults.")
        # Minimal placeholder domains for demo/UI dev
        label_domains, label_sizes = _default_metadata()

    if args.checkpoint:
        from app.inference import load_model
        model_kwargs = {}
        if args.use_semitone_spectrum:
            model_kwargs["in_channels"] = (84,)
        model_kwargs["nhead"] = args.nhead   # always set, defaults to 4
        model = load_model(args.checkpoint, label_sizes, device, model_kwargs)
    else:
        logger.warning("No checkpoint provided - running in DEMO mode with random predictions.")
        model = build_dummy_model(label_sizes)
        model.to(device)

    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QFont

    app = QApplication(sys.argv)
    app.setApplicationName("PARC")
    app.setApplicationDisplayName("PARC - Roman Numeral Analysis")

    # Prefer IBM Plex Mono if available
    default_font = QFont("IBM Plex Sans", 13)
    app.setFont(default_font)

    from app.ui.main_window import MainWindow
    win = MainWindow(model, label_domains, label_sizes, device, args)
    win.show()

    sys.exit(app.exec())


def _default_metadata():
    """
    Minimal label domains / sizes for demo mode without real dataset metadata.
    """
    from app.inference import CHROMATIC_SCALE

    DEGREE_MAP = {"I": 0, "II": 1, "III": 2, "IV": 3, "V": 4, "VI": 5, "VII": 6}

    rn_labels = []
    for deg in ["I","II","III","IV","V","VI","VII"]:
        for q in ["","m","°","ø","7","m7","°7","M7"]:
            rn_labels.append(f"{deg}{q}")

    key_labels = []
    for pc in CHROMATIC_SCALE:
        key_labels.append(f"{pc} major")
        key_labels.append(f"{pc} minor")

    label_domains = {
        "global_key":           key_labels,
        "tonicization":         [""] + [f"V/{d}" for d in ["I","II","III","IV","V","VI","VII"]],
        "root_scale_degree":    list(DEGREE_MAP.keys()),
        "quality":              ["M","m","d","h7","D7","M7","m7","d7","a","a7","aM7","mM7","oM7"],
        "inversion":            ["0","1","2"],
        "root_pitch_class":     CHROMATIC_SCALE,
        "bass_pitch_class":     CHROMATIC_SCALE,
        "tonicized_pitch_class":CHROMATIC_SCALE,
        "roman_numeral":        rn_labels,
    }
    label_sizes = {k: len(v) for k, v in label_domains.items()}
    return label_domains, label_sizes


if __name__ == "__main__":
    main()