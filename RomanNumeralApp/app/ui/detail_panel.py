"""
app/ui/detail_panel.py - Chord detail sidebar and settings tab.

Changes from v1
- _small_label defined before use (was called in __init__ before definition)
- PDF export button added
- Settings panel includes "Bars per row" for PDF export
- Cleaner layout with proper spacing
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFrame, QHBoxLayout,
                               QLabel, QPushButton, QScrollArea,
                               QSpinBox, QVBoxLayout, QWidget)

from app.ui.theme import AMBER, BG1, BG2, BORDER, FG0, FG1, FG2


# helpers

def _small_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color:{FG2}; font-size:9px; letter-spacing:1.5px;"
        f" font-family:'IBM Plex Mono'; background:transparent;")
    return lbl


def _sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"background:{BORDER}; max-height:1px;")
    return f


class InfoRow(QWidget):
    """One label + value row."""
    def __init__(self, label: str, value: str = "", parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 1, 0, 1)
        lay.setSpacing(4)
        self._lbl = QLabel(label)
        self._lbl.setStyleSheet(
            f"color:{FG2}; font-size:10px; min-width:80px; max-width:80px;")
        self._val = QLabel(value)
        self._val.setStyleSheet(
            f"color:{FG0}; font-size:11px; font-family:'IBM Plex Mono';")
        lay.addWidget(self._lbl)
        lay.addWidget(self._val, 1)

    def set_value(self, v: str):
        self._val.setText(v)


# detail panel

class DetailPanel(QWidget):
    """Right-side panel showing chord details and export controls."""

    export_json_clicked = pyqtSignal()
    export_csv_clicked  = pyqtSignal()
    export_pdf_clicked  = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{BG1};")

        # Scrollable area so it works at small heights
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background:transparent;")

        inner = QWidget()
        inner.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(14, 14, 14, 10)
        lay.setSpacing(6)

        # Chord info
        lay.addWidget(_small_label("CHORD DETAILS"))
        lay.addWidget(_sep())

        self.row_chord   = InfoRow("Chord",     "-")
        self.row_key     = InfoRow("Key",       "-")
        self.row_quality = InfoRow("Quality",   "-")
        self.row_inv     = InfoRow("Inversion", "-")
        self.row_root    = InfoRow("Root",      "-")
        self.row_bass    = InfoRow("Bass",      "-")
        self.row_tones   = InfoRow("Tones",     "-")
        self.row_tonic   = InfoRow("Tonicizes", "-")
        self.row_conf    = InfoRow("Confidence","-")
        self.row_func    = InfoRow("Function",  "-")
        self.row_time    = InfoRow("Time",      "-")

        for row in [self.row_chord, self.row_key, self.row_quality,
                    self.row_inv, self.row_root, self.row_bass,
                    self.row_tones, self.row_tonic, self.row_conf,
                    self.row_func, self.row_time]:
            lay.addWidget(row)

        # Export
        lay.addSpacing(8)
        lay.addWidget(_small_label("EXPORT"))
        lay.addWidget(_sep())

        def _export_btn(text: str) -> QPushButton:
            b = QPushButton(text)
            b.setStyleSheet(f"""
                QPushButton {{
                    background:{BG2}; color:{FG1}; border:1px solid {BORDER};
                    border-radius:5px; padding:5px 10px; font-size:11px;
                    text-align:left;
                }}
                QPushButton:hover {{ border-color:{AMBER}; color:{AMBER}; }}
                QPushButton:disabled {{ color:{FG2}; border-color:{BORDER}; }}
            """)
            return b

        self.btn_json = _export_btn("⬇  Export JSON")
        self.btn_csv  = _export_btn("⬇  Export CSV")
        self.btn_pdf  = _export_btn("⬇  Export Lead Sheet PDF")

        self.btn_json.clicked.connect(self.export_json_clicked)
        self.btn_csv.clicked.connect(self.export_csv_clicked)
        self.btn_pdf.clicked.connect(self.export_pdf_clicked)

        for btn in (self.btn_json, self.btn_csv, self.btn_pdf):
            lay.addWidget(btn)

        lay.addStretch()
        scroll.setWidget(inner)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll)


    def set_chord(self, event: Optional[dict]):
        if not event:
            for row in [self.row_chord, self.row_key, self.row_quality,
                        self.row_inv, self.row_root, self.row_bass,
                        self.row_tones, self.row_tonic, self.row_conf,
                        self.row_func, self.row_time]:
                row.set_value("-")
            return

        from app.inference import CHROMATIC_SCALE

        self.row_chord.set_value(event.get("chord_label", "?"))
        self.row_key.set_value(event.get("global_key", "?"))
        self.row_quality.set_value(event.get("quality", "?"))

        inv = int(event.get("inversion", 0))
        self.row_inv.set_value(
            {0: "Root position", 1: "1st inversion",
             2: "2nd inversion"}.get(inv, str(inv)))

        self.row_root.set_value(
            CHROMATIC_SCALE[int(event.get("root_pitch_class", 0)) % 12])
        self.row_bass.set_value(
            CHROMATIC_SCALE[int(event.get("bass_pitch_class", 0)) % 12])

        tones = event.get("chord_tones", [])
        self.row_tones.set_value("  ".join(tones) if tones else "-")

        tonic = str(event.get("tonicization", ""))
        self.row_tonic.set_value(
            tonic if tonic and tonic not in ("None", "0", "") else "-")

        conf = event.get("confidence")
        self.row_conf.set_value(f"{conf:.0%}" if conf is not None else "-")

        func = event.get("function", "other")
        self.row_func.set_value(
            {"tonic": "Tonic", "subdominant": "Subdominant",
             "dominant": "Dominant", "other": "-"}.get(func, func.title()))

        t = event.get("time", 0)
        end = event.get("end", t)
        self.row_time.set_value(
            f"{int(t)//60}:{int(t)%60:02d} – {int(end)//60}:{int(end)%60:02d}")

    def set_export_enabled(self, enabled: bool):
        for btn in (self.btn_json, self.btn_csv, self.btn_pdf):
            btn.setEnabled(enabled)


# settings panel

class SettingsPanel(QWidget):
    """Settings: compute device, feature type, PDF export options."""

    settings_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{BG1};")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 10)
        lay.setSpacing(8)

        # Compute device
        lay.addWidget(_small_label("COMPUTE DEVICE"))
        self.combo_device = QComboBox()
        self.combo_device.addItems(self._available_devices())
        self.combo_device.setStyleSheet(
            f"background:{BG2}; color:{FG0}; border:1px solid {BORDER};"
            f" border-radius:4px; padding:3px 8px;")
        lay.addWidget(self.combo_device)

        # Feature type
        lay.addSpacing(4)
        lay.addWidget(_small_label("FEATURE TYPE"))
        self.combo_feat = QComboBox()
        self.combo_feat.addItems(["Chroma + Bass chroma", "Semitone spectrum (84-bin)"])
        self.combo_feat.setStyleSheet(self.combo_device.styleSheet())
        lay.addWidget(self.combo_feat)

        # PDF: bars per row
        lay.addSpacing(4)
        lay.addWidget(_small_label("PDF BARS PER ROW"))
        self.spin_bars = QSpinBox()
        self.spin_bars.setRange(4, 16)
        self.spin_bars.setValue(8)
        self.spin_bars.setStyleSheet(
            f"background:{BG2}; color:{FG0}; border:1px solid {BORDER};"
            f" border-radius:4px; padding:3px 6px;")
        lay.addWidget(self.spin_bars)

        # Apply
        lay.addSpacing(8)
        btn = QPushButton("Apply Settings")
        btn.setObjectName("primary")
        btn.clicked.connect(self._emit)
        lay.addWidget(btn)

        lay.addStretch()

        # About block
        lay.addWidget(_sep())
        about = QLabel(
            "MAJOR TOM - Roman Numeral Analysis\n"
            "Multi-task Transformer · PARC dataset\n"
            "github.com/uai-ufmg/parc"
        )
        about.setStyleSheet(
            f"color:{FG2}; font-size:9px; font-family:'IBM Plex Mono';"
            f" background:transparent;")
        about.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(about)

    def _available_devices(self):
        try:
            import torch
            devs = ["cpu"]
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    devs.append(f"cuda:{i}")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                devs.append("mps")
            return devs
        except ImportError:
            return ["cpu"]

    def _emit(self):
        self.settings_changed.emit({
            "device":               self.combo_device.currentText(),
            "use_semitone_spectrum":self.combo_feat.currentIndex() == 1,
            "pdf_bars_per_row":     self.spin_bars.value(),
        })

    def pdf_bars_per_row(self) -> int:
        return self.spin_bars.value()