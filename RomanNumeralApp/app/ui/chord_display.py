"""
app/ui/chord_display.py - Large animated chord display panel.

Shows:
  - Global key (top)
  - History strip (last 4 chords, fading)
  - Current chord (massive, center)
  - Chord tones & bass note
  - Tonicization label
  - Confidence glow
  - Circle of fifths indicator
"""
from __future__ import annotations

import math
from typing import List, Optional

from PyQt6.QtCore import (QEasingCurve, QPoint, QPropertyAnimation, QRect,
                           QRectF, Qt, QTimer, pyqtProperty)
from PyQt6.QtGui import (QColor, QFont, QFontMetrics, QLinearGradient, QPainter,
                          QPainterPath, QPen, QRadialGradient)
from PyQt6.QtWidgets import QSizePolicy, QWidget

from app.ui.theme import FUNCTION_COLORS, BG0, BG1, BG2, BORDER, FG0, FG1, FG2, AMBER



DEGREE_ORDER = ["I", "II", "III", "IV", "V", "VI", "VII"]
COF_ORDER    = ["C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F"]

HISTORY_MAX = 5


class ChordDisplayWidget(QWidget):
    """Central chord display with animated transitions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current: Optional[dict] = None
        self._history: List[dict]     = []
        self._alpha_anim: float       = 1.0
        self._fade_timer              = QTimer(self)
        self._fade_timer.setInterval(16)
        self._fade_timer.timeout.connect(self._tick_fade)
        self._fade_direction: int     = 0   # 0=stable, 1=in, -1=out
        self._fade_alpha: float       = 1.0
        self._pending: Optional[dict] = None

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(260)

    def set_chord(self, event: Optional[dict]):
        # Ignore if this chord is already the one shown or already queued.
        # Without the _pending guard, fast scrubbing re-triggers the fade for
        # the same target repeatedly and pollutes the history strip.
        if event is self._current or event == self._current:
            return
        if event is self._pending or event == self._pending:
            return
        # NOTE: history is appended at the crossfade swap point (see
        # _tick_fade), NOT here. Appending here recorded _current on every
        # call - and during a scrub _current hasn't swapped yet, so the same
        # chord got pushed many times, corrupting the strip.
        self._pending = event
        self._start_crossfade()

    def clear(self):
        self._current = None
        self._history = []
        self.update()

    # animation

    def _start_crossfade(self):
        self._fade_direction = -1
        self._fade_timer.start()

    def _tick_fade(self):
        step = 0.08
        if self._fade_direction == -1:
            self._fade_alpha = max(0.0, self._fade_alpha - step)
            if self._fade_alpha <= 0.0:
                # Swap point: the outgoing chord is now leaving the display, so
                # this is the correct moment to record it in history - exactly
                # once, and only chords that were actually shown.
                if self._current is not None:
                    self._history.append(self._current)
                    if len(self._history) > HISTORY_MAX:
                        self._history.pop(0)
                self._current = self._pending
                self._pending = None
                self._fade_direction = 1
        else:
            self._fade_alpha = min(1.0, self._fade_alpha + step)
            if self._fade_alpha >= 1.0:
                self._fade_timer.stop()
                self._fade_direction = 0
        self.update()

    # painting

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        W, H = self.width(), self.height()
        rect = QRectF(0, 0, W, H)

        # Background gradient
        bg = QLinearGradient(0, 0, 0, H)
        bg.setColorAt(0, QColor(BG1))
        bg.setColorAt(1, QColor(BG0))
        p.fillRect(rect, bg)

        cur  = self._current
        func = cur.get("function", "other") if cur else "other"
        accent, accent_dim = FUNCTION_COLORS.get(func, (AMBER, "#3D2A00"))
        accent_col     = QColor(accent)
        accent_dim_col = QColor(accent_dim)

        # Confidence glow
        if cur:
            conf = cur.get("confidence", 0.5)
            glow = QRadialGradient(W / 2, H * 0.45, W * 0.45)
            glow_col = QColor(accent)
            glow_col.setAlphaF(conf * 0.18 * self._fade_alpha)
            glow.setColorAt(0, glow_col)
            glow.setColorAt(1, QColor(0, 0, 0, 0))
            p.fillRect(rect, glow)

        alpha = int(255 * self._fade_alpha)
        p.setOpacity(self._fade_alpha)

        # Key label
        y_key = 22
        if cur:
            key_text = cur.get("global_key", "")
            font_key = QFont("IBM Plex Mono", 11, QFont.Weight.Medium)
            p.setFont(font_key)
            p.setPen(QColor(FG2))
            p.drawText(QRectF(0, y_key, W, 20), Qt.AlignmentFlag.AlignHCenter, "KEY")
            p.setPen(accent_col)
            p.drawText(QRectF(0, y_key + 18, W, 24), Qt.AlignmentFlag.AlignHCenter, key_text.upper())

        # History strip
        hist_y = 76
        hist_h = 32
        if self._history:
            n = len(self._history)
            slot_w = min(W / (HISTORY_MAX + 1), 90)
            x_start = (W - slot_w * n) / 2 - slot_w / 2

            for i, hchord in enumerate(self._history):
                fade = (i + 1) / (n + 1) * 0.7
                hfunc = hchord.get("function", "other")
                hcolor_s = FUNCTION_COLORS.get(hfunc, (FG1, BORDER))[0]
                hcolor = QColor(hcolor_s)
                hcolor.setAlphaF(fade * self._fade_alpha)
                p.setPen(hcolor)
                font_hist = QFont("IBM Plex Mono", 11 + i, QFont.Weight.Normal)
                p.setFont(font_hist)
                label = hchord.get("chord_label", "?")
                p.drawText(
                    QRectF(x_start + i * slot_w, hist_y, slot_w, hist_h),
                    Qt.AlignmentFlag.AlignCenter,
                    label,
                )

        # Current chord
        chord_y = hist_y + hist_h + 10
        chord_h = H - chord_y - 80

        if cur:
            label = cur.get("chord_label", "?")
            # Dynamic font size
            max_fs = min(int(chord_h * 0.72), 110)
            fs = max(32, max_fs)
            font_chord = QFont("IBM Plex Mono", fs, QFont.Weight.Bold)
            fm = QFontMetrics(font_chord)
            while fm.horizontalAdvance(label) > W * 0.88 and fs > 28:
                fs -= 2
                font_chord.setPointSize(fs)
                fm = QFontMetrics(font_chord)

            p.setFont(font_chord)
            p.setPen(accent_col)
            p.drawText(QRectF(0, chord_y, W, chord_h), Qt.AlignmentFlag.AlignCenter, label)

            # Chord tones
            tones = cur.get("chord_tones", [])
            bass_pc = cur.get("bass_pitch_class", None)
            from app.inference import CHROMATIC_SCALE
            bass_name = CHROMATIC_SCALE[bass_pc % 12] if bass_pc is not None else None

            font_tones = QFont("IBM Plex Mono", 12)
            p.setFont(font_tones)
            tones_str = "  ".join(tones)
            p.setPen(QColor(FG1))
            p.drawText(QRectF(0, H - 72, W, 22), Qt.AlignmentFlag.AlignHCenter, tones_str)

            if bass_name:
                bass_str = f"bass: {bass_name}"
                p.setPen(QColor(FG2))
                font_bass = QFont("IBM Plex Mono", 10)
                p.setFont(font_bass)
                p.drawText(QRectF(0, H - 50, W, 18), Qt.AlignmentFlag.AlignHCenter, bass_str)

            # Tonicization
            tonic = str(cur.get("tonicization", ""))
            if tonic and tonic not in ("", "None", "0"):
                font_ton = QFont("IBM Plex Sans", 10)
                p.setFont(font_ton)
                p.setPen(QColor(FG2))
                p.drawText(QRectF(0, H - 30, W, 18),
                           Qt.AlignmentFlag.AlignHCenter,
                           f"-> {tonic}")

        p.end()



class CircleOfFifths(QWidget):
    """Compact circle of fifths showing current key and chord degree.

    Highlight semantics
    - Amber wedge              -> the song's global key root
    - Blue wedge               -> the current chord's root
    - Amber wedge, blue outline -> both coincide (chord root == key root, e.g.
                                  a tonic chord in the home key): the wedge is
                                  filled amber for the key and framed with a
                                  blue outline so the chord-root state stays
                                  legible without clutter.
    """

    NOTES = ["C", "G", "D", "A", "E", "B", "F♯", "D♭", "A♭", "E♭", "B♭", "F"]

    _CHORD_BLUE = "#4A8FF4"

    # Enharmonic-aware note-name -> pitch-class (0–11) lookup.
    # Covers sharps, flats, Unicode accidentals, and the rare B#/E#/Cb/Fb.
    _NAME_TO_PC = {
        "C": 0,  "B#": 0,
        "C#": 1, "DB": 1,
        "D": 2,
        "D#": 3, "EB": 3,
        "E": 4,  "FB": 4,
        "F": 5,  "E#": 5,
        "F#": 6, "GB": 6,
        "G": 7,
        "G#": 8, "AB": 8,
        "A": 9,
        "A#": 10, "BB": 10,
        "B": 11, "CB": 11,
    }

    @classmethod
    def _name_to_pc(cls, name: str) -> Optional[int]:
        """Convert a note name (e.g. 'Eb', 'F♯', 'c#') to a pitch class 0–11."""
        if not name:
            return None
        # Normalise Unicode accidentals and case before lookup.
        n = name.replace("♯", "#").replace("♭", "b").strip()
        # Uppercase the whole token so 'eb' and 'Eb' both resolve; the map
        # keys are stored uppercase to match.
        return cls._NAME_TO_PC.get(n.upper())

    def __init__(self, parent=None):
        super().__init__(parent)
        self._key_pc:   Optional[int] = None
        self._chord_pc: Optional[int] = None
        self.setMinimumSize(160, 160)
        self.setMaximumSize(220, 220)

    def set_chord(self, event: Optional[dict]):
        if not event:
            return
        # Key: parse the root name out of e.g. "Eb major" -> pitch class.
        key = event.get("global_key", "")
        self._key_pc = self._name_to_pc(key.split()[0]) if key else None
        # Chord root: pitch class is already an integer in the event, so use
        # it directly instead of round-tripping through CHROMATIC_SCALE.
        rpc = event.get("root_pitch_class", None)
        self._chord_pc = (int(rpc) % 12) if rpc is not None else None
        self.update()

    def paintEvent(self, event):
        p   = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H  = self.width(), self.height()
        cx, cy = W / 2, H / 2
        r_outer = min(cx, cy) - 14
        r_inner = r_outer * 0.52
        r_text  = r_outer * 0.78

        # Outer disc
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(BG1))
        p.drawEllipse(QRectF(cx - r_outer - 2, cy - r_outer - 2,
                              (r_outer + 2) * 2, (r_outer + 2) * 2))

        # Wedge geometry: 30 degrees per slot, 2 degrees gap on each side for clean separation.
        WEDGE_SPAN = 28.0   # degrees of fill (30 degrees slot − 1 degrees gap each side)
        WEDGE_HALF = WEDGE_SPAN / 2

        for i, note in enumerate(self.NOTES):
            angle  = math.radians(-90 + i * 30)
            note_pc = self._name_to_pc(note)
            # Match by pitch class so enharmonic spelling (Db vs C#) does not
            # break highlighting on black-key roots/keys.
            is_key   = (note_pc is not None and note_pc == self._key_pc)
            is_chord = (note_pc is not None and note_pc == self._chord_pc)

            if is_key:
                fill = QColor(AMBER); fill.setAlphaF(0.85)
            elif is_chord:
                fill = QColor(self._CHORD_BLUE); fill.setAlphaF(0.75)
            else:
                fill = QColor(BG2)

            # Build wedge path
            path = QPainterPath()
            path.moveTo(cx, cy)
            path.arcTo(QRectF(cx - r_outer, cy - r_outer, r_outer * 2, r_outer * 2),
                       90 - i * 30 - WEDGE_HALF, WEDGE_SPAN)
            path.lineTo(cx, cy)

            # Fill the wedge
            p.setBrush(fill)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPath(path)

            # Overlap state: key + chord on the same wedge -> amber fill framed
            # with a clean blue outline. The inner disc (drawn afterward) masks
            # the inner half of the stroke, leaving a tidy band on the rim.
            if is_key and is_chord:
                pen = QPen(QColor(self._CHORD_BLUE))
                pen.setWidthF(2.2)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPath(path)

            # Note label
            tx = cx + r_text * math.cos(angle)
            ty = cy + r_text * math.sin(angle)
            highlighted = is_key or is_chord
            font = QFont("IBM Plex Mono", 8,
                         QFont.Weight.Bold if highlighted else QFont.Weight.Normal)
            p.setFont(font)
            p.setPen(QColor(FG0) if highlighted else QColor(FG2))
            p.drawText(QRectF(tx - 14, ty - 9, 28, 18),
                       Qt.AlignmentFlag.AlignCenter, note)

        # Hollow inner disc (drawn once, after all wedges)
        p.setBrush(QColor(BG0))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QRectF(cx - r_inner, cy - r_inner, r_inner * 2, r_inner * 2))

        # Center dot
        p.setBrush(QColor(BG1))
        p.setPen(QColor(BORDER))
        p.drawEllipse(QRectF(cx - 4, cy - 4, 8, 8))
        p.end()