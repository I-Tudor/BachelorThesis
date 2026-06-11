"""
app/export_pdf.py - Generate a lead-sheet style PDF from a chord timeline.

Dependencies: reportlab  (pip install reportlab)

Layout
  Title + key at top.
  Chords arranged in rows of 8 bars.
  Each bar = one chord event (or subdivided if multiple chords fit a bar).
  Color-coded rectangles behind chord symbols match harmonic function.
  Roman numeral in bold, quality/inversion as smaller text.
  Bar number at top-left of each cell.
  Timestamp (mm:ss) at bottom-left of each cell.
  Confidence dot (filled circle) at top-right, radius ∝ confidence.
"""
from __future__ import annotations

from typing import List

# colour palette (matches theme.py)

_FUNCTION_RGB = {
    "tonic":       (0.957, 0.659, 0.290),   # amber
    "subdominant": (0.290, 0.561, 0.957),   # blue
    "dominant":    (0.957, 0.416, 0.290),   # red-orange
    "other":       (0.333, 0.361, 0.435),   # slate
}
_BG_ALPHA = 0.18   # fill alpha for coloured cells
_TEXT_RGB = (0.933, 0.941, 0.969)    # FG0
_DIM_RGB  = (0.659, 0.678, 0.745)    # FG1
_DARK_RGB = (0.071, 0.075, 0.094)    # BG0


def export_lead_sheet(
    timeline:  List[dict],
    path:      str,
    title:     str = "Chord Analysis",
    bpm:       Optional[float] = None,
    bars_per_row: int = 8,
) -> None:
    """
    Render a lead-sheet PDF and write it to *path*.

    Parameters
    ----------
    timeline     : chord event list from build_chord_timeline()
    path         : output file path (should end with .pdf)
    title        : document title printed at the top
    bpm          : optional tempo annotation
    bars_per_row : number of chord cells per row
    """
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.units import cm, mm
        from reportlab.pdfgen.canvas import Canvas
        from reportlab.lib.colors import Color, HexColor, black, white
    except ImportError as exc:
        raise ImportError(
            "reportlab is required for PDF export.\n"
            "Install it with:  pip install reportlab"
        ) from exc

    if not timeline:
        raise ValueError("Empty chord timeline - nothing to export.")

    # page setup
    PAGE_W, PAGE_H = landscape(A4)
    MARGIN_X = 1.5 * cm
    MARGIN_Y = 1.5 * cm
    USABLE_W = PAGE_W - 2 * MARGIN_X

    CELL_W = USABLE_W / bars_per_row
    CELL_H = 2.2 * cm
    HEADER_H = 2.8 * cm

    c = Canvas(path, pagesize=landscape(A4))
    c.setTitle(title)
    c.setAuthor("PARC - Roman Numeral Chord Analyzer")

    # dark page background
    c.setFillColorRGB(*_DARK_RGB)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

    # title block
    c.setFillColorRGB(*_TEXT_RGB)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(MARGIN_X, PAGE_H - MARGIN_Y - 0.5 * cm, title)

    # Key + BPM
    key = timeline[0].get("global_key", "")
    meta_parts = []
    if key:
        meta_parts.append(f"Key: {key}")
    if bpm:
        meta_parts.append(f"♩ = {bpm:.0f}")
    meta_parts.append(f"{len(timeline)} chord events")

    c.setFont("Helvetica", 10)
    c.setFillColorRGB(*_DIM_RGB)
    c.drawString(MARGIN_X, PAGE_H - MARGIN_Y - 1.15 * cm, "   ".join(meta_parts))

    # Colour legend (small, top-right)
    legend_x = PAGE_W - MARGIN_X - 4 * cm
    legend_y = PAGE_H - MARGIN_Y - 0.4 * cm
    legend_items = [
        ("Tonic",       "tonic"),
        ("Subdominant", "subdominant"),
        ("Dominant",    "dominant"),
        ("Other",       "other"),
    ]
    for i, (label, func) in enumerate(legend_items):
        rx = legend_x + i * (cm * 1.0)
        ry = legend_y
        r, g, b = _FUNCTION_RGB.get(func, _FUNCTION_RGB["other"])
        c.setFillColorRGB(r, g, b, 0.85)
        c.roundRect(legend_x + i * (cm * 1.05), ry - 0.28 * cm,
                    cm * 0.9, 0.28 * cm, 3, fill=1, stroke=0)
        c.setFillColorRGB(*_DARK_RGB)
        c.setFont("Helvetica", 6)
        c.drawCentredString(legend_x + i * (cm * 1.05) + 0.45 * cm,
                            ry - 0.22 * cm, label)

    # lay out events across rows
    events = timeline
    n_events = len(events)
    row = 0
    col = 0

    def cell_rect(row_idx: int, col_idx: int):
        x = MARGIN_X + col_idx * CELL_W
        y = PAGE_H - MARGIN_Y - HEADER_H - (row_idx + 1) * CELL_H
        return x, y, CELL_W, CELL_H

    def new_page():
        nonlocal row, col
        c.showPage()
        # re-draw background
        c.setFillColorRGB(*_DARK_RGB)
        c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        row = 0
        col = 0

    def rows_on_page() -> int:
        usable_h = PAGE_H - MARGIN_Y - HEADER_H - MARGIN_Y
        return max(1, int(usable_h // CELL_H))

    MAX_ROWS = rows_on_page()

    for idx, ev in enumerate(events):
        if row >= MAX_ROWS:
            new_page()
            MAX_ROWS = rows_on_page()

        x, y, w, h = cell_rect(row, col)
        func = ev.get("function", "other")
        r, g, b = _FUNCTION_RGB.get(func, _FUNCTION_RGB["other"])

        # Cell background fill
        c.setFillColorRGB(r, g, b, _BG_ALPHA)
        c.roundRect(x + 1, y + 1, w - 2, h - 2, 4, fill=1, stroke=0)

        # Top accent line
        c.setStrokeColorRGB(r, g, b, 0.80)
        c.setLineWidth(1.5)
        c.line(x + 1, y + h - 1, x + w - 1, y + h - 1)

        # Cell border (dim)
        c.setStrokeColorRGB(0.18, 0.20, 0.27, 1)
        c.setLineWidth(0.4)
        c.roundRect(x + 1, y + 1, w - 2, h - 2, 4, fill=0, stroke=1)

        # Bar number (top-left)
        c.setFont("Helvetica", 6.5)
        c.setFillColorRGB(*_DIM_RGB)
        c.drawString(x + 3, y + h - 9, f"#{idx + 1}")

        # Confidence dot (top-right)
        conf = ev.get("confidence", 0.5)
        dot_r = 2.5 + conf * 2.5
        dot_x = x + w - 5
        dot_y = y + h - 7
        c.setFillColorRGB(r, g, b, 0.5 + conf * 0.5)
        c.circle(dot_x, dot_y, dot_r, fill=1, stroke=0)

        # Main chord label (centre)
        label = ev.get("chord_label", ev.get("roman_numeral", "?"))
        font_size = _fit_font_size(label, w - 8, "Helvetica-Bold",
                                   min_size=10, max_size=20)
        c.setFont("Helvetica-Bold", font_size)
        c.setFillColorRGB(r, g, b)
        c.drawCentredString(x + w / 2, y + h / 2 - font_size * 0.35, label)

        # Chord tones (below label)
        tones = ev.get("chord_tones", [])
        if tones:
            c.setFont("Helvetica", 6)
            c.setFillColorRGB(*_DIM_RGB)
            c.drawCentredString(x + w / 2, y + 8, "  ".join(tones))

        # Timestamp (bottom-left)
        t_sec = ev.get("time", 0)
        ts = f"{int(t_sec)//60}:{int(t_sec)%60:02d}"
        c.setFont("Helvetica", 5.5)
        c.setFillColorRGB(*_DIM_RGB, 0.7)
        c.drawString(x + 3, y + 2, ts)

        col += 1
        if col >= bars_per_row:
            col = 0
            row += 1

    c.save()


# helpers

def _fit_font_size(text: str, max_width: float, font_name: str,
                   min_size: int = 8, max_size: int = 24) -> int:
    """Binary-search the largest font size that fits text within max_width."""
    try:
        from reportlab.pdfbase.pdfmetrics import stringWidth
    except ImportError:
        return min_size

    lo, hi = min_size, max_size
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if stringWidth(text, font_name, mid) <= max_width:
            lo = mid
        else:
            hi = mid - 1
    return lo


# allow running standalone for quick testing

if __name__ == "__main__":
    from typing import Optional
    _DEMO = [
        {"time": i * 2.0, "end": (i + 1) * 2.0,
         "chord_label": lbl, "roman_numeral": lbl,
         "global_key": "C major", "function": fn,
         "chord_tones": tones, "confidence": 0.85}
        for i, (lbl, fn, tones) in enumerate([
            ("I",    "tonic",       ["C","E","G"]),
            ("V⁷",  "dominant",    ["G","B","D","F"]),
            ("vi",   "tonic",       ["A","C","E"]),
            ("IV",   "subdominant", ["F","A","C"]),
        ] * 8)
    ]
    export_lead_sheet(_DEMO, "/tmp/parc_demo_lead_sheet.pdf", title="C Major - I V vi IV")
    print("Wrote /tmp/parc_demo_lead_sheet.pdf")
else:
    from typing import Optional