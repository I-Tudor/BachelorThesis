"""
app/ui/theme.py - Design tokens and PyQt6 stylesheets.

Aesthetic direction: refined dark luxury - deep slate backgrounds,
warm amber accents, musical editorial typography, subtle grain texture
via gradient layering.
"""

# palette

BG0      = "#0B0C10"    # deepest background
BG1      = "#12141A"    # card / panel background
BG2      = "#1C1F28"    # raised surfaces
BG3      = "#252936"    # subtle hover / border
BORDER   = "#2E3347"
SUBTLE   = "#3D4460"

FG0      = "#EEF0F7"    # primary text
FG1      = "#A8ADBE"    # secondary text
FG2      = "#666C84"    # muted text

AMBER    = "#F4A84A"    # tonic / highlight
AMBER_DIM= "#7A5420"
BLUE     = "#4A8FF4"    # subdominant
BLUE_DIM = "#1E3D7A"
RED      = "#F46A4A"    # dominant
RED_DIM  = "#7A2A1E"
GREY     = "#555B70"    # other / neutral

# Function -> color mapping
FUNCTION_COLORS = {
    "tonic":       (AMBER,    AMBER_DIM),
    "subdominant": (BLUE,     BLUE_DIM),
    "dominant":    (RED,      RED_DIM),
    "other":       (FG1,      BORDER),
}

# global QSS

BASE_QSS = f"""
QMainWindow, QDialog, QWidget {{
    background-color: {BG0};
    color: {FG0};
    font-family: "IBM Plex Sans", "SF Pro Text", "Segoe UI", sans-serif;
    font-size: 13px;
}}

QLabel {{
    color: {FG0};
    background: transparent;
}}

QPushButton {{
    background-color: {BG2};
    color: {FG0};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 16px;
    font-size: 12px;
    letter-spacing: 0.5px;
}}
QPushButton:hover {{
    background-color: {BG3};
    border-color: {SUBTLE};
}}
QPushButton:pressed {{
    background-color: {BORDER};
}}
QPushButton:disabled {{
    color: {FG2};
    border-color: {BORDER};
}}

QPushButton#primary {{
    background-color: {AMBER};
    color: {BG0};
    border: none;
    font-weight: 600;
}}
QPushButton#primary:hover {{
    background-color: #F8B85A;
}}

QSlider::groove:horizontal {{
    height: 3px;
    background: {BG3};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {AMBER};
    border: none;
    width: 12px;
    height: 12px;
    margin: -5px 0;
    border-radius: 6px;
}}
QSlider::sub-page:horizontal {{
    background: {AMBER};
    border-radius: 2px;
}}

QScrollBar:horizontal {{
    height: 6px;
    background: {BG1};
}}
QScrollBar::handle:horizontal {{
    background: {BORDER};
    border-radius: 3px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {SUBTLE};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

QSplitter::handle {{
    background: {BORDER};
    width: 1px;
    height: 1px;
}}

QComboBox {{
    background-color: {BG2};
    color: {FG0};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px 10px;
}}
QComboBox::drop-down {{
    border: none;
}}
QComboBox QAbstractItemView {{
    background-color: {BG2};
    color: {FG0};
    border: 1px solid {BORDER};
    selection-background-color: {BG3};
}}

QProgressBar {{
    background-color: {BG2};
    border: 1px solid {BORDER};
    border-radius: 4px;
    height: 6px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background-color: {AMBER};
    border-radius: 3px;
}}

QToolTip {{
    background-color: {BG2};
    color: {FG0};
    border: 1px solid {BORDER};
    padding: 4px 8px;
    border-radius: 4px;
}}
"""