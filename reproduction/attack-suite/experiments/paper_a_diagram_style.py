"""Shared palette and drawing primitives for the Paper A diagrams.

Every diagram is drawn on one axes whose data coordinates are inches of the
final figure, so positions and font sizes are print sizes. Colours follow
make_v3_figures.py (grey = activation-only attacker, blue = joint attacker)
with one accent colour reserved for errors.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "paper_a" / "figures"
TEXT_WIDTH = 6.5

INK = "#333333"
MUTED = "#6B6B6B"
GREY = "#8C8C8C"
JOINT = "#3B6EA5"
ACCENT = "#C4452C"
ACCENT_FILL = "#F7DCD4"
CLIENT_FILL = "#EAF0F7"
SERVER_FILL = "#F1F1F1"
ATTACKER_FILL = "#FBEBD9"
ATTACKER = "#9A5B1E"
PAPER_FILL = "#FFFFFF"
RULE = "#BBBBBB"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8,
    "pdf.fonttype": 42, "axes.linewidth": 0.6, "axes.edgecolor": "#555555",
    "xtick.color": "#555555", "xtick.labelsize": 7.5, "xtick.major.width": 0.6,
    "text.hinting": "none", "text.hinting_factor": 1,
})


def canvas(width: float, height: float) -> tuple[Figure, Axes]:
    """Figure with one invisible axes whose coordinates are inches."""
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, width), ylim=(0, height))
    ax.axis("off")
    return fig, ax


def box(ax: Axes, x: float, y: float, w: float, h: float, fill: str, edge: str = RULE,
        lw: float = 0.7, ls: str = "-", radius: float = 0.05, z: float = 1) -> None:
    """Rounded rectangle with lower-left corner (x, y)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                fc=fill, ec=edge, lw=lw, ls=ls, zorder=z))


def pill(ax: Axes, x: float, y: float, text: str, colour: str, size: float = 7.5,
         ha: str = "left", fg: str = "white", weight: str = "bold") -> None:
    """Filled rounded label centred vertically on y."""
    ax.text(x, y, text, ha=ha, va="center", fontsize=size, color=fg, weight=weight, zorder=6,
            bbox={"boxstyle": "round,pad=0.25,rounding_size=0.6", "fc": colour, "ec": "none"})


def arrow(ax: Axes, start: tuple[float, float], end: tuple[float, float], colour: str = INK,
          lw: float = 1.0, ls: str = "-", rad: float = 0.0, z: float = 4) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>,head_length=4,head_width=2.2",
                                 connectionstyle=f"arc3,rad={rad}", color=colour, lw=lw, ls=ls,
                                 shrinkA=0, shrinkB=0, zorder=z))


def title(ax: Axes, x: float, y: float, letter: str, text: str, note: str = "") -> None:
    """Panel title: bold '(a) Title' with an optional grey note on the line below."""
    ax.text(x, y, f"({letter}) {text}", fontsize=8.5, weight="bold", color=INK, va="baseline")
    if note:
        ax.text(x, y - 0.14, note, fontsize=7, color=MUTED, va="baseline")


def _extent(ax: Axes, s: str, size: float, weight: str) -> float:
    fig = ax.figure
    t = ax.text(0, 0, s, fontsize=size, weight=weight)
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


def text_width(ax: Axes, s: str, size: float, weight: str = "normal") -> float:
    """Advance width of s in inches, trailing spaces included."""
    return _extent(ax, s + "|", size, weight) - _extent(ax, "|", size, weight) if s else 0.0


def save(fig: Figure, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    return path
