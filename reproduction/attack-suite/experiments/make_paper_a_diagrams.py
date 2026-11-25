#!/usr/bin/env python
"""Method diagrams of Paper A: the logical architecture and the attack pipeline.

Wording follows Sections 3 and 5 of paper/paper_a/main.tex; the diagrams carry
no measured values. Palette and primitives are shared with Figure 1
(experiments/paper_a_diagram_style.py).

Usage: uv run --no-project --with matplotlib python experiments/make_paper_a_diagrams.py
"""

from matplotlib.axes import Axes

from paper_a_diagram_style import (ACCENT, ATTACKER, ATTACKER_FILL, CLIENT_FILL, GREY, INK, JOINT, MUTED,
                                   PAPER_FILL, RULE, SERVER_FILL, TEXT_WIDTH, arrow, box, canvas, save,
                                   text_width)

BODY, NOTE = 8, 7.5


def stage(ax: Axes, x: float, y: float, w: float, h: float, head: str, detail: str = "",
          fill: str = PAPER_FILL, edge: str = RULE, ls: str = "-") -> None:
    """Box with a bold heading near the top and an optional muted detail below it."""
    box(ax, x, y, w, h, fill, edge=edge, ls=ls)
    top = y + h - 0.08 if detail else y + h / 2
    ax.text(x + w / 2, top, head, ha="center", va="top" if detail else "center", fontsize=BODY,
            weight="bold", color=INK, linespacing=1.1, zorder=5)
    if detail:
        lines = head.count("\n") + 1
        ax.text(x + w / 2, top - 0.15 * lines - 0.02, detail, ha="center", va="top", fontsize=NOTE,
                color=MUTED, linespacing=1.15, zorder=5)


def badge(ax: Axes, x: float, y: float, n: int) -> None:
    """Step number matching the caption's (1)-(4)."""
    ax.text(x, y, str(n), ha="center", va="center", fontsize=6.5, weight="bold", color="white", zorder=7,
            bbox={"boxstyle": "circle,pad=0.18", "fc": INK, "ec": "none"})


def candidates(ax: Axes, x: float, y: float, w: float, h: float) -> None:
    """Candidate stage: the activation list, and the half the joint attacker takes from the gradient."""
    box(ax, x, y, w, h, PAPER_FILL, edge=RULE)
    cx = x + w / 2
    ax.text(cx, y + h - 0.08, "candidates per position", ha="center", va="top", fontsize=BODY,
            weight="bold", color=INK)
    ax.text(cx, y + h - 0.3, "k closest tokens from\nthe activation", ha="center", va="top", fontsize=NOTE,
            color=INK, linespacing=1.1)
    ax.text(cx, y + h - 0.58, "joint attacker: half of them\nreplaced from the gradient", ha="center",
            va="top", fontsize=NOTE, color=JOINT, linespacing=1.1)
    badge(ax, x, y + h - 0.35, 2)
    badge(ax, x, y + h - 0.67, 3)


def pipeline() -> None:
    fig, ax = canvas(TEXT_WIDTH, 1.72)
    y, h, gap = 0.5, 0.92, 0.075
    widths = [0.76, 0.88, 1.58, 1.36, 0.56, 0.95]
    xs = [0.02]
    for w in widths[:-1]:
        xs.append(xs[-1] + w + gap)
    stage(ax, xs[0], y, widths[0], h, "32 public\ncalibration\ndocuments")
    stage(ax, xs[1], y, widths[1], h, "ridge fit", "maps activations\nback to token\nembeddings")
    candidates(ax, xs[2], y, widths[2], h)
    stage(ax, xs[3], y, widths[3], h, "beam search, width 2", "keeps the guesses whose\nactivations, recomputed\n"
          "through its copy of the\nclient's layers, best\nmatch the observed ones")
    stage(ax, xs[4], y, widths[4], h, "recon-\nstruction")
    stage(ax, xs[5], y, widths[5], h, "scored\nthree ways", "token\nwhole-document\nnames, numbers,\nidentifiers")
    badge(ax, xs[1], y + h, 1)
    badge(ax, xs[3], y + h, 4)
    for x, w in zip(xs[:-1], widths[:-1]):
        arrow(ax, (x + w + 0.005, y + h / 2), (x + w + gap - 0.005, y + h / 2))
    top = y + h
    observed(ax, xs[2] + widths[2] * 0.3, top, "observed\nactivation", INK)
    observed(ax, xs[2] + widths[2] * 0.72, top, "observed\ngradient", JOINT)
    observed(ax, xs[3] + widths[3] / 2, top, "observed\nactivation", INK)
    controls(ax, xs[2], widths[2], y)
    path = save(fig, "fig_attack_pipeline.pdf")
    print("wrote", path)


def observed(ax: Axes, x: float, top: float, label: str, colour: str) -> None:
    """A tensor observed at the split, entering a stage from above."""
    ax.text(x, top + 0.07, label, ha="center", va="bottom", fontsize=7, color=colour, linespacing=1.0)
    arrow(ax, (x, top + 0.07), (x, top + 0.01), colour=colour, lw=0.8)


def controls(ax: Axes, cx0: float, cw: float, y: float) -> None:
    """Dashed side branches: the two controls that remove the gradient's information."""
    specs = [("shuffled-gradient control:", "another document's gradient", "left"),
             ("random-candidate control:", "random tokens replace the gradient half", "right")]
    for head, detail, side in specs:
        w = max(text_width(ax, head, NOTE, "bold"), text_width(ax, detail, NOTE)) + 0.14
        x = 0.02 if side == "left" else cx0 + cw + 0.25
        box(ax, x, 0.02, w, 0.36, PAPER_FILL, edge=GREY, ls=(0, (3, 2)))
        ax.text(x + 0.07, 0.28, head, fontsize=NOTE, weight="bold", color=MUTED, va="center")
        ax.text(x + 0.07, 0.12, detail, fontsize=NOTE, color=MUTED, va="center")
        start = (x + w, 0.2) if side == "left" else (x, 0.2)
        end = (cx0 + cw * (0.3 if side == "left" else 0.76), y - 0.01)
        arrow(ax, start, end, colour=GREY, lw=0.8, ls=(0, (3, 2)), rad=0.3 if side == "left" else -0.3)


def architecture() -> None:
    fig, ax = canvas(TEXT_WIDTH, 2.3)
    split = 2.55
    client_side(ax)
    server_side(ax)
    ax.plot([split, split], [0.76, 2.17], color=MUTED, lw=0.9, ls=(0, (3, 2)), zorder=2)
    ax.text(split, 2.19, "split", ha="center", va="bottom", fontsize=NOTE, color=MUTED)
    arrow(ax, (1.76, 1.62), (3.33, 1.62), lw=1.1)
    ax.text(split, 1.82, "activations", ha="center", va="bottom", fontsize=BODY, weight="bold", color=INK)
    ax.text(split, 1.67, "one vector per token", ha="center", va="bottom", fontsize=NOTE, color=INK)
    ax.text(split, 1.57, "training and inference", ha="center", va="top", fontsize=NOTE, color=MUTED)
    arrow(ax, (3.33, 1.17), (1.76, 1.17), colour=JOINT, lw=1.1)
    ax.text(split, 1.22, "gradients", ha="center", va="bottom", fontsize=BODY, weight="bold", color=JOINT)
    ax.text(split, 1.12, "training only", ha="center", va="top", fontsize=NOTE, color=JOINT)
    attacker(ax, split)
    offloading_inset(ax)
    path = save(fig, "fig_architecture.pdf")
    print("wrote", path)


def client_side(ax: Axes) -> None:
    box(ax, 0.02, 0.86, 1.74, 1.31, CLIENT_FILL, edge=JOINT, lw=0.7)
    ax.text(0.09, 2.08, "CLIENT", fontsize=BODY, weight="bold", color=JOINT, va="top")
    ax.text(0.09, 1.93, "holds the private text", fontsize=NOTE, color=INK, va="top")
    stage(ax, 0.07, 1.0, 0.47, 0.76, "private\ntext")
    stage(ax, 0.61, 1.0, 0.5, 0.76, "token\nembed-\ndings")
    stage(ax, 1.19, 1.0, 0.5, 0.76, "layers\n1 … L", "client's\nlayers")
    arrow(ax, (0.54, 1.38), (0.6, 1.38))
    arrow(ax, (1.12, 1.38), (1.18, 1.38))
    ax.text(0.09, 0.9, "GPT-2: 12 layers; L = 3 or 6", va="bottom", fontsize=7, color=MUTED)


def server_side(ax: Axes) -> None:
    box(ax, 3.34, 0.86, 1.26, 1.31, SERVER_FILL, edge=RULE, lw=0.7)
    ax.text(3.41, 2.08, "SERVER", fontsize=BODY, weight="bold", color=INK, va="top")
    stage(ax, 3.4, 1.0, 0.66, 0.76, "layers\nL+1 … 12")
    stage(ax, 4.12, 1.0, 0.42, 0.76, "next-\ntoken\nloss")
    arrow(ax, (4.065, 1.38), (4.115, 1.38))


def attacker(ax: Axes, split: float) -> None:
    box(ax, 0.02, 0.02, 6.46, 0.66, ATTACKER_FILL, edge=ATTACKER, lw=0.7)
    ax.plot([split, split], [0.68, 0.78], color=ATTACKER, lw=0.9, ls=":", zorder=2)
    ax.text(0.09, 0.58, "ATTACKER", fontsize=BODY, weight="bold", color=ATTACKER, va="top")
    ax.text(0.86, 0.58, "passive observer at the split", fontsize=NOTE, color=INK, va="top")
    ax.text(0.09, 0.39, "holds", fontsize=NOTE, weight="bold", color=INK, va="top")
    holds = ("a copy of the client's layers: the public copy,\n"
             "or (stronger case) the client's trained copy\n32 public calibration documents")
    ax.text(0.55, 0.39, holds, fontsize=NOTE, color=INK, va="top", linespacing=1.15)
    ax.text(3.75, 0.39, "never gets", fontsize=NOTE, weight="bold", color=ACCENT, va="top")
    ax.text(4.5, 0.39, "the private text\nthe server's layers\nthe labels of the next-token loss", fontsize=NOTE, color=INK,
            va="top", linespacing=1.15)


def offloading_inset(ax: Axes) -> None:
    """Partial-offloading follow-up (Section 5): the server trains layers b .. b+s-1."""
    x0, y0, cell, b, s = 4.8, 1.12, 0.137, 2, 4
    box(ax, 4.68, 0.86, 1.8, 1.31, PAPER_FILL, edge=RULE, lw=0.7, ls=(0, (3, 2)))
    ax.text(4.75, 2.08, "Follow-up: partial offloading", fontsize=NOTE, weight="bold", color=INK, va="top")
    ax.text(4.75, 1.92, "the server trains only s\nconsecutive layers, starting\nat layer b (counted from 0)",
            fontsize=7, color=INK, va="top", linespacing=1.1)
    for i in range(12):
        remote = b <= i < b + s
        box(ax, x0 + i * cell, y0, cell * 0.86, 0.18, GREY if remote else PAPER_FILL, edge=GREY, lw=0.6,
            radius=0.015)
        ax.text(x0 + i * cell + cell * 0.43, y0 - 0.03, str(i), ha="center", va="top", fontsize=6, color=MUTED)
    left, right = x0 + b * cell, x0 + (b + s) * cell - cell * 0.14
    ax.plot([left, left, right, right], [y0 + 0.22, y0 + 0.26, y0 + 0.26, y0 + 0.22], color=INK, lw=0.7)
    ax.text((left + right) / 2, y0 + 0.28, f"example: b = {b}, s = {s}", ha="center", va="bottom",
            fontsize=7, color=INK)


if __name__ == "__main__":
    pipeline()
    architecture()
