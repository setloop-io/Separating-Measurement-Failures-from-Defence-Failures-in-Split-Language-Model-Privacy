#!/usr/bin/env python
"""Figure 1 of Paper A: what leaves the client, what the attacker reads back, and how often.

The example document, both reconstructions and every number are computed from
the committed campaign shards (experiments/paper_a_example.py); intervals are
the paper's crossed bootstrap (experiments/v3_campaign_data.py). The only
display change is WikiText's hyphen markup " @-@ ", drawn as "-".

Usage:
  uv run --no-project --with matplotlib --with numpy --with tokenizers \
      python experiments/make_paper_a_hero.py
"""

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle, FancyArrowPatch
from tokenizers import Tokenizer

from paper_a_diagram_style import (ACCENT, ACCENT_FILL, ATTACKER, ATTACKER_FILL, CLIENT_FILL, GREY,
                                   INK, JOINT, MUTED, PAPER_FILL, ROOT, RULE, SERVER_FILL,
                                   TEXT_WIDTH, arrow, box, canvas, pill, save, text_width, title)
from paper_a_example import TOKENS_PER_DOCUMENT, Aggregate, Example, load_example
from v3_campaign_data import Estimate, level, load_campaign

NOTE = 7.5
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
HEIGHT = 2.9
TOP = HEIGHT - 0.11
BODY = 8
LINE = 0.14
WRONG_PAD = 0.02
WIKITEXT_HYPHEN = [" @", "-", "@"]
PAPER_VALUES = {"act_token": 94.20, "joint_token": 97.38, "act_exact": 13.71, "joint_exact": 37.77}
PAPER_FOLD = 2.75


@dataclass(frozen=True)
class Word:
    text: str
    gap: bool
    wrong: bool


def visible(s: str) -> str:
    """Replace control characters with their Unicode control pictures."""
    return "".join(chr(0x2400 + ord(c)) if ord(c) < 32 else "␡" if ord(c) == 127 else c for c in s)


def display_words(tok: Tokenizer, pred: list[int], truth: list[int]) -> list[Word]:
    """Tokens as drawn: wrong tokens flagged, a spurious leading space shown as an open box."""
    words, attach, i = [], False, 0
    while i < len(pred):
        if [tok.decode([t]) for t in truth[i:i + 3]] == WIKITEXT_HYPHEN and pred[i:i + 3] == truth[i:i + 3]:
            words.append(Word("-", False, False))
            attach, i = True, i + 3
            continue
        s, t = tok.decode([pred[i]]), tok.decode([truth[i]])
        wrong, gap = pred[i] != truth[i], s.startswith(" ")
        body = visible(s[1:] if gap else s)
        if wrong and gap and not t.startswith(" "):
            body, gap = "␣" + body, i > 0
        words.append(Word(body, gap and not attach, wrong))
        attach, i = False, i + 1
    return words


def groups(words: list[Word]) -> list[list[int]]:
    """Indices of words joined without a gap; a line may only break between groups."""
    out: list[list[int]] = []
    for k, w in enumerate(words):
        if w.gap or not out:
            out.append([])
        out[-1].append(k)
    return out


def flow(ax: Axes, words: list[Word], x0: float, y: float, width: float, size: float = BODY,
         colour: str = INK) -> float:
    """Draw words left to right from baseline y, wrapping between groups; return the last baseline.

    Each word is placed at the measured width of the line before it, so spacing
    and kerning match a single run of text.
    """
    line, shift = "", 0.0
    for group in groups(words):
        sep = " " if line else ""
        if line and shift + text_width(ax, line + sep + "".join(words[k].text for k in group), size) > width:
            line, sep, shift, y = "", "", 0.0, y - LINE
        for k in group:
            w = words[k]
            if w.wrong and line:
                shift += WRONG_PAD
            x = x0 + shift + text_width(ax, line + sep, size)
            if w.wrong:
                box(ax, x - 0.012, y - 0.03, text_width(ax, w.text, size) + 0.024, 0.13, ACCENT_FILL,
                    edge=ACCENT, lw=0.6, radius=0.02, z=2)
                shift += WRONG_PAD
            ax.text(x, y, w.text, fontsize=size, color=ACCENT if w.wrong else colour, va="baseline", zorder=3)
            line, sep = line + sep + w.text, ""
    return y


def activation_grid(ax: Axes, x: float, y: float, cols: int = 7, rows: int = 4, cell: float = 0.052) -> None:
    """Schematic activation block: a fixed pattern of shades, not data."""
    shades = ["#C9D7E8", "#9DB6D3", "#6F93BD", "#E3EAF3"]
    for r in range(rows):
        for c in range(cols):
            box(ax, x + c * cell, y + r * cell, cell * 0.9, cell * 0.9, shades[(3 * r + 2 * c + r * c) % 4],
                edge="none", radius=0.006, z=3)


def eye(ax: Axes, x: float, y: float, colour: str) -> None:
    """Small eye icon centred on (x, y)."""
    t = np.linspace(0, np.pi, 40)
    ax.plot(x + 0.09 * np.cos(t), y + 0.045 * np.sin(t), color=colour, lw=0.9, zorder=6)
    ax.plot(x + 0.09 * np.cos(t), y - 0.045 * np.sin(t), color=colour, lw=0.9, zorder=6)
    ax.add_patch(Circle((x, y), 0.026, color=colour, zorder=6))


def panel_client(ax: Axes, tok: Tokenizer, ex: Example) -> None:
    x0, w, split, mid = 0.02, 1.98, 1.30, 1.23
    title(ax, x0, TOP, "a", "What leaves the client")
    box(ax, x0, 1.74, w, 0.94, PAPER_FILL, edge=RULE)
    ax.text(x0 + 0.07, 2.57, "private text", fontsize=NOTE, color=MUTED, va="baseline", style="italic")
    flow(ax, display_words(tok, ex.truth, ex.truth), x0 + 0.07, 2.42, w - 0.14)
    box(ax, x0, mid - 0.2, 0.62, 0.4, CLIENT_FILL, edge=JOINT, lw=0.6)
    ax.text(x0 + 0.31, mid, "client's\nlayers 1–6", ha="center", va="center", fontsize=BODY, color=INK,
            linespacing=1.1, zorder=4)
    box(ax, 1.42, mid - 0.2, 0.58, 0.4, SERVER_FILL, edge=RULE, lw=0.6)
    ax.text(1.71, mid, "server's\nlayers 7–12", ha="center", va="center", fontsize=BODY, color=INK,
            linespacing=1.1, zorder=4)
    arrow(ax, (x0 + 0.31, 1.74), (x0 + 0.31, mid + 0.2))
    activation_grid(ax, 0.77, mid - 0.105)
    ax.text(0.9, 1.58, "activations", ha="center", va="baseline", fontsize=BODY, color=INK, weight="bold")
    ax.text(0.9, 1.47, "768 numbers/token", ha="center", va="baseline", fontsize=6.5, color=MUTED)
    arrow(ax, (x0 + 0.62, mid), (0.76, mid))
    arrow(ax, (1.13, mid), (1.41, mid))
    ax.plot([split, split], [0.72, 1.66], color=MUTED, lw=0.9, ls=(0, (3, 2)), zorder=2)
    ax.text(split + 0.04, 1.64, "split", fontsize=NOTE, color=MUTED, va="top")
    ax.add_patch(FancyArrowPatch((1.71, mid - 0.2), (x0 + 0.31, mid - 0.2), color=JOINT, lw=1.1, zorder=4,
                                 arrowstyle="-|>,head_length=4,head_width=2.2", shrinkA=0, shrinkB=0,
                                 connectionstyle="arc,angleA=-90,angleB=-90,armA=7,armB=7,rad=4"))
    ax.text(0.95, 0.88, "gradients (training only)", ha="center", va="top", fontsize=BODY, color=JOINT)
    box(ax, x0, 0.02, w, 0.7, ATTACKER_FILL, edge=ATTACKER, lw=0.6)
    eye(ax, x0 + 0.16, 0.6, ATTACKER)
    ax.text(x0 + 0.32, 0.6, "Attacker at the split", fontsize=BODY, weight="bold", color=ATTACKER,
            va="center")
    lines = "passive observer\nholds a public copy of the\nclient's layers\nnever sees the text"
    ax.text(x0 + 0.08, 0.5, lines, fontsize=NOTE, color=INK, va="top", linespacing=1.1)


def recon_block(ax: Axes, tok: Tokenizer, x0: float, y: float, w: float, label: str, colour: str,
                pred: list[int], truth: list[int]) -> float:
    """Pill, right-aligned score tag and the reconstruction; return the next free baseline."""
    right = sum(p == t for p, t in zip(pred, truth))
    tag = f"{right}/{len(truth)} tokens · {'exact' if pred == truth else 'not exact'}"
    pill(ax, x0 + 0.03, y, label, colour, size=7, weight="normal")
    ax.text(x0 + w, y, tag, ha="right", va="center", fontsize=7, color=MUTED)
    last = flow(ax, display_words(tok, pred, truth), x0, y - 0.19, w)
    return last - 0.25


def panel_readback(ax: Axes, tok: Tokenizer, ex: Example) -> None:
    x0, w = 2.16, 2.5
    title(ax, x0, TOP, "b", "What the attacker reads back", "one document, one run")
    ax.text(x0, 2.46, "true text", fontsize=NOTE, color=MUTED, va="baseline", style="italic")
    y = flow(ax, display_words(tok, ex.truth, ex.truth), x0, 2.31, w, colour=MUTED) - 0.25
    y = recon_block(ax, tok, x0, y, w, "activation-only attacker", GREY, ex.act, ex.truth)
    recon_block(ax, tok, x0, y, w, "joint attacker", JOINT, ex.joint, ex.truth)
    box(ax, x0, 0.035, 0.1, 0.1, ACCENT_FILL, edge=ACCENT, lw=0.6, radius=0.02)
    ax.text(x0 + 0.15, 0.085, "wrong token      ␣ = extra leading space", fontsize=NOTE, color=MUTED,
            va="center")


def dot_row(ax: Axes, act: Estimate, joint: Estimate, lim: tuple[float, float], ticks: list[float],
            heading: str) -> None:
    ax.axvspan(act.lower, act.upper, color=GREY, alpha=0.22, lw=0, zorder=1)
    for est, colour in ((act, GREY), (joint, JOINT)):
        ax.plot([est.lower, est.upper], [0, 0], color=colour, lw=1.6, zorder=3, solid_capstyle="butt")
        ax.plot(est.value, 0, "o", ms=5.5, color=colour, zorder=4)
        ax.text(est.value, 0.55, f"{est.value:.2f}", ha="center", va="bottom", fontsize=10,
                weight="bold", color=colour)
    ax.annotate("", (joint.lower, 0), (act.upper, 0),
                arrowprops={"arrowstyle": "-|>,head_length=0.35,head_width=0.2", "color": INK, "lw": 0.8,
                            "shrinkA": 3, "shrinkB": 3})
    ax.set(xlim=lim, ylim=(-0.6, 1.6), yticks=[], xticks=ticks)
    ax.set_title(heading, fontsize=NOTE, color=INK, loc="left", pad=2)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", length=2, pad=1.5)


def panel_aggregate(fig, ax: Axes, token: tuple[Estimate, Estimate], exact: tuple[Estimate, Estimate],
                    fold: float) -> None:
    x0, w = 4.84, 1.62
    title(ax, x0, TOP, "c", "Across all runs", "256 documents × 20 runs")
    rect = lambda y, h: ((x0 + 0.04) / TEXT_WIDTH, y / HEIGHT, (w - 0.14) / TEXT_WIDTH, h / HEIGHT)  # noqa: E731
    dot_row(fig.add_axes(rect(1.68, 0.62)), *token, (92.5, 99.0), [93, 95, 97, 99], "token recovery (%)")
    dot_row(fig.add_axes(rect(0.72, 0.62)), *exact, (0, 50), [0, 25, 50], "whole-document recovery (%)")
    ax.text(x0, 0.2, f"{fold:.2f}×", ha="left", va="baseline", fontsize=13, weight="bold", color=JOINT)
    ax.text(x0 + text_width(ax, f"{fold:.2f}× ", 13, "bold"), 0.2, "more documents\nrecovered exactly",
            ha="left", va="baseline", fontsize=NOTE, color=MUTED, linespacing=1.1)


def check_numbers(agg: Aggregate, token: tuple[Estimate, Estimate], exact: tuple[Estimate, Estimate]) -> float:
    """Assert the plotted values equal the prediction-file means and round to the paper's values."""
    got = {"act_token": token[0].value, "joint_token": token[1].value,
           "act_exact": exact[0].value, "joint_exact": exact[1].value}
    for name, value in got.items():
        if abs(value - getattr(agg, name)) > 1e-9:
            raise ValueError(f"{name}: bootstrap level {value} differs from prediction mean {getattr(agg, name)}")
        if round(value, 2) != PAPER_VALUES[name]:
            raise ValueError(f"{name}: {value} does not round to the paper's {PAPER_VALUES[name]}")
    fold = got["joint_exact"] / got["act_exact"]
    if round(fold, 2) != PAPER_FOLD:
        raise ValueError(f"fold {fold} does not round to {PAPER_FOLD}")
    return fold


def main() -> None:
    ex, agg = load_example(ROOT / "results")
    camp = load_campaign()
    token = tuple(level(camp, [arm], 6, ["public"], "token_accuracy") for arm in ("forward_k32", "joint_true_k32"))
    exact = tuple(level(camp, [arm], 6, ["public"], "sequence_exact") for arm in ("forward_k32", "joint_true_k32"))
    fold = check_numbers(agg, token, exact)
    tok = Tokenizer.from_pretrained("openai-community/gpt2", revision=GPT2_REVISION)
    fig, ax = canvas(TEXT_WIDTH, HEIGHT)
    panel_client(ax, tok, ex)
    panel_readback(ax, tok, ex)
    panel_aggregate(fig, ax, token, exact, fold)
    path = save(fig, "fig_hero.pdf")
    report(ex, agg, token, exact, fold, tok)
    print("wrote", path)


def report(ex: Example, agg: Aggregate, token: tuple, exact: tuple, fold: float, tok: Tokenizer) -> None:
    print(f"runs per attacker: {agg.runs_per_attacker}")
    print(f"token recovery    act {agg.act_token:.4f}  joint {agg.joint_token:.4f}")
    print(f"whole-document    act {agg.act_exact:.4f}  joint {agg.joint_exact:.4f}  fold {fold:.4f}")
    for name, (a, j) in (("token", token), ("whole-document", exact)):
        print(f"95% CI {name:15s} act [{a.lower:.2f}, {a.upper:.2f}]  joint [{j.lower:.2f}, {j.upper:.2f}]")
    print(f"example: doc #{ex.doc_index} {ex.doc_id}, {TOKENS_PER_DOCUMENT} tokens")
    for f in (ex.act_file, ex.joint_file):
        print(f"sha256 {f.path.relative_to(ROOT)}\n  recorded {f.recorded_sha256}\n  actual   {f.actual_sha256}")
    for name, pred in (("act", ex.act), ("joint", ex.joint)):
        wrong = [(i, tok.decode([t]), tok.decode([p])) for i, (p, t) in enumerate(zip(pred, ex.truth)) if p != t]
        print(f"{name} wrong (position, true, predicted): {wrong}")


if __name__ == "__main__":
    main()
