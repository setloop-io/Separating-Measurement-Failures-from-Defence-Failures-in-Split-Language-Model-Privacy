#!/usr/bin/env python
"""Regenerate the v3 campaign figures for Paper A.

Every plotted value is computed from the committed shard evidence
results/validation-heldout-v3-r3-shard*/validation.json via
experiments/v3_campaign_data.py. Intervals are the paper's 95% crossed
bootstrap (victim seed x document, attack seeds averaged first, protocol
replicate count and seed). Figures are sized at their final width in the
paper (6.5 in text width), so font sizes are print sizes.

Usage: python experiments/make_v3_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from v3_campaign_data import (Campaign, Estimate, level, load_campaign,  # noqa: E402
                              paired_difference, perplexity_increase, point)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "paper_a" / "figures"
TEXT_WIDTH = 6.5
TOKEN, SEQ = "token_accuracy", "sequence_exact"
JOINT, GREY, ORANGE, INK = "#3B6EA5", "#8C8C8C", "#C0793A", "#333333"
BASE_ARMS = ["forward_k8", "forward_k16", "forward_k32",
             "sipit_steps32", "sipit_steps64", "sipit_steps128"]
DEFENCES = {"replica-median": ("Replica-median", JOINT, "o"),
            "secret-mixup": ("Secret mixup", ORANGE, "s")}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#555555", "axes.linewidth": 0.6,
    "xtick.color": "#555555", "ytick.color": INK, "xtick.major.width": 0.6,
    "ytick.major.size": 0, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "figure.constrained_layout.use": True,
})


def dot(ax, y: float, est: Estimate, colour: str, marker: str = "o", size: float = 6.0,
        hollow: bool = False) -> None:
    ax.errorbar(est.value, y, xerr=[[est.value - est.lower], [est.upper - est.value]],
                fmt=marker, ms=size, color=colour, mfc="white" if hollow else colour,
                mew=1.2, elinewidth=1.4, capsize=0, zorder=3)


def xgrid(ax) -> None:
    ax.grid(axis="x", lw=0.4, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)


def teaser(camp: Campaign) -> list[tuple[str, float]]:
    """Primary contrast: activation-only vs joint, depth 6, public head, k=32."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 1.55))
    rows = [("Activation only", "forward_k32", GREY), ("Activations + gradients", "joint_true_k32", JOINT)]
    report = []
    panels = ((axes[0], TOKEN, "Token recovery (%)", (92.5, 99.0)),
              (axes[1], SEQ, "Whole-document recovery (%)", (0, 50)))
    for ax, field, name, lim in panels:
        ests = [level(camp, [arm], 6, ["public"], field) for _, arm, _ in rows]
        for y, ((label, _, colour), est) in enumerate(zip(rows, ests)):
            dot(ax, 1 - y, est, colour)
            ax.annotate(f"{est.value:.2f}", (est.upper, 1 - y), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=9, color=INK)
            report.append((f"{label} {field}", est.value))
        ax.set(xlim=lim, ylim=(-0.6, 1.6), xlabel=name)
        xgrid(ax)
    diff = paired_difference(camp, "joint_true_k32", "forward_k32", 6, "public", TOKEN)
    fold = report[3][1] / report[2][1]
    axes[0].text(0.03, 0.5, f"+{diff.value:.2f} points\n95% CI [{diff.lower:.2f}, {diff.upper:.2f}]",
                 transform=axes[0].transAxes, va="center", fontsize=8.5, color=INK)
    axes[1].text(0.97, 0.5, f"{fold:.2f}$\\times$", transform=axes[1].transAxes,
                 ha="right", va="center", fontsize=11, color=JOINT, weight="bold")
    axes[0].set_yticks([1, 0], [r[0] for r in rows])
    axes[1].set_yticks([1, 0], [])
    fig.savefig(OUT / "fig_teaser.pdf")
    plt.close(fig)
    return report + [("paired token difference", diff.value), ("difference lower", diff.lower),
                     ("difference upper", diff.upper), ("whole-document fold", fold)]


CONDITIONS = [("joint_true_k32", "True gradient"), ("joint_noise_0.1_k32", r"Noise $\sigma$ = 0.1"),
              ("joint_bits_8_k32", "8-bit quantised"), ("joint_noise_1.0_k32", r"Noise $\sigma$ = 1.0"),
              ("joint_bits_4_k32", "4-bit quantised"), ("joint_shuffled_k32", "Shuffled (control)"),
              ("joint_random_k32", "Random candidates (control)")]


def gradient_conditions(camp: Campaign) -> None:
    """Joint attacker under each observed-gradient condition, depth 6, public head, k=32."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.3))
    n = len(CONDITIONS)
    for ax, field, name in ((axes[0], TOKEN, "Token recovery (%)"),
                            (axes[1], SEQ, "Whole-document recovery (%)")):
        base = level(camp, ["forward_k32"], 6, ["public"], field)
        ax.axvspan(base.lower, base.upper, color=GREY, alpha=0.18, lw=0, zorder=1)
        ax.axvline(base.value, color=GREY, lw=1.0, ls="--", zorder=2)
        ax.text(base.value, n - 0.35, " activation only", color="#555555", fontsize=8.5,
                ha="right" if field == TOKEN else "left", va="bottom")
        for i, (arm, _) in enumerate(CONDITIONS):
            dot(ax, n - 1 - i, level(camp, [arm], 6, ["public"], field),
                GREY if "control" in CONDITIONS[i][1] else JOINT)
        ax.set(ylim=(-0.6, n + 0.45), xlabel=name)
        xgrid(ax)
    axes[0].set_yticks(range(n - 1, -1, -1), [lbl for _, lbl in CONDITIONS])
    axes[1].set_yticks(range(n - 1, -1, -1), [])
    fig.savefig(OUT / "fig_gradient_conditions.pdf")
    plt.close(fig)


def vdot(ax, x: float, est: Estimate, colour: str, marker: str, hollow: bool = False) -> None:
    ax.errorbar(x, est.value, yerr=[[est.value - est.lower], [est.upper - est.value]],
                fmt=marker, ms=6.5, color=colour, mfc="white" if hollow else colour,
                mew=1.2, elinewidth=1.4, capsize=0, zorder=3)


def privacy_utility(camp: Campaign) -> None:
    """Residual leakage of the six defended attacks vs held-out perplexity cost, per depth."""
    cost = perplexity_increase(camp)
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.5))
    both = ["public", "exact"]
    for ax, field, name in ((axes[0], TOKEN, "Token recovery (%)"),
                            (axes[1], SEQ, "Whole-document recovery (%)")):
        for depth in (3, 6):
            base = level(camp, BASE_ARMS, depth, both, field)
            vdot(ax, 0.0, base, GREY, "D", hollow=True)
            ax.annotate(f"L{depth}", (0.0, base.value), xytext=(-7, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=8.5, color=INK)
            for defence, (_, colour, marker) in DEFENCES.items():
                est = level(camp, [f"{a}@{defence}" for a in BASE_ARMS], depth, both, field)
                x = cost[defence, depth]
                vdot(ax, x, est, colour, marker)
                if defence == "secret-mixup":
                    ax.annotate(f"L{depth}", (x, est.value), xytext=(7 if depth == 3 else -7, 0),
                                textcoords="offset points", ha="left" if depth == 3 else "right",
                                va="center", fontsize=8.5, color=INK)
        ax.set(xlabel="Held-out perplexity increase (%)", ylabel=name, xlim=(-6, 38))
        ax.grid(lw=0.4, color="#DDDDDD", zorder=0)
        ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], marker="D", ls="", ms=6, mfc="white", mec=GREY, mew=1.2, label="Undefended")]
    handles += [plt.Line2D([], [], marker=m, ls="", ms=6, color=c, label=lbl) for lbl, c, m in DEFENCES.values()]
    fig.legend(handles=handles, loc="outside upper center", ncols=3,
               frameon=False, fontsize=9, handletextpad=0.3, columnspacing=1.6)
    fig.savefig(OUT / "fig_privacy_utility.pdf")
    plt.close(fig)


GRADIENT_NAMES = {"true": "true gradient", "noise_0.1": r"noise $\sigma$ = 0.1", "bits_8": "8-bit gradient",
                  "noise_1.0": r"noise $\sigma$ = 1.0", "bits_4": "4-bit gradient",
                  "shuffled": "shuffled gradient", "random": "random candidates"}
DEFENCE_NAMES = {"replica-median": "replica-median", "secret-mixup": "secret mixup"}


def arm_label(arm: str) -> str:
    """Plain name of a campaign arm identifier, following the paper's wording."""
    base, _, defence = arm.partition("@")
    suffix = f" + {DEFENCE_NAMES[defence]}" if defence else ""
    if base == "backward":
        return "Backward-only heuristic" + suffix
    if base.startswith("sipit_steps"):
        return f"SipIt, {base.removeprefix('sipit_steps')} steps" + suffix
    stem, _, k = base.rpartition("_k")
    if stem == "forward":
        return f"Activation-only, $k$ = {k}" + suffix
    condition = stem.removeprefix("joint_")
    return f"Joint, {GRADIENT_NAMES[condition]}, $k$ = {k}" + suffix


def heatmap_rows() -> list[str]:
    undefended = ["backward"] + [f"forward_k{k}" for k in (8, 16, 32)] + [f"sipit_steps{n}" for n in (32, 64, 128)]
    undefended += [f"joint_{c}_k{k}" for c in GRADIENT_NAMES for k in (8, 16, 32)]
    return undefended + [f"{a}@{d}" for d in DEFENCE_NAMES for a in BASE_ARMS]


COPY_LABELS = {"public": "public copy", "exact": "trained copy"}


def arm_heatmap(camp: Campaign) -> list[tuple[str, str]]:
    """All 40 arms x (depth, head) token recovery; appendix figure."""
    arms = heatmap_rows()
    if set(arms) != {k[5] for k in camp.cells if k[4] == "training"}:
        raise ValueError("heatmap rows differ from the campaign arms")
    conds = [(3, "public"), (3, "exact"), (6, "public"), (6, "exact")]
    mat = np.array([[point(camp, a, d, k) for d, k in conds] for a in arms])
    cmap = matplotlib.colors.ListedColormap(plt.get_cmap("Blues")(np.linspace(0.05, 0.8, 256)))
    cmap.set_under("#EFEFEF")
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH - 0.3, 7.3))
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=75, vmax=100)
    ax.set_xticks(range(len(conds)), [f"L = {d}\n{COPY_LABELS[k]}" for d, k in conds], fontsize=8.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(arms)), [arm_label(a) for a in arms], fontsize=8)
    ax.tick_params(length=0)
    for i, j in np.ndindex(mat.shape):
        dark = np.dot(im.cmap(im.norm(mat[i, j]))[:3], [0.2126, 0.7152, 0.0722]) < 0.55
        ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=8,
                color="white" if dark else INK)
    ax.axhline(len(arms) - 12.5, color="white", lw=3)
    for side in ax.spines.values():
        side.set_visible(False)
    bar = fig.colorbar(im, ax=ax, shrink=0.45, extend="min", location="bottom", aspect=30, pad=0.01)
    bar.set_label("Token recovery (%)", fontsize=8.5)
    bar.ax.tick_params(labelsize=8)
    fig.savefig(OUT / "fig_arm_heatmap.pdf")
    plt.close(fig)
    return [(a, arm_label(a)) for a in arms]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    camp = load_campaign()
    for name, value in teaser(camp):
        print(f"teaser  {name:40s} {value:8.4f}")
    gradient_conditions(camp)
    privacy_utility(camp)
    for key, value in sorted(perplexity_increase(camp).items()):
        print(f"utility {key[0]:15s} L{key[1]}  +{value:.2f}% perplexity")
    for arm, label in arm_heatmap(camp):
        print(f"arm     {arm:32s} {label}")
    print("wrote", *sorted(p.name for p in OUT.glob("fig_*.pdf")), sep="\n  ")


if __name__ == "__main__":
    main()
