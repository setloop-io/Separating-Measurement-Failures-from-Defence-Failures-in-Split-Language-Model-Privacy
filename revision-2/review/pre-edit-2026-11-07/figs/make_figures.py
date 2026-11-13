# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8", "numpy>=1.26"]
# ///
"""Generate Paper 9 revision 2 figures from saved experimental evidence.

Regenerate from any working directory:
    uv run --no-project path/to/revision-2/figs/make_figures.py

The inputs are researcher-generated, independently recomputed CSVs under
revision-2/data. The R1 campaign is cross-checked against the original CSV,
and P4 states are cross-checked against the original generated pilot report.
No synthetic data, new model runs, or statistical intervals on campaign counts
are used. A local provenance JSON records each input's hash and plotted values.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

import orx_figstyle as style
from orx_figstyle import PALETTE, figure, figure_grid, panel_labels, save, use_style

HERE = Path(__file__).resolve().parent
REVISION = HERE.parent
PACKAGE = REVISION.parent
INPUTS = {
    "r1_cells": REVISION / "data/r1_cells.csv",
    "p4_cells": REVISION / "data/p4_cells.csv",
    "original_r1_cells": PACKAGE / "evidence/trilemma-campaign/r1_cell_results.csv",
    "original_r1_seeds": PACKAGE / "evidence/trilemma-campaign/r1_seed_results.csv",
    "original_p4_report": PACKAGE / "reproduction/defense-redesign-pilot/outputs/runs/pilot-20260922-a/report.md",
}
WIDTH = 6.5  # The actual 11pt article uses letter paper and 1-inch margins.
# Update the style module's document-width registry, not the generated files.
style.TEXT = WIDTH


def rows(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def truth(value):
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false", "1", "0", "pass", "fail"}:
        raise ValueError(f"Unrecognised truth value: {value!r}")
    return normalized in {"true", "1", "pass"}


def configure():
    use_style()
    mpl.rcParams.update({
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "mathtext.default": "regular",
    })


def load_and_verify():
    r1, p4 = rows(INPUTS["r1_cells"]), rows(INPUTS["p4_cells"])
    originals = {r["cell"]: r for r in rows(INPUTS["original_r1_cells"])}
    assert len(r1) == len(originals) == 19
    assert len(p4) == 7
    assert all(r["verdict"] == "REJECTED" for r in r1)
    seed_rows = rows(INPUTS["original_r1_seeds"])
    for row in r1:
        original = originals[row["cell"]]
        utility_seeds = np.array([float(s["replacement_penalty"]) for s in seed_rows if s["cell"] == row["cell"]])
        assert len(utility_seeds) == int(row["n_seeds"]) == 4
        independently_checked_mean = utility_seeds.mean()
        half_width = 3.182446305284263 * utility_seeds.std(ddof=1) / math.sqrt(4)
        for field, calculated in [("utility_mean", independently_checked_mean),
                                  ("utility_lower", independently_checked_mean-half_width),
                                  ("utility_upper", independently_checked_mean+half_width)]:
            assert math.isclose(float(row[field]), calculated, abs_tol=1e-10)
        assert row["reasons"] == original["reasons"]
        for field in ("utility_mean", "utility_lower", "utility_upper"):
            assert math.isclose(float(row[field]), float(original[field]), abs_tol=1e-10)
        assert truth(row["utility_pass"]) == (
            float(row["utility_mean"]) <= 0.10 and float(row["utility_upper"]) <= 0.12
        )
    reason_counts = Counter(reason for row in r1 for reason in row["reasons"].split(";"))
    assert reason_counts == {"privacy_qualified_breach": 13, "utility_gate": 5, "gain_gate": 7}

    # The generated pilot report is the primary declaration of cell-level states.
    # Parse only the seven-row summary, rather than using rounded values as data.
    primary_p4 = {}
    for line in INPUTS["original_p4_report"].read_text().splitlines():
        fields = [field.strip() for field in line.split("|")[1:-1]]
        if len(fields) == 7 and fields[0] in {r["cell"] for r in p4}:
            primary_p4[fields[0]] = fields
    assert len(primary_p4) == 7
    for row in p4:
        report = primary_p4[row["cell"]]
        assert row["privacy_status"] == report[6]
        assert truth(row["utility_pass"]) == (float(report[2]) <= 0.10 and float(report[3]) <= 0.12)
        assert truth(row["gain_pass"]) == (float(report[4]) >= 0.01 and report[5] == "True")
        assert not truth(row["pilot_eligible"])
    return r1, p4, reason_counts


def campaign_gates(r1, p4, counts):
    fig, axes = figure_grid(
        1, 2, width=WIDTH, ratio=0.54,
        gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.15},
    )
    ax, matrix = axes
    labels = ["Qualified privacy breach", "Utility gate", "Gain gate"]
    values = [counts[k] for k in ("privacy_qualified_breach", "utility_gate", "gain_gate")]
    positions = np.arange(3)
    bars = ax.barh(positions, values, height=0.55,
                   color=PALETTE["blue"], edgecolor="black", linewidth=0.5)
    for bar, value in zip(bars, values):
        ax.text(value + 0.5, bar.get_y() + bar.get_height()/2,
                f"{value}/19", ha="left", va="center", fontsize=8)
    ax.set_yticks(positions, labels)
    ax.set_ylim(2.8, -0.8)
    ax.set_xlim(0, 19)
    ax.set_xticks([0, 5, 10, 15, 19])
    ax.set_xlabel("R1 cells (count; reasons overlap)")
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)

    order = ["d0-q8", "d1-s0.1-q8", "d1-s0.5-q8", "d1-s1.0-q8", "d3-q1", "d3-q2", "d3-q3"]
    by_cell = {r["cell"]: r for r in p4}
    styles = {
        "P": {"facecolor": "#D6EAF5", "hatch": "", "edgecolor": "#404040"},
        "F": {"facecolor": "#F8E0CF", "hatch": "///", "edgecolor": "#555555"},
        "U": {"facecolor": "#F3F3F3", "hatch": "...", "edgecolor": "#888888"},
    }
    outcomes = []
    for y, cell in enumerate(order):
        row = by_cell[cell]
        states = ["P" if truth(row["utility_pass"]) else "F",
                  "P" if truth(row["gain_pass"]) else "F",
                  {"failed": "F", "undetermined": "U"}[row["privacy_status"]]]
        outcomes.append({"cell": cell, "states": states})
        for x, state in enumerate(states):
            box = Rectangle((x - 0.47, y - 0.41), 0.94, 0.82,
                            linewidth=0.45, **styles[state])
            matrix.add_patch(box)
            matrix.text(x, y, state, ha="center", va="center", fontsize=8.5,
                        fontweight="bold", bbox={"facecolor": styles[state]["facecolor"],
                                                 "edgecolor": "none", "pad": 0.3})
    matrix.set_xlim(-0.5, 2.5)
    matrix.set_ylim(6.6, -0.7)
    matrix.set_xticks([0, 1, 2], ["Utility", "Gain", "Privacy"])
    matrix.set_yticks(range(7), order)
    matrix.set_xlabel("P4 gate outcome")
    matrix.tick_params(length=0)
    matrix.grid(visible=False)
    for spine in matrix.spines.values():
        spine.set_visible(False)
    panel_labels(axes, pad=4)
    # The categorical states are repeated as glyphs; hatching preserves meaning
    # in greyscale. One figure legend is outside both panels.
    handles = [Patch(label=label, **styles[key]) for key, label in
               [("P", "P: pass"), ("F", "F: failed"), ("U", "U: undetermined")]]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, -0.002),
               ncol=3, handlelength=1.6, columnspacing=1.5)
    fig.set_layout_engine("constrained", rect=(0, 0.09, 1, 0.92))
    save(fig, str(HERE / "campaign-gates"))
    return {"r1_reason_counts": dict(counts), "r1_cells": len(r1), "p4_outcomes": outcomes,
            "uncertainty": "None: complete enumeration of saved exploratory campaign cells."}


def utility_boundary(r1):
    by_cell = {r["cell"]: r for r in r1}
    selected = [by_cell[cell] for cell in ("base", "depth-4", "span-6")]
    fig, ax = figure(width=WIDTH, ratio=0.435)
    # These references are separate clauses of the same gate: the mean is
    # compared with 0.10, and the upper endpoint is compared with 0.12.
    mean_line = ax.axvline(0.10, color="#595959", linewidth=1.0, linestyle="--", zorder=1)
    upper_line = ax.axvline(0.12, color="#595959", linewidth=1.0, linestyle=":", zorder=1)
    formats = [("#4D4D4D", "o"), (PALETTE["blue"], "D"), (PALETTE["orange"], "s")]
    plotted = []
    for y, (row, (color, marker)) in enumerate(zip(selected, formats)):
        mean, lo, hi = (float(row[f]) for f in ("utility_mean", "utility_lower", "utility_upper"))
        assert int(row["n_seeds"]) == 4
        assert truth(row["utility_pass"])
        ax.errorbar(mean, y, xerr=np.array([[mean-lo], [hi-mean]]),
                    fmt=marker, markersize=5, markerfacecolor=color, markeredgecolor="black",
                    markeredgewidth=0.5, color=color, elinewidth=1.5, capsize=3, capthick=0.8,
                    zorder=3)
        # Exact means and upper limits are separate, readable columns. The text
        # is kept outside the interval region to avoid hiding the near-boundary
        # span-6 endpoints, and the x axis is extended to reserve this space.
        ax.text(0.149, y, f"{mean:.6f}", ha="right", va="center", fontsize=8)
        ax.text(0.184, y, f"{hi:.6f}", ha="right", va="center", fontsize=8)
        ax.text(0.195, y, "P", ha="center", va="center", fontsize=8.5, fontweight="bold")
        plotted.append({"cell": row["cell"], "n_seeds": 4, "mean": mean,
                        "lower_two_sided_95_t": lo, "upper_two_sided_95_t": hi,
                        "utility_pass": True})
    ax.text(0.149, -0.62, "Mean", ha="right", va="center", fontsize=8)
    ax.text(0.184, -0.62, "Upper 95%", ha="right", va="center", fontsize=8)
    ax.text(0.195, -0.62, "Gate", ha="center", va="center", fontsize=8)
    ax.set_yticks(range(3), ["Base", "Depth 4", "Span 6"])
    ax.set_ylim(2.65, -0.9)
    ax.set_xlim(-0.025, 0.208)
    ax.set_xticks([-0.02, 0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12],
                  ["−0.02", "0", "0.02", "0.04", "0.06", "0.08", "0.10", "0.12"])
    ax.set_xlabel("Held-out utility penalty (nats)")
    ax.spines["bottom"].set_bounds(-0.025, 0.125)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    # Only grid the quantitative plotting region; don't turn the exact-value
    # columns into numeric plot area.
    ax.grid(axis="x", visible=False)
    for x in [-0.02, 0, 0.02, 0.04, 0.06, 0.08]:
        ax.axvline(x, ymin=0.12, ymax=0.78, color="#E2E2E2", linewidth=0.45, zorder=0)
    ax.plot([0.129, 0.129], [-0.42, 2.35], color="#C4C4C4", linewidth=0.5)
    ax.legend([mean_line, upper_line], ["Mean gate: ≤ 0.10", "Upper-limit gate: ≤ 0.12"],
              ncol=2, loc="upper center", bbox_to_anchor=(0.48, -0.21),
              handlelength=2.2, columnspacing=2)
    save(fig, str(HERE / "r1-utility-boundary"))
    return {"cells": plotted, "interval_definition": "two-sided 95% t-interval over four saved seeds; df=3",
            "mean_gate_nats": 0.10, "upper_endpoint_gate_nats": 0.12,
            "meaning": "P is utility pass only; these three cells remain rejected for privacy and/or gain."}


def main():
    configure()
    r1, p4, counts = load_and_verify()
    campaign = campaign_gates(r1, p4, counts)
    utility = utility_boundary(r1)
    manifest = {
        "inputs": [{"path": str(path.relative_to(PACKAGE)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for path in INPUTS.values()],
        "figures": {"campaign-gates": campaign, "r1-utility-boundary": utility},
        "rendering": {"width_inches": WIDTH, "minimum_font_points": 8,
                      "matplotlib_version": mpl.__version__, "numpy_version": np.__version__},
    }
    (HERE / "figure-provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Source verification and figure generation complete.")


if __name__ == "__main__":
    main()
