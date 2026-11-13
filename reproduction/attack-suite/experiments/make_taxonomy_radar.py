#!/usr/bin/env python
"""Competence radar for Paper B: registered attack modules per family and
the subset with real-model or matched-reproduction evidence.

Module counts are parsed from the Table 2 longtable in
paper/paper_b/main.tex, so the figure always matches the printed
taxonomy (64 modules across 16 families). The evidence classification
follows the table caption and the four-kinds inventory paragraph in the
same section: families outside the toy-world list are real-model or
matched-reproduction evaluations; deployment/eval is an orchestration
tier.

Usage: .venv/bin/python experiments/make_taxonomy_radar.py
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "paper" / "paper_b" / "main.tex"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

ROW_PATTERNS = [
    ("Trained inversion", r"^Trained inversion &"),
    ("Matching attacks", r"^Matching attacks &"),
    ("Key/budget solves", r"^Key / budget solves &"),
    ("Known-plaintext", r"^Known-plaintext &"),
    ("Gradient/output", r"^Gradient / output &"),
    ("Membership", r"^Membership &"),
    ("Structure/blind", r"^Structure / blind &"),
    ("Active/malicious", r"^Active / malicious &"),
    ("Model-IP theft", r"^Model-IP theft &"),
    ("Hardware/link", r"^Hardware / link &"),
    ("Reasoning theft", r"^Reasoning theft &"),
    ("Integrity/availability", r"^Integrity / availability &"),
    ("Tenant isolation", r"^Tenant isolation / composition &"),
    ("Application profile", r"^Application profile \(GenAI\) &"),
    ("Agentic profile", r"^Agentic profile \(OWASP ASI\) &"),
    ("Deployment/eval", r"^Deployment / eval &"),
]
TOY_WORLD = {"Integrity/availability", "Tenant isolation",
             "Application profile", "Agentic profile"}
ORCHESTRATION = {"Deployment/eval"}

lines_all = TEX.read_text().split("\n")
start = next(i for i, l in enumerate(lines_all)
             if "\\label{tab:taxonomy}" in l)
end = next(i for i in range(start, len(lines_all))
           if "\\end{longtable}" in lines_all[i])
lines = lines_all[start:end]
starts = [(next(i for i, l in enumerate(lines) if re.match(pat, l)), name)
          for name, pat in ROW_PATTERNS]
families, counts = [], []
for (a, name), (b, _) in zip(starts, starts[1:] + [(len(lines), "END")]):
    seg = "\n".join(lines[a:b])
    mods = set(re.findall(r"\\attack\{([^}]*)\}", seg))
    families.append(name)
    counts.append(len(mods))
assert sum(counts) == 64, f"expected 64 modules, counted {sum(counts)}"

measured = [
    0 if (f in TOY_WORLD or f in ORCHESTRATION) else c
    for f, c in zip(families, counts)
]

plt.rcParams.update({"font.size": 7, "figure.dpi": 200, "savefig.bbox": "tight"})
labels = [f.replace("/", " / ") for f in families]
N = len(families)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
c_loop = counts + counts[:1]
m_loop = measured + measured[:1]
a_loop = angles + angles[:1]

fig, ax = plt.subplots(figsize=(4.6, 4.6), subplot_kw={"polar": True})
ax.plot(a_loop, c_loop, color="#1f5fa8", lw=1.6, zorder=3,
        label=f"registered modules ({sum(counts)})")
ax.fill(a_loop, c_loop, color="#1f5fa8", alpha=0.15, zorder=2)
ax.plot(a_loop, m_loop, color="#b03030", lw=1.4, ls="--", zorder=4,
        label="with real-model / matched evidence")
ax.set_xticks(angles)
ax.set_xticklabels(labels, fontsize=6.2)
ax.set_rlabel_position(90)
rmax = max(counts) + 2
ax.set_ylim(0, rmax)
ticks = list(range(2, rmax, 2))
ax.set_yticks(ticks)
ax.set_yticklabels([str(t) for t in ticks], fontsize=5.5)
ax.grid(lw=0.3, alpha=0.6)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), frameon=False, ncols=1)
fig.savefig(OUT / "fig_competence_radar.pdf")
plt.close(fig)
print("wrote fig_competence_radar.pdf;", N, "families,", sum(counts), "modules,",
      sum(1 for m in measured if m), "families with real-model evidence")
