# Evidence audit update — 17 September 2026

This supplement qualifies the historical manuscript/evidence snapshot. It does not establish completion of Phase 2.

- External P0–P5 records were recovered from the five-node search, but their owning project explicitly retired them from active scientific claims on 10 September because execution-source provenance was missing. Byte verification does not restore claim admissibility. The replacement R1 confirmation is incomplete.
- The original P6 grid completed 192 cells but used pilot-corpus VMI-only evaluation. It does not satisfy the original full protocol or establish the Mode-1 P5 claim. The corrected prospective grid is running under source commit d5ec03eaa561241bf71564bdf08e7fb1501dde78; its results are not included here.
- BiSR fidelity is NOT_ESTABLISHED. The five-node search found only the raw-article substitute, not the original SensiReplaced dataset. The matched comparison remains blocked.

The original MANIFEST.sha256 covers the baseline snapshot. MANIFEST-AUDIT-20260917.sha256 separately covers this supplement. Repositories remain private. No source evidence was deleted. See audit-20260917/ for exact scope, hashes, source paths, arithmetic corrections, and reproduction scripts.

Extract archives at the repository root; they preserve project-relative paths. The external audit runs with `python experiments/audit_external_p2_20260917.py`. The BiSR audit runs with `python experiments/audit_bisr_fidelity_20260917.py`. Scripts and required small evidence are included in the respective archives. These are retrospective checks, not new experimental confirmation.
