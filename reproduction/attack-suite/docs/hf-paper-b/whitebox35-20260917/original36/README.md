# Corrected PR35 white-box evidence

The original 36 adaptive-e2e results omitted the client head and are invalid. This release uses token embeddings -> client head -> true encoder -> observed wire. Recorded head forward/backward work is checked for every complete cell. Historical outputs are retained at their source.

The original36 tier reuses unchanged original captures/checkpoints at depths3/6, seeds11/22/33, PCA/frozen bottlenecks64/128/256. The full tier covers internal GPT2 boundaries1–11 with the same seeds/dimensions. Depth12 is excluded, and per-depth canaries are excluded from scientific summaries.

The18 historical co-trained training cells remain NOT_EVALUABLE. Any new cotrain-final-inference records use a matched final victim head and final encoder on newly captured held-out inference observations. This changes the observation phase and grants stronger access; these results must remain separate.

Verify each file against manifest.json, concatenate artifacts.tar.gz.part* in lexical order, then extract the resulting gzip tar archive. Archives preserve project-relative paths and include code, pinned corpus, checkpoints, captures, provenance, journals and measured work. Install from archived pyproject.toml using requirements-research.lock. Use the pinned GPT2 revision in protocol.json for fresh captures.

To regenerate corrected36 use `python -m experiments.run_whitebox_sweep --root results/whitebox35-original36-v1 --adaptive-e2e --device cuda --threads 4` on a disposable copy (the legacy runner overwrites its output tier). To regenerate fresh layers use commands in the full tier journal. Co-trained CLI is `python -m experiments.whitebox35.cotrain --help`. Hash-only cell verification is available as verify_cell in cotrain.py.

Source code Apache-2.0. Corpus/model assets retain upstream licences. Repositories remain private. No global privacy guarantee is asserted.
