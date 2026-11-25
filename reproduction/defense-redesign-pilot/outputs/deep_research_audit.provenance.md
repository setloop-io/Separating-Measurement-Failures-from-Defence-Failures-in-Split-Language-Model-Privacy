# Provenance: supplied research narrative audit

Prepared/accessed 2026-09-10. Researcher-role output from a bounded deep-research workflow. This is a corrective source audit, not an exhaustive literature review or experimental replication.

## Local sources

| Source | Identity | Use |
|---|---|---|
| `Deep-research.md` | SHA-256 `c6d4f90929b37d94c8b0bedf74d83868a19bff69d02d20b19caedb103e773bd3` | Supplied narrative, read in character chunks because it contains very long lines. Preserved unchanged. No bibliographic reference list was supplied with its assertions. |
| Paper 3 `docs/CLAIM_EVIDENCE_LEDGER.md` | Repository revision `52191c430110bd8d9dee80538fde22bf1726e56a`; file SHA-256 `08b4411fd2ae86df7623b6ffb47a374a3977c7dcde2899ea66492119abbbf83f` | Inherited evidence boundary. Final manuscript clarifications and reviewer-response revision override broader historical decision prose. |
| Paper 4 `AGENTS.md` | Read at task start | Prospective cohort freeze, seed-level inference, inherited Paper 3 evidence, explicit channel scope, and active evaluation before a positive defense claim. |

## Primary-source checks

The following URLs were opened or returned with primary-source text by web retrieval. Search snippets from third-party summaries were used only to locate primary documents and were not relied on as evidence. Abstract-only checks are explicitly marked. No numerical external results were imported into the experiment plan.

| Source | Inspected location and support | Boundary |
|---|---|---|
| [Makhdoumi et al., Privacy Funnel](https://arxiv.org/pdf/1402.1774) | Full PDF retrieved; formulation of privacy mappings, log-loss and mutual-information objectives. | Not an LLM-specific converse or a calibration rule for correlation differences. Author-hosted PDF retrieval errored; arXiv PDF used. |
| [MIT 6.441 notes](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/5d8f16adc3385c9ff2975b121bd620e4_MIT6_441S16_course_notes.pdf) | Theorem 5.3, PDF page index 53; finite-alphabet conditional-entropy bound. | Technical teaching primary material; used to check the direction of the inequality, not as new empirical research. |
| [Pasquini et al., FSHA](https://arxiv.org/pdf/2012.02670) | §3, PDF pages 3–4; malicious learning objective, public autoencoder, discriminator, client feature-space steering. | Active server intervention; not a guarantee of failure for every defense. |
| [Chen et al., BiSR](https://arxiv.org/html/2409.00960v1) | §§2–3 threat model, pretrained priors, semi-white-box access, forward/backward reconstruction. | Method verification only; no reproduction or direct transfer to from-initialization byte models asserted. |
| [Gu et al., PIDI/ADMI](https://arxiv.org/html/2606.14210v1) | §3, generation and threat-model paragraphs: head/body activations, honest-but-curious inference, no gradients. | Preprint version 1; no venue upgrade claimed. |
| [Chen et al., MIXGUARD](https://arxiv.org/abs/2606.16801) | Author abstract: token/representation obfuscation, gradient perturbation, reconstruction and adaptive evaluations. | Abstract-level scope only; no exhaustive control audit or claim about our property gate. |
| [Liu et al., DualGuard](https://aclanthology.org/2025.acl-long.835.pdf) | Abstract/introduction and system diagram: local warmup transformation, global retention, bidirectional reconstruction. | Official ACL proceedings; no assertion that specific controls are absent throughout the paper. |
| [InfoDecom](https://ojs.aaai.org/index.php/AAAI/article/download/39212/43173) | Design and threat-model sections: honest-but-curious split inference, reconstruction, visual recognizability. | Theorems were not independently re-proved; no claim about performance on Paper 4's byte properties. |
| [Rafailov et al., DPO](https://proceedings.neurips.cc/paper_files/paper/2023/file/a85b405ed65c6477a4fe8302b5e06ce7-Paper-Conference.pdf) | Official NeurIPS 2023 paper; policy fine-tuning objective and method description. | Corrects final-layer-only claim; does not establish boundary privacy. |
| [Shao et al., DeepSeekMath](https://arxiv.org/html/2402.03300v3) | §4.1.1 and Algorithm 1: update policy parameters using group-relative objective. | Does not constrain training to softmax or establish a split-boundary defense. |
| [Gentry, FHE thesis](https://crypto.stanford.edu/craig/craig-thesis.pdf) | Author-hosted thesis; §2.2 computational-security game and malleability, §4.2 correctness. | Foundational construction, not a contemporary performance benchmark. |
| [Intel TDX assurance](https://www.intel.com/content/www/us/en/developer/articles/technical/software-security-guidance/technical-documentation/tdx-security-research-and-assurance.html) | Security objectives, threat model and TCB discussion. | Vendor primary documentation; no universal assurance outside its assumptions. |
| [Intel MKTME side-channel guidance](https://www.intel.com/content/www/us/en/developer/articles/technical/software-security-guidance/best-practices/mktme-side-channel-impact-on-intel-tdx.html) | Overview and impact; memory-access-pattern threats under untrusted VMM. | Does not establish that any proposed Paper 4 deployment is vulnerable; shows why scope must be explicit. |

## Reasoning provenance and limits

- The independent `(Y,S)` example, permutation/mean identity, per-position averaging observation, and the distinction between exact surrogate execution and zero replacement loss are elementary analysis performed for this audit. They are not reported experiments or attributed empirical findings.
- No outside defense was installed, run, or scored. No Paper 3 experiment was rerun. No claimed Paper 4 hypothesis was tested by this audit.
- No full paper-code audit was performed for MIXGUARD, DualGuard, InfoDecom, DPO, GRPO, or the proposed hybrid. Absence of a source-backed result in this audit is not evidence that the result cannot exist.
- SafeSplit, SynIB, proposed LoRA/FHE speedups, and asserted domain-specific reconstruction guarantees remain unverified leads. Before adding them as baselines or formal premises, identify the exact paper, assumptions, implementation, and evaluation endpoint.
- The brief uses paraphrases and no extended quotations. Links point to primary papers, official proceedings, university-authored material, or vendor documentation.
