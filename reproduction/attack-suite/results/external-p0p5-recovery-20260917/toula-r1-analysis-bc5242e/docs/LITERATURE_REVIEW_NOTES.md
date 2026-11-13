# Related-work verification — 2026-09-09

Scope: finalize the existing manuscript's related work and verify the supplied review's
claims against primary sources. This is a focused literature check, not a claim of an
exhaustive global survey. The alpha CLI was unavailable; primary publisher and arXiv pages
were checked directly. The manuscript uses the existing BibTeX citation mechanism.

| Reference key | Primary source checked | Supported comparison |
|---|---|---|
| `vepakomma2018split` | https://arxiv.org/abs/1812.00564 | Split learning with local raw data and distributed model components. |
| `vflairllm` | https://arxiv.org/abs/2508.03097 ; https://kdd.org/kdd2025/datasets-and-benchmarks-track-papers-2/ | Split-LLM framework, partitions, tasks and attack/defense benchmarking; this paper adds causal remote-learning estimands and segment prediction. |
| `bisr` | https://arxiv.org/abs/2409.00960 ; https://arxiv.org/html/2409.00960v2 | Pretrained-weight prior, fine-tuning drift, autoregressive dependencies, forward/backward reconstruction. |
| `dualguard` | https://aclanthology.org/2025.acl-long.835/ | Local parameter-space transformation plus retention during global fine-tuning. |
| `men2024shortgpt` | https://arxiv.org/abs/2403.03853 | Block influence and layer removal motivate, but do not prove, segment substitutability. Corrected first author to Xin Men. |
| `xu2024fora` | https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_A_Stealthy_Wrongdoer_Feature-Oriented_Reconstruction_Attack_against_Split_Learning_CVPR_2024_paper.pdf ; https://github.com/X1aoyangXu/FORA | Public auxiliary data and substitute-client feature transfer for reconstruction in vision split learning. |
| `mai2024snd` | https://proceedings.mlr.press/v235/mai24a.html | Inference-stage perturbation and client-side output denoising; not server-side denoising as the supplied review asserted. |
| `liu2026privtune` | https://arxiv.org/abs/2512.08809v3 | Crafted perturbations toward indirect neighbors; evaluates attribute inference as well as embedding inversion. First posted 2025, version 3 in 2026, marked accepted at INFOCOM 2026. |
| `gu2026pidi` | https://arxiv.org/abs/2606.14210 | PIDI attacks prompts and responses; ADMI combines adapter warmup and MI regularization. Source marks acceptance at ICML 2026. |
| `chen2026mixguard` | https://arxiv.org/abs/2606.16801 | Token/representation obfuscation plus adaptive gradient perturbation; cited as a 2026 preprint. |
| `politis2026gradient` | https://arxiv.org/abs/2609.04382 | Prior calibrated audit of returned gradients motivates the P5 defense axis; different system, not pooled evidence. |
| `tishby2000ib` | https://arxiv.org/abs/physics/0004057 | Compression retaining predictive information; no bound transferred to the current empirical study. |
| `makhdoumi2014funnel` | https://arxiv.org/abs/1402.1774 | Utility versus inference of sensitive variables under a specified disclosure model. |

The novelty claim is restricted to the **combination of causal remote-work controls,
declared-transcript calibration, and segment-property prediction**. No priority claim is
made for benchmarking, bidirectional attacks, calibration, pruning, or privacy funnels in
isolation. The segment-prediction contribution includes its held-out failure.

The manuscript does not infer that the tested D0/D1 bundles defeat or refute SnD, PrivTune,
DualGuard, ADMI, or MIXGUARD: those systems were not implemented in this study. It does not
import the supplied review's speculative information-necessity claims, broad regulatory
conclusions, or cryptographic cost comparisons.

## Citation and presentation follow-up

The requested arXiv:2609.04382 was already present as `politis2026gradient`. Its primary
arXiv entry was rechecked against the title, authors, year, and research scope. The
manuscript now cites it in the introduction, the P5 defense-design explanation, and related
work, with the full title visible in the introduction and related-work paragraph. No
duplicate bibliography entry was added.

All 13 citation keys remain distinct and resolve. References now appear in citation order
(`unsrt`); arXiv entries display explicit linked identifiers rather than unexplained URLs.
Publisher entries retain their publisher/DOI links. DualGuard, Split-N-Denoise, and PrivTune
primary entries were also rechecked during this follow-up. The source comparisons above
remain the bibliographic audit trail for the complete list.

## Official proceedings follow-up — 2026-09-09

The four requested publication records were checked against the matching titles and authors.
The alpha CLI remains unavailable, so this check used primary web sources directly.

| Reference | Verified publication | Bibliography action |
|---|---|---|
| ShortGPT | [Findings of ACL 2025](https://aclanthology.org/2025.findings-acl.1035/), pp. 20192–20204, DOI `10.18653/v1/2025.findings-acl.1035` | Replaced the arXiv entry with proceedings metadata and added Qianhao Yuan to match the published author list. Retained the existing citation key `men2024shortgpt` for compatibility; the displayed year is 2025. |
| FORA | [CVPR 2024 proceedings](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_A_Stealthy_Wrongdoer_Feature-Oriented_Reconstruction_Attack_against_Split_Learning_CVPR_2024_paper.html), pp. 12130–12139 | Conference, authors, and pages were already present. Replaced the arXiv link with the CVF proceedings link. |
| BiSR | ACM CCS 2024, DOI [10.1145/3658644.3690295](https://doi.org/10.1145/3658644.3690295), corroborated by the authors' [arXiv journal-reference and related-DOI fields](https://arxiv.org/abs/2409.00960) | Already cited as CCS 2024. Added a structured `url` field while retaining the rendered DOI link. Direct ACM retrieval returned HTTP 403; no new metadata was inferred from that inaccessible page. |
| Split-and-Denoise | [ICML 2024, PMLR 235](https://proceedings.mlr.press/v235/mai24a.html), pp. 34281–34302 | Already cited as ICML 2024 with the official link. Added publisher and structured URL fields. |

The matching ShortGPT publication is in Findings of ACL 2025; the lookup did not establish
the suggested ICLR 2026 attribution. ICLR 2026 papers that cite ShortGPT do not establish
ShortGPT's own venue. Likewise, this BiSR title is identified specifically with CCS 2024,
not a range of security/software-engineering venues. Acceptance annotations on other entries
are not converted into invented proceedings metadata. No scientific claims or results changed.

### User-supplied OpenReview decision

For [OpenReview record JMNht3SmcG](https://openreview.net/forum?id=JMNht3SmcG), the user
supplied the decision text: **Reject**, posted by Program Chairs on 22 January 2025 at
05:28 and modified on 5 February 2025 at 05:45, with
[decision revisions](https://openreview.net/revisions?id=fZxQA9Gi5i).
This is user-provided evidence; direct forum/API retrieval encountered browser verification.
The supplied decision does not support an ICLR acceptance attribution. It does not negate
the separately verified Findings of ACL 2025 publication, which remains the cited version.
