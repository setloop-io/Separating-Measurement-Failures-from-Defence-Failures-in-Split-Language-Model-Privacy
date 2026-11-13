# Independent review of completed v4 adaptation analysis

Verdict: **No numerical or interpretation blocker found in the five analysis files.** Full deterministic reproduction and immutable HF verification remain release checks owned by the main workflow.

Reviewed `integrity.json`, `document-metrics.jsonl`, `contrasts.json`, `arm-summaries.json` and `result.json`. Independently recomputed all 20 arm means, all 16 paired contrast estimates and every per-victim estimate from the 102,400 document-metric rows; all agree exactly. No duplicate metric identities were found. The result's four output digests match their files. Integrity records contain 80 candidate joint cells, 240 frozen control cells and 256 documents per cell. The separate main-workflow reproduction will independently recheck raw journals, rescore predictions and regenerate bootstrap intervals; this review did not rerun the bootstrap.

## Statistical specification

The primary is depth 6/public, BiSR-bf minus joint k32. The secondary family has the prespecified 15 comparisons: four bf–forward, four bf–SIP-it128, four SIP–forward and three remaining bf–joint strata. Primary alpha is 0.05; secondary per-comparison alpha is 0.05/15. All stored intervals bracket their point estimates. The reused implementation resamples victim and document axes independently, after averaging the two attack seeds, with 100,000 replicates and seed 93281. Secondary tails contain approximately 166.7 expected bootstrap draws, so tail precision remains finite.

Primary recovery is 93.206787% for adapted BiSR-bf and 97.375488% for joint k32. The difference is **−4.168701 percentage points**, with a 95% crossed-bootstrap interval **[−4.789429, −3.529053] percentage points**. The whole interval is below the negative 2-point practical threshold. This supports a practically meaningful deficit in this declared setting, not a general ranking across BiSR settings.

The corrected secondary results are heterogeneous. In particular, bf exceeds forward at depth 6/exact, while bf versus joint there crosses zero. Do not describe that latter result as equivalence. SIP versus forward compares initialisation methods and does not isolate BRE. All 40 public/exact SIP pairs have identical predictions over all 256 documents; they add no independent attacker replications.

## Compute and scope

Measured stage time totals 29.580616 GPU-hours; cold loads add 0.093407 hours. The two host elapsed times total **29.700103 GPU-hours**, including remaining launch/process overhead. Adding the conservatively reserved 3 hours for prior checks and preparation gives **32.700103 hours**, below the approved 50-hour ceiling. The 3 hours are a reservation, not a measured training-time total. Layer work must not be labelled FLOPs; this comparison does not match compute budgets.

The report correctly conditions uncertainty on the fixed victim-training corpus and four shared GRUs (two attack seeds × two depths). There are ten victim-training runs, not 80 independent trained attackers. It explicitly excludes original dual-cut/published-setting reproduction, auxiliary-training uncertainty and equivalence claims from zero-crossing intervals. No manuscript should broaden those statements.
