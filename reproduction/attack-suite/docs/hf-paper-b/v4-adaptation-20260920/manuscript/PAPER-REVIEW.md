# Independent review of the Paper B adapted-comparison revision

Review date: 20 September 2026. Verdict: **no scientific claim/evidence blocker
identified in the new adapted-comparison material**. The numerical results,
declared scope and uncertainty statements agree with the completed analysis.
The intentional immutable-HF placeholder and the pending full reproduction and
publication checks remain release tasks, not resolved facts in this review.

## Material inspected

- `paper/paper_b/main.tex`, particularly the abstract, §§ on BiSR conformance and
  the frozen adapted comparison, Table `tab:bisr-adapted`, related work,
  conclusion and evidence availability.
- `paper/paper_b/response-to-review-2026-09-17.md`.
- All summary and contrast outputs in
  `build/v4-release-20260920/analysis/`, including the complete-set integrity
  record and the result report's internal artifact hashes.
- `support/REPRODUCE.md`, the frozen adaptation amendment, original v4 protocol,
  final execution admission and the preparation result in the adaptation
  checkout.

This review checks the new claims against the completed evidence. It does not
rerun GPU prediction or independently re-audit every older result in Paper B.
Compilation, visual inspection, fresh CPU reproduction and immutable HF
verification are being handled separately.

## Checks and findings

| Claim | Evidence check | Finding |
|---|---|---|
| Complete adapted matrix | Integrity report contains 80 candidate joint cells and 240 frozen control cells; design is 10 victims × 2 depths × 2 grants × 2 attack seeds, with 256 documents per candidate cell | Correct: 20,480 document executions and 160 reported candidate arm cells |
| Primary bf-minus-joint contrast | Exact difference −4.168701171875 percentage points; interval [−4.7894287109375, −3.529052734375] | Correctly rounded to −4.17 [−4.79, −3.53]; interval is wholly below −2 points |
| Primary recovery means | bf 93.206787109375%; joint 97.37548828125% | Correctly rounded to 93.21% and 97.38% |
| Direction in each victim run | All ten primary victim-level differences are negative | Manuscript statement is supported |
| Table of means | All 20 depth/grant/arm means checked against `arm-summaries.json` | Every displayed value agrees to two decimal places |
| Secondary family | Fifteen contrasts, α = 0.05/15, 100,000 draws, seed 93281 | Ten intervals wholly negative, one wholly positive and four crossing zero, as stated |
| Positive secondary contrast | bf-minus-forward, depth 6/exact: +3.438720703125 points, corrected interval approximately [2.59013, 4.32027] | Correctly reported as +3.44 points; no uniform superiority claim made |
| Near-zero joint contrast | bf-minus-joint, depth 6/exact: −0.0244140625 points, corrected interval approximately [−0.62805, +0.57943] | Correctly reported as unresolved, not equivalent |
| Shared SIP measurements | `sip_grant_pair_identity` reports all 40 pairs identical for all 256 documents | Correctly excludes extra independent attacker replications |
| Prediction resource use | Host allocation time = 29.7001026343 GPU-host hours; measured stage time = 29.5806155549 hours | Paper correctly reports 29.70 allocation hours, not stage time or FLOPs |
| Training/preparation resource use | Preparation supervisor `resource.json.elapsed_seconds` records 2101.638643110986 seconds = 0.5837885120 hours; worker `result.json.wall_seconds` records 2099.7251023519784 seconds = 0.5832569729 hours | Paper's additional 0.58 allocation hours is supported; use the outer supervisor timer for exact allocation accounting |

All hashes listed in the completed analysis `result.json` matched the local
artifacts during this review. No numerical correction is required.

## Scientific scope and interpretation

The manuscript consistently distinguishes three propositions:

1. C-A established implementation agreement within a fixed, single-seam
   configuration, sharing saved checkpoints and inputs between engines.
2. The separately frozen amendment establishes the reported comparison against
   the original v3 controls at declared, distinct budgets.
3. The original published-setting B1 requirement remains unmet; neither result
   establishes original dual-cut or SensiReplaced reproduction.

These distinctions survive the abstract, main result, related work, conclusion
and response letter. The paper does not silently replace the original fidelity
gate or retrospectively validate historical substitute runs.

The numerical-kernel explanation is appropriately narrow: SDPA reproduces the
original activation captures; attack optimisation retains the qualified eager
path. The text explicitly says this does not reproduce downstream gradients and
does not assert equality between the two kernels. The factor-256 gradient
conversion matches the recorded loss-normalisation rule.

The uncertainty statement correctly averages attack seeds before resampling
documents and victim runs, conditions on the four shared trained inverters and
fixed auxiliary corpus, and acknowledges approximate inference with ten victim
runs. It does not count the 20,480 document executions as independent training
replications. Token accuracy alone receives inferential contrasts; overlapping
or zero-crossing intervals are not treated as equivalence. The SIP-versus-forward
contrast is correctly described as a method comparison rather than isolating
the BRE contribution.

Compute is described as allocation wall time, and the paper explicitly rejects
equal-compute and FLOP interpretations. The primary result supports a lower
recovery conclusion for this adaptation at depth 6/public; the prose does not
extend it to all grants, other boundaries, all BiSR settings, or a privacy
guarantee.

## Minor clarification recommended

The frozen initial amendment retains the inherited parenthetical
`bisr_sip vs forward_k32 (isolates the stage-2 BRE contribution)`. That
parenthetical is scientifically incorrect, and the revised manuscript already
corrects its interpretation. Add a short explicit note to the supplement's
reproduction instructions explaining that the frozen wording is retained for
provenance, while the contrast does not isolate BRE. Do not modify the frozen
amendment or change the contrast family. This is a documentation consistency
correction; it does not affect any result or require additional experiments.

Follow-up: the release owner added a standalone `support/PROTOCOL-ERRATUM.md`
to make this correction explicit while preserving already verified reproduction
inputs. This resolves the clarification recommendation.

For completeness, the stated 29.70 and 0.58 hours account for prediction and
inverter preparation respectively. They should not later be relabelled as an
all-in total that includes earlier capture checks, resource canaries, the C-A
campaign or CPU analysis. The current manuscript does not make that stronger
claim, so no correction is required.

The two preparation timers differ by 1.913540759 seconds. The supervisor timer
starts before the worker subprocess and ends after it exits; the worker timer
starts after imports and ends before its final result and journal writes. This
is lifecycle overhead, not a conflicting measurement. The authoritative
allocation record is `experiments/v4-final-admission/preparation/resource.json`
in the execution source (also `preparation/resource.json` in the raw archive),
SHA-256 `6d607f7e466ccd955f4a5e620fda0a8ad165b767603c1f72cf0923d80a761c10`.

## Release conditions

- Replace `IMMUTABLE_V4_REVISION` in both manuscript and response letter only
  after the complete evidence prefix has passed the immutable remote digest
  verification. The current placeholder is intentional and is not counted as a
  scientific defect.
- Retain the distinction between raw-archive verification and a successful
  full CPU analysis reproduction. Do not describe the latter as complete until
  the reproduction helper succeeds.
- Confirm the final rendered manuscript contains the table and all scope
  qualifiers without clipping or unresolved references. This review did not
  perform visual PDF inspection.

## Reviewed file identities

| File | SHA-256 |
|---|---|
| `paper/paper_b/main.tex` | `876a89d0a483afc1168dac93b31b61b2a071ec1219fbfd017e35e4afe56a9b82` |
| `paper/paper_b/response-to-review-2026-09-17.md` | `8737540a3a56e81340a4e0ab995fb9d871af5666fdaa70b160971255f2815297` |
| `support/REPRODUCE.md` | `53ded1769d88c6596250872206a212e0633bb90ba288411b54a28ec475ba9f1a` |
| `analysis/result.json` | `1394d4411aebab119393e0d67dd9b27108febe0ff28341749b507c649fb48f08` |
