# V4 adaptation resource approval

The user approved a **50 GPU-hour aggregate ceiling** on 19 September 2026. This supersedes the original30-hour allowance and the pending40-hour proposal. The ceiling covers adaptation preparation, training, validation, predictions and failed attempts across toula and koula combined. It does not change the frozen comparison matrix or waive scientific admission checks.

The measured online prediction extrapolation is34.6207 GPU-hours; setup, fresh training and contingency must fit within the remaining allowance. No statement of automatic full admission follows from budget approval. The original frozen v4 protocol and initial adaptation scope records remain unchanged; `experiments/v4_resource_authorization_20260919.json` records this additional authorisation.

The exact-head activation replay blocker is resolved: all20 original captures/5120 document-depth rows match bit-for-bit using SDPA. The qualified candidate attack remains eager. The next experiment is fresh seed17/29 SIP training at depths3/6, followed by mechanical gates and final manifests.

At this check koula failed SSH twice (host down, then timeout); toula, soula, voula and sasa responded. Koula holds the prepared auxiliary inputs/cache. No new training or full comparison run has been launched. The next preparation experiment is4426e8db-84d0-461d-86ca-530e8e04481b, branch`orx/v4-adapted-sip-preparation`.


## Resumed after koula recovery

Koula is reachable and resumed preparation in ORX run `91effbd8-9efb-4489-9299-9611380b67ac`, source `844adea`. The run verifies frozen auxiliary inputs and caches, trains four new GRUs (seeds17/29 × depths3/6), then runs80 trained mechanical gates against the pinned reference. No final comparative scoring or full prediction campaign is launched by this stage.

The supervisor limits the whole process group to7200 seconds. Reservations sum to50 GPU-hours:1 conservative hour for earlier adaptation checks,2 for this preparation, and23.5 per GPU node for later work. Any later preparation/retry must reduce the remaining reservations; these are aggregate limits, not per-node50-hour grants. The runtime writes a resource record even on timeout/failure. The first logged input verification passed for15604 auxiliary windows. Canonical sample order matches the original pinned v3 input locally.

`orx exp wake` could not be enabled because this session is not an `orx up` agent session. The training-and-gates job continues autonomously, but there is no registered automatic agent continuation or automatic full-comparison launch.
