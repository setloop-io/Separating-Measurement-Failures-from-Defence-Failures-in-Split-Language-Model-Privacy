# Pilot freeze records

`p4-defense-pilot-1.0.json` is produced at freeze time by the orchestrator via
`scripts/make_freeze_record.py` after the qualification campaign verifies and the
seed audit passes. It is intentionally absent until then: the exploratory pilot
refuses to execute without a committed freeze record binding its config hash,
code commit, seed audit, and qualification verification.
