# R1 workload preparation — 2026-09-16 (owned-node engineering)

Prepared from the clean master checkout at `592ef40` (source inventory seal
`9e39b3db26b21a2e975d8159c0f6e6f62faca8449060fff2ad227afb36f8b2a0`, 97 bound files)
for engineering execution on the owned GB10 nodes toula/koula, authorized by the
owner on 2026-09-16. Supersedes nothing; the 2026-09-13 package remains the CPU
preparation record. **Engineering only: `scientific_execution_admitted: false`.**

- `workload.tar.gz` — the complete prepared bundle (183 files), archive SHA-256
  `f3515e8312c21eb294ef91d36a0576b568663812668ac813c5252377188c1d30`.
- `r1_bundle.json` — bundle inventory; bundle SHA-256
  `45d7682eaf0ef8a3c2445901df8c521fa9f9dbcbff3ed1d04cfd982a33ac041d`.
- `plan.json` — the four planned workloads (all engineering/blocked-draft).
- `validation.json` — archive/restore/verify record: the archive was extracted to a
  disposable directory and that copy's `verify-bundle` reported `status: verified`.

The archive is xattr-free (COPYFILE_DISABLE=1) so GNU-tar extraction on Linux stays byte-clean; the first xattr-carrying archive was rejected by the Linux verify-bundle and replaced (see validation.json).

Restoration check: extract `workload.tar.gz` to a new directory and run
`python <restored>/r1-bundle-eng-20260916/scripts/run_r1.py verify-bundle` with a
Python environment carrying the project's runtime dependencies.
