# Search for Paper 3 historical source in Papers 1 and 2

Checked on 2026-09-10. This is a read-only follow-up to the evidence availability
audit. The question was whether another local paper repository preserves the
missing source snapshots used to execute Paper 3's historical experiments.

**Paper 1 has its own E1 runtime source archive and original, checksum-matching
patch. No matching Paper 3 execution snapshot or patch was found in either
Paper 1 repository or Paper 2.** The unresolved provenance statement should
therefore refer specifically to Paper 3, not imply that Paper 1 has no preserved
source.

## Repositories inspected

| Repository | Search coverage | Finding |
|---|---|---|
| `paper1_arxiv_submission` | 747 working files, archive members, and all 1,335 locally stored Git objects, including unreachable objects | Preserves Paper 1's E1 runtime archive. No Paper 3 source or patch match. |
| `Paper-1-When-the-Privacy-Probe-Passes-but-the-Transcript-Leaks` | 1,801 working files inventoried; text/archive scan plus all 3,124 local Git objects and archive payloads | Preserves the same E1 archive and the original Paper 1 patch matching its manifest. No Paper 3 match. |
| `Paper-2-How-Much-of-an-LLM-Can-a-Protected-Boundary-Replace` | 1,350 working files inventoried and all 1,891 local Git objects, including unreachable objects | No candidate source archive or matching Paper 3 source/patch found. |

The search included ignored evidence files and local Git objects beyond the
current branch. Virtual environments and caches were excluded from the main
text scan; separate checks found no installed `pcr` source candidate. Archive
members were inspected without extracting or executing their contents.

## What was recovered in Paper 1

The original working repository contains:

- [E1 runtime source archive](/Users/geo/Paper-1-When-the-Privacy-Probe-Passes-but-the-Transcript-Leaks/paper-data/provenance/e1_source_of_record/e1_runtime_sources.tar.gz),
  SHA-256 `7bbd4e3d653ee7eb93b4eac5017f124f9c545ffaf568cb649854c53a8acd0061`.
- [Original execution patch](/Users/geo/Paper-1-When-the-Privacy-Probe-Passes-but-the-Transcript-Leaks/paper-data/provenance/odysseus_dtraining_b1915da.patch),
  SHA-256 `21ea9908f412cded7f2e4086c2e3918a412700b6fb9b387bd0b038379426f6ad`.
- [Their provenance manifest](/Users/geo/Paper-1-When-the-Privacy-Probe-Passes-but-the-Transcript-Leaks/paper-data/provenance/e1_source_of_record/MANIFEST.json),
  which identifies the August 22 E1/gradfix experiment lineage and upstream
  revision `b1915da`.

Both files match the hashes in that manifest. The arXiv submission contains the
same runtime archive, but its current patch has documented node-name
substitutions. The original patch also survives in that repository's Git blob
`17ff6b29dbc2337c40261840a2c810169a7aaecf`. Comparing the two patches attributes
the difference to those substitutions; neither version matches a Paper 3 patch.
This verifies these particular Paper 1 records, not every historical Paper 1 run.

## Why this does not resolve Paper 3's provenance

Paper 3's 768 original manifests reference scaffold commit
`be32144570e6fdb6cbdbdf367c9e71ffa685d24a` and nine distinct dirty-tree hashes.
No candidate patch matched any of those nine hashes. Searches also found no
Paper 3 source identifiers or exact source-file matches to its first committed
implementation (`9aac3d0`) or current implementation. Neither of those Paper 3
commits is present as a commit object in the three inspected repositories.

Paper 1's archived `privacy_runtime`, attacker, and split-training files concern
its own earlier experiment system. Their existence does not identify the
historical `src/pcr` implementation used by Paper 3. Moreover, Paper 3's recorded
bare `git diff` hashes exclude staged changes and untracked files: even a matching
patch alone would not necessarily recover all execution source.

The defensible statement is: **Paper 3's exact historical execution snapshots
remain unrecovered after searching these local Paper 1 and Paper 2 repositories.
Paper 1's E1 source archive and original patch are available.** Remote hosts,
external backups, and repositories not stored locally were not searched; this
finding does not establish that recovery elsewhere is impossible.

## Records and changes

Machine-readable search records are [Paper 1 submission](paper1_submission_search.json),
[Paper 1 working repository](paper1_working_search.json), and
[Paper 2](paper2_search.json). [summary.json](summary.json) records their checksums.

No archived manuscript, experiment source, original manifest, or result was
changed. No model training or probe fitting was performed. This separate
addendum preserves the previously published audit files and their checksums.
