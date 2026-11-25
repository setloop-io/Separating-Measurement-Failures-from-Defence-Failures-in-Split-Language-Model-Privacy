# Invalid E-A4 deterministic FedAvg repeats

These artifacts record seeds 43 and 44 correctly, but the FedAvg data path did
not use the seed when constructing its document shards. The resulting feature
files were byte-identical to seed 42 and are not independent repetitions.

They are retained for provenance only. The superseding protocol shuffles the
selected member-document pool by training seed before assigning disjoint
even/odd peer shards, and reruns seeds 42, 43, and 44 consistently.
