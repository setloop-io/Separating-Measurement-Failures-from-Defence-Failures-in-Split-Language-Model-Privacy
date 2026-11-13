# Run manifests

Every executed cell must have one immutable manifest containing at least:

- study/protocol version and exploratory/confirmatory status;
- git commit, dirty-tree patch digest, environment/container digest;
- complete command line and resolved configuration;
- model, tokenizer, corpus, and split digests;
- independent seed and all derived random streams;
- start/end timestamps and host/device identifiers;
- transcript schema and captured-channel inventory;
- output paths and SHA-256 checksums;
- success/failure status with a predeclared operational reason;
- measured FLOPs, bytes, wall time, memory, and energy where available.

Manifests must be written before scoring. Outcome-dependent metadata edits are forbidden; use a
versioned correction record instead.

