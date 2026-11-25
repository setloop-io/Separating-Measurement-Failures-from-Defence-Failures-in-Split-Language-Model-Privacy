# Rejected and invalid latent-v6 diagnostics

These aggregate-only artifacts are retained for the research audit trail and
must not be pooled into accepted timings or privacy conclusions.

- `latent_v6_remote_batch_*`, `latent_v6_active_d050_*`, and
  `latent_v6_graph_*` predate per-connection cloud-model isolation. Their
  nominal seeds shared accumulated Koula model state.
- `latent_v6_iso_*n030*seed43*` is a valid isolated run that failed the
  strengthened graph-attacker gate.
- `latent_v6_defgraph_*` is a valid isolated negative result: adding the graph
  attacker to the minimax loop did not improve held-out graph recovery.
- `latent_v6_split8_*` and `latent_v6_split20_*` are valid isolated boundary
  searches that failed the binding utility gate. The external attacker was
  intentionally skipped once utility failed.

No raw tensor bundle, wire capture, secret, checkpoint, or model weight is
stored here.
