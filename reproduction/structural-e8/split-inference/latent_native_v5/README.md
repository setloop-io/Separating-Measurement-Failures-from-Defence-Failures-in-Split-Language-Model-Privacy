# Latent-native v5 inference candidate

This directory reserves the isolated inference contract corresponding to the
training prototype. It is intentionally non-launchable: the 1024-to-128
private boundary and D-only cloud module now execute, but the live two-node
request protocol and end-to-end generation quality have not yet passed.

Koula must never receive the private encoder, private decoder, ratchet master,
uncompressed boundary activation, prompt template manifest, or canonical
replay labels. A root-level Koula can still inspect each live 128-dimensional
release, so all privacy claims remain empirical and attacker-gated.
