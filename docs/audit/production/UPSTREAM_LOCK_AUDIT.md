# Upstream lock audit — 2026-09-20

Read-only result: **DIFFERENT; NOT RE-FROZEN**.

- Supplied and isolated version-lock SHA256:
  `1928f0954b92e2063dc301e2184331b7add7c9e7d238eb2d1ea3490d784f2d06`
- Current live version-lock SHA256:
  `15f446e3e05c7c4dc74df686278fd5f4f316c2ae75bb984eece1ad3ed8a00757`
- Supplied and current target-manifest SHA256:
  `7b3262723358cdb4debef01b933b5a55d51cae316e79dbad9e99b7eaa0f33c81`
- Protected live files changed by this audit: **0**.

The live lock moved from `e8-open6-reviewed-v1.2-prerun-summary-fix` to
`e8-open6-reviewed-v1.3-targeted-c70-total-capacity-fix`. It changes the recorded
hashes for `g4a_local_execution.py`, `e8_open6_execution.py`, `e8_g3_runner.py`
and `e8_g3_postprocess.py`, adds `scripts/repair_c70_capacity.py`, and changes the
scope text to require targeted run-by-zone C70 supersession receipts. The physical
target-manifest hash remains unchanged.

No expected hash, approval file, live source, state, attempt ledger, cache index or
result was modified to make the audit pass. Before production completion work can
start, the v1.3 repair and receipts must be reviewed, then the isolated whitelist,
source snapshot, tests and approval contract must be explicitly re-frozen against
that accepted identity.
