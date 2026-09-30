# E8 completion software delivery

Delivery state: **CODE_COMPLETE**. Numerical state at final doctor: **WAITING_FOR_UPSTREAM**.

Continuation verification on 2026-09-20 completed with 77 passed, 2 skipped and
0 failed tests. Real EnergyPlus calls remained zero. The lock probe added after the
v4 review package is now read-only and detects a lock held by another process.

The current live upstream is not the frozen identity used by this delivery. Its
`config/e8_open6_versions.json` SHA256 is
`15f446e3e05c7c4dc74df686278fd5f4f316c2ae75bb984eece1ad3ed8a00757`, not the
supplied `1928f0954b92e2063dc301e2184331b7add7c9e7d238eb2d1ea3490d784f2d06`.
Doctor therefore remains blocked and the approval/execute command must not be used
until the v1.3 C70 repair is reviewed and the isolated source snapshot is explicitly
re-frozen. The target-manifest SHA256 still matches the supplied value.

The isolated entry implements the approval gate, read-only Open6 receipt, six-event
building-level lockbox prediction freeze, two lockbox manifests, four-factor residential
mechanism manifests with physical deduplication, exact group metrics, E8 integration,
fixed-stock propagation, simultaneous component recovery, city simultaneous power,
district links, 30/32/34 C sensitivity, BERI, LOEO8, T1-T3, simple baselines, tie-aware
screening, evidence compatibility, figure Source Data contracts and final evidence gate.

The final label is not available from code completion or solver completion. It requires
all task states, core cardinalities, stock coverage, validation outputs and every retained
figure source table to pass current-file verification.

Development used synthetic fixtures and small read-only G4-a evidence. It made zero
EnergyPlus calls and did not write the running Open6 state, attempts, ledgers, manifests,
inputs or caches. Windows hard-shutdown and true lockbox batch behavior remain to be
confirmed by the later user-started production run; these are not reported as completed.
