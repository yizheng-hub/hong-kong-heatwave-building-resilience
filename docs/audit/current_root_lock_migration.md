# Unified-root lock migration

The immutable accepted-production locks remain under `docs/audit/production_protocol/`.
The active locks under `config/` were refreshed after consolidating the project into one
portable root. The reviewed changes are limited to project-relative input locations,
the shortened superseded-C70 archive location, the `src` import root, and the mixed-use
residential-share eligibility rule required to reproduce the accepted 330,761-building
cohort. No accepted simulation output was rewritten and no EnergyPlus call was made
during this lock migration.

The accepted tables are independently protected by their production receipts and are
compared with a clean, solver-free post-processing run under `result/verification/`.
