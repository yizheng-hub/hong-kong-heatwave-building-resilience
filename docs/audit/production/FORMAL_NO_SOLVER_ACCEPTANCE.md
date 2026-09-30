# Formal no-solver acceptance

Date: 2026-09-21 (Asia/Shanghai)

## Outcome

- Full isolated test suite: **105 passed, 2 skipped, 0 failed**.
- JUnit: `test-results/completion_formal_production_current.xml`.
- Skips: tests requiring intentionally absent large real G3 caches; they are not converted to passes.
- Warnings: two pandas FutureWarnings in test-only boolean assertions; no production failure.
- Real EnergyPlus calls during development and acceptance: **0**.
- Live Open6 writes performed by this workspace: **0**.

## End-to-end acceptance profile

Persistent fixture output: `fixture_acceptance_results_formal_final`.

- status: `ACCEPTANCE_PROFILE_COMPLETE`
- production_ready: `false`
- events: all eight event identities
- five-scenario logical rows: 160
- core groups: 32
- stage-score rows: 128
- fixture buildings: 3
- full Panel contract: 29/29 rows verified
- review ZIP SHA256: `bba54e29348cd809c4be5ba8a72b2eba54553adc6f426acf03ed5e6b59e1a457`

The fixture runs the same numerical analysis and final-gate functions with small
synthetic inputs. It does not contain paper results and cannot emit
`READY_FOR_FIGURES_AND_MANUSCRIPT`.

## Covered formal boundaries

The suite covers the five decision schemas and numerical functions, missing-field
rejection, decision-hash invalidation, same-zone humidity, complete multi-zone
recovery support, static-region denominators, per-building construction-year
unknowns, full Panel schema/key/domain checks, all four targeted C70
effective-source migrations and the 8,910-row current source index,
immutable lockbox prediction freeze, interruption/resume and final package
self-exclusion. Windows hard power loss and real lockbox solver behavior remain
production-run validations, not mocked scientific successes.

## Current production preparation boundary

Open6 is now read-only qualified: 8,910 logical runs, 1,782 complete groups,
7,128 stage rows and no active writer. The completion code is
`CODE_COMPLETE_WAITING_FOR_USER_APPROVAL`; this is not a paper-result
completion label. Doctor reports a conservative storage estimate of
18,040,478,241 bytes against the unapproved 16 GiB setting, so the approval
entry point remains blocked until the resource decision is updated.
