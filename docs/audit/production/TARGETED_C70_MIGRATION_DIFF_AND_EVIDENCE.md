# Targeted C70 migration evidence

Status: technical evidence complete; scientific acceptance remains part of the
single explicit user approval. This report does not release lockboxes and does
not invoke EnergyPlus.

- Effective Open6 version-lock SHA256:
  `2bd819a72bb73dec7515627abd1659a3e4957e6f5b78d4218880614cd2062f41`
- Open6 manifest SHA256 remains:
  `7b3262723358cdb4debef01b933b5a55d51cae316e79dbad9e99b7eaa0f33c81`
- Reviewed physical replacements: four HW_C70 runs.

The replacements are 0049/B0127 and 0068/B0206, B0213 and B0260. For each
run, the audit verifies the archived superseded input identity, current
byte-level physics hash, IDF/EPW/schedule/raw/parsed evidence, affected
five-scenario group dependency receipt, current group-table hashes and the
declared and zone-level C70 capacity QC. The final combined Open6 QC contains
1,782 passing declared-capacity checks and 1,782 passing zone-capacity checks.

The completion source resolver writes an effective 8,910-row index. It fills
the deterministic result path for every newly solved Open6 run and replaces
the four archived C70 hashes with their current hashes. Other compatible runs
retain their existing physical identity and are not re-solved.

Machine-readable evidence is stored under:
`D:\HKBFETD\release\HK_heatwave_E8_completion_results\00_protocol\targeted_c70_migration`.
