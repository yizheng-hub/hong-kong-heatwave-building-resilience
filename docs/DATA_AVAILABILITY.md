# Data availability and redistribution boundary

The complete local project root contains the provider inputs actually used by the production chain, the necessary generated intermediates, compiled physical inputs, accepted solver records and accepted results. The private GitHub repository intentionally contains only the redistributable subset of this same directory tree.

## Included in the Git repository

- Prepared building-component and spatial-link tables.
- Prepared map boundaries and simplified road context used by the figures.
- Accepted building-event outcomes for all eight assessment events.
- Compact panel source tables, cross-event analyses and validation summaries.
- The full editable plotting notebook and all final PNG figures.
- The compact accepted eight-event review archive and immutable audit receipts.

## Original inputs

`data/` is the exclusive location for original/provider inputs in the complete local root. `data/SOURCE_MANIFEST.csv` records the actual local path, provider file identity and access status for HK-BFETD, HKO, NASA POWER, CMIP6, Census, EMSD, roads and external validation datasets. Generated mappings and prepared weather products are under `intermediate/source_derived/`, not under `data/`.

## Not redistributed

- Restricted campus meter records and any provider-controlled raw observations.
- Multi-gigabyte compiled physical inputs and accepted EnergyPlus output directories.
- The pinned local EnergyPlus installation.
- Original provider files whose redistribution conditions are source-specific or whose archives are impractically large for Git.
- Manuscript or Supplementary Information files.

The absence of these provider-controlled or very large files from GitHub does not prevent regeneration of the published analysis and figures. It does mean that a GitHub-only run begins from accepted derived simulation outputs rather than rerunning the physical solver. The complete local root retains the full current physical record and can be checked with `--full-local`.

Consult `SOURCE_AND_LICENSES.csv`, `PROJECT_CONTENTS.csv` and `../data/SOURCE_MANIFEST.csv` before redistributing third-party material.
