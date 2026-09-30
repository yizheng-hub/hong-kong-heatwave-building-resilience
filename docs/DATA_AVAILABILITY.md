# Data availability and redistribution boundary

## Included in the Git repository

- Prepared building-component and spatial-link tables.
- Prepared map boundaries and simplified road context used by the figures.
- Accepted building-event outcomes for all eight assessment events.
- Compact panel source tables, cross-event analyses and validation summaries.
- The full editable plotting notebook and all final PNG figures.

## Original inputs

`data/` is the exclusive location for original/external inputs. `data/SOURCE_MANIFEST.csv` records the expected repository location, provider file identity and access status for HK-BFETD, HKO, NASA POWER, CMIP6, Census, EMSD, roads and external validation datasets. Where redistribution or repository size is problematic, the source remains with its provider and a source-specific access note is included instead of a copied archive.

## Not redistributed

- Restricted campus meter records and any provider-controlled raw observations.
- Multi-gigabyte raw EnergyPlus output directories and temporary solver files.
- Original provider files whose redistribution conditions are source-specific or whose archives are impractically large for Git.
- Manuscript or Supplementary Information files.

The absence of these provider-controlled or very large raw files does not prevent regeneration of the distributed result tables' published visualizations. It does mean that a repository run begins from the accepted derived simulation outputs rather than rerunning the physical solver.

Consult `SOURCE_AND_LICENSES.csv` and `../data/SOURCE_MANIFEST.csv` before redistributing third-party material.
