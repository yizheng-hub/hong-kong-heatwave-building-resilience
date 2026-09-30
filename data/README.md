# Data: source inputs only

`data/` is reserved exclusively for original or externally supplied inputs. It never contains cleaned tables, joins, model results, panel source data or figure outputs.

The repository records every required source in `SOURCE_MANIFEST.csv`. A source is represented in one of two ways:

- `vendored`: an original provider file may be distributed with the repository under its source terms;
- `external` or `restricted`: the source remains with its provider and this directory contains an access note, expected file identity and citation instead of a copied archive.

Prepared boundaries and simplified roads used by the figures are under `../intermediate/spatial/`. Accepted simulation outcomes and panel data are under `../result/`; final PNG files are under `../output/`.

Provider identity and redistribution conditions are summarized in `../docs/SOURCE_AND_LICENSES.csv`. The repository's MIT licence applies to code, not to third-party data.
