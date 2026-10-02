# Hong Kong heatwave building resilience

This is the current project root for the citywide assessment of heatwave-related electricity demand, cooling-service shortfalls, indoor overheating and recovery across Hong Kong's building stock. The complete local folder contains the original/provider inputs, frozen configurations, current code, necessary intermediate files, accepted solver records, accepted results, panel data and all 23 final PNG figures.

The private GitHub repository is the redistributable mirror of this same project tree. It includes code, configuration, compact prepared inputs, manuscript-aligned result tables, panel data, the editable plotting notebook, final figures and the compact accepted-results archive. Large solver records, the EnergyPlus installation and provider-controlled original data remain in the complete local root and are intentionally ignored by Git; their identities and locations are documented in `data/SOURCE_MANIFEST.csv` and `docs/PROJECT_CONTENTS.csv`.

## Quick start

Git and Git LFS are required.

```bash
git lfs install
conda env create -f environment.yml
conda activate hk-heatwave
python -m ipykernel install --user --name hk-heatwave --display-name "HK heatwave"
python src/run_pipeline.py
```

To validate the GitHub-distributed analysis inputs without redrawing the figures:

```bash
python src/run_pipeline.py --check-only
```

Within the complete local root, use the stronger check below. It verifies the original-input inventory, compiled inputs, solver-output inventory, exact accepted-result hashes, current code locks and all accepted figure hashes without invoking EnergyPlus:

```bash
python src/run_pipeline.py --check-only --full-local --report docs/audit/UNIFIED_PROJECT_VALIDATION.json
```

The default workflow writes 23 PNG files to `output/figures/`, refreshes `docs/DATA_DICTIONARY.csv`, updates the redistributable-file manifest `SHA256SUMS.txt`, and reruns validation. Large local-only inputs and solver trees are checked through the full-local inventory and accepted-result hashes rather than rehashed on every figure run. The editable notebook is `plot/plot.ipynb`; every figure has one complete executable cell containing data reads, transformations and styling. This analysis/figure workflow invokes EnergyPlus zero times.

## Verified scientific rebuild

The source-to-analysis chain can be checked without running a new physical
simulation. Use separate verification directories so the accepted production
record remains immutable:

```bash
python src/workflows/stages/prepare_building_stock.py --output-dir intermediate/verification/clean_chain/stock
python src/workflows/stages/prepare_population_context.py --foundation-dir intermediate/verification/clean_chain/stock --output-dir intermediate/verification/clean_chain/population
python src/workflows/stages/prepare_stock_archetypes.py --foundation-dir intermediate/verification/clean_chain/stock --output-dir intermediate/verification/clean_chain/archetypes
python src/workflows/stages/build_building_response_and_exposure.py --prepare-inputs-only --building-stock intermediate/verification/clean_chain/stock/building_stock_repaired.geoparquet --census-context intermediate/verification/clean_chain/population/residential_building_census_context.geoparquet --archetype-assignments intermediate/verification/clean_chain/archetypes/building_archetype_assignments.csv --archetype-catalog intermediate/verification/clean_chain/archetypes/archetype_catalog_simulation.csv --output-dir intermediate/verification/clean_chain/model_inputs
python src/data_preparation/restore_prepared_stock.py
python src/run_analysis.py --config config/analysis.json --output-dir result/population_revision_run
python src/validation/compare_results.py --generated result/population_revision_run --accepted result/accepted --report-dir result/population_revision_comparison
python src/data_preparation/promote_population_revision.py --source-root result/population_revision_run
```

The fourth command exits before engine validation or simulation. The analysis
command has a zero-call EnergyPlus budget and consumes only preserved physical
outputs. `result/population_revision_comparison/comparison_summary.json` reports the
table-level comparison with the frozen accepted results.

The provider-derived rebuild reproduces the accepted 330,761 building IDs and
top-level class counts. The manuscript-aligned population is now rebuilt from
the primary `ssug_landuse_residential_gfa` allocation and activated by
`src/data_preparation/restore_prepared_stock.py`. It assigns 7,405,391.46
reference residents; across 540 estates, the normalized absolute and root-square
errors are 4.22% and 6.45% (Spearman's rho = 0.986). The previously retained
tenure-capacity allocation remains only in the frozen accepted archive for audit
comparison and is not used by the current compact manuscript results.

All active code and configuration use project-relative paths. Original
production receipts and preserved solver/compiled-input manifests retain their
historical absolute path strings as immutable provenance; those strings are not
resolved by the current workflow and are not rewritten because doing so would
invalidate the original evidence hashes.

## Repository layout

```text
config/        Frozen event, condition and evaluation definitions
data/          Original/provider inputs only; never generated analysis output
intermediate/  Prepared mappings, compiled inputs and accepted solver records
result/        Accepted scientific outputs, analysis tables and panel source data
output/        Final PNG figures
plot/          Fully editable figure notebook
software/      Pinned local EnergyPlus runtime (local only)
src/           Current preparation, simulation, analysis, validation and workflow code
docs/          Data dictionary, provenance, contracts, inventories and workflow notes
```

The manuscript and Supplementary Information are intentionally not part of this repository.

## Scientific scope

The assessment covers eight complete historical heatwaves and five modeled conditions: a recent-weather reference with unconstrained cooling demand (`CF_UNL`), heatwave demand without a cooling-capacity constraint (`HW_UNL`), rated cooling capacity (`HW_RATED`), 70% rated total cooling capacity (`HW_C70`), and a synchronized 24-hour grid interruption (`HW_O24_SYNC`). In filenames and code, `E8` identifies this eight-event experiment; it is not a software-version label. The synchronized interruption starts at the heatwave-period interval with the highest outdoor temperature in the common event weather series; occupancy and prototype cooling schedules do not select this shared clock.

No generated or derived table is stored under `data/`. See `data/SOURCE_MANIFEST.csv` for source identities, `docs/WORKFLOW.md` for the calculation flow, `docs/PROJECT_CONTENTS.csv` for the full local/GitHub boundary and `docs/DATA_AVAILABILITY.md` for redistribution conditions.

## Licensing

Repository code is released under the MIT License. Third-party data and derived records remain subject to the terms of their original providers; see `docs/SOURCE_AND_LICENSES.csv` before redistributing data.
