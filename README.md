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
