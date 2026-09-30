# Hong Kong heatwave building resilience

This repository contains source-input manifests, accepted building-event results, analysis utilities and the editable plotting notebook used to generate the final scientific results and 23 static figures for a citywide assessment of heatwave-related electricity demand, cooling-service shortfalls, indoor overheating and recovery across Hong Kong's building stock.

The repository is self-contained for validating the distributed result tables and regenerating all six main and seventeen supplementary PNG figures. It does **not** rerun EnergyPlus. The accepted building-event and city-trajectory outputs are distributed as analysis inputs because the original simulator outputs are substantially larger and some observational comparison data are access-restricted.

## Quick start

Git and Git LFS are required.

```bash
git lfs install
conda env create -f environment.yml
conda activate hk-heatwave
python -m ipykernel install --user --name hk-heatwave --display-name "HK heatwave"
python src/run_pipeline.py
```

To validate the repository without redrawing the figures:

```bash
python src/run_pipeline.py --check-only
```

The full workflow writes 23 PNG files to `output/figures/`, refreshes `docs/DATA_DICTIONARY.csv`, and updates `SHA256SUMS.txt`. The editable notebook is `plot/plot.ipynb`; every figure has one complete executable cell containing data reads, transformations and styling.

## Repository layout

```text
config/        Frozen event, condition and evaluation definitions
data/          Original/external inputs or provider access manifests only
intermediate/  Prepared stock, spatial mappings and accepted trajectories
result/        Building-event outcomes, analysis tables and panel source data
output/        Final PNG figures
plot/          Fully editable figure notebook
src/           Input checks, analysis metrics and workflow entry point
docs/          Data dictionary, provenance, contracts and workflow notes
```

The manuscript and Supplementary Information are intentionally not part of this repository.

## Scientific scope

The distributed assessment covers eight complete historical heatwaves and five modeled conditions: a recent-weather reference with unconstrained cooling demand (`CF_UNL`), heatwave demand without a cooling-capacity constraint (`HW_UNL`), rated cooling capacity (`HW_RATED`), 70% rated total cooling capacity (`HW_C70`), and a synchronized 24-hour grid interruption (`HW_O24_SYNC`). The synchronized interruption starts at the heatwave-period interval with the highest outdoor temperature in the common event weather series; occupancy and prototype cooling schedules do not select this shared clock.

No generated or derived table is stored under `data/`. See `data/SOURCE_MANIFEST.csv` for source identities, `docs/WORKFLOW.md` for the calculation flow and `docs/DATA_AVAILABILITY.md` for the boundary between included records and provider-controlled inputs.

## Licensing

Repository code is released under the MIT License. Third-party data and derived records remain subject to the terms of their original providers; see `docs/SOURCE_AND_LICENSES.csv` before redistributing data.
