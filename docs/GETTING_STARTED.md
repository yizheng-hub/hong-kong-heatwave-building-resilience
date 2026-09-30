# Getting started

## Requirements

- Git and Git LFS
- Conda or Mamba
- A system capable of holding approximately 2 GB of working memory for the densest map panels

## Installation

```bash
git lfs install
git clone <repository-url>
cd hong-kong-heatwave-building-resilience
git lfs pull
conda env create -f environment.yml
conda activate hk-heatwave
python -m ipykernel install --user --name hk-heatwave --display-name "HK heatwave"
```

## Validate and run

```bash
python src/run_pipeline.py --check-only
python src/run_pipeline.py
```

The check verifies the eight event files, the 540-estate population comparison, all five annual-energy calibration groups, Hong Kong time formatting, and the complete figure-cell order. The full run clears stale PNGs from `output/figures/`, executes `plot/plot.ipynb`, requires exactly 23 new PNGs, refreshes the field dictionary and computes SHA-256 checksums for the redistributable repository files. Full-local inventory checks and accepted-result hashes cover the large local-only inputs and solver trees.

No EnergyPlus executable is called by this workflow.

## Complete local root

The author's complete local root also contains provider inputs, compiled inputs, accepted solver records, unpacked production results and the pinned EnergyPlus runtime. Verify that larger record with:

```bash
python src/run_pipeline.py --check-only --full-local --report docs/audit/UNIFIED_PROJECT_VALIDATION.json
```

This command still invokes EnergyPlus zero times. It additionally validates exact production hashes and the local content inventory. A GitHub clone is not expected to pass `--full-local` unless the local-only inputs and records listed in `docs/PROJECT_CONTENTS.csv` have been restored.
