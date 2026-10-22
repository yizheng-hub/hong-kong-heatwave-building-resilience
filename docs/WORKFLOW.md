# Workflow

## Scientific data flow

1. **Frozen definitions** — `config/` records the eight assessment events, five weather/supply conditions, thermal thresholds, controlled-mechanism levels, contracts and current-root code locks.
2. **Original source inputs** — `data/` is reserved for provider files and source manifests. Generated tables are never written there.
3. **Source-adjacent preparation** — `intermediate/source_derived/` contains generated census mappings, address mappings, boundaries and prepared weather series. The five semantic stage scripts are under `src/workflows/stages/`.
4. **Compiled physical inputs** — `intermediate/compiled_simulation_inputs/` preserves the frozen lockbox and mechanism IDF/EPW/schedule inputs. Open-event compilation and retained upstream records are under `intermediate/simulation_outputs/upstream/`.
5. **Physical outputs** — accepted solver records are under `intermediate/simulation_outputs/`. The complete local root retains these records; GitHub omits them because of size.
6. **Accepted production outputs** — `result/accepted/` is the unpacked final completion tree verified against the original production receipt. `result/building_event_outcomes/`, `result/analysis/`, `result/validation/` and `result/figure_source_data/` are the compact manuscript-aligned analysis products.
7. **Figures** — `plot/plot.ipynb` reads the compact included inputs and generates six main and seventeen supplementary PNG figures. Every figure is implemented in one complete notebook cell.
8. **Audit artifacts** — validation checks result hashes, row counts, code locks, HKT formatting, figure-cell order and accepted PNG hashes. The full local mode also checks source, compiled-input, solver-output and runtime inventories.

## Reproduction levels

### Manuscript-aligned analysis and figures

```bash
python src/run_pipeline.py
```

This is the normal GitHub workflow. It invokes EnergyPlus zero times and regenerates all 23 PNG figures from accepted derived inputs.

### Complete local-root verification

```bash
python src/run_pipeline.py --check-only --full-local --report docs/audit/UNIFIED_PROJECT_VALIDATION.json
```

This also invokes EnergyPlus zero times. It verifies the full local scientific record, including exact hashes from the final production receipt and minimum inventories for original inputs, compiled physical inputs, solver outputs, unpacked accepted results and the pinned runtime.

### Source-to-analysis verification in clean directories

The stock, population context, archetype assignment and final pre-weather model
inputs can be rebuilt as an isolated chain:

```bash
python src/workflows/stages/prepare_building_stock.py --output-dir intermediate/verification/clean_chain/stock
python src/workflows/stages/prepare_population_context.py --foundation-dir intermediate/verification/clean_chain/stock --output-dir intermediate/verification/clean_chain/population
python src/workflows/stages/prepare_stock_archetypes.py --foundation-dir intermediate/verification/clean_chain/stock --output-dir intermediate/verification/clean_chain/archetypes
python src/workflows/stages/build_building_response_and_exposure.py --prepare-inputs-only --building-stock intermediate/verification/clean_chain/stock/building_stock_repaired.geoparquet --census-context intermediate/verification/clean_chain/population/residential_building_census_context.geoparquet --archetype-assignments intermediate/verification/clean_chain/archetypes/building_archetype_assignments.csv --archetype-catalog intermediate/verification/clean_chain/archetypes/archetype_catalog_simulation.csv --output-dir intermediate/verification/clean_chain/model_inputs
```

This chain is solver-free. It reproduces 341,153 source footprints, 283,296
residential/mixed-use contextual records and the accepted 330,761-building
candidate membership (251,838 residential, 49,110 commercial and 29,813
mixed-use buildings). The unresolved floor-model holdout gate remains reported
as failed and is not hidden by the successful cohort reproduction.

The complete post-processing run uses preserved physical outputs and a blank
result directory. First activate the current official-source population rebuild,
then run and promote the verified population-dependent outputs:

```bash
python src/data_preparation/restore_prepared_stock.py
python src/run_analysis.py --config config/analysis.json --output-dir result/population_revision_run
python src/validation/compare_results.py --generated result/population_revision_run --accepted result/accepted --report-dir result/population_revision_comparison
python src/data_preparation/promote_population_revision.py --source-root result/population_revision_run
```

The post-processing audit records zero EnergyPlus calls. The strict field-level
audit confirms that all non-population physical outcomes are unchanged; expected
differences are confined to population weights and their derived exposure and
screening quantities. Project-path and source-provenance fields may also differ
after consolidation.

### Physical rebuild

The physical workflow is retained for auditability and future reruns. Source preparation is implemented by the five scripts in `src/workflows/stages/`; eight-event compilation/execution is implemented by `src/utils/e8_open6_prepare.py`, `src/workflows/run_g4a_local.py`, `src/workflows/run_e8_local.py` and `src/workflows/run_e8_completion.py`. These entry points have explicit resource and approval gates. Consult each command's `--help` before use. A physical rebuild is not part of the default figure workflow and was not invoked during repository consolidation.

## Synchronized interruption clock

`HW_O24_SYNC` uses the event-wide `outage_common_24h` branch. For each event, the common start is the heatwave-period interval with the highest outdoor temperature in the shared event weather series. It is then applied to all designs. Prototype occupancy and cooling schedules do not select this common clock. Schedule-qualified peak selection belongs to a separate prototype-specific outage branch and is not the rule used for `HW_O24_SYNC`.

## Time support

The distributed high-frequency curves use interval-ending Hong Kong timestamps. A record labelled 03:00 therefore summarizes its preceding output interval; it is not an instantaneous measurement.
