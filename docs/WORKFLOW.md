# Workflow

1. **Frozen definitions** — `config/` records the eight assessment events, five weather/supply conditions, thermal thresholds and controlled-mechanism levels.
2. **Original source inputs** — `data/` is reserved for provider files or access manifests. It contains no generated analysis output.
3. **Prepared mappings and trajectories** — `intermediate/` links buildings to components and spatial units, stores prepared map layers and retains the accepted city trajectory needed by the peak diagnostic.
4. **Accepted building-event results** — `result/building_event_outcomes/` stores the final building-level outcomes for each assessment event. These files are inputs to the public analysis workflow, not newly simulated during a repository run.
5. **Analysis and validation** — `result/analysis/`, `result/validation/` and `src/analysis/metrics.py` define the distributed summaries and reusable estimators, including normalized population errors and tie-aware screening capture.
6. **Figures** — `plot/plot.ipynb` reads the included inputs and generates six main and seventeen supplementary PNG figures.
7. **Audit artifacts** — the workflow refreshes the data dictionary and repository checksums.

## Synchronized interruption clock

`HW_O24_SYNC` uses the event-wide `outage_common_24h` branch. For each event, the common start is the heatwave-period interval with the highest outdoor temperature in the shared event weather series. It is then applied to all designs. Prototype occupancy and cooling schedules do not select this common clock. Schedule-qualified peak selection belongs to a separate prototype-specific outage branch and is not the rule used for `HW_O24_SYNC`.

## Time support

The distributed high-frequency curves use interval-ending Hong Kong timestamps. A record labelled 03:00 therefore summarizes its preceding output interval; it is not an instantaneous measurement.
