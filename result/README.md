# Results

- `building_event_outcomes/` contains one accepted building-level Parquet table for each of the eight assessment events.
- `analysis/` contains cross-event, city, recovery and screening summaries.
- `validation/` contains held-out comparison and sensitivity tables.
- `figure_source_data/` contains the compact tables read directly by the main-figure cells and selected supplementary-figure cells.
- `accepted/` contains the complete unpacked final production outputs in the local root; it is excluded from Git because of size.
- `archive/` contains the compact accepted eight-event review archive and checksum retained through Git LFS.

These are scientific outputs, not primary observations. Variable definitions are indexed in `../docs/DATA_DICTIONARY.csv`; panel-level use is listed in `../docs/FIGURE_SOURCE_DATA_CONTRACT.csv`.
