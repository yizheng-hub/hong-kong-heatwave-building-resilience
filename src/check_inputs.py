from __future__ import annotations

import json
import sys
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import nbformat
import pandas as pd
import pyarrow

from analysis.metrics import (
    normalized_absolute_error,
    normalized_root_square_error,
    spearman_rank_correlation,
)
from data_preparation.paths import project_paths
from simulation.conditions import EVENT_IDS
from validation.checks import require_columns, require_files


def main() -> int:
    paths = project_paths()
    figure_data = paths.figure_source_data
    event_directory = paths.result / "building_event_outcomes"
    required = [
        paths.notebook,
        figure_data / "Figure1a_heatwave_catalog.csv",
        figure_data / "Figure2_estate_population.csv",
        figure_data / "Figure3_2022_territory_buildings.csv",
        figure_data / "Figure_building_geometry.parquet",
        paths.intermediate / "prepared_stock" / "building_spatial_links.parquet",
        paths.intermediate / "prepared_stock" / "building_component_map.parquet",
        paths.intermediate / "spatial" / "boundaries" / "housing_market_areas.geoparquet",
        paths.intermediate / "spatial" / "roads" / "centerline_00000_simplified5m.geojson",
    ]
    required.extend(event_directory / f"{event_id}.parquet" for event_id in EVENT_IDS)
    require_files(required)

    require_columns(
        figure_data / "Figure2_estate_population.csv",
        {
            "official_population",
            "allocated_population__ssug_landuse_residential_gfa",
        },
    )
    population = pd.read_csv(figure_data / "Figure2_estate_population.csv")
    observed = population["official_population"]
    estimated = population["allocated_population__ssug_landuse_residential_gfa"]
    checks = {
        "estate_count": int(len(population)),
        "normalized_absolute_error_percent": normalized_absolute_error(observed, estimated),
        "normalized_root_square_error_percent": normalized_root_square_error(observed, estimated),
        "spearman_rho": spearman_rank_correlation(observed, estimated),
        "event_count": len(EVENT_IDS),
        "pyarrow": pyarrow.__version__,
        "python": sys.version.split()[0],
    }
    if checks["estate_count"] != 540:
        raise ValueError(f"Expected 540 estates, found {checks['estate_count']}.")
    if checks["normalized_absolute_error_percent"] >= 5:
        raise ValueError("Population-allocation normalized absolute error is outside the expected range.")
    if not 6.3 < checks["normalized_root_square_error_percent"] < 6.5:
        raise ValueError("Population-allocation normalized root-square error is outside the expected range.")

    annual_energy = pd.read_csv(figure_data / "Figure2_annual_energy.csv")
    expected_groups = {
        "commercial_sector_total",
        "private_housing",
        "public_housing",
        "subsidised_sale",
        "other_housing",
    }
    observed_groups = set(annual_energy["calibration_group"].astype(str))
    if observed_groups != expected_groups:
        raise ValueError(f"Annual-energy groups differ from the expected five groups: {observed_groups}")
    checks["annual_energy_groups"] = len(observed_groups)

    catalog = pd.read_csv(figure_data / "Figure1a_heatwave_catalog.csv")
    selected_events = set(catalog.loc[catalog["selected"].astype(bool), "event_id"].astype(str))
    if selected_events != set(EVENT_IDS):
        raise ValueError("Selected heatwave events do not match the configured assessment events.")

    timestamp = pd.Timestamp("2022-07-19 00:15:00", tz="Asia/Hong_Kong")
    formatter = mdates.DateFormatter("%d %b %H:%M", tz=ZoneInfo("Asia/Hong_Kong"))
    displayed = formatter(mdates.date2num(timestamp.to_pydatetime()))
    if displayed != "19 Jul 00:15":
        raise ValueError(f"Hong Kong time display check failed: {displayed}")
    checks["hkt_display_check"] = displayed

    notebook = nbformat.read(paths.notebook, as_version=4)
    labels = []
    for cell in notebook.cells[1:]:
        source = cell.source
        if "SUPPLEMENTARY FIGURE S" in source:
            labels.append(source.split("SUPPLEMENTARY FIGURE S", 1)[1].split()[0])
        elif "# FIGURE " in source:
            labels.append(source.split("# FIGURE ", 1)[1].split()[0])
    expected_labels = [str(number) for number in range(1, 7)] + [str(number) for number in range(1, 18)]
    if labels != expected_labels:
        raise ValueError(f"Notebook figure cells are out of order: {labels}")
    checks["figure_cells"] = len(labels)
    print(json.dumps(checks, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
