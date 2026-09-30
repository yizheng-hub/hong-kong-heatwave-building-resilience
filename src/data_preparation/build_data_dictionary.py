from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from paths import project_paths


EXACT_DESCRIPTIONS = {
    "BUILDINGSTRUCTUREID": "Stable building-structure identifier used for joins.",
    "event_id": "Stable identifier for a heatwave event.",
    "scenario_id": "Identifier for the weather and cooling-supply condition.",
    "timestamp_hkt": "Interval-ending timestamp in Hong Kong time (UTC+08:00).",
    "main_class": "Broad building-use class used for stock summaries.",
    "detailed_type": "Detailed building-use category.",
    "geometry": "Building or boundary geometry encoded by the source format.",
    "selected": "Whether the heatwave was selected for building assessment.",
    "assessment_role": "Public assessment role for the event; blank for non-selected catalogue events.",
    "thermal_support_state": "Completeness state for the building thermal comparison.",
}


def infer_unit(column: str) -> str:
    lower = column.lower()
    units = (
        ("person_degree_hours", "person K h"),
        ("population_degree_hours", "person K h"),
        ("kwh_thermal", "kWh_th"),
        ("kwh_m2_year", "kWh m-2 yr-1"),
        ("kwh", "kWh"),
        ("_gw", "GW"),
        ("_mw", "MW"),
        ("_kw", "kW"),
        ("power_w", "W"),
        ("w_m2", "W m-2"),
        ("gfa_m2", "m2"),
        ("area_m2", "m2"),
        ("_kh", "K h"),
        ("degree_hours", "K h"),
        ("_c_days", "degC d"),
        ("_c", "degC"),
        ("_pct", "%"),
        ("fraction", "fraction"),
        ("_h", "h"),
        ("_km", "km"),
    )
    for token, unit in units:
        if token in lower:
            return unit
    return ""


def infer_description(column: str) -> str:
    if column in EXACT_DESCRIPTIONS:
        return EXACT_DESCRIPTIONS[column]
    return column.replace("_", " ").strip().capitalize() + "."


def schema_for(path: Path) -> list[tuple[str, str]]:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, nrows=0)
        return [(column, str(dtype)) for column, dtype in frame.dtypes.items()]
    schema = pq.read_schema(path)
    return [(field.name, str(field.type)) for field in schema]


def main() -> None:
    paths = project_paths()
    records = []
    for parent in (paths.data, paths.intermediate, paths.result):
        for path in sorted(parent.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".csv", ".parquet", ".geoparquet"}:
                continue
            relative = path.relative_to(paths.root).as_posix()
            for column, dtype in schema_for(path):
                records.append(
                    {
                        "path": relative,
                        "column": column,
                        "data_type": dtype,
                        "unit": infer_unit(column),
                        "description": infer_description(column),
                    }
                )
    destination = paths.root / "docs" / "DATA_DICTIONARY.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "column", "data_type", "unit", "description"])
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} field records to {destination}")


if __name__ == "__main__":
    main()
