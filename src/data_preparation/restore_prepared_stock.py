"""Prepare the active fixed stock from the audited current source chain.

The provider-derived rebuild supplies building attributes and the primary
land-use-constrained residential-GFA population allocation.  The accepted
fixed-stock table is used only to verify the immutable 330,761-building cohort
and to retain the already validated WRF cell links required by downstream
weather propagation.  No accepted population value is copied into the active
analysis inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


POPULATION_FIELD = "allocated_population__ssug_landuse_residential_gfa"
EXPECTED_COUNTS = {
    "Residential": 251_838,
    "Commercial": 49_110,
    "Mixed-use": 29_813,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accepted-fixed-stock",
        type=Path,
        default=root / "result/accepted/03_stock/fixed_stock_buildings.parquet",
    )
    parser.add_argument(
        "--rebuilt-all-sector",
        type=Path,
        default=root / "intermediate/verification/clean_chain/model_inputs/all_sector_building_model_inputs.parquet",
    )
    parser.add_argument(
        "--source-stock",
        type=Path,
        default=root / "intermediate/prepared_stock/source/building_stock_repaired.geoparquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "intermediate/prepared_stock",
    )
    args = parser.parse_args(argv)

    accepted = pd.read_parquet(args.accepted_fixed_stock)
    if len(accepted) != sum(EXPECTED_COUNTS.values()):
        raise ValueError(f"accepted fixed stock has {len(accepted):,} rows")
    if accepted["BUILDINGSTRUCTUREID"].duplicated().any():
        raise ValueError("accepted fixed stock has duplicate building IDs")
    counts = accepted["main_class"].value_counts().to_dict()
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"building-class counts differ from the frozen cohort: {counts}")

    source = pd.read_parquet(args.source_stock, columns=["BUILDINGSTRUCTUREID"])
    accepted_ids = set(accepted["BUILDINGSTRUCTUREID"])
    if not accepted_ids <= set(source["BUILDINGSTRUCTUREID"]):
        raise ValueError("accepted stock contains IDs absent from the provider-derived stock")

    rebuilt = pd.read_parquet(args.rebuilt_all_sector)
    required_rebuilt = {
        "BUILDINGSTRUCTUREID",
        "pre_weather_simulation_candidate",
        POPULATION_FIELD,
        "main_class",
        "sub_class",
        "geometry",
    }
    missing = sorted(required_rebuilt - set(rebuilt.columns))
    if missing:
        raise ValueError(f"rebuilt all-sector table is missing required fields: {missing}")
    candidate = rebuilt.loc[rebuilt["pre_weather_simulation_candidate"].fillna(False)].copy()
    rebuilt_ids = set(candidate["BUILDINGSTRUCTUREID"])
    if rebuilt_ids != accepted_ids:
        raise ValueError(
            "the independently rebuilt 330,761-building cohort differs from the accepted cohort"
        )
    rebuilt_counts = candidate["main_class"].value_counts().to_dict()
    if rebuilt_counts != EXPECTED_COUNTS:
        raise ValueError(f"rebuilt building-class counts differ from the frozen cohort: {rebuilt_counts}")
    if candidate["BUILDINGSTRUCTUREID"].duplicated().any():
        raise ValueError("rebuilt candidate stock has duplicate building IDs")
    if pd.to_numeric(candidate[POPULATION_FIELD], errors="coerce").isna().any():
        raise ValueError(f"primary population field has missing values: {POPULATION_FIELD}")

    population = candidate[["BUILDINGSTRUCTUREID", POPULATION_FIELD]].copy()
    population["population_reference_identity"] = (
        "current_primary_2021_ssug_landuse_residential_gfa"
    )

    map_columns = [
        "BUILDINGSTRUCTUREID",
        "main_class",
        "sub_class",
        "gfa_value_m2",
        "residential_gfa_m2",
        "commercial_gfa_m2",
        "floors_value",
        "residential_prototype_id",
        "commercial_prototype_id",
        "pre_weather_simulation_candidate",
        "wrf_south_north",
        "wrf_west_east",
    ]
    # Retain the accepted physical component areas, prototypes and WRF links.
    # This revision changes population allocation only; it must not introduce
    # a second physical-stock version through small source-rebuild differences.
    building_map = accepted[map_columns].copy()
    attributes = candidate.copy()

    output = args.output_dir.resolve()
    population_path = output / "population_reference.parquet"
    map_path = output / "building_hourly_component_map.parquet"
    attributes_path = output / "all_sector_building_model_inputs.parquet"
    atomic_parquet(population, population_path)
    atomic_parquet(building_map, map_path)
    atomic_parquet(attributes, attributes_path)

    provenance = {
        "status": "recomputed_from_current_official_source_chain",
        "raw_data_status": "provider inputs remain under data/ and were not replaced",
        "reason": (
            "the manuscript analysis was aligned to the reproducible primary population "
            "allocation declared by the current production chain"
        ),
        "independent_cohort_check": (
            "the provider-derived rebuild produced the same 330,761 building IDs and class counts"
        ),
        "population_scheme": "ssug_landuse_residential_gfa",
        "population_field": POPULATION_FIELD,
        "accepted_population_used": False,
        "accepted_fixed_stock": args.accepted_fixed_stock.resolve().relative_to(root).as_posix(),
        "accepted_fixed_stock_sha256": sha256(args.accepted_fixed_stock),
        "source_stock": args.source_stock.resolve().relative_to(root).as_posix(),
        "source_stock_sha256": sha256(args.source_stock),
        "rebuilt_all_sector": args.rebuilt_all_sector.resolve().relative_to(root).as_posix(),
        "rebuilt_all_sector_sha256": sha256(args.rebuilt_all_sector),
        "building_counts": EXPECTED_COUNTS,
        "population_sum": float(population[POPULATION_FIELD].sum()),
        "outputs": {
            path.resolve().relative_to(root).as_posix(): sha256(path)
            for path in (population_path, map_path, attributes_path)
        },
    }
    atomic_json(provenance, output / "population_reference_provenance.json")
    print(json.dumps(provenance, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
