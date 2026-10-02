# -*- coding: utf-8 -*-
"""Build stable residential heat archetypes from the qualified foundation layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FOUNDATION = ROOT / "intermediate" / "prepared_stock" / "source"
DEFAULT_OUTPUT = ROOT / "intermediate" / "prepared_stock" / "archetypes"
ARCHETYPE_KEYS = [
    "residential_type",
    "era_class",
    "height_class",
    "morphology_class",
    "gfa_evidence_class",
]


def residential_type(category: pd.Series) -> pd.Series:
    return category.astype("string").map(
        {
            "2": "private_housing_landsd",
            "3": "public_housing_landsd",
        }
    ).fillna("other_direct_residential")


def era_class(year: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [
                year.lt(1980),
                year.lt(2000),
                year.lt(2010),
            ],
            ["pre_1980", "1980_1999", "2000_2009"],
            default="2010_plus",
        ),
        index=year.index,
    )


def height_class(floors: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [floors.le(6), floors.le(15), floors.le(30)],
            ["low_1_6", "mid_7_15", "high_16_30"],
            default="very_high_31_plus",
        ),
        index=floors.index,
    )


def morphology_class(frame: pd.DataFrame) -> pd.Series:
    compactness = (
        4 * np.pi * frame["footprint_area_m2"] / frame["perimeter_m"].pow(2)
    ).clip(0, 1)
    return pd.Series(
        np.select(
            [compactness.lt(0.35), compactness.lt(0.60)],
            ["articulated", "moderate"],
            default="compact",
        ),
        index=frame.index,
    )


def assign_stable_archetypes(
    frame: pd.DataFrame, minimum_group_size: int = 30
) -> pd.DataFrame:
    if len(frame) < minimum_group_size:
        raise ValueError(
            "The training stock is smaller than the requested minimum archetype size"
        )

    result = frame.copy()
    result["residential_type"] = residential_type(result["CATEGORY"])
    result["era_class"] = era_class(result["construction_year_value"])
    result["height_class"] = height_class(result["floors_value"])
    result["morphology_class"] = morphology_class(result)
    result["gfa_evidence_class"] = np.where(
        result["gfa_measurement_class"].isin(
            ["observed_building_record", "permit_constrained_allocation"]
        ),
        "observed_or_permit",
        "capacity_proxy",
    )
    result["archetype_pooling_level"] = 0
    pooling_order = [
        "morphology_class",
        "gfa_evidence_class",
        "era_class",
        "height_class",
    ]

    for housing_type in result["residential_type"].unique():
        housing_mask = result["residential_type"].eq(housing_type)
        for level, column in enumerate(pooling_order, start=1):
            group_sizes = result.loc[housing_mask].groupby(ARCHETYPE_KEYS).size()
            if group_sizes.min() >= minimum_group_size:
                break
            result.loc[housing_mask, column] = "pooled"
            result.loc[housing_mask, "archetype_pooling_level"] = level

    group_sizes = result.groupby(ARCHETYPE_KEYS).size()
    if group_sizes.min() < minimum_group_size:
        result.loc[:, ARCHETYPE_KEYS] = "pooled"
        result["archetype_pooling_level"] = len(pooling_order) + 1

    definitions = result[ARCHETYPE_KEYS].drop_duplicates().sort_values(ARCHETYPE_KEYS)
    definitions["archetype_id"] = [f"A{number:03d}" for number in range(1, len(definitions) + 1)]
    result = result.merge(definitions, on=ARCHETYPE_KEYS, how="left", validate="many_to_one")
    result["compactness"] = (
        4 * np.pi * result["footprint_area_m2"] / result["perimeter_m"].pow(2)
    ).clip(0, 1)
    result["wall_area_proxy_m2"] = result["perimeter_m"] * result["height_value_m"]
    result["wall_to_floor_ratio_proxy"] = (
        result["wall_area_proxy_m2"] / result["gfa_capacity_proxy_m2"]
    ).clip(0.1, 4.0)
    result["roof_to_floor_ratio_proxy"] = (
        result["footprint_area_m2"] / result["gfa_capacity_proxy_m2"]
    ).clip(0.01, 1.0)
    return result


def add_physics_priors(catalog: pd.DataFrame) -> pd.DataFrame:
    result = catalog.copy()
    year = result["median_construction_year"]
    result["u_wall_w_m2k"] = np.select(
        [year.lt(1980), year.lt(2000), year.lt(2010)],
        [2.4, 2.0, 1.6],
        default=1.3,
    )
    result["u_roof_w_m2k"] = np.select(
        [year.lt(1980), year.lt(2000), year.lt(2010)],
        [2.2, 1.8, 1.5],
        default=1.1,
    )
    result["window_to_wall_ratio"] = result["residential_type"].map(
        {
            "private_housing_landsd": 0.30,
            "public_housing_landsd": 0.25,
            "other_direct_residential": 0.28,
        }
    ).fillna(0.28)
    result["window_shgc"] = np.select(
        [year.lt(1980), year.lt(2000), year.lt(2010)],
        [0.75, 0.70, 0.65],
        default=0.55,
    )
    result["infiltration_ach"] = np.select(
        [year.lt(1980), year.lt(2000), year.lt(2010)],
        [1.00, 0.80, 0.65],
        default=0.50,
    )
    result["thermal_capacitance_kj_m2k"] = np.where(
        result["residential_type"].eq("public_housing_landsd"), 180.0, 150.0
    )
    result["baseline_ac_cop"] = np.select(
        [year.lt(1980), year.lt(2000), year.lt(2010)],
        [2.6, 2.8, 3.0],
        default=3.2,
    )
    result["physics_parameter_status"] = "literature_prior_not_building_calibrated"
    return result


def build_catalog(assignments: pd.DataFrame, core: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    training = assignments.groupby(["archetype_id", *ARCHETYPE_KEYS], as_index=False).agg(
        training_buildings=("BUILDINGSTRUCTUREID", "size"),
        median_floors=("floors_value", "median"),
        median_height_m=("height_value_m", "median"),
        median_construction_year=("construction_year_value", "median"),
        median_footprint_m2=("footprint_area_m2", "median"),
        median_compactness=("compactness", "median"),
        median_wall_to_floor_ratio=("wall_to_floor_ratio_proxy", "median"),
        median_roof_to_floor_ratio=("roof_to_floor_ratio_proxy", "median"),
        maximum_pooling_level=("archetype_pooling_level", "max"),
    )
    core_assignments = core[["BUILDINGSTRUCTUREID", "residential_gfa_estimate_m2"]].merge(
        assignments[["BUILDINGSTRUCTUREID", "archetype_id"]],
        on="BUILDINGSTRUCTUREID",
        how="left",
        validate="one_to_one",
    )
    stock = core_assignments.groupby("archetype_id", as_index=False).agg(
        core_buildings=("BUILDINGSTRUCTUREID", "size"),
        core_residential_gfa_m2=("residential_gfa_estimate_m2", "sum"),
        median_core_residential_gfa_m2=("residential_gfa_estimate_m2", "median"),
    )
    catalog = training.merge(stock, on="archetype_id", how="left", validate="one_to_one")
    catalog["core_buildings"] = catalog["core_buildings"].fillna(0).astype(int)
    catalog["core_residential_gfa_m2"] = catalog["core_residential_gfa_m2"].fillna(0)
    catalog["median_core_residential_gfa_m2"] = catalog[
        "median_core_residential_gfa_m2"
    ].fillna(0)
    catalog = add_physics_priors(catalog)
    simulation = catalog[catalog["core_buildings"].gt(0)].copy()
    simulation["core_gfa_weight"] = (
        simulation["core_residential_gfa_m2"]
        / simulation["core_residential_gfa_m2"].sum()
    )
    return catalog, simulation


def parameter_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["u_wall_w_m2k", "era prior", 0.65, 3.0, "sensitivity only; not measured"],
            ["u_roof_w_m2k", "era prior", 0.45, 2.6, "sensitivity only; not measured"],
            ["window_to_wall_ratio", "housing-type prior", 0.18, 0.40, "sensitivity only; not measured"],
            ["window_shgc", "era prior", 0.35, 0.80, "sensitivity only; not measured"],
            ["infiltration_ach", "era prior", 0.30, 1.40, "sensitivity only; not measured"],
            ["thermal_capacitance_kj_m2k", "construction prior", 100.0, 240.0, "sensitivity only; not measured"],
            ["baseline_ac_cop", "era prior", 2.2, 3.8, "scenario assumption; AC presence is unknown"],
        ],
        columns=[
            "parameter",
            "assignment_basis",
            "uncertainty_lower",
            "uncertainty_upper",
            "allowed_use_and_limitation",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation-dir", type=Path, default=DEFAULT_FOUNDATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--weather-validation",
        type=Path,
        default=ROOT / "intermediate" / "prepared_weather" / "weather_validation.json",
    )
    parser.add_argument("--minimum-group-size", type=int, default=30)
    args = parser.parse_args()
    foundation = args.foundation_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    training = gpd.read_parquet(foundation / "building_stock_archetype_training.geoparquet")
    core = gpd.read_parquet(foundation / "building_stock_core_simulation.geoparquet")
    assignments = assign_stable_archetypes(training, args.minimum_group_size)
    catalog, simulation = build_catalog(assignments, core)

    assignments[
        [
            "BUILDINGSTRUCTUREID",
            "archetype_id",
            *ARCHETYPE_KEYS,
            "archetype_pooling_level",
        ]
    ].to_csv(output / "building_archetype_assignments.csv", index=False)
    catalog.to_csv(output / "archetype_catalog_all.csv", index=False)
    simulation.to_csv(output / "archetype_catalog_simulation.csv", index=False)
    parameter_manifest().to_csv(output / "archetype_parameter_manifest.csv", index=False)

    group_sizes = assignments.groupby("archetype_id").size()
    assigned_core = core[["BUILDINGSTRUCTUREID"]].merge(
        assignments[["BUILDINGSTRUCTUREID", "archetype_id"]],
        on="BUILDINGSTRUCTUREID",
        how="left",
        validate="one_to_one",
    )
    core_area_expected = float(core["residential_gfa_estimate_m2"].sum())
    core_area_catalog = float(simulation["core_residential_gfa_m2"].sum())
    weather_validation = json.loads(
        args.weather_validation.resolve().read_text(encoding="utf-8")
    )
    foundation_summary = json.loads(
        (foundation / "audit_summary.json").read_text(encoding="utf-8")
    )
    validation = {
        "training_buildings": int(len(training)),
        "core_buildings": int(len(core)),
        "archetypes_all": int(len(catalog)),
        "archetypes_with_core_stock": int(len(simulation)),
        "minimum_training_group_size": int(group_sizes.min()),
        "core_buildings_with_assignment": int(assigned_core["archetype_id"].notna().sum()),
        "core_residential_gfa_expected_m2": core_area_expected,
        "core_residential_gfa_catalog_m2": core_area_catalog,
        "core_residential_gfa_conservation_error_m2": abs(
            core_area_expected - core_area_catalog
        ),
        "training_input_only_official_floors": bool(
            training["floors_source"].eq("landsd_official_storeys").all()
        ),
        "training_input_only_observed_years": bool(
            training["construction_year_source"].ne("spatial_quantile_model").all()
        ),
        "core_excludes_category_5_9": bool(
            ~core["CATEGORY"].astype(str).isin(["5", "9"]).any()
        ),
        "core_excludes_ml_only": bool(
            ~core["res_evidence_hkbfetd_ml_only"].fillna(False).any()
        ),
        "citywide_floor_propagation_ready": bool(
            foundation_summary["validation_gates"]["citywide_floor_propagation_ready"]
        ),
    }
    validation["gates"] = {
        "minimum_30_training_buildings_per_archetype": bool(
            validation["minimum_training_group_size"] >= args.minimum_group_size
        ),
        "all_core_buildings_assigned": bool(
            validation["core_buildings_with_assignment"] == validation["core_buildings"]
        ),
        "core_gfa_conservation_le_1e_6_m2": bool(
            validation["core_residential_gfa_conservation_error_m2"] <= 1e-6
        ),
        "official_floor_training_only": validation["training_input_only_official_floors"],
        "observed_year_training_only": validation["training_input_only_observed_years"],
        "core_excludes_category_5_9": validation["core_excludes_category_5_9"],
        "core_excludes_ml_only": validation["core_excludes_ml_only"],
    }
    validation["gates"]["day3_archetypes_ready"] = bool(
        all(validation["gates"].values())
    )
    (output / "archetype_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    readiness = {
        "foundation_ready": foundation_summary["validation_gates"][
            "day3_data_foundation_ready"
        ],
        "weather_ready": weather_validation["gates"]["day3_weather_ready"],
        "archetypes_ready": validation["gates"]["day3_archetypes_ready"],
        "citywide_floor_propagation_ready": validation[
            "citywide_floor_propagation_ready"
        ],
        "claim_scope": "core-stock reduced-order scenario simulation; no measured-energy calibration",
    }
    readiness["day3_ready_for_core_simulation"] = bool(
        readiness["foundation_ready"]
        and readiness["weather_ready"]
        and readiness["archetypes_ready"]
    )
    readiness["day5_citywide_deterministic_propagation_allowed"] = bool(
        readiness["day3_ready_for_core_simulation"]
        and readiness["citywide_floor_propagation_ready"]
    )
    (output / "day3_readiness.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"validation": validation, "readiness": readiness}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
