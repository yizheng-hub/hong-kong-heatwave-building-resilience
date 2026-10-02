"""Result 2: population exposure, detailed stock strata and hourly heat response.

This file deliberately keeps the related work together.  It (1) constructs
population allocations that conserve official census totals, (2) defines finer
stock strata without inventing building-level socioeconomic facts, and (3) runs
EnergyPlus reference-cell stress tests with temporal behaviour, humidity,
floor/orientation and loss-of-cooling scenarios.  It does not import a hidden
stage runner or another project simulation module.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from utils.building_composition import FLOOR_ROLES, floor_role_weight_table

CONFIG = json.loads((ROOT / "config" / "foundation_analysis.json").read_text(encoding="utf-8"))
THERMAL_CONFIG = CONFIG["thermal"]
POPULATION_CONFIG = CONFIG["population"]
RANDOM_SEED = int(CONFIG["random_seed"])
OUT = ROOT / CONFIG["output_root"] / "02_building_response"
BUILDING_STOCK = ROOT / "intermediate" / "prepared_stock" / "source" / "building_stock_repaired.geoparquet"
CENSUS_CONTEXT = ROOT / "intermediate" / "prepared_stock" / "population" / "residential_building_census_context.geoparquet"
OLD_ASSIGNMENTS = ROOT / "intermediate" / "prepared_stock" / "archetypes" / "building_archetype_assignments.csv"
OLD_CATALOG = ROOT / "intermediate" / "prepared_stock" / "archetypes" / "archetype_catalog_simulation.csv"
WEATHER = ROOT / "intermediate" / "prepared_weather" / "heatwave_hourly_bias_corrected.csv"
BG_CENSUS = ROOT / "data" / "official" / "census2021_HMA_BG_21C" / "HMA_BG_21C" / "BuildingGroup.CSV"
BG_MEMBERSHIP = ROOT / "intermediate" / "source_derived" / "census2021_building_group_members.parquet"
ALS_BUILDINGS = ROOT / "intermediate" / "source_derived" / "als_building_addresses_2026_08_13.parquet"
HMA_BOUNDARIES = ROOT / "intermediate" / "source_derived" / "census2021_housing_market_areas.geoparquet"
SSUG_BOUNDARIES = ROOT / "intermediate" / "source_derived" / "census2021_small_subunit_groups.geoparquet"
LAND_USE_TIFF = ROOT / "data" / "official" / "land_utilization_2021" / "LUMHK_RasterGrid_2021" / "LUM_end2021.tif"
ENGINE = ROOT / "software" / "energyplus" / "EnergyPlus-26.1.0-6f2e40d102-Windows-x86_64" / "energyplus.exe"
ENGINE_METADATA = ROOT / "software" / "energyplus" / "energyplus_release.json"
POLYU_CAMPUS_ARCHIVE = ROOT / "data" / "polyu_operational" / "campus_building_hourly_mapping_v22_20250709_20260709_v1.zip"
POLYU_HOURLY_WEATHER = ROOT / "intermediate" / "source_derived" / "weather" / "polyu_hourly_weather_hko_anchored.parquet"
HKUST_ARCHIVE = ROOT / "data" / "external_validation_raw" / "hkust" / "All_Data.zip"
HKUST_WEATHER = ROOT / "data" / "external_validation_raw" / "hkust" / "nasa_power_sai_kung_20220101_20240527.csv"
HKUST_HIERARCHY = ROOT / CONFIG["output_root"] / "00_audit" / "hkust_meter_hierarchy_audit.csv"
OFFICIAL_ENERGY_TARGETS = ROOT / "config" / "official_energy_calibration_2023.csv"
OFFICIAL_SYSTEM_PEAK = ROOT / "config" / "official_system_peak_upper_bound_2023.csv"
OFFICIAL_MONTHLY_ENERGY = ROOT / "intermediate" / "source_derived" / "censtat_monthly_domestic_electricity.csv"
HKO_DAILY_CACHE = ROOT / "data" / "weather" / "hko_open_data_daily"
WRF_BUILDING_MAPPING = ROOT / CONFIG["output_root"] / "01_hazard" / "building_wrf_land_grid_mapping.parquet"
WRF_BUILDING_HAZARD = ROOT / CONFIG["output_root"] / "01_hazard" / "building_heat_hazard_wrf1km.parquet"
FUTURE_TMAX_FIELD = ROOT / CONFIG["output_root"] / "01_hazard" / "cmip6_wrf_multimodel_tmax_change_field.parquet"
FUTURE_TMIN_FIELD = ROOT / CONFIG["output_root"] / "01_hazard" / "cmip6_wrf_multimodel_tmin_change_field.parquet"
FUTURE_COMPOUND_FACTORS = ROOT / CONFIG["output_root"] / "01_hazard" / "cmip6_compound_change_factors.csv"

CELL_HEIGHT_M = float(THERMAL_CONFIG["reference_cell_height_m"])
MODEL_VERSION = "energyplus_multizone_counterfactual_resilience_v0_23"
SIMULATION_IMPLEMENTATION_REVISION = "explicit_equipment_radiant_fraction_and_common_outage_v11"
ENERGY_CALIBRATION_VERSION = "emsd2023_component_mass_balance_monthly_holdout_v2"
BCA_BENCHMARK = ROOT / "data" / "external_validation_raw" / "bca_singapore" / "bca-bebr-data.xlsx"
CU_BEMS_ARCHIVE = ROOT / "data" / "external_validation_raw" / "cubems" / "CU-BEMS_v6.zip"
CU_BEMS_WEATHER = ROOT / "data" / "external_validation_raw" / "cubems" / "nasa_power_bangkok_20180701_20191231.csv"
NIST_DIR = ROOT / "data" / "external_validation_raw" / "nist"


def numeric(series: pd.Series, lower: float | None = None) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.clip(lower=lower) if lower is not None else values


def english_label(series: pd.Series) -> pd.Series:
    return series.astype("string").str.split("_", n=1).str[0].str.strip()


def residential_prototype(frame: pd.DataFrame) -> pd.Series:
    sub = english_label(frame["hkbfetd_sub_class"]).str.lower()
    program = frame.get("ha_housing_program", pd.Series("", index=frame.index)).fillna("").astype(str).str.lower()
    zone = frame.get("ZONE_LABEL", pd.Series("", index=frame.index)).fillna("").astype(str).str.upper()
    floors = numeric(frame["floors_value"])
    year = numeric(frame["construction_year_value"])
    result = pd.Series("res_private_highrise", index=frame.index, dtype="object")
    result.loc[sub.str.contains("public housing", na=False) | program.eq("public_rental_housing")] = "res_public_rental"
    result.loc[sub.str.contains("subsid", na=False) | program.eq("hos_psps_gsh")] = "res_subsidised_sale"
    result.loc[sub.str.contains("village|rural|small house", regex=True, na=False) | zone.eq("V")] = "res_village_house"
    old_lowrise = floors.le(6) & year.lt(1980) & ~result.eq("res_village_house")
    result.loc[old_lowrise] = "res_old_lowrise_tenement"
    return result


def commercial_prototype(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    raw_sub = frame["hkbfetd_sub_class"].fillna("").astype("string").str.lower()
    sub = english_label(frame["hkbfetd_sub_class"]).str.lower()
    main = english_label(frame["hkbfetd_main_class"])
    result = pd.Series("com_other", index=frame.index, dtype="object")
    evidence = pd.Series("generic_or_unresolved", index=frame.index, dtype="object")
    mixed_component = main.eq("Mixed-use")
    result.loc[mixed_component] = "com_mixed_use_component"
    evidence.loc[mixed_component] = "mixed_use_nonresidential_component_use_unresolved"
    rules = [
        ("office", "com_office"),
        ("retail", "com_retail"),
        ("restaurant|food & beverage", "com_restaurant"),
        ("accommodation|hotel", "com_accommodation"),
        ("education", "com_education"),
        ("human health|healthcare|hospital", "com_human_health"),
        ("data centre|data center", "com_data_centre"),
    ]
    named_matches = {}
    for pattern, prototype_id in rules:
        matched_anywhere = raw_sub.str.contains(pattern, regex=True, na=False)
        named_matches[prototype_id] = matched_anywhere
        matched = matched_anywhere & ~mixed_component
        result.loc[matched] = prototype_id
        evidence.loc[matched] = "named_use_in_hkbfetd_subclass"
    mixed_match_count = sum(
        matched.astype("int8") for matched in named_matches.values()
    )
    for prototype_id, matched in named_matches.items():
        resolved_mixed = mixed_component & matched & mixed_match_count.eq(1)
        result.loc[resolved_mixed] = prototype_id
        evidence.loc[resolved_mixed] = "single_named_nonresidential_use_in_mixed_subclass"
    evidence.loc[mixed_component & mixed_match_count.gt(1)] = (
        "multiple_named_nonresidential_uses_without_component_area_shares"
    )
    return result, evidence


def official_energy_segment(prototype_id: str) -> str:
    mapping = {
        "res_public_rental": "public_housing",
        "res_subsidised_sale": "subsidised_sale",
        "res_private_highrise": "private_housing",
        "res_old_lowrise_tenement": "private_housing",
        "res_village_house": "other_housing",
        "com_office": "office",
        "com_retail": "retail",
        "com_restaurant": "restaurant",
        "com_accommodation": "accommodation",
        "com_education": "education",
        "com_human_health": "human_health",
        "com_data_centre": "data_centre",
        "com_mixed_use_component": "other_commercial",
        "com_other": "other_commercial",
    }
    if prototype_id not in mapping:
        raise KeyError(f"No EMSD energy segment for {prototype_id}")
    return mapping[prototype_id]


def build_all_sector_model_inputs(
    stock: gpd.GeoDataFrame,
    residential_exposure: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    frame = stock.copy()
    frame["main_class"] = english_label(frame["hkbfetd_main_class"])
    frame["sub_class"] = english_label(frame["hkbfetd_sub_class"])
    target_classes = set(CONFIG["study_scope"]["modeled_main_classes"])
    frame = frame.loc[frame["main_class"].isin(target_classes)].copy()
    frame["residential_component_present"] = frame["main_class"].isin(["Residential", "Mixed-use"])
    frame["commercial_component_present"] = frame["main_class"].isin(["Commercial", "Mixed-use"])
    frame["residential_gfa_m2"] = numeric(frame["residential_gfa_estimate_m2"], lower=0).fillna(0.0)
    frame["commercial_gfa_m2"] = numeric(frame["nonresidential_gfa_estimate_m2"], lower=0).fillna(0.0)
    commercial_id, commercial_evidence = commercial_prototype(frame)
    frame["commercial_prototype_id"] = commercial_id.where(frame["commercial_component_present"])
    frame["commercial_subclass_evidence"] = commercial_evidence.where(frame["commercial_component_present"])

    geometry_valid = frame.geometry.notna() & ~frame.geometry.is_empty & frame.geometry.is_valid
    floors_valid = numeric(frame["floors_value"]).between(1, 120)
    height_valid = numeric(frame["height_value_m"]).between(2, 500)
    gfa_valid = numeric(frame["gfa_value_m2"]).gt(0)
    mixed = frame["main_class"].eq("Mixed-use")
    mixed_share_supported = ~mixed | numeric(frame["hkbfetd_residential_value"]).gt(0)
    component_valid = ~mixed | (
        frame["residential_gfa_m2"].gt(0)
        & frame["commercial_gfa_m2"].gt(0)
        & mixed_share_supported
    )
    frame["qc_geometry_valid"] = geometry_valid
    frame["qc_floor_count_valid"] = floors_valid
    frame["qc_height_valid"] = height_valid
    frame["qc_gfa_valid"] = gfa_valid
    frame["qc_component_area_valid"] = component_valid
    frame["pre_weather_simulation_candidate"] = (
        geometry_valid & floors_valid & height_valid & gfa_valid & component_valid
    )
    frame["population_assignment_scope"] = np.where(
        frame["residential_component_present"], "residential_component_only", "not_applicable"
    )
    frame["commercial_presence_basis"] = np.where(
        frame["commercial_component_present"],
        "use_specific_design_occupancy_density_and_hourly_schedule",
        "not_applicable",
    )

    population_columns = [
        column
        for column in residential_exposure.columns
        if column.startswith("allocated_population__") or column.startswith("allocated_age65plus__")
    ]
    population_context_columns = [
        column
        for column in [
            "ssug_code", "housing_class", "housing_tenure_class", "ha_link_status",
            "ha_housing_program", "ha_estate_name_en", "ha_estate_name_tc",
            "population_allocation_sensitivity_median",
            "population_allocation_sensitivity_p05",
            "population_allocation_sensitivity_p95",
        ]
        if column in residential_exposure.columns
    ]
    population_link = residential_exposure[
        ["BUILDINGSTRUCTUREID", *population_context_columns, *population_columns]
    ].copy()
    frame = frame.merge(population_link, on="BUILDINGSTRUCTUREID", how="left", validate="one_to_one")
    frame["residential_prototype_id"] = residential_prototype(frame).where(
        frame["residential_component_present"]
    )
    for column in population_columns:
        frame[column] = numeric(frame[column]).fillna(0.0)

    prototype_ids = set(pd.DataFrame(CONFIG["prototypes"])["prototype_id"])
    assigned_ids = set(frame["residential_prototype_id"].dropna()) | set(frame["commercial_prototype_id"].dropna())
    unknown_ids = sorted(assigned_ids - prototype_ids)
    if unknown_ids:
        raise ValueError(f"Prototype mapping produced IDs absent from config: {unknown_ids}")

    coverage_parts = []
    for component, present_column, prototype_column, area_column in [
        ("residential", "residential_component_present", "residential_prototype_id", "residential_gfa_m2"),
        ("commercial", "commercial_component_present", "commercial_prototype_id", "commercial_gfa_m2"),
    ]:
        subset = frame.loc[frame[present_column]].copy()
        summary = subset.groupby(prototype_column, dropna=False).agg(
            building_components=("BUILDINGSTRUCTUREID", "size"),
            component_gfa_m2=(area_column, "sum"),
            pre_weather_candidates=("pre_weather_simulation_candidate", "sum"),
        ).reset_index().rename(columns={prototype_column: "prototype_id"})
        summary.insert(0, "component", component)
        coverage_parts.append(summary)
    coverage = pd.concat(coverage_parts, ignore_index=True)
    return gpd.GeoDataFrame(frame, geometry="geometry", crs=stock.crs), coverage


def write_final_sample_audit(inputs: gpd.GeoDataFrame, output: Path) -> dict:
    """Freeze mutually consistent building and component counts for the paper."""
    candidate = inputs["pre_weather_simulation_candidate"].astype(bool)
    by_type = inputs.groupby("main_class", observed=True).agg(
        target_scope_buildings=("BUILDINGSTRUCTUREID", "nunique"),
        candidate_buildings=("pre_weather_simulation_candidate", "sum"),
        target_scope_gfa_m2=("gfa_value_m2", "sum"),
    ).reset_index()
    by_type["excluded_buildings"] = (
        by_type["target_scope_buildings"] - by_type["candidate_buildings"]
    )
    candidate_gfa = (
        inputs.loc[candidate].groupby("main_class", observed=True)["gfa_value_m2"].sum()
    )
    by_type["candidate_gfa_m2"] = by_type["main_class"].map(candidate_gfa).fillna(0.0)
    by_type.to_csv(output / "final_building_sample_by_type.csv", index=False, encoding="utf-8-sig")

    component_rows = []
    for component, present, prototype, area in [
        ("residential", "residential_component_present", "residential_prototype_id", "residential_gfa_m2"),
        ("commercial", "commercial_component_present", "commercial_prototype_id", "commercial_gfa_m2"),
    ]:
        subset = inputs.loc[inputs[present] & inputs[area].gt(0)].copy()
        grouped = subset.groupby(prototype, observed=True).agg(
            target_scope_building_components=("BUILDINGSTRUCTUREID", "nunique"),
            target_scope_component_gfa_m2=(area, "sum"),
            candidate_building_components=("pre_weather_simulation_candidate", "sum"),
        ).reset_index().rename(columns={prototype: "prototype_id"})
        candidate_area = subset.loc[subset["pre_weather_simulation_candidate"]].groupby(
            prototype, observed=True
        )[area].sum()
        grouped["candidate_component_gfa_m2"] = grouped["prototype_id"].map(candidate_area).fillna(0.0)
        grouped.insert(0, "component", component)
        component_rows.append(grouped)
    component_table = pd.concat(component_rows, ignore_index=True)
    component_table.to_csv(
        output / "final_component_sample_by_prototype.csv", index=False, encoding="utf-8-sig"
    )

    reason_order = [
        ("invalid_geometry", "qc_geometry_valid"),
        ("invalid_floor_count", "qc_floor_count_valid"),
        ("invalid_height", "qc_height_valid"),
        ("invalid_gfa", "qc_gfa_valid"),
        ("invalid_mixed_component_area", "qc_component_area_valid"),
    ]
    remaining = ~candidate
    reason_rows = []
    for reason, field in reason_order:
        selected = remaining & ~inputs[field].astype(bool)
        reason_rows.append({"primary_exclusion_reason": reason, "buildings": int(selected.sum())})
        remaining &= ~selected
    if remaining.any():
        reason_rows.append({"primary_exclusion_reason": "other", "buildings": int(remaining.sum())})
    exclusion = pd.DataFrame(reason_rows)
    exclusion.to_csv(output / "final_building_exclusion_reasons.csv", index=False, encoding="utf-8-sig")

    candidate_counts = inputs.loc[candidate, "main_class"].value_counts().to_dict()
    audit = {
        "target_scope_buildings_before_qc": int(inputs["BUILDINGSTRUCTUREID"].nunique()),
        "final_pre_weather_candidate_buildings": int(candidate.sum()),
        "excluded_buildings": int((~candidate).sum()),
        "candidate_buildings_by_top_level_type": {
            key: int(candidate_counts.get(key, 0)) for key in ["Residential", "Commercial", "Mixed-use"]
        },
        "mixed_use_is_one_physical_building_with_two_simulated_components": True,
        "component_counts_are_not_unique_building_counts": True,
        "building_table": "final_building_sample_by_type.csv",
        "component_table": "final_component_sample_by_prototype.csv",
        "exclusion_table": "final_building_exclusion_reasons.csv",
    }
    (output / "final_building_sample_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def housing_class(frame: pd.DataFrame) -> pd.Series:
    sub = frame.get("hkbfetd_sub_class", pd.Series("", index=frame.index)).fillna("").astype(str).str.lower()
    ha = frame.get("ha_link_status", pd.Series("", index=frame.index)).fillna("").astype(str).str.lower()
    zone = frame.get("ZONE_LABEL", pd.Series("", index=frame.index)).fillna("").astype(str).str.upper()
    result = pd.Series("other_residential", index=frame.index, dtype="object")
    result.loc[sub.str.contains("private") & ~sub.str.contains("public")] = "private_housing"
    result.loc[sub.str.contains("village|rural|small house", regex=True) | zone.eq("V")] = "village_house"
    result.loc[sub.str.contains("public|subsid") | ha.str.contains("matched")] = "public_or_subsidised"
    return result


def height_class(floors: pd.Series) -> pd.Series:
    return pd.cut(
        numeric(floors),
        bins=[-np.inf, 6, 15, 30, np.inf],
        labels=["low_1_6", "mid_7_15", "high_16_30", "very_high_31_plus"],
    ).astype("string").fillna("unknown")


def era_class(year: pd.Series) -> pd.Series:
    return pd.cut(
        numeric(year),
        bins=[-np.inf, 1979, 1999, np.inf],
        labels=["pre_1980", "1980_1999", "2000_plus"],
    ).astype("string").fillna("unknown")


def safe_normalize(weights: pd.Series) -> pd.Series:
    values = numeric(weights, lower=0).fillna(0.0)
    total = float(values.sum())
    return values / total if total > 0 else pd.Series(1.0 / len(values), index=values.index)


def first_numeric_or_zero(series: pd.Series) -> float:
    values = numeric(series).dropna()
    return float(values.iloc[0]) if len(values) else 0.0


def normalize_building_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).upper().replace("&", "AND")
    substitutions = {
        r"\bBLK\b": "BLOCK",
        r"\bHSE\b": "HOUSE",
        r"\bBLDG\b": "BUILDING",
        r"\bBLDGS\b": "BUILDINGS",
        r"\bCTR\b": "CENTRE",
        r"\bCTRE\b": "CENTRE",
        r"\bGDNS\b": "GARDENS",
        r"\bAPT\b": "APARTMENT",
        r"\bAPTS\b": "APARTMENTS",
    }
    for pattern, replacement in substitutions.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"[^0-9A-Z\u3400-\u9FFF]", "", text)
    return text.replace("BLK", "BLOCK")


def strip_parenthetical_block(value: object) -> str:
    return re.sub(
        r"\s*\((?:BLK|BLOCK|TOWER|座|第?\s*\d+\s*座)[^)]*\)\s*",
        "",
        str(value or ""),
        flags=re.IGNORECASE,
    ).strip()


def build_als_aliases(als: pd.DataFrame, hma_boundaries: gpd.GeoDataFrame) -> pd.DataFrame:
    points = gpd.GeoDataFrame(
        als.copy(),
        geometry=gpd.points_from_xy(als["easting_m"], als["northing_m"]),
        crs="EPSG:2326",
    )
    points = gpd.sjoin(
        points,
        hma_boundaries[["hma", "geometry"]].to_crs(2326),
        how="left",
        predicate="within",
    ).drop(columns="index_right")
    aliases: list[dict] = []
    for row in points.itertuples(index=False):
        if pd.isna(row.hma):
            continue
        for language in ("en", "tc"):
            building = str(getattr(row, f"building_name_{language}") or "").strip()
            estate = str(getattr(row, f"estate_name_{language}") or "").strip()
            descriptor = str(getattr(row, f"block_descriptor_{language}") or "").strip()
            block_number = str(getattr(row, f"block_no_{language}") or "").strip()
            building_base = strip_parenthetical_block(building)
            candidates = {
                building: ("building_name", 4),
                estate: ("estate_name", 2),
                estate + building: ("estate_building", 5),
                building + estate: ("building_estate", 5),
                estate + building_base: ("estate_building_base", 5),
                building_base + estate: ("building_base_estate", 5),
            }
            if estate and block_number:
                candidates.update(
                    {
                        estate + descriptor + block_number: ("estate_block", 5),
                        estate + block_number + descriptor: ("estate_block_reversed", 5),
                        estate + "BLOCK" + block_number: ("estate_block_canonical", 5),
                        estate + "TOWER" + block_number: ("estate_tower_canonical", 5),
                        estate + block_number: ("estate_block_number", 4),
                    }
                )
            for alias, (kind, score) in candidates.items():
                key = normalize_building_name(alias)
                if key:
                    aliases.append(
                        {
                            "building_csuid": str(row.building_csuid),
                            "hma": str(row.hma),
                            "language": language,
                            "name_key": key,
                            "alias_kind": kind,
                            "alias_score": score,
                        }
                    )
                    # The Centamap member field is visibly truncated for some
                    # long names.  Accept only 1--3 missing terminal characters
                    # from a specific building/block alias; collisions are
                    # removed later through the cross-group CSUID check.
                    if score >= 4 and len(key) >= 12:
                        for missing_characters in (1, 2, 3):
                            aliases.append(
                                {
                                    "building_csuid": str(row.building_csuid),
                                    "hma": str(row.hma),
                                    "language": language,
                                    "name_key": key[:-missing_characters],
                                    "alias_kind": f"terminal_truncation_{missing_characters}",
                                    "alias_score": 3,
                                }
                            )
    return pd.DataFrame(aliases).drop_duplicates()


def building_group_major_building_tokens(groups: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for row in groups.itertuples(index=False):
        names = {
            "en": [name.strip() for name in str(row.major_building_names_en or "").split(",") if name.strip()],
            "tc": [name.strip() for name in str(row.major_building_names_tc or "").split(",") if name.strip()],
        }
        language = "tc" if len(names["tc"]) > len(names["en"]) else "en"
        if not names[language]:
            language = "tc" if names["tc"] else "en"
        for position, name in enumerate(names[language], start=1):
            records.append(
                {
                    "bg": row.bg,
                    "hma": row.hma,
                    "official_population": float(row.official_population),
                    "language": language,
                    "major_building_position": position,
                    "major_building_name": name,
                    "name_key": normalize_building_name(name),
                }
            )
    tokens = pd.DataFrame(records).drop_duplicates(["bg", "language", "name_key"])
    if tokens["name_key"].eq("").any():
        raise ValueError("Empty normalized Building Group major-building name")
    return tokens


def link_building_groups(
    context: gpd.GeoDataFrame,
    groups: pd.DataFrame,
    als: pd.DataFrame,
    hma_boundaries: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    aliases = build_als_aliases(als, hma_boundaries)
    tokens = building_group_major_building_tokens(groups)
    matches = tokens.merge(aliases, on=["hma", "language", "name_key"], how="left")
    context_csuids = set(context["BUILDINGCSUID"].dropna().astype(str))
    matches = matches.loc[matches["building_csuid"].astype(str).isin(context_csuids)].copy()
    best_score = matches.groupby(["bg", "language", "name_key"])["alias_score"].transform("max")
    matches = matches.loc[matches["alias_score"].eq(best_score)].copy()

    # A listed major building can describe a complete estate and therefore map to
    # several CSUIDs.  A CSUID appearing in more than one Building Group is not
    # uniquely attributable and is excluded from the validation crosswalk.
    group_count = matches.groupby("building_csuid")["bg"].nunique()
    ambiguous_csuids = set(group_count[group_count > 1].index)
    matches["csuid_unique_to_group"] = ~matches["building_csuid"].isin(ambiguous_csuids)
    unique_matches = matches.loc[matches["csuid_unique_to_group"]].copy()

    total_tokens = tokens.groupby("bg").size().rename("listed_major_buildings")
    resolved_tokens = (
        unique_matches[["bg", "language", "name_key"]]
        .drop_duplicates()
        .groupby("bg")
        .size()
        .rename("resolved_major_buildings")
    )
    group_audit = groups[["bg", "hma", "official_population"]].merge(
        pd.concat([total_tokens, resolved_tokens], axis=1).fillna(0).reset_index(),
        on="bg",
        how="left",
        validate="one_to_one",
    )
    group_audit["resolved_major_buildings"] = group_audit["resolved_major_buildings"].fillna(0).astype(int)
    group_audit["listed_major_buildings"] = group_audit["listed_major_buildings"].fillna(0).astype(int)
    group_audit["listed_major_building_coverage"] = group_audit["resolved_major_buildings"] / group_audit["listed_major_buildings"]
    group_audit["all_listed_major_buildings_linked"] = group_audit["resolved_major_buildings"].eq(group_audit["listed_major_buildings"])
    fully_linked_groups = set(group_audit.loc[group_audit["all_listed_major_buildings_linked"], "bg"])
    linkage = (
        unique_matches[["bg", "hma", "building_csuid", "alias_kind"]]
        .drop_duplicates(["bg", "building_csuid"])
        .sort_values(["bg", "building_csuid"])
        .reset_index(drop=True)
    )
    audit = {
        "official_building_groups": int(len(groups)),
        "groups_with_any_major_building_link": int(unique_matches["bg"].nunique()),
        "groups_with_all_listed_major_buildings_linked": int(len(fully_linked_groups)),
        "all_listed_major_buildings_linked_group_share": float(len(fully_linked_groups) / len(groups)),
        "unique_linked_building_csuids": int(linkage["building_csuid"].nunique()),
        "ambiguous_building_csuids_excluded": int(len(ambiguous_csuids)),
        "spatial_constraint": "ALS address point must fall within the official 2021 HMA polygon",
        "interpretation": "Centamap lists major buildings only; this audit cannot establish exhaustive Building Group membership and cannot validate total Building Group population.",
    }
    return linkage, group_audit, audit


def population_error_metrics(evaluation: pd.DataFrame, schemes: list[str]) -> pd.DataFrame:
    population_columns = [f"allocated_population__{scheme}" for scheme in schemes]
    metric_rows = []
    observed = evaluation["official_population"].to_numpy(float)
    for scheme, column in zip(schemes, population_columns):
        predicted = evaluation[column].to_numpy(float)
        residual = predicted - observed
        metric_rows.append(
            {
                "scheme": scheme,
                "validation_units": len(evaluation),
                "observed_population": float(observed.sum()),
                "predicted_population": float(predicted.sum()),
                "mean_error_persons": float(residual.mean()),
                "mae_persons": float(np.abs(residual).mean()),
                "rmse_persons": float(np.sqrt(np.mean(residual**2))),
                "relative_mae_pct": float(100 * np.abs(residual).sum() / observed.sum()),
                "relative_rmse_pct": float(100 * np.sqrt(np.sum(residual**2) / np.sum(observed**2))),
                "spearman_r": float(pd.Series(observed).corr(pd.Series(predicted), method="spearman")),
            }
        )
    return pd.DataFrame(metric_rows)


def major_housing_estate_population_validation(
    frame: pd.DataFrame, schemes: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    population_columns = [f"allocated_population__{scheme}" for scheme in schemes]
    linked = frame.loc[frame["mhe_code"].notna()].copy()
    evaluation = linked.groupby("mhe_code", as_index=False).agg(
        official_population=("mhe_population", "first"),
        **{column: (column, "sum") for column in population_columns},
    )
    if len(evaluation) != 540:
        raise ValueError(f"Expected all 540 official Major Housing Estates, found {len(evaluation)}")
    return evaluation, population_error_metrics(evaluation, schemes)


def attach_ssug_and_land_use(
    frame: gpd.GeoDataFrame, ssug_boundaries: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Attach official source zones and the 10 m land-use cell at each building."""
    if frame.crs is None or ssug_boundaries.crs is None:
        raise ValueError("Building and SSUG layers must declare a CRS")
    frame = frame.to_crs(2326).copy()
    points = gpd.GeoDataFrame(
        {"_row_order": np.arange(len(frame), dtype=int)},
        geometry=frame.geometry.representative_point(),
        crs=frame.crs,
    )
    source = ssug_boundaries[
        ["ssbg", "official_population", "domestic_households", "occupied_quarters", "geometry"]
    ].to_crs(2326)
    linked = gpd.sjoin(points, source, how="left", predicate="within")
    if linked["_row_order"].duplicated().any():
        raise ValueError("A building representative point intersects multiple SSUG polygons")
    linked = linked.set_index("_row_order").reindex(range(len(frame)))
    frame["ssug_code"] = linked["ssbg"].astype("string").to_numpy()
    frame["ssug_population"] = numeric(linked["official_population"]).to_numpy()
    frame["ssug_domestic_households"] = numeric(linked["domestic_households"]).to_numpy()
    frame["ssug_occupied_quarters"] = numeric(linked["occupied_quarters"]).to_numpy()

    with Image.open(LAND_USE_TIFF) as image:
        tiepoint = tuple(float(value) for value in image.tag_v2[33922])
        scale = tuple(float(value) for value in image.tag_v2[33550])
        raster = np.asarray(image)
    x = points.geometry.x.to_numpy(float)
    y = points.geometry.y.to_numpy(float)
    columns = np.floor((x - tiepoint[3]) / scale[0]).astype(int)
    rows = np.floor((tiepoint[4] - y) / scale[1]).astype(int)
    inside = (
        (rows >= 0)
        & (rows < raster.shape[0])
        & (columns >= 0)
        & (columns < raster.shape[1])
    )
    land_use = np.full(len(frame), np.nan)
    land_use[inside] = raster[rows[inside], columns[inside]]
    frame["land_utilization_code_2021"] = pd.array(land_use, dtype="Int16")
    return frame


def population_weights(group: pd.DataFrame, scheme: str) -> tuple[pd.Series, bool]:
    """Return SSUG-internal weights and whether a masked scheme needed fallback."""
    gfa = numeric(group["residential_gfa_estimate_m2"], lower=0).fillna(0.0)
    volume = (
        numeric(group["footprint_area_m2"], lower=0).fillna(0.0)
        * numeric(group["height_value_m"], lower=0).fillna(0.0)
    )
    residential_land = group["land_utilization_code_2021"].isin(
        POPULATION_CONFIG["residential_land_use_codes"]
    )
    if scheme == "ssug_all_residential_gfa":
        return safe_normalize(gfa), False
    if scheme == "ssug_landuse_residential_gfa":
        weights = gfa.where(residential_land, 0.0)
        fallback = float(weights.sum()) <= 0
        return safe_normalize(gfa if fallback else weights), fallback
    if scheme == "ssug_landuse_residential_volume":
        weights = volume.where(residential_land, 0.0)
        fallback = float(weights.sum()) <= 0
        return safe_normalize(volume if fallback else weights), fallback
    if scheme == "ssug_constrained_housing_capacity":
        capacity = numeric(group["housing_capacity_persons_raw"], lower=0).fillna(0.0)
        fallback = float(capacity.sum()) <= 0
        return safe_normalize(gfa if fallback else capacity), fallback
    if scheme == "ssug_stpug_tenure_household_capacity":
        capacity = numeric(group["tenure_capacity_persons_raw"], lower=0).fillna(0.0)
        fallback = float(capacity.sum()) <= 0
        return safe_normalize(gfa if fallback else capacity), fallback
    raise ValueError(f"Unknown population allocation scheme: {scheme}")


def add_constrained_housing_capacity(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Estimate dwelling capacity while preserving each SSUG's occupied-quarter total."""
    result = frame.copy()
    flat_text = result.get("ha_number_of_flats", pd.Series("", index=result.index)).fillna("").astype(str)
    parsed_flats = pd.to_numeric(
        flat_text.str.extract(r"(\d[\d ,]*)", expand=False).str.replace(" ", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    )
    exact_ha = result.get("ha_link_status", pd.Series("", index=result.index)).eq(
        "exact_unique_normalized_name"
    )
    estate_name = result.get("ha_estate_name_en", pd.Series("", index=result.index)).fillna("").astype(str)
    program = result.get("ha_housing_program", pd.Series("", index=result.index)).fillna("").astype(str)
    estate_key = program + "::" + estate_name
    usable_ha = exact_ha & parsed_flats.gt(0) & estate_name.ne("")
    gfa = numeric(result["residential_gfa_estimate_m2"], lower=0).fillna(0.0)
    estate_gfa = gfa.where(usable_ha, 0.0).groupby(estate_key).transform("sum")
    estate_flat_total = parsed_flats.where(usable_ha).groupby(estate_key).transform("first")
    result["ha_estimated_dwelling_units"] = (
        estate_flat_total * gfa / estate_gfa.replace(0.0, np.nan)
    ).where(usable_ha, 0.0).fillna(0.0)
    result["estimated_dwelling_units"] = 0.0

    ssug_group_key = result["ssug_code"].astype("string").fillna("__UNLINKED_SSUG__")
    for ssug_code, indices in result.groupby(ssug_group_key).groups.items():
        group = result.loc[indices]
        if ssug_code == "__UNLINKED_SSUG__":
            result.loc[indices, "estimated_dwelling_units"] = safe_normalize(gfa.loc[indices])
            continue
        occupied_quarters = first_numeric_or_zero(group["ssug_occupied_quarters"])
        if occupied_quarters <= 0:
            continue
        ha_units = numeric(group["ha_estimated_dwelling_units"], lower=0).fillna(0.0)
        ha_total = float(ha_units.sum())
        if ha_total > occupied_quarters:
            ha_units = ha_units * occupied_quarters / ha_total
            ha_total = occupied_quarters
        remaining = max(occupied_quarters - ha_total, 0.0)
        non_ha = ~usable_ha.loc[indices]
        non_ha_gfa = gfa.loc[indices].where(non_ha, 0.0)
        if remaining > 0 and float(non_ha_gfa.sum()) <= 0:
            non_ha_gfa = gfa.loc[indices]
        other_units = safe_normalize(non_ha_gfa) * remaining if remaining > 0 else non_ha_gfa * 0.0
        result.loc[indices, "estimated_dwelling_units"] = ha_units + other_units

    household_size = pd.Series(2.7, index=result.index)
    household_size.loc[program.eq("hos_psps_gsh")] = 2.8
    result["capacity_household_size"] = household_size
    result["housing_capacity_persons_raw"] = result["estimated_dwelling_units"] * household_size
    result["housing_capacity_source"] = np.where(
        usable_ha,
        "HA_estate_flat_count_constrained_to_2021_SSUG_occupied_quarters",
        "2021_SSUG_remaining_occupied_quarters_allocated_by_residential_GFA",
    )
    return result


def add_stpug_tenure_capacity(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Use 2021 tenure-specific household totals before conserving finer SSUG population."""
    result = frame.copy()
    sub = english_label(result["hkbfetd_sub_class"]).str.lower()
    program = result.get("ha_housing_program", pd.Series("", index=result.index)).fillna("").astype(str)
    tenure = pd.Series("private_permanent", index=result.index, dtype="object")
    tenure.loc[sub.str.contains("public housing", na=False) | program.eq("public_rental_housing")] = "public_rental"
    tenure.loc[sub.str.contains("subsid", na=False) | program.eq("hos_psps_gsh")] = "subsidised_sale"
    result["housing_tenure_class"] = tenure
    result["tenure_capacity_households"] = 0.0
    gfa = numeric(result["residential_gfa_estimate_m2"], lower=0).fillna(0.0)
    target_columns = {
        "public_rental": "census_public_rental_households",
        "subsidised_sale": "census_subsidised_sale_households",
        "private_permanent": "census_private_permanent_households",
    }
    stpug_key = result.get("stpug_code", pd.Series(pd.NA, index=result.index)).astype("string").fillna(
        "__UNLINKED_STPUG__"
    )
    for stpug_code, indices in result.groupby(stpug_key).groups.items():
        if stpug_code == "__UNLINKED_STPUG__":
            continue
        group = result.loc[indices]
        for tenure_name, target_column in target_columns.items():
            target_households = first_numeric_or_zero(group[target_column])
            if target_households <= 0:
                continue
            eligible = tenure.loc[indices].eq(tenure_name)
            weights = gfa.loc[indices].where(eligible, 0.0)
            if float(weights.sum()) <= 0:
                weights = gfa.loc[indices]
            result.loc[indices, "tenure_capacity_households"] += safe_normalize(weights) * target_households
    household_size = pd.Series(2.7, index=result.index)
    household_size.loc[tenure.eq("subsidised_sale")] = 2.8
    result["tenure_capacity_household_size"] = household_size
    result["tenure_capacity_persons_raw"] = result["tenure_capacity_households"] * household_size
    result["tenure_capacity_source"] = (
        "2021_STPUG_tenure_households_allocated_within_tenure_by_residential_GFA_then_"
        "normalized_to_2021_SSUG_population"
    )
    return result


def build_population_and_archetypes(
    context: gpd.GeoDataFrame,
    old_assignments: pd.DataFrame,
    old_catalog: pd.DataFrame,
    groups: pd.DataFrame,
    als: pd.DataFrame,
    hma_boundaries: gpd.GeoDataFrame,
    ssug_boundaries: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = attach_ssug_and_land_use(context, ssug_boundaries)
    frame["housing_class"] = housing_class(frame)
    frame = add_constrained_housing_capacity(frame)
    frame = add_stpug_tenure_capacity(frame)
    frame["height_class_v0_3"] = height_class(frame["floors_value"])
    frame["era_class_v0_3"] = era_class(frame["construction_year_value"])
    compactness = 4 * math.pi * numeric(frame["footprint_area_m2"], lower=0) / numeric(frame["perimeter_m"], lower=0).pow(2)
    frame["morphology_class_v0_3"] = pd.cut(compactness, [-np.inf, 0.35, 0.65, np.inf], labels=["articulated", "intermediate", "compact"]).astype("string").fillna("unknown")
    frame = frame.merge(old_assignments[["BUILDINGSTRUCTUREID", "archetype_id"]], on="BUILDINGSTRUCTUREID", how="left", validate="one_to_one")
    frame["detailed_archetype_id"] = (
        frame["archetype_id"].fillna("AUNK").astype(str)
        + "__" + frame["housing_class"].astype(str)
        + "__" + frame["era_class_v0_3"].astype(str)
        + "__" + frame["height_class_v0_3"].astype(str)
        + "__" + frame["morphology_class_v0_3"].astype(str)
    )

    primary_scheme = str(POPULATION_CONFIG["primary_allocation"])
    candidate_schemes = [
        str(value) for value in POPULATION_CONFIG["candidate_capacity_allocations"]
    ]
    schemes = [
        primary_scheme,
        *candidate_schemes,
        *[str(value) for value in POPULATION_CONFIG["sensitivity_allocations"]],
    ]
    if len(schemes) != len(set(schemes)):
        raise ValueError(f"Duplicate population allocation schemes: {schemes}")
    ssug_group_key = frame["ssug_code"].astype("string").fillna("__UNLINKED_SSUG__")
    fallback_counts = {scheme: 0 for scheme in schemes}
    for scheme in schemes:
        column = f"allocated_population__{scheme}"
        frame[column] = 0.0
        for ssug_code, indices in frame.groupby(ssug_group_key).groups.items():
            if ssug_code == "__UNLINKED_SSUG__":
                continue
            group = frame.loc[indices]
            total_population = first_numeric_or_zero(group["ssug_population"])
            if total_population <= 0:
                continue
            weights, used_fallback = population_weights(group, scheme)
            fallback_counts[scheme] += int(used_fallback)
            frame.loc[indices, column] = weights * total_population
        frame[f"allocated_age65plus__{scheme}"] = frame[column] * numeric(frame["census_age65plus_share"]).fillna(0.0)

    allocation_columns = [f"allocated_population__{scheme}" for scheme in schemes]
    allocation_ensemble = frame[allocation_columns].apply(pd.to_numeric, errors="coerce")
    frame["population_allocation_sensitivity_median"] = allocation_ensemble.median(axis=1)
    frame["population_allocation_sensitivity_p05"] = allocation_ensemble.quantile(0.05, axis=1)
    frame["population_allocation_sensitivity_p95"] = allocation_ensemble.quantile(0.95, axis=1)

    group_columns = ["detailed_archetype_id", "archetype_id", "housing_class", "era_class_v0_3", "height_class_v0_3", "morphology_class_v0_3"]
    catalog = frame.groupby(group_columns, dropna=False).agg(
        buildings=("BUILDINGSTRUCTUREID", "size"),
        residential_gfa_m2=("residential_gfa_estimate_m2", "sum"),
        median_floors=("floors_value", "median"),
        median_height_m=("height_value_m", "median"),
        median_construction_year=("construction_year_value", "median"),
        median_footprint_m2=("footprint_area_m2", "median"),
    ).reset_index()
    parameters = old_catalog[["archetype_id", "u_wall_w_m2k", "u_roof_w_m2k", "window_to_wall_ratio", "window_shgc", "infiltration_ach", "thermal_capacitance_kj_m2k", "baseline_ac_cop", "physics_parameter_status"]]
    catalog = catalog.merge(parameters, on="archetype_id", how="left", validate="many_to_one")
    catalog["v0_3_parameter_status"] = (
        "hong_kong_literature_priors_with_emsd_mass_balance_and_polyu_hkust_load_validation"
    )
    catalog = catalog.sort_values("residential_gfa_m2", ascending=False).reset_index(drop=True)

    errors = []
    for scheme in schemes:
        allocated = frame.groupby(ssug_group_key)[f"allocated_population__{scheme}"].sum()
        target = frame.groupby(ssug_group_key)["ssug_population"].first().fillna(0.0)
        errors.append(float((allocated - target).abs().max()))
    represented_ssugs = set(frame["ssug_code"].dropna().astype(str))
    unrepresented_ssugs = ssug_boundaries.loc[
        ~ssug_boundaries["ssbg"].astype(str).isin(represented_ssugs)
    ].copy()
    represented_population = float(
        numeric(
            ssug_boundaries.loc[
                ssug_boundaries["ssbg"].astype(str).isin(represented_ssugs),
                "official_population",
            ]
        ).sum()
    )
    unrepresented_population = float(numeric(unrepresented_ssugs["official_population"]).sum())
    linkage, group_audit, linkage_audit = link_building_groups(frame, groups, als, hma_boundaries)
    mhe_evaluation, population_metrics = major_housing_estate_population_validation(frame, schemes)
    primary_mhe = population_metrics.loc[population_metrics["scheme"].eq(primary_scheme)].iloc[0]
    benchmark_mae = float(POPULATION_CONFIG["published_mhe_benchmark_relative_mae_pct"])
    benchmark_rmse = float(POPULATION_CONFIG["published_mhe_benchmark_relative_rmse_pct"])
    validation = {
        "residential_or_mixed_use_buildings_before_qc": len(frame),
        "detailed_archetypes": len(catalog),
        "primary_population_scheme": primary_scheme,
        "population_schemes": schemes,
        "official_ssug_units": int(len(ssug_boundaries)),
        "represented_ssug_units": int(len(represented_ssugs)),
        "unrepresented_ssug_units": int(len(unrepresented_ssugs)),
        "unrepresented_ssug_codes": sorted(unrepresented_ssugs["ssbg"].astype(str).tolist()),
        "official_land_population": float(represented_population + unrepresented_population),
        "represented_ssug_population": represented_population,
        "unrepresented_ssug_population": unrepresented_population,
        "represented_population_share_pct": float(100 * represented_population / (represented_population + unrepresented_population)),
        "allocated_population_primary_scheme": float(frame[f"allocated_population__{primary_scheme}"].sum()),
        "maximum_represented_ssug_population_conservation_error_persons": max(errors),
        "masked_source_zones_requiring_unmasked_fallback": fallback_counts,
        "population_is_observed_at_building_level": False,
        "population_allocation_sensitivity_envelope": (
            "building-wise p05, median and p95 across the five pre-specified constrained "
            "dasymetric allocation models; this is a model-sensitivity envelope, not a sampling confidence interval"
        ),
        "building_level_population_validation_complete": False,
        "building_group_external_validation_complete": False,
        "building_group_major_building_linkage": linkage_audit,
        "major_housing_estate_external_validation_executed": True,
        "major_housing_estate_population_metrics": population_metrics.to_dict(orient="records"),
        "primary_major_housing_estate_relative_mae_pct": float(primary_mhe["relative_mae_pct"]),
        "primary_major_housing_estate_relative_rmse_pct": float(primary_mhe["relative_rmse_pct"]),
        "primary_major_housing_estate_spearman_r": float(primary_mhe["spearman_r"]),
        "published_major_housing_estate_benchmark_relative_mae_pct": benchmark_mae,
        "published_major_housing_estate_benchmark_relative_rmse_pct": benchmark_rmse,
        "published_major_housing_estate_benchmark_met": bool(primary_mhe["relative_mae_pct"] <= benchmark_mae and primary_mhe["relative_rmse_pct"] <= benchmark_rmse),
        "income_or_luxury_assigned_to_individual_building": False,
        "unlinked_ssug_buildings": int(ssug_group_key.eq("__UNLINKED_SSUG__").sum()),
        "population_method": "official 2021 SSUG population allocated within each represented source polygon by estimated residential GFA at building representative points falling in official 10 m private-residential, public-residential or rural-settlement cells; zones without a positive masked weight fall back to all residential GFA",
        "population_conservation_scope": "Every SSUG containing at least one modeled residential or mixed-use structure is conserved exactly. Population in SSUGs with no modeled structure remains explicitly unallocated rather than being moved to a neighbouring zone.",
        "source_population_scope_note": "SSUG total population includes residents of residential buildings, residential quarters in non-residential buildings and collective living quarters. The Building Group residential-building total is reported separately and the difference is retained as a sensitivity limitation, not silently removed.",
        "sources": {
            "census_building_group": "https://www.census2021.gov.hk/en/terms_and_definitions.html",
            "census_ssug": "https://data.gov.hk/en-data/dataset/hk-censtatd-census_geo-2021-population-census-by-ssg",
            "planning_department_land_utilization": "https://data.gov.hk/en-data/dataset/hk-pland-pland1-land-utilization-in-hong-kong-raster-grid",
            "hong_kong_ann_dasymetric": "https://doi.org/10.1016/j.compenvurbsys.2024.102072",
            "building_volume_dasymetric": "https://doi.org/10.1016/j.compenvurbsys.2021.101594",
            "hong_kong_mhe_accuracy_benchmark": "https://doi.org/10.1016/j.scs.2026.107233",
        },
    }
    return frame, catalog, validation, linkage, group_audit, mhe_evaluation


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def simulation_contract_sha256() -> str:
    """Return the versioned EnergyPlus-physics contract fingerprint.

    The v0.22 fingerprint was frozen when its 6,237 event cases completed.  Its
    dependency hashes keep configuration, building composition and engine
    metadata changes cache-invalidating, while audit/report-only source edits do
    not silently change the simulated physics.  Any subsequent physics edit must
    advance ``MODEL_VERSION`` or ``SIMULATION_IMPLEMENTATION_REVISION``.
    """
    dependency_paths = {
        "foundation_analysis.json": ROOT / "config" / "foundation_analysis.json",
        "building_composition.py": ROOT / "utils" / "building_composition.py",
        "energyplus_release.json": ENGINE_METADATA,
    }
    dependency_hashes = {
        name: sha256_file(path) for name, path in dependency_paths.items()
    }
    frozen_v022_dependencies = {
        "foundation_analysis.json": "10f7a6f98ab2bdf4f2c90f3289a182c717b9a1eb44e57c6c9e8fa926a56d3bc9",
        "building_composition.py": "89eb58e9bb9927044ddf3b5fbe0b0a023a8e7ec11ac41cce02f1769a046296d3",
        "energyplus_release.json": "1a48574751b647a0fb95f1719f2ed2467e4957b0912124de1172ea06e7c965e2",
    }
    if (
        MODEL_VERSION == "energyplus_multizone_counterfactual_resilience_v0_22"
        and SIMULATION_IMPLEMENTATION_REVISION
        == "humidistat_data_centre_redundancy_and_diagnostics_v10"
        and dependency_hashes == frozen_v022_dependencies
    ):
        return "7a2b99bcbe6a8e6ee254f15a47285332a99bc818320326cc048a3bfc49bcf72e"
    digest = hashlib.sha256()
    digest.update(MODEL_VERSION.encode("utf-8"))
    digest.update(SIMULATION_IMPLEMENTATION_REVISION.encode("utf-8"))
    for name, file_hash in dependency_hashes.items():
        digest.update(name.encode("utf-8"))
        digest.update(file_hash.encode("utf-8"))
    return digest.hexdigest()


def weather_frame_sha256(weather: pd.DataFrame) -> str:
    """Return a deterministic digest of the complete ordered weather input."""
    frame = weather.copy()
    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column].dtype):
            frame[column] = pd.to_datetime(frame[column], utc=True).astype(str)
    payload = frame.to_csv(index=False, lineterminator="\n", float_format="%.12g")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def payload_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def validate_engine(executable: Path) -> dict:
    if not executable.is_file():
        raise FileNotFoundError(executable)
    metadata = json.loads(ENGINE_METADATA.read_text(encoding="utf-8"))
    result = subprocess.run([str(executable), "--version"], check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    version = (result.stdout + result.stderr).strip()
    if str(metadata["version"]) not in version:
        raise RuntimeError(f"EnergyPlus version mismatch: {version}")
    return {**metadata, "reported_version": version, "executable_sha256": sha256_file(executable)}


def dewpoint_c(temperature_c: float, relative_humidity_pct: float) -> float:
    humidity = min(max(relative_humidity_pct, 0.1), 100.0) / 100.0
    alpha = math.log(humidity) + 17.625 * temperature_c / (243.04 + temperature_c)
    return 243.04 * alpha / (17.625 - alpha)


def erbs_solar_decomposition(
    timestamp_hkt: pd.Timestamp,
    global_horizontal_w_m2: float,
    latitude_deg: float = 22.31,
    longitude_deg: float = 114.17,
) -> tuple[float, float, float, float]:
    """Split GHI into DNI and DHI with the hourly Erbs correlation.

    NASA POWER supplies global horizontal irradiance for the event weather.
    EnergyPlus requires global, direct-normal and diffuse-horizontal fields.
    The decomposition retains the measured/modelled GHI exactly while making
    facade orientation physically active instead of treating all solar gain as
    isotropic diffuse radiation.
    """
    stamp = pd.Timestamp(timestamp_hkt)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("Asia/Hong_Kong")
    else:
        stamp = stamp.tz_convert("Asia/Hong_Kong")
    ghi = max(float(global_horizontal_w_m2), 0.0)
    day = int(stamp.dayofyear)
    fractional_hour = stamp.hour + stamp.minute / 60.0
    gamma = 2.0 * math.pi / 365.0 * (day - 1 + (fractional_hour - 12.0) / 24.0)
    equation_of_time_min = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2.0 * gamma)
        - 0.040849 * math.sin(2.0 * gamma)
    )
    declination = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2.0 * gamma)
        + 0.000907 * math.sin(2.0 * gamma)
        - 0.002697 * math.cos(3.0 * gamma)
        + 0.00148 * math.sin(3.0 * gamma)
    )
    local_standard_meridian_deg = 120.0
    solar_time_min = (
        fractional_hour * 60.0
        + equation_of_time_min
        + 4.0 * (longitude_deg - local_standard_meridian_deg)
    )
    hour_angle = math.radians(solar_time_min / 4.0 - 180.0)
    latitude = math.radians(latitude_deg)
    cos_zenith = max(
        math.sin(latitude) * math.sin(declination)
        + math.cos(latitude) * math.cos(declination) * math.cos(hour_angle),
        0.0,
    )
    if ghi <= 0.0:
        return ghi, 0.0, 0.0, cos_zenith
    if cos_zenith <= 1e-6:
        # Hourly products can retain a small interval-mean twilight flux even
        # when the point solar geometry is below the horizon. Treat it as
        # diffuse so the measured GHI is conserved without inventing DNI.
        return ghi, 0.0, ghi, cos_zenith
    extraterrestrial_normal = 1367.0 * (
        1.0 + 0.033 * math.cos(2.0 * math.pi * day / 365.0)
    )
    extraterrestrial_horizontal = extraterrestrial_normal * cos_zenith
    clearness = min(max(ghi / max(extraterrestrial_horizontal, 1e-6), 0.0), 1.0)
    if clearness <= 0.22:
        diffuse_fraction = 1.0 - 0.09 * clearness
    elif clearness <= 0.80:
        diffuse_fraction = (
            0.9511
            - 0.1604 * clearness
            + 4.388 * clearness**2
            - 16.638 * clearness**3
            + 12.336 * clearness**4
        )
    else:
        diffuse_fraction = 0.165
    diffuse_horizontal = min(max(ghi * diffuse_fraction, 0.0), ghi)
    direct_normal = min(
        max((ghi - diffuse_horizontal) / cos_zenith, 0.0),
        extraterrestrial_normal,
    )
    # Capping DNI at the extraterrestrial-normal bound transfers any residual
    # low-sun interval energy to DHI while retaining the supplied GHI exactly.
    diffuse_horizontal = min(max(ghi - direct_normal * cos_zenith, 0.0), ghi)
    return ghi, direct_normal, diffuse_horizontal, cos_zenith


def audit_polyu_campus_response(output: Path, seed: int) -> dict:
    """Build leakage-safe local load-response targets from complete PolyU meter hours."""
    if not POLYU_CAMPUS_ARCHIVE.exists() or not POLYU_HOURLY_WEATHER.exists():
        raise FileNotFoundError(
            "PolyU campus archive or its step-00 hourly weather is missing; run 00_PREPARE_AND_AUDIT_INPUTS.py first"
        )
    with zipfile.ZipFile(POLYU_CAMPUS_ARCHIVE) as archive:
        member = next(name for name in archive.namelist() if name.endswith("/data/building_hourly.parquet"))
        campus = pd.read_parquet(io.BytesIO(archive.read(member)))
    campus["timestamp_hkt"] = pd.to_datetime(campus["timestamp"], errors="coerce")
    weather = pd.read_parquet(POLYU_HOURLY_WEATHER)
    weather["timestamp_hkt"] = pd.to_datetime(weather["timestamp_hkt"]).dt.tz_localize(None)
    complete = campus.loc[campus["quality_status"].eq("complete_inputs")].copy()
    complete["energy_kwh"] = numeric(complete["complete_input_energy_kwh"], lower=0)
    complete = complete.merge(
        weather[["timestamp_hkt", "temperature_c", "relative_humidity_pct", "solar_radiation_imputed"]],
        on="timestamp_hkt",
        how="inner",
        validate="many_to_one",
    )
    complete["date"] = complete["timestamp_hkt"].dt.normalize()
    complete["hour"] = complete["timestamp_hkt"].dt.hour
    complete["day_of_week"] = complete["timestamp_hkt"].dt.dayofweek
    complete["hour_of_week"] = complete["day_of_week"] * 24 + complete["hour"]
    phase_values = sorted(complete["phase"].dropna().unique())
    holdout_phase = {
        phase for phase in phase_values
        if int(hashlib.sha256(f"{seed}:{phase}".encode()).hexdigest()[:8], 16) % 5 == 0
    }
    if not holdout_phase or len(holdout_phase) == len(phase_values):
        holdout_phase = set(phase_values[::5])
    complete["building_split"] = np.where(complete["phase"].isin(holdout_phase), "building_holdout", "calibration")
    final_time_start = complete["timestamp_hkt"].max() - pd.Timedelta(days=28) + pd.Timedelta(hours=1)
    complete["time_split"] = np.where(complete["timestamp_hkt"].ge(final_time_start), "final_28_day_holdout", "calibration_period")

    signatures = []
    for building_id, group in complete.groupby("building_id"):
        ordered = group.sort_values("timestamp_hkt")
        scale = float(ordered["energy_kwh"].median())
        normalized = ordered["energy_kwh"].to_numpy(dtype=float) / scale if scale > 0 else ordered["energy_kwh"].to_numpy(dtype=float)
        payload = ordered["timestamp_hkt"].astype("int64").to_numpy().tobytes() + np.round(normalized, 8).tobytes()
        signatures.append({"building_id": building_id, "profile_equivalence_id": hashlib.sha256(payload).hexdigest()[:16]})
    signatures = pd.DataFrame(signatures)
    signatures["profile_equivalence_cluster_size"] = signatures.groupby("profile_equivalence_id")["building_id"].transform("size")
    complete = complete.merge(signatures, on="building_id", how="left", validate="many_to_one")
    complete["building_split"] = complete.groupby("profile_equivalence_id")["building_split"].transform(
        lambda values: "building_holdout"
        if values.eq("building_holdout").any()
        else "calibration"
    )
    split_lookup = complete[["building_id", "building_split"]].drop_duplicates()

    audit = campus.groupby(["building_id", "phase", "building_code"], as_index=False).agg(
        expected_hours=("timestamp", "size"),
        complete_input_hours=("quality_status", lambda values: int(values.eq("complete_inputs").sum())),
        partial_input_hours=("quality_status", lambda values: int(values.eq("partial_inputs").sum())),
        no_data_hours=("quality_status", lambda values: int(values.eq("no_data").sum())),
        observed_energy_kwh=("energy_kwh", "sum"),
    )
    audit["complete_input_coverage_pct"] = 100 * audit["complete_input_hours"] / audit["expected_hours"]
    audit["primary_dynamic_target"] = audit["complete_input_coverage_pct"].ge(70.0)
    audit = audit.merge(signatures, on="building_id", how="left", validate="one_to_one")
    audit = audit.merge(split_lookup, on="building_id", how="left", validate="one_to_one")
    audit.to_csv(output / "polyu_campus_building_quality_audit.csv", index=False, encoding="utf-8-sig")

    daily = complete.groupby(["building_id", "phase", "building_code", "building_split", "date"], as_index=False).agg(
        energy_kwh=("energy_kwh", "sum"),
        valid_hours=("energy_kwh", "size"),
        mean_temperature_c=("temperature_c", "mean"),
        maximum_temperature_c=("temperature_c", "max"),
        mean_rh_pct=("relative_humidity_pct", "mean"),
    )
    primary_ids = set(audit.loc[audit["primary_dynamic_target"], "building_id"])
    daily = daily.loc[daily["valid_hours"].eq(24) & daily["building_id"].isin(primary_ids)].copy()
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["median_daily_kwh"] = daily.groupby("building_id")["energy_kwh"].transform("median")
    daily["relative_daily_load"] = daily["energy_kwh"] / daily["median_daily_kwh"].replace(0.0, np.nan)
    daily["time_split"] = np.where(
        daily["date"].ge(final_time_start.tz_localize(None).normalize()),
        "final_28_day_holdout",
        "calibration_period",
    )
    daily = daily.merge(
        signatures[["building_id", "profile_equivalence_id", "profile_equivalence_cluster_size"]],
        on="building_id",
        how="left",
        validate="many_to_one",
    )

    calibration_daily = (
        daily.loc[daily["building_split"].eq("calibration")]
        .dropna(subset=["relative_daily_load", "mean_temperature_c"])
        .sort_values(["profile_equivalence_id", "date"])
        .drop_duplicates(["profile_equivalence_id", "date"])
        .reset_index(drop=True)
    )
    if calibration_daily["profile_equivalence_id"].nunique() < 5:
        raise ValueError("PolyU calibration requires at least five independent load-profile clusters")
    quantiles = np.linspace(0.10, 0.70, 13)
    balance_candidates = np.unique(
        np.round(calibration_daily["mean_temperature_c"].quantile(quantiles).to_numpy(), 2)
    )
    dates = np.asarray(sorted(calibration_daily["date"].unique()))
    date_to_block = {
        date: block
        for block, block_dates in enumerate(np.array_split(dates, 4))
        for date in block_dates
    }
    cv_blocks = calibration_daily["date"].map(date_to_block).to_numpy()

    def pooled_design(frame: pd.DataFrame, balance_point_c: float) -> np.ndarray:
        day_of_year = frame["date"].dt.dayofyear.to_numpy(dtype=float)
        controls = pd.concat(
            [
                (frame["mean_temperature_c"] - balance_point_c).clip(lower=0).rename("cooling_degree_c"),
                pd.Series(np.sin(2 * np.pi * day_of_year / 365.25), index=frame.index, name="annual_sine"),
                pd.Series(np.cos(2 * np.pi * day_of_year / 365.25), index=frame.index, name="annual_cosine"),
                pd.get_dummies(frame["day_of_week"], prefix="dow", drop_first=True, dtype=float),
                pd.get_dummies(frame["profile_equivalence_id"], prefix="profile", drop_first=True, dtype=float),
            ],
            axis=1,
        )
        return np.column_stack([np.ones(len(frame)), controls.to_numpy(dtype=float)])

    cv_rows = []
    y_calibration = calibration_daily["relative_daily_load"].to_numpy(dtype=float)
    for candidate in balance_candidates:
        matrix = pooled_design(calibration_daily, float(candidate))
        fold_mae = []
        for block in sorted(np.unique(cv_blocks)):
            test = cv_blocks == block
            train = ~test
            coefficients, *_ = np.linalg.lstsq(matrix[train], y_calibration[train], rcond=None)
            fold_mae.append(float(np.mean(np.abs(y_calibration[test] - matrix[test] @ coefficients))))
        cv_rows.append(
            {
                "balance_point_c": float(candidate),
                "blocked_time_folds": len(fold_mae),
                "mean_absolute_error": float(np.mean(fold_mae)),
            }
        )
    balance_cv = pd.DataFrame(cv_rows).sort_values(
        ["mean_absolute_error", "balance_point_c"], kind="mergesort"
    )
    selected_balance_point_c = float(balance_cv.iloc[0]["balance_point_c"])
    balance_cv["selected"] = balance_cv["balance_point_c"].eq(selected_balance_point_c)
    balance_cv.to_csv(
        output / "polyu_campus_balance_point_blocked_cv.csv",
        index=False,
        encoding="utf-8-sig",
    )

    response_rows = []
    for keys, group in daily.groupby(["building_id", "phase", "building_code", "building_split"]):
        valid = group.dropna(subset=["relative_daily_load", "mean_temperature_c"])
        if len(valid) < 120 or valid["mean_temperature_c"].nunique() < 20:
            continue
        cooling_degree = (
            valid["mean_temperature_c"] - selected_balance_point_c
        ).clip(lower=0).rename("cooling_degree_c")
        controls = pd.concat(
            [
                cooling_degree,
                pd.get_dummies(valid["day_of_week"], prefix="dow", drop_first=True, dtype=float),
                pd.get_dummies(valid["month"], prefix="month", drop_first=True, dtype=float),
            ],
            axis=1,
        )
        matrix = np.column_stack([np.ones(len(valid)), controls.to_numpy(dtype=float)])
        y = valid["relative_daily_load"].to_numpy(dtype=float)
        coefficients, *_ = np.linalg.lstsq(matrix, y, rcond=None)
        fitted = matrix @ coefficients
        denominator = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - float(np.sum((y - fitted) ** 2)) / denominator if denominator > 0 else np.nan
        response_rows.append(
            {
                "building_id": keys[0],
                "phase": keys[1],
                "building_code": keys[2],
                "building_split": keys[3],
                "profile_equivalence_id": valid["profile_equivalence_id"].iloc[0],
                "profile_equivalence_cluster_size": int(valid["profile_equivalence_cluster_size"].iloc[0]),
                "selected_balance_point_c": selected_balance_point_c,
                "valid_daily_observations": int(len(valid)),
                "hot_daily_observations": int(cooling_degree.gt(0).sum()),
                "calendar_adjusted_relative_daily_load_slope_per_c_above_balance_point": float(coefficients[1]),
                "calendar_adjusted_regression_r2": r2,
                "calendar_controls": "day_of_week_and_month_fixed_effects",
                "median_daily_kwh": float(group["median_daily_kwh"].iloc[0]),
            }
        )
    response = pd.DataFrame(response_rows)
    response.to_csv(output / "polyu_campus_temperature_load_response.csv", index=False, encoding="utf-8-sig")

    def transferable_validation(train: pd.DataFrame, test: pd.DataFrame) -> dict:
        train = train.dropna(subset=["relative_daily_load", "mean_temperature_c"]).copy()
        test = test.dropna(subset=["relative_daily_load", "mean_temperature_c"]).copy()
        combined = pd.concat([train, test], ignore_index=True)
        day_of_year = combined["date"].dt.dayofyear.to_numpy(float)
        features = pd.concat(
            [
                (combined["mean_temperature_c"] - selected_balance_point_c)
                .clip(lower=0)
                .rename("cooling_degree_c"),
                pd.Series(np.sin(2 * np.pi * day_of_year / 365.25), name="annual_sine"),
                pd.Series(np.cos(2 * np.pi * day_of_year / 365.25), name="annual_cosine"),
                pd.get_dummies(combined["day_of_week"], prefix="dow", drop_first=True, dtype=float),
            ],
            axis=1,
        ).fillna(0.0)
        matrix = np.column_stack([np.ones(len(combined)), features.to_numpy(float)])
        coefficients, *_ = np.linalg.lstsq(
            matrix[: len(train)], train["relative_daily_load"].to_numpy(float), rcond=None
        )
        observed = test["relative_daily_load"].to_numpy(float)
        predicted = matrix[len(train) :] @ coefficients
        error = predicted - observed
        return {
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_buildings": int(train["building_id"].nunique()),
            "test_buildings": int(test["building_id"].nunique()),
            "nmbe_pct": float(100 * error.sum() / observed.sum()),
            "cvrmse_pct": float(100 * np.sqrt(np.mean(error**2)) / observed.mean()),
            "mae_relative_load": float(np.mean(np.abs(error))),
            "temperature_slope_per_c": float(coefficients[1]),
        }

    building_holdout_validation = transferable_validation(
        daily.loc[
            daily["building_split"].eq("calibration")
            & daily["time_split"].eq("calibration_period")
        ],
        daily.loc[
            daily["building_split"].eq("building_holdout")
            & daily["time_split"].eq("calibration_period")
        ],
    )
    time_holdout_validation = transferable_validation(
        daily.loc[
            daily["building_split"].eq("calibration")
            & daily["time_split"].eq("calibration_period")
        ],
        daily.loc[
            daily["building_split"].eq("calibration")
            & daily["time_split"].eq("final_28_day_holdout")
        ],
    )
    complete[[
        "timestamp_hkt", "building_id", "phase", "building_code", "energy_kwh",
        "temperature_c", "relative_humidity_pct", "solar_radiation_imputed",
        "building_split", "time_split", "hour_of_week",
    ]].to_parquet(output / "polyu_campus_complete_hourly_calibration.parquet", index=False)
    return {
        "archive_rows": int(len(campus)),
        "complete_joined_hours": int(len(complete)),
        "building_units": int(campus["building_id"].nunique()),
        "primary_units_at_least_70pct_complete": int(audit["primary_dynamic_target"].sum()),
        "independent_profile_clusters_in_primary_units": int(audit.loc[audit["primary_dynamic_target"], "profile_equivalence_id"].nunique()),
        "primary_units_in_nonindependent_profile_clusters": int(audit.loc[audit["primary_dynamic_target"] & audit["profile_equivalence_cluster_size"].gt(1)].shape[0]),
        "calibration_phases": sorted(set(phase_values) - holdout_phase),
        "building_holdout_phases": sorted(holdout_phase),
        "final_time_holdout_start_hkt": final_time_start.isoformat(),
        "temperature_response_units": int(len(response)),
        "selected_balance_point_c_from_calibration_only": selected_balance_point_c,
        "balance_point_selection": "four contiguous time blocks; independent calibration profile clusters only; minimum held-block MAE",
        "profile_equivalence_clusters_do_not_cross_split": bool(
            complete.groupby("profile_equivalence_id")["building_split"].nunique().max() == 1
        ),
        "building_holdout_validation": building_holdout_validation,
        "final_28_day_time_holdout_validation": time_holdout_validation,
        "calibration_slope_median_per_c": float(
            response.loc[
                response["building_split"].eq("calibration"),
                "calendar_adjusted_relative_daily_load_slope_per_c_above_balance_point",
            ].median()
        ),
        "building_holdout_slope_median_per_c": float(
            response.loc[
                response["building_split"].eq("building_holdout"),
                "calendar_adjusted_relative_daily_load_slope_per_c_above_balance_point",
            ].median()
        ),
        "primary_hour_rule": "quality_status equals complete_inputs; no coverage rescaling or imputation",
    }


def load_hko_anchored_nasa_csv(
    path: Path,
    station_code: str,
) -> pd.DataFrame:
    """Read a NASA POWER CSV and anchor its daily range to a nearby HKO station."""
    header = next(
        index
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines())
        if line.startswith("YEAR,")
    )
    raw = pd.read_csv(path, skiprows=header)
    raw = raw.replace(-999.0, np.nan)
    raw["timestamp_hkt"] = pd.to_datetime(
        dict(year=raw["YEAR"], month=raw["MO"], day=raw["DY"], hour=raw["HR"])
    )
    raw["date"] = raw["timestamp_hkt"].dt.normalize()
    hko_parts = []
    for data_type, field in [("CLMMAXT", "hko_tmax_c"), ("CLMMINT", "hko_tmin_c")]:
        payload = json.loads(
            (HKO_DAILY_CACHE / f"{data_type}_{station_code}.json").read_text(encoding="utf-8")
        )
        frame = pd.DataFrame(
            payload["data"], columns=["year", "month", "day", field, "completeness"]
        )
        frame["date"] = pd.to_datetime(frame[["year", "month", "day"]], errors="coerce")
        frame[field] = numeric(frame[field])
        hko_parts.append(frame[["date", field]])
    hko = hko_parts[0].merge(hko_parts[1], on="date", how="inner", validate="one_to_one")
    weather = raw.merge(hko, on="date", how="left", validate="many_to_one")
    daily = weather.groupby("date")["T2M"].agg(nasa_tmin_c="min", nasa_tmax_c="max")
    weather = weather.merge(daily, on="date", how="left", validate="many_to_one")
    fraction = (
        (weather["T2M"] - weather["nasa_tmin_c"])
        / (weather["nasa_tmax_c"] - weather["nasa_tmin_c"]).replace(0.0, np.nan)
    )
    weather["temperature_c"] = (
        weather["hko_tmin_c"]
        + fraction * (weather["hko_tmax_c"] - weather["hko_tmin_c"])
    ).fillna(weather["T2M"])
    weather["relative_humidity_pct"] = numeric(weather["RH2M"]).clip(0, 100)
    weather["solar_radiation_w_m2"] = numeric(weather["ALLSKY_SFC_SW_DWN"], lower=0)
    return weather[
        ["timestamp_hkt", "date", "temperature_c", "relative_humidity_pct", "solar_radiation_w_m2"]
    ]


def hkust_building_scope(name: str) -> str:
    value = str(name).lower()
    if any(token in value for token in ["student_hall", "staff_quarters", "postgraduate_hall"]):
        return "campus_residence"
    if any(token in value for token in ["plant", "pump_house", "tunnel"]):
        return "infrastructure_excluded"
    return "campus_nonresidential"


def audit_hkust_hourly_response(output: Path, seed: int) -> dict:
    """Construct strict, non-overlapping whole-building HKUST hourly loads."""
    for path in [HKUST_ARCHIVE, HKUST_WEATHER, HKUST_HIERARCHY]:
        if not path.exists():
            raise FileNotFoundError(path)
    hierarchy = pd.read_csv(HKUST_HIERARCHY)
    candidates = hierarchy.loc[
        hierarchy["metered_entity_types"].eq("ACB_Incomer")
        & hierarchy["clean_interval_minutes"].astype(str).str.contains("60", na=False)
        & hierarchy["clean_timeseries_file_present"].fillna(False)
    ].copy()
    candidates["leaf_building"] = candidates["buildings"].fillna("").map(
        lambda value: "|".join(
            part
            for part in str(value).split("|")
            if part and part != "HKUST_Campus_Zone_1_8"
        )
    )
    candidates = candidates.loc[
        candidates["leaf_building"].ne("")
        & ~candidates["leaf_building"].str.contains(r"\|", regex=True)
    ].copy()
    candidates["building_scope"] = candidates["leaf_building"].map(hkust_building_scope)
    candidates = candidates.loc[
        ~candidates["building_scope"].eq("infrastructure_excluded")
    ].copy()
    candidates["file_key"] = candidates["timeseries_files"].map(lambda value: Path(str(value)).name)

    meter_frames = []
    quality_rows = []
    with zipfile.ZipFile(HKUST_ARCHIVE) as archive:
        members = {
            Path(name).name.removeprefix("GUI_NO."): name
            for name in archive.namelist()
            if "Clean Dataset/Resappled data/T60/" in name
            and name.lower().endswith(".xlsx")
        }
        for row in candidates.itertuples(index=False):
            member = members.get(row.file_key)
            if member is None:
                continue
            frame = pd.read_excel(io.BytesIO(archive.read(member)), usecols=["time", "number"])
            frame["timestamp_hkt"] = pd.to_datetime(frame["time"], errors="coerce")
            frame["reading_kwh"] = numeric(frame["number"])
            frame = frame.sort_values("timestamp_hkt")
            frame["energy_kwh"] = frame["reading_kwh"].diff()
            consecutive = frame["timestamp_hkt"].diff().eq(pd.Timedelta(hours=1))
            valid = consecutive & frame["energy_kwh"].ge(0)
            positive = frame.loc[valid & frame["energy_kwh"].gt(0), "energy_kwh"]
            upper = float(positive.quantile(0.9995)) if len(positive) else math.nan
            if math.isfinite(upper) and upper > 0:
                valid &= frame["energy_kwh"].le(upper * 3)
            quality_rows.append(
                {
                    "meter_id": row.meter_id,
                    "leaf_building": row.leaf_building,
                    "building_scope": row.building_scope,
                    "rows": int(len(frame)),
                    "nonmissing_readings": int(frame["reading_kwh"].notna().sum()),
                    "valid_consecutive_nonnegative_hours": int(valid.sum()),
                    "negative_or_reset_intervals": int((consecutive & frame["energy_kwh"].lt(0)).sum()),
                    "extreme_increment_upper_kwh": upper,
                }
            )
            selected = frame.loc[valid, ["timestamp_hkt", "energy_kwh"]].copy()
            selected["meter_id"] = row.meter_id
            selected["leaf_building"] = row.leaf_building
            selected["building_scope"] = row.building_scope
            meter_frames.append(selected)
    meter_quality = pd.DataFrame(quality_rows)
    meter_quality.to_csv(
        output / "hkust_whole_building_meter_quality.csv", index=False, encoding="utf-8-sig"
    )
    meter_hours = pd.concat(meter_frames, ignore_index=True)
    expected = (
        candidates.groupby(["leaf_building", "building_scope"], as_index=False)
        .agg(expected_meters=("meter_id", "nunique"))
    )
    building_hourly = (
        meter_hours.groupby(["leaf_building", "building_scope", "timestamp_hkt"], as_index=False)
        .agg(energy_kwh=("energy_kwh", "sum"), valid_meters=("meter_id", "nunique"))
        .merge(expected, on=["leaf_building", "building_scope"], how="left", validate="many_to_one")
    )
    building_hourly = building_hourly.loc[
        building_hourly["valid_meters"].eq(building_hourly["expected_meters"])
    ].copy()
    weather = load_hko_anchored_nasa_csv(HKUST_WEATHER, "SKG")
    building_hourly = building_hourly.merge(
        weather[["timestamp_hkt", "temperature_c", "relative_humidity_pct"]],
        on="timestamp_hkt",
        how="inner",
        validate="many_to_one",
    )
    building_hourly["date"] = building_hourly["timestamp_hkt"].dt.normalize()
    building_hourly["hour_of_week"] = (
        building_hourly["timestamp_hkt"].dt.dayofweek * 24
        + building_hourly["timestamp_hkt"].dt.hour
    )
    building_hourly["building_split"] = building_hourly["leaf_building"].map(
        lambda value: "building_holdout"
        if int(hashlib.sha256(f"{seed}:HKUST:{value}".encode()).hexdigest()[:8], 16) % 5 == 0
        else "calibration"
    )
    if building_hourly["building_split"].nunique() < 2:
        ordered = sorted(building_hourly["leaf_building"].unique())
        building_hourly.loc[
            building_hourly["leaf_building"].eq(ordered[0]), "building_split"
        ] = "building_holdout"
    daily = building_hourly.groupby(
        ["leaf_building", "building_scope", "building_split", "date"], as_index=False
    ).agg(
        energy_kwh=("energy_kwh", "sum"),
        valid_hours=("energy_kwh", "size"),
        mean_temperature_c=("temperature_c", "mean"),
        mean_rh_pct=("relative_humidity_pct", "mean"),
    )
    daily = daily.loc[daily["valid_hours"].eq(24)].copy()
    eligible = daily.groupby("leaf_building")["date"].transform("size").ge(180)
    daily = daily.loc[eligible].copy()
    daily["median_daily_kwh"] = daily.groupby("leaf_building")["energy_kwh"].transform("median")
    daily["relative_daily_load"] = (
        daily["energy_kwh"] / daily["median_daily_kwh"].replace(0.0, np.nan)
    )
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["cooling_degree_c_24"] = (daily["mean_temperature_c"] - 24.0).clip(lower=0)

    train = daily.loc[daily["building_split"].eq("calibration")].dropna(
        subset=["relative_daily_load", "cooling_degree_c_24"]
    )
    test = daily.loc[daily["building_split"].eq("building_holdout")].dropna(
        subset=["relative_daily_load", "cooling_degree_c_24"]
    )
    feature_columns = ["cooling_degree_c_24"]
    calendar = pd.get_dummies(
        pd.concat([train[["day_of_week", "month"]], test[["day_of_week", "month"]]]),
        columns=["day_of_week", "month"],
        drop_first=True,
        dtype=float,
    )
    train_calendar = calendar.iloc[: len(train)].reset_index(drop=True)
    test_calendar = calendar.iloc[len(train) :].reset_index(drop=True)
    x_train = np.column_stack(
        [np.ones(len(train)), train[feature_columns].to_numpy(float), train_calendar.to_numpy(float)]
    )
    x_test = np.column_stack(
        [np.ones(len(test)), test[feature_columns].to_numpy(float), test_calendar.to_numpy(float)]
    )
    coefficients, *_ = np.linalg.lstsq(
        x_train, train["relative_daily_load"].to_numpy(float), rcond=None
    )
    predicted = x_test @ coefficients
    observed = test["relative_daily_load"].to_numpy(float)
    errors = predicted - observed
    mean_observed = float(np.mean(observed))
    scope_slopes = {}
    for scope, scope_frame in train.groupby("building_scope", observed=True):
        scope_frame = scope_frame.dropna(
            subset=["relative_daily_load", "cooling_degree_c_24"]
        ).copy()
        if len(scope_frame) < 365 or scope_frame["leaf_building"].nunique() < 3:
            continue
        scope_calendar = pd.get_dummies(
            scope_frame[["day_of_week", "month"]],
            columns=["day_of_week", "month"],
            drop_first=True,
            dtype=float,
        )
        scope_x = np.column_stack(
            [
                np.ones(len(scope_frame)),
                scope_frame[["cooling_degree_c_24"]].to_numpy(float),
                scope_calendar.to_numpy(float),
            ]
        )
        scope_coefficients, *_ = np.linalg.lstsq(
            scope_x, scope_frame["relative_daily_load"].to_numpy(float), rcond=None
        )
        scope_slopes[str(scope)] = {
            "calibration_buildings": int(scope_frame["leaf_building"].nunique()),
            "calibration_days": int(len(scope_frame)),
            "calendar_adjusted_relative_daily_load_slope_per_c_above_24": float(
                scope_coefficients[1]
            ),
        }
    validation = {
        "selected_acb_incomer_meters": int(len(candidates)),
        "whole_buildings": int(building_hourly["leaf_building"].nunique()),
        "residential_proxy_buildings": int(
            building_hourly.loc[building_hourly["building_scope"].eq("campus_residence"), "leaf_building"].nunique()
        ),
        "nonresidential_buildings": int(
            building_hourly.loc[building_hourly["building_scope"].eq("campus_nonresidential"), "leaf_building"].nunique()
        ),
        "strict_all_incomers_complete_hours": int(len(building_hourly)),
        "complete_daily_observations": int(len(daily)),
        "calibration_buildings": int(train["leaf_building"].nunique()),
        "held_out_buildings": int(test["leaf_building"].nunique()),
        "held_out_daily_observations": int(len(test)),
        "calendar_adjusted_temperature_slope_per_c_above_24": float(coefficients[1]),
        "scope_specific_calibration_slopes": scope_slopes,
        "held_out_nmbe_pct": float(100 * errors.sum() / observed.sum()),
        "held_out_cvrmse_pct": float(100 * np.sqrt(np.mean(errors**2)) / mean_observed),
        "held_out_mae_relative_load": float(np.mean(np.abs(errors))),
        "weather": "NASA POWER hourly shape anchored daily to HKO Sai Kung Tmax/Tmin",
        "aggregation": "ACB incomers only; campus ancestor removed; all building incomers required in an hour",
    }
    building_hourly.to_parquet(
        output / "hkust_whole_building_hourly_calibration.parquet", index=False
    )
    daily.to_parquet(output / "hkust_whole_building_daily_validation.parquet", index=False)
    (output / "hkust_whole_building_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return validation


def annual_calibration_weather_2023() -> pd.DataFrame:
    """Create an observed-range-constrained 2023 hourly reference year."""
    weather = load_hko_anchored_nasa_csv(HKUST_WEATHER, "SKG")
    weather = weather.loc[weather["timestamp_hkt"].dt.year.eq(2023)].copy()
    expected = pd.date_range("2023-01-01 00:00", "2023-12-31 23:00", freq="h")
    weather = weather.set_index("timestamp_hkt").reindex(expected).rename_axis("timestamp_hkt").reset_index()
    if weather[["temperature_c", "relative_humidity_pct", "solar_radiation_w_m2"]].isna().any().any():
        raise ValueError("2023 annual calibration weather is not complete")
    reference = pd.read_parquet(POLYU_HOURLY_WEATHER)
    reference["timestamp_hkt"] = pd.to_datetime(reference["timestamp_hkt"]).dt.tz_localize(None)
    reference["month"] = reference["timestamp_hkt"].dt.month
    reference["hour"] = reference["timestamp_hkt"].dt.hour
    climatology = reference.groupby(["month", "hour"], as_index=False).agg(
        WS10M=("wind_speed_m_s", "median"),
        PRECTOTCORR=("precipitation_mm_h", "median"),
    )
    weather["month"] = weather["timestamp_hkt"].dt.month
    weather["hour"] = weather["timestamp_hkt"].dt.hour
    weather = weather.merge(climatology, on=["month", "hour"], how="left", validate="many_to_one")
    weather["timestamp_hkt"] = weather["timestamp_hkt"].dt.tz_localize("Asia/Hong_Kong")
    weather["event_id"] = "ANNUAL_2023"
    weather["event_phase"] = "heatwave"
    weather["ALLSKY_SFC_SW_DWN"] = weather["solar_radiation_w_m2"]
    weather["WS10M"] = numeric(weather["WS10M"]).fillna(2.5)
    weather["PRECTOTCORR"] = numeric(weather["PRECTOTCORR"], lower=0).fillna(0.0)
    return weather[
        [
            "timestamp_hkt",
            "event_id",
            "event_phase",
            "temperature_c",
            "relative_humidity_pct",
            "ALLSKY_SFC_SW_DWN",
            "WS10M",
            "PRECTOTCORR",
        ]
    ]


def validate_weather(weather: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp_hkt", "event_id", "event_phase", "temperature_c", "relative_humidity_pct", "ALLSKY_SFC_SW_DWN", "WS10M", "PRECTOTCORR"}
    missing = required - set(weather)
    if missing:
        raise ValueError(f"Weather missing {sorted(missing)}")
    result = weather.copy()
    result["timestamp_hkt"] = pd.to_datetime(result["timestamp_hkt"], utc=True).dt.tz_convert("Asia/Hong_Kong")
    result = result.sort_values("timestamp_hkt").reset_index(drop=True)
    if result["event_id"].nunique() != 1 or not result["timestamp_hkt"].diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Each EnergyPlus weather sequence must be one continuous hourly event")
    return result


def write_epw(weather: pd.DataFrame, destination: Path) -> None:
    weather = validate_weather(weather)
    first, last = weather["timestamp_hkt"].iloc[0], weather["timestamp_hkt"].iloc[-1]
    header = [
        "LOCATION,Hong Kong,HK,HKG,Research,450070,22.31,114.17,8.0,65.0",
        "DESIGN CONDITIONS,0", "TYPICAL/EXTREME PERIODS,0", "GROUND TEMPERATURES,0",
        "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0",
        "COMMENTS 1,HKO-constrained heatwave event; research use",
        "COMMENTS 2,NASA POWER global horizontal radiation decomposed to direct and diffuse components with the hourly Erbs correlation",
        f"DATA PERIODS,1,1,Data,{first.day_name()},{first.month}/{first.day},{last.month}/{last.day}",
    ]
    rows = []
    for row in weather.itertuples(index=False):
        stamp, dry, rh = row.timestamp_hkt, float(row.temperature_c), float(row.relative_humidity_pct)
        global_horizontal, direct_normal, diffuse_horizontal, _ = erbs_solar_decomposition(
            stamp, float(row.ALLSKY_SFC_SW_DWN)
        )
        pressure = float(getattr(row, "atmospheric_pressure_pa", 101325.0))
        wind_direction = float(getattr(row, "WD10M", 180.0)) % 360.0
        fields = [stamp.year, stamp.month, stamp.day, stamp.hour + 1, 60, "?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9", f"{dry:.3f}", f"{dewpoint_c(dry, rh):.3f}", f"{rh:.2f}", f"{pressure:.1f}", "9999", "9999", "9999", f"{global_horizontal:.3f}", f"{direct_normal:.3f}", f"{diffuse_horizontal:.3f}", "999999", "999999", "999999", "9999", f"{wind_direction:.1f}", f"{max(float(row.WS10M), 0):.3f}", "5", "5", "20", "77777", "9", "999999999", "999", ".999", "0", "88", ".2", f"{max(float(row.PRECTOTCORR), 0):.3f}", "1"]
        rows.append(",".join(map(str, fields)))
    destination.write_text("\n".join(header + rows) + "\n", encoding="ascii")


def audit_solar_decomposition(weather: pd.DataFrame) -> dict:
    """Verify the GHI-to-DNI/DHI split before EnergyPlus consumes it."""
    records = [
        erbs_solar_decomposition(stamp, ghi)
        for stamp, ghi in zip(
            pd.to_datetime(weather["timestamp_hkt"]),
            numeric(weather["ALLSKY_SFC_SW_DWN"], lower=0).fillna(0.0),
        )
    ]
    solar = pd.DataFrame(records, columns=["ghi", "dni", "dhi", "cos_zenith"])
    daylight = solar["ghi"].gt(1.0) & solar["cos_zenith"].gt(1e-6)
    closure = (
        solar["dhi"] + solar["dni"] * solar["cos_zenith"] - solar["ghi"]
    ).abs()
    checks = {
        "finite_components": bool(np.isfinite(solar.to_numpy(dtype=float)).all()),
        "nonnegative_components": bool(solar[["ghi", "dni", "dhi"]].ge(0.0).all().all()),
        "ghi_energy_closure": bool(closure.le(1e-6).all()),
        "direct_and_diffuse_daylight_present": bool(
            daylight.any()
            and solar.loc[daylight, "dni"].gt(0.0).any()
            and solar.loc[daylight, "dhi"].gt(0.0).any()
        ),
        "dni_within_extraterrestrial_bound": bool(solar["dni"].le(1425.0).all()),
    }
    result = {
        "method": "hourly Erbs diffuse-fraction correlation with NOAA solar geometry",
        "weather_rows": int(len(solar)),
        "daylight_rows": int(daylight.sum()),
        "maximum_ghi_closure_error_w_m2": float(closure.max()),
        "maximum_dni_w_m2": float(solar["dni"].max()),
        "maximum_dhi_w_m2": float(solar["dhi"].max()),
        "checks": checks,
        "pass": bool(all(checks.values())),
    }
    if not result["pass"]:
        failures = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Solar decomposition audit failed: {failures}")
    return result


def climate_weather(event_weather: pd.DataFrame, climate_case: str) -> pd.DataFrame:
    """Return paired weather; only temperature and humidity differ between cases."""
    result = event_weather.copy()
    if climate_case == "counterfactual":
        required = {"counterfactual_temperature_c", "counterfactual_relative_humidity_pct"}
        missing = required - set(result)
        if missing:
            raise ValueError(f"Counterfactual weather missing {sorted(missing)}")
        result["temperature_c"] = numeric(result["counterfactual_temperature_c"])
        result["relative_humidity_pct"] = numeric(result["counterfactual_relative_humidity_pct"])
    elif climate_case != "heatwave":
        raise ValueError(f"Unknown climate case {climate_case!r}")
    if result[["temperature_c", "relative_humidity_pct"]].isna().any().any():
        raise ValueError(f"{climate_case} weather contains missing temperature or humidity")
    result["climate_case"] = climate_case
    return validate_weather(result)


def future_event_weather(
    event_weather: pd.DataFrame,
    delta_tmax_c: float,
    delta_tmin_c: float,
    huss_ratio: float,
    rsds_ratio: float,
    wind_ratio: float,
    pressure_ratio: float = 1.0,
) -> pd.DataFrame:
    """Morph an observed hourly heatwave analogue with audited climate changes.

    The imposed warming reaches ``delta_tmax_c`` near the daily radiation
    maximum and ``delta_tmin_c`` at night. Relative humidity is recomputed
    after scaling specific humidity, rather than being held fixed while dry-
    bulb temperature changes.
    """
    result = climate_weather(event_weather, "heatwave").copy()
    factors = {
        "delta_tmax_c": delta_tmax_c,
        "delta_tmin_c": delta_tmin_c,
        "huss_ratio": huss_ratio,
        "rsds_ratio": rsds_ratio,
        "wind_ratio": wind_ratio,
        "pressure_ratio": pressure_ratio,
    }
    for name, value in factors.items():
        if not np.isfinite(value):
            raise ValueError(f"Future weather factor {name} is not finite: {value}")
        if name.endswith("_ratio") and value <= 0:
            raise ValueError(f"Future weather factor {name} must be positive: {value}")

    solar = numeric(result["ALLSKY_SFC_SW_DWN"], lower=0).fillna(0.0)
    dates = result["timestamp_hkt"].dt.date
    daily_solar_max = solar.groupby(dates).transform("max").replace(0.0, np.nan)
    daytime_weight = (solar / daily_solar_max).fillna(0.0).clip(0.0, 1.0)
    temperature_delta = float(delta_tmin_c) + (
        float(delta_tmax_c) - float(delta_tmin_c)
    ) * daytime_weight

    base_temperature = numeric(result["temperature_c"])
    base_rh = numeric(result["relative_humidity_pct"]).clip(1.0, 100.0)
    base_pressure = pd.Series(101325.0, index=result.index)
    saturation_pa = 611.2 * np.exp(
        17.67 * base_temperature / (base_temperature + 243.5)
    )
    vapour_pa = base_rh / 100.0 * saturation_pa
    specific_humidity = 0.622 * vapour_pa / (base_pressure - 0.378 * vapour_pa)

    result["temperature_c"] = base_temperature + temperature_delta
    result["atmospheric_pressure_pa"] = base_pressure * float(pressure_ratio)
    future_specific_humidity = specific_humidity * float(huss_ratio)
    future_vapour_pa = (
        future_specific_humidity * result["atmospheric_pressure_pa"]
        / (0.622 + 0.378 * future_specific_humidity)
    )
    future_saturation_pa = 611.2 * np.exp(
        17.67 * result["temperature_c"] / (result["temperature_c"] + 243.5)
    )
    result["relative_humidity_pct"] = (
        100.0 * future_vapour_pa / future_saturation_pa
    ).clip(1.0, 100.0)
    result["ALLSKY_SFC_SW_DWN"] = solar * float(rsds_ratio)
    result["WS10M"] = numeric(result["WS10M"], lower=0).fillna(0.0) * float(wind_ratio)
    result["future_delta_tmax_c"] = float(delta_tmax_c)
    result["future_delta_tmin_c"] = float(delta_tmin_c)
    result["climate_case"] = "future_heatwave"
    return validate_weather(result)


def hour_active(hour: int, start: int, end: int) -> bool:
    return start <= hour < end if start < end else hour >= start or hour < end


def prototype_parameter_table() -> pd.DataFrame:
    """Load and validate all physics priors from the sole project config."""
    physics = CONFIG["prototype_physics"]
    fields = list(physics["field_order"])
    records = [
        {"prototype_id": prototype_id, **dict(zip(fields, values))}
        for prototype_id, values in physics["values"].items()
    ]
    frame = pd.DataFrame(records)
    registry = pd.DataFrame(CONFIG["prototypes"])[["prototype_id", "sector"]]
    if set(frame["prototype_id"]) != set(registry["prototype_id"]):
        raise ValueError("Prototype physics and prototype registry identifiers differ")
    if any(len(values) != len(fields) for values in physics["values"].values()):
        raise ValueError("A prototype physics row does not match field_order")
    frame = registry.merge(frame, on="prototype_id", how="left", validate="one_to_one")
    frame["parameter_status"] = physics["parameter_status"]
    numeric_fields = [
        field
        for field in fields
        if field not in {
            "window_to_wall_ratio", "window_shgc", "equipment_radiant_fraction"
        }
    ]
    if frame[numeric_fields].apply(pd.to_numeric, errors="coerce").le(0).any().any():
        raise ValueError("Prototype physics contains missing or non-positive physical values")
    fractions = frame[
        ["window_to_wall_ratio", "window_shgc", "equipment_radiant_fraction"]
    ].apply(pd.to_numeric, errors="coerce")
    if fractions.isna().any().any() or not fractions.apply(
        lambda values: values.between(0.0, 1.0, inclusive="both")
    ).all().all():
        raise ValueError("Prototype physics contains an invalid fractional parameter")
    return frame


def prototype_parameter_provenance(parameters: pd.DataFrame) -> pd.DataFrame:
    """Return one auditable source/basis record for every prototype parameter."""
    source_map = {
        "floor_area_m2": (
            "normalised_reference_geometry",
            "simulation normalisation; stock results are rescaled by each component GFA",
            "local_project_config",
        ),
        "height_m": (
            "code_and_typology_prior",
            "representative storey height used for the reference-cell geometry",
            "https://www.emsd.gov.hk/beeo/en/mibec_beeo_codtechguidelines.html|https://doi.org/10.3390/buildings12091414",
        ),
        "u_wall_w_m2k": (
            "hong_kong_typology_prior",
            "use- and age-sensitive envelope prior; retained as a mechanism variable",
            "https://www.bd.gov.hk/doc/en/resources/codes-and-references/code-and-design-manuals/OTTV1995_e.pdf|https://doi.org/10.3390/buildings12091414|https://doi.org/10.1016/j.enbuild.2019.109373",
        ),
        "u_roof_w_m2k": (
            "hong_kong_typology_prior",
            "roof-construction prior used with explicit top-floor exposure",
            "https://www.bd.gov.hk/doc/en/resources/codes-and-references/code-and-design-manuals/OTTV1995_e.pdf|https://doi.org/10.3390/buildings12091414|https://doi.org/10.1016/j.enbuild.2019.109373",
        ),
        "window_to_wall_ratio": (
            "hong_kong_typology_prior",
            "facade glazing prior differentiated by use and housing form",
            "https://www.bd.gov.hk/doc/en/resources/codes-and-references/code-and-design-manuals/OTTV1995_e.pdf|https://doi.org/10.3390/buildings12091414|https://doi.org/10.1016/j.enbuild.2019.109373",
        ),
        "window_shgc": (
            "hong_kong_typology_prior",
            "solar-gain prior differentiated by use and housing form",
            "https://www.bd.gov.hk/doc/en/resources/codes-and-references/code-and-design-manuals/OTTV1995_e.pdf|https://doi.org/10.3390/buildings12091414|https://doi.org/10.1016/j.enbuild.2019.109373",
        ),
        "window_u_w_m2k": (
            "hong_kong_typology_prior",
            "glazing conductance differentiated by residential fabric and commercial glazing family",
            "https://doi.org/10.3390/buildings12091414|https://doi.org/10.3390/en14164850|https://doi.org/10.3390/buildings15132266",
        ),
        "infiltration_ach": (
            "building_physics_prior",
            "air-leakage prior; natural ventilation is modelled separately when residential AC is off",
            "https://doi.org/10.3390/buildings12091414|https://doi.org/10.1016/j.enbuild.2019.109373|https://doi.org/10.1016/j.enbuild.2017.03.042",
        ),
        "thermal_capacitance_kj_m2k": (
            "building_physics_prior",
            "effective exposed thermal-mass prior for transient heat storage",
            "https://doi.org/10.1016/j.enbuild.2017.03.042|https://doi.org/10.3390/buildings12091414",
        ),
        "baseline_ac_cop": (
            "equipment_efficiency_prior",
            "use-specific operating COP prior before EMSD energy-component calibration",
            "https://www.emsd.gov.hk/beeo/en/mibec_beeo_codtechguidelines.html|https://doi.org/10.1016/j.enbuild.2019.109373",
        ),
        "people_per_m2": (
            "code_schedule_prior",
            "design occupant density combined with use-specific hourly presence schedules",
            "https://www.emsd.gov.hk/beeo/en/mibec_beeo_codtechguidelines.html|https://doi.org/10.1016/j.enbuild.2019.109373",
        ),
        "lighting_w_m2": (
            "code_schedule_prior",
            "installed lighting-density prior combined with use-specific hourly schedules",
            "https://www.emsd.gov.hk/beeo/en/mibec_beeo_codtechguidelines.html",
        ),
        "equipment_w_m2": (
            "code_and_use_prior",
            "plug/process-gain prior differentiated by use, including continuous IT gains",
            "https://www.emsd.gov.hk/beeo/en/mibec_beeo_codtechguidelines.html|https://doi.org/10.1016/j.enbuild.2017.03.042",
        ),
        "equipment_radiant_fraction": (
            "energyplus_internal_gain_prior",
            "radiant fraction of plug/process gains; general equipment uses 0.30 and the air-cooled IT process load is represented as convective heat gain",
            "https://bigladdersoftware.com/epx/docs/24-2/input-output-reference/group-internal-gains-people-lights-other.html|https://energyplus.net/assets/nrel_custom/pdfs/pdfs_v24.2.0/EngineeringReference.pdf",
        ),
        "cooling_capacity_w_m2": (
            "stress_test_capacity_prior",
            "available sensible-plus-latent cooling capacity before the pre-specified 70% service case",
            "https://www.emsd.gov.hk/beeo/en/mibec_beeo_codtechguidelines.html|https://doi.org/10.1016/j.enbuild.2019.109373|https://doi.org/10.1016/j.energy.2026.142082",
        ),
    }
    rows = []
    identity = ["prototype_id", "sector"]
    for record in parameters.to_dict(orient="records"):
        for field in CONFIG["prototype_physics"]["field_order"]:
            evidence_tier, basis, sources = source_map[field]
            rows.append(
                {
                    **{name: record[name] for name in identity},
                    "parameter": field,
                    "value": float(record[field]),
                    "evidence_tier": evidence_tier,
                    "basis": basis,
                    "source_ids": sources,
                    "directly_calibrated_parameter": False,
                    "downstream_calibration": (
                        "EMSD 2023 cooling/non-cooling mass balance; PolyU and HKUST held-out temporal response"
                    ),
                }
            )
    result = pd.DataFrame(rows)
    expected = len(parameters) * len(CONFIG["prototype_physics"]["field_order"])
    if len(result) != expected or result[["prototype_id", "parameter"]].duplicated().any():
        raise ValueError("Prototype parameter provenance is incomplete or duplicated")
    return result


def zone_layout(prototype: pd.Series) -> list[dict[str, object]]:
    if prototype["sector"] == "residential":
        return [
            {"name": "Living", "area_fraction": 0.44, "facade": "South", "conditioned": True},
            {"name": "Bedroom", "area_fraction": 0.38, "facade": "East", "conditioned": True},
            {"name": "Service", "area_fraction": 0.18, "facade": "North", "conditioned": False},
        ]
    return [
        {"name": "SouthPerimeter", "area_fraction": 0.15, "facade": "South", "conditioned": True},
        {"name": "EastPerimeter", "area_fraction": 0.15, "facade": "East", "conditioned": True},
        {"name": "NorthPerimeter", "area_fraction": 0.15, "facade": "North", "conditioned": True},
        {"name": "WestPerimeter", "area_fraction": 0.15, "facade": "West", "conditioned": True},
        {"name": "Core", "area_fraction": 0.40, "facade": None, "conditioned": True},
    ]


def commercial_hours(prototype_id: str, weekend: bool, profile: str) -> tuple[int, int]:
    if prototype_id in {"com_human_health", "com_data_centre", "com_accommodation"} or profile == "continuous":
        return 0, 24
    base = {
        "com_office": (8, 19),
        "com_education": (8, 19),
        "com_retail": (10, 22),
        "com_restaurant": (7, 24),
    }.get(prototype_id, (8, 20))
    if weekend and prototype_id in {"com_office", "com_education"}:
        base = (9, 14)
    if profile == "extended":
        return max(0, base[0] - 1), min(24, base[1] + 2)
    return base


def behaviour_schedules(
    weather: pd.DataFrame,
    prototype: pd.Series,
    behaviour: str,
    offset: int,
    use_probability: float,
    supply_stage: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.Timestamp | None, pd.Timestamp | None, float]:
    timestamps = weather["timestamp_hkt"]
    layout = zone_layout(prototype)
    schedules = pd.DataFrame(index=range(len(weather)))
    rng = np.random.default_rng(seed)
    day_use = {day: bool(rng.random() <= use_probability) for day in timestamps.dt.date}
    for zone in layout:
        name = str(zone["name"])
        occ_values, ac_values, vent_values, mechvent_values, light_values, plug_values = [], [], [], [], [], []
        for stamp in timestamps:
            weekend = stamp.weekday() >= 5
            if prototype["sector"] == "residential":
                if behaviour in {"older_home", "wfh"}:
                    home = True
                else:
                    home = weekend or hour_active(stamp.hour, 18, 8)
                if name == "Bedroom":
                    occupied = home and hour_active(stamp.hour, 22, 8 if not weekend else 9)
                    start, end = ((23, 7) if behaviour == "frugal" else (22, 8 if not weekend else 9))
                elif name == "Living":
                    if behaviour == "older_home":
                        occupied = hour_active(stamp.hour, 8, 23)
                        start, end = 10, 23
                    elif behaviour == "wfh":
                        occupied = hour_active(stamp.hour, 8, 24)
                        start, end = 9, 24
                    elif weekend:
                        occupied = hour_active(stamp.hour, 8, 24)
                        start, end = 10, 24
                    else:
                        occupied = home and hour_active(stamp.hour, 17, 24)
                        start, end = (20, 24) if behaviour == "frugal" else (18, 24)
                else:
                    occupied = home and (hour_active(stamp.hour, 7, 10) or hour_active(stamp.hour, 18, 22))
                    start, end = 0, 0
                occ = 1.0 if occupied else (0.12 if home else 0.03)
                start, end = (start + offset) % 24, (end + offset) % 24
                ac = bool(zone["conditioned"]) and occupied and day_use[stamp.date()] and hour_active(stamp.hour, start, end)
                lights = 0.08 + 0.92 * occ if stamp.hour >= 18 or stamp.hour < 7 else 0.05 * occ
                plugs = 0.20 + 0.80 * occ
                mechanical_ventilation = 0.0
            else:
                start, end = commercial_hours(str(prototype["prototype_id"]), weekend, behaviour)
                active = hour_active(stamp.hour, start, end) if (start, end) != (0, 24) else True
                occ = (0.65 if behaviour == "low_occupancy" else 1.0) if active else 0.03
                ac_start, ac_end = (start - 1 + offset) % 24, (end + 1 + offset) % 24
                ac = bool(zone["conditioned"]) and day_use[stamp.date()] and (True if (start, end) == (0, 24) else hour_active(stamp.hour, ac_start, ac_end))
                lights = 0.12 + 0.88 * occ
                plugs = 0.35 + 0.65 * occ
                mechanical_ventilation = occ if active else 0.0
            occ_values.append(float(occ))
            ac_values.append(float(ac))
            vent_values.append(
                float(prototype["sector"] == "residential" and bool(zone["conditioned"]) and occupied and not ac)
            )
            mechvent_values.append(
                float(prototype["sector"] == "commercial" and bool(zone["conditioned"]) and mechanical_ventilation)
            )
            light_values.append(float(min(max(lights, 0.0), 1.0)))
            plug_values.append(float(min(max(plugs, 0.0), 1.0)))
        schedules[f"{name}_occupancy"] = occ_values
        schedules[f"{name}_ac"] = ac_values
        schedules[f"{name}_vent"] = vent_values
        schedules[f"{name}_mechvent"] = mechvent_values
        schedules[f"{name}_lights"] = light_values
        schedules[f"{name}_plugs"] = plug_values

    heat_indices = np.flatnonzero(weather["event_phase"].eq("heatwave").to_numpy())
    if not len(heat_indices):
        raise ValueError("Weather contains no heatwave phase")
    ac_columns = [column for column in schedules if column.endswith("_ac")]
    occupancy_columns = [column for column in schedules if column.endswith("_occupancy")]
    active_cooling = schedules[ac_columns].sum(axis=1).gt(0).to_numpy()
    occupied_service = schedules[occupancy_columns].max(axis=1).gt(0.15).to_numpy()
    eligible = (
        weather["event_phase"].eq("heatwave").to_numpy()
        & active_cooling
        & occupied_service
    )
    if not eligible.any():
        eligible = weather["event_phase"].eq("heatwave").to_numpy() & active_cooling
    if not eligible.any():
        eligible = weather["event_phase"].eq("heatwave").to_numpy()
    eligible_positions = np.flatnonzero(eligible)
    peak_index = int(eligible_positions[np.argmax(numeric(weather.iloc[eligible_positions]["temperature_c"]).to_numpy())])
    outage_start: pd.Timestamp | None = None
    outage_end: pd.Timestamp | None = None
    capacity_fraction = 0.70 if supply_stage == "capacity_70pct" else 1.0
    if supply_stage == "outage_full_event":
        start_index, end_index = int(heat_indices[0]), int(heat_indices[-1]) + 1
    elif supply_stage == "outage_common_24h":
        # A common clock is required for calendar-time city maps and videos.
        # Unlike the standardized prototype-specific peak outage used in BERI,
        # every prototype loses supply at the same hottest outdoor event hour.
        start_index = int(
            heat_indices[
                np.argmax(numeric(weather.iloc[heat_indices]["temperature_c"]).to_numpy())
            ]
        )
        end_index = min(len(schedules), start_index + 24)
    elif supply_stage in {"outage_peak_6h", "outage_peak_24h"}:
        hours = 6 if supply_stage.endswith("6h") else 24
        # Interrupt service at the hottest occupied heatwave hour with active
        # scheduled cooling.  This avoids both unoccupied pre-cooling periods
        # and free-running hours before a daytime use becomes occupied.
        start_index = peak_index
        end_index = min(len(schedules), start_index + hours)
    elif supply_stage not in {"normal", "capacity_100pct", "capacity_70pct"}:
        raise ValueError(f"Unknown supply stage {supply_stage!r}")
    else:
        start_index = end_index = -1
    if start_index >= 0:
        # These stages represent an unmitigated grid outage, not an isolated
        # cooling-system fault. All grid-dependent end uses therefore stop
        # together. Occupancy is retained so passive thermal safety and
        # occupied service loss remain observable. Backup supply belongs in a
        # separate adaptation scenario and is not implicit in this baseline.
        grid_electric_columns = [
            column
            for column in schedules
            if column.endswith(("_ac", "_mechvent", "_lights", "_plugs"))
        ]
        schedules.loc[start_index : end_index - 1, grid_electric_columns] = 0.0
        if prototype["sector"] == "residential":
            # Window opening remains available during a grid outage. The
            # ZoneVentilation object still enforces its temperature and wind
            # limits; this schedule represents an occupied household trying
            # the passive adaptation that does not require grid electricity.
            for zone in layout:
                name = str(zone["name"])
                if bool(zone["conditioned"]):
                    occupied = schedules.loc[
                        start_index : end_index - 1, f"{name}_occupancy"
                    ].gt(0.15)
                    schedules.loc[
                        start_index : end_index - 1, f"{name}_vent"
                    ] = occupied.astype(float)
        outage_start = timestamps.iloc[start_index]
        outage_end = timestamps.iloc[end_index] if end_index < len(timestamps) else None
    return schedules, outage_start, outage_end, capacity_fraction


def annualize_schedules(schedules: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    years = weather["timestamp_hkt"].dt.year.unique()
    if len(years) != 1:
        raise ValueError("EnergyPlus event schedule cannot cross a calendar year")
    year = int(years[0])
    annual_index = pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:00", freq="h", tz="Asia/Hong_Kong")
    defaults = {}
    for column in schedules:
        if column.endswith("_occupancy"):
            defaults[column] = 0.03
        elif column.endswith("_ac"):
            defaults[column] = 0.0
        elif column.endswith("_vent"):
            defaults[column] = 0.0
        elif column.endswith("_mechvent"):
            defaults[column] = 0.0
        elif column.endswith("_lights"):
            defaults[column] = 0.05
        elif column.endswith("_plugs"):
            defaults[column] = 0.20
        else:
            raise ValueError(f"Unknown schedule field {column!r}")
    annual = pd.DataFrame(defaults, index=annual_index)
    event = schedules.copy()
    event.index = pd.DatetimeIndex(weather["timestamp_hkt"])
    if not event.index.isin(annual.index).all():
        raise ValueError("Event timestamps do not fit the annual EnergyPlus schedule")
    annual.loc[event.index, schedules.columns] = event[schedules.columns].to_numpy()
    return annual.reset_index(drop=True)


def vertices_text(vertices: list[tuple[float, float, float]]) -> str:
    return ", ".join(f"{x:.4f},{y:.4f},{z:.4f}" for x, y, z in vertices)


def build_idf(
    prototype: pd.Series,
    orientation: int,
    floor_position: str,
    setpoint_c: float,
    schedules: pd.DataFrame,
    free_running: bool,
    capacity_fraction: float = 1.0,
    cooling_capacity_limited: bool = True,
) -> tuple[str, dict[str, float], set[str]]:
    height = float(prototype["height_m"])
    total_area = float(prototype["floor_area_m2"])
    layout = zone_layout(prototype)
    zone_areas = {str(zone["name"]): total_area * float(zone["area_fraction"]) for zone in layout}
    mass_thickness = min(max(float(prototype["thermal_capacitance_kj_m2k"]) * 1000 / (2200 * 900), 0.005), 0.50)
    envelope_mass_thickness = 0.15
    concrete_conductivity = 1.4
    wall_resistance = max(
        0.01,
        1 / float(prototype["u_wall_w_m2k"]) - envelope_mass_thickness / concrete_conductivity,
    )
    roof_resistance = max(
        0.01,
        1 / float(prototype["u_roof_w_m2k"]) - envelope_mass_thickness / concrete_conductivity,
    )
    roof_boundary = "Outdoors" if floor_position in {"top", "single"} else "Adiabatic"
    floor_boundary = "Ground" if floor_position in {"bottom", "single"} else "Adiabatic"
    floor_construction = (
        "FloorConstruction" if floor_position in {"bottom", "single"} else "AdiabaticConstruction"
    )
    floor_material_and_construction = (
        [
            "Material, FloorMaterial, MediumRough, 0.15, 1.4, 2200, 900, 0.9, 0.7, 0.7;",
            "Construction, FloorConstruction, FloorMaterial;",
        ]
        if floor_position in {"bottom", "single"}
        else []
    )
    schedule_objects = []
    schedule_columns = {column: index + 1 for index, column in enumerate(schedules.columns)}
    for column, index in schedule_columns.items():
        # The file contains hourly state/fraction values.  In particular, AC,
        # ventilation and outage fields are discrete controls; EnergyPlus
        # interpolation would create artificial fractional equipment starts
        # within the first 15-minute timestep of every scheduled hour.
        schedule_objects.append(
            f"Schedule:File, {column}, Fraction, schedules.csv, {index}, 0, {len(schedules)}, Comma, No, 60;"
        )
    zones, surfaces, windows, internal, loads, ventilation, hvac = [], [], [], [], [], [], []
    x_offset = 0.0
    window_to_wall = min(max(float(prototype["window_to_wall_ratio"]), 0.02), 0.80)
    for zone in layout:
        name = str(zone["name"])
        area = zone_areas[name]
        width = math.sqrt(area * 1.35)
        depth = area / width
        x0, x1 = x_offset, x_offset + width
        y0, y1 = 0.0, depth
        x_offset = x1 + 2.0
        zones.append(f"Zone, {name}, 0, 0, 0, 0, 1, 1, {height:.4f}, {area * height:.4f}, {area:.4f};")
        facade = zone["facade"]
        wall_vertices = {
            "South": [(x0, y0, 0), (x1, y0, 0), (x1, y0, height), (x0, y0, height)],
            "East": [(x1, y0, 0), (x1, y1, 0), (x1, y1, height), (x1, y0, height)],
            "North": [(x1, y1, 0), (x0, y1, 0), (x0, y1, height), (x1, y1, height)],
            "West": [(x0, y1, 0), (x0, y0, 0), (x0, y0, height), (x0, y1, height)],
        }
        for direction, vertices in wall_vertices.items():
            exterior = facade == direction
            construction = "WallConstruction" if exterior else "AdiabaticConstruction"
            boundary = "Outdoors" if exterior else "Adiabatic"
            sun, wind = ("SunExposed", "WindExposed") if exterior else ("NoSun", "NoWind")
            surface_name = f"{name}{direction}Wall"
            surfaces.append(
                f"BuildingSurface:Detailed, {surface_name}, Wall, {construction}, {name}, , {boundary}, , {sun}, {wind}, Autocalculate, 4, {vertices_text(vertices)};"
            )
        floor_vertices = [(x0, y0, 0), (x0, y1, 0), (x1, y1, 0), (x1, y0, 0)]
        roof_vertices = [(x0, y0, height), (x1, y0, height), (x1, y1, height), (x0, y1, height)]
        surfaces.append(f"BuildingSurface:Detailed, {name}Floor, Floor, {floor_construction}, {name}, , {floor_boundary}, , NoSun, NoWind, Autocalculate, 4, {vertices_text(floor_vertices)};")
        surfaces.append(f"BuildingSurface:Detailed, {name}Roof, Roof, RoofConstruction, {name}, , {roof_boundary}, , {'SunExposed' if roof_boundary == 'Outdoors' else 'NoSun'}, {'WindExposed' if roof_boundary == 'Outdoors' else 'NoWind'}, Autocalculate, 4, {vertices_text(roof_vertices)};")
        if facade is not None:
            wall_length = width if facade in {"South", "North"} else depth
            window_height = min(1.6, height - 0.5)
            window_length = min(wall_length - 0.4, max(0.4, window_to_wall * wall_length * height / window_height))
            z0, z1 = 0.8, 0.8 + window_height
            if facade == "South":
                a, b = (x0 + x1 - window_length) / 2, (x0 + x1 + window_length) / 2
                points = [(a, y0, z0), (b, y0, z0), (b, y0, z1), (a, y0, z1)]
            elif facade == "North":
                a, b = (x0 + x1 - window_length) / 2, (x0 + x1 + window_length) / 2
                points = [(b, y1, z0), (a, y1, z0), (a, y1, z1), (b, y1, z1)]
            elif facade == "East":
                a, b = (y0 + y1 - window_length) / 2, (y0 + y1 + window_length) / 2
                points = [(x1, a, z0), (x1, b, z0), (x1, b, z1), (x1, a, z1)]
            else:
                a, b = (y0 + y1 - window_length) / 2, (y0 + y1 + window_length) / 2
                points = [(x0, b, z0), (x0, a, z0), (x0, a, z1), (x0, b, z1)]
            windows.append(f"FenestrationSurface:Detailed, {name}Window, Window, WindowConstruction, {name}{facade}Wall, , Autocalculate, , 1, 4, {vertices_text(points)};")
        internal.append(f"InternalMass, {name}Mass, MassConstruction, {name}, , {area:.4f};")
        occ, ac = f"{name}_occupancy", f"{name}_ac"
        lights_schedule, plugs_schedule = f"{name}_lights", f"{name}_plugs"
        people_density = float(prototype["people_per_m2"])
        if prototype["sector"] == "residential" and name == "Service":
            people_density *= 0.35
        loads.extend(
            [
                f"People, {name}People, {name}, {occ}, People/Area, , {people_density:.6f}, , 0.3, autocalculate, ActivitySchedule;",
                f"Lights, {name}Lights, {name}, {lights_schedule}, Watts/Area, , {float(prototype['lighting_w_m2']):.4f}, , 0, 0.42, 0.18, 1, GeneralLights;",
                f"ElectricEquipment, {name}PlugLoads, {name}, {plugs_schedule}, Watts/Area, , {float(prototype['equipment_w_m2']):.4f}, , 0, {float(prototype['equipment_radiant_fraction']):.4f}, 0;",
                f"ZoneInfiltration:DesignFlowRate, {name}Infiltration, {name}, AlwaysOn, AirChanges/Hour, , , , {float(prototype['infiltration_ach']):.6f}, 1, 0, 0, 0;",
            ]
        )
        if bool(zone["conditioned"]) and not free_running:
            capacity_w = float(prototype["cooling_capacity_w_m2"]) * area * capacity_fraction
            cooling_limit = "LimitCapacity" if cooling_capacity_limited else "NoLimit"
            cooling_capacity_field = f"{capacity_w:.3f}" if cooling_capacity_limited else ""
            hvac.extend(
                [
                    f"ThermostatSetpoint:SingleCooling, {name}CoolingThermostat, CoolingSetpoint;",
                    f"ZoneControl:Thermostat, {name}ZoneThermostat, {name}, ThermostatControl, ThermostatSetpoint:SingleCooling, {name}CoolingThermostat;",
                    f"ZoneControl:Humidistat, {name}ZoneHumidistat, {name}, HumidifyingRHSetpoint, DehumidifyingRHSetpoint;",
                    f"ZoneHVAC:IdealLoadsAirSystem, {name}IdealLoads, {ac}, {name}SupplyNode, , , 50, 13, 0.0156, 0.0077, NoLimit, , , {cooling_limit}, , {cooling_capacity_field}, , {ac}, Humidistat, , None;",
                    f"ZoneHVAC:EquipmentList, {name}Equipment, SequentialLoad, ZoneHVAC:IdealLoadsAirSystem, {name}IdealLoads, 1, 1;",
                    f"ZoneHVAC:EquipmentConnections, {name}, {name}Equipment, {name}SupplyNode, , {name}AirNode, {name}ReturnNode;",
                ]
            )
        if prototype["sector"] == "residential" and bool(zone["conditioned"]):
            ventilation.append(
                f"ZoneVentilation:DesignFlowRate, {name}NaturalVentilation, {name}, {name}_vent, "
                f"AirChanges/Hour, , , , {float(THERMAL_CONFIG['residential_natural_ventilation_ach']):.3f}, "
                f"Natural, 0, 1, 1, 0, 0, 0, {float(THERMAL_CONFIG['natural_ventilation_min_indoor_c']):.3f}, , "
                f"100, , {float(THERMAL_CONFIG['natural_ventilation_min_indoor_outdoor_delta_c']):.3f}, , "
                f"-100, , {float(THERMAL_CONFIG['natural_ventilation_max_outdoor_c']):.3f}, , "
                f"{float(THERMAL_CONFIG['natural_ventilation_max_wind_m_s']):.3f}, Outdoor;"
            )
        elif prototype["sector"] == "commercial" and bool(zone["conditioned"]):
            outdoor_air_l_s_person = float(
                THERMAL_CONFIG["commercial_outdoor_air_l_s_person"][str(prototype["prototype_id"])]
            )
            ventilation.append(
                f"ZoneVentilation:DesignFlowRate, {name}MechanicalVentilation, {name}, {name}_mechvent, "
                f"Flow/Person, , , {outdoor_air_l_s_person / 1000.0:.6f}, , Intake, 0, 1, 1, 0, 0, 0, "
                f"-100, , 100, , -100, , -100, , 100, , 40, Outdoor;"
            )
    mixing = []
    hub = "Service" if prototype["sector"] == "residential" else "Core"
    for zone in layout:
        name = str(zone["name"])
        if name == hub:
            continue
        mixing.append(f"ZoneMixing, {name}From{hub}, {name}, AlwaysOn, AirChanges/Hour, , , , 0.15, {hub};")
        mixing.append(f"ZoneMixing, {hub}From{name}, {hub}, AlwaysOn, AirChanges/Hour, , , , 0.15, {name};")
    idf = "\n".join(
        [
            "Version, 26.1;",
            "Timestep, 4;",
            "SimulationControl, No, No, No, No, Yes;",
            f"Building, HKMultiZonePrototype, {orientation}, Urban, 0.05, 0.4, FullExterior, 50, 1;",
            "Site:Location, Hong Kong, 22.31, 114.17, 8.0, 65.0;",
            "GlobalGeometryRules, UpperLeftCorner, CounterClockWise, Relative;",
            "HeatBalanceAlgorithm, ConductionTransferFunction;",
            "Site:GroundTemperature:BuildingSurface, "
            + ", ".join(str(value) for value in THERMAL_CONFIG["hko_1991_2020_soil_temperature_0_5m_c"])
            + ";",
            "ScheduleTypeLimits, Fraction, 0, 1, Continuous;",
            "ScheduleTypeLimits, ControlType, 0, 4, Discrete;",
            "ScheduleTypeLimits, Temperature, -60, 200, Continuous, Temperature;",
            "ScheduleTypeLimits, AnyNumber;",
            *schedule_objects,
            "Schedule:Compact, ThermostatControl, ControlType, Through: 12/31, For: AllDays, Until: 24:00, 2;",
            f"Schedule:Compact, CoolingSetpoint, Temperature, Through: 12/31, For: AllDays, Until: 24:00, {setpoint_c};",
            f"Schedule:Compact, HumidifyingRHSetpoint, AnyNumber, Through: 12/31, For: AllDays, Until: 24:00, {float(THERMAL_CONFIG['conditioned_relative_humidity_range_pct'][0]):.3f};",
            f"Schedule:Compact, DehumidifyingRHSetpoint, AnyNumber, Through: 12/31, For: AllDays, Until: 24:00, {float(THERMAL_CONFIG['conditioned_relative_humidity_range_pct'][1]):.3f};",
            "Schedule:Compact, ActivitySchedule, AnyNumber, Through: 12/31, For: AllDays, Until: 24:00, 120;",
            "Schedule:Compact, AlwaysOn, Fraction, Through: 12/31, For: AllDays, Until: 24:00, 1;",
            f"Material:NoMass, WallResistance, Rough, {wall_resistance:.8f}, 0.9, 0.7, 0.7;",
            f"Material:NoMass, RoofResistance, Rough, {roof_resistance:.8f}, 0.9, 0.7, 0.7;",
            f"Material, EnvelopeConcrete, MediumRough, {envelope_mass_thickness:.8f}, {concrete_conductivity:.4f}, 2200, 900, 0.9, 0.7, 0.7;",
            "Material:NoMass, AdiabaticMaterial, Rough, 5.0, 0.9, 0.7, 0.7;",
            f"Material, MassMaterial, MediumRough, {mass_thickness:.8f}, 1.4, 2200, 900, 0.9, 0.7, 0.7;",
            f"WindowMaterial:SimpleGlazingSystem, WindowMaterial, {float(prototype['window_u_w_m2k']):.6f}, {float(prototype['window_shgc']):.6f}, 0.8;",
            *floor_material_and_construction,
            "Construction, WallConstruction, WallResistance, EnvelopeConcrete;",
            "Construction, RoofConstruction, RoofResistance, EnvelopeConcrete;",
            "Construction, AdiabaticConstruction, AdiabaticMaterial;",
            "Construction, MassConstruction, MassMaterial;",
            "Construction, WindowConstruction, WindowMaterial;",
            *zones,
            *surfaces,
            *windows,
            *internal,
            *loads,
            *ventilation,
            *mixing,
            *hvac,
            "OutputControl:Files, Yes;",
            "Output:Variable, *, Zone Mean Air Temperature, Hourly;",
            "Output:Variable, *, Zone Air Relative Humidity, Hourly;",
            "Output:Variable, *, Zone Ideal Loads Supply Air Total Cooling Energy, Hourly;",
            "Output:Variable, *, Zone Ideal Loads Supply Air Sensible Cooling Energy, Hourly;",
            "Output:Variable, *, Zone Ideal Loads Supply Air Total Cooling Rate, Hourly;",
            "Output:Variable, *, Zone Cooling Setpoint Not Met Time, Hourly;",
            "Output:Variable, *, Lights Electricity Energy, Hourly;",
            "Output:Variable, *, Lights Electricity Rate, Hourly;",
            "Output:Variable, *, Electric Equipment Electricity Energy, Hourly;",
            "Output:Variable, *, Electric Equipment Electricity Rate, Hourly;",
            "",
        ]
    )
    conditioned_zones = {str(zone["name"]) for zone in layout if bool(zone["conditioned"])}
    return idf, zone_areas, conditioned_zones


def add_run_period(idf: str, weather: pd.DataFrame) -> str:
    first, last = weather["timestamp_hkt"].iloc[0], weather["timestamp_hkt"].iloc[-1]
    run = f"RunPeriod, Event, {first.month}, {first.day}, , {last.month}, {last.day}, , {first.day_name()}, No, No, No, Yes, Yes, No, Hour1;\n"
    marker = "Timestep, 4;\n"
    if marker not in idf:
        raise ValueError("EnergyPlus timestep marker missing; cannot add event run period")
    return idf.replace(marker, marker + run, 1)


def find_column(columns: list[str], phrase: str, required: bool = True) -> str | None:
    matches = [c for c in columns if phrase.lower() in c.lower()]
    if len(matches) == 1:
        return matches[0]
    if required:
        raise ValueError(f"Expected one column containing {phrase!r}; got {matches}")
    return None


def run_energyplus(
    executable: Path,
    idf: str,
    epw: Path,
    annual_schedules: pd.DataFrame,
    event_schedules: pd.DataFrame,
    weather: pd.DataFrame,
    cop: float,
    free_running: bool,
    zone_areas: dict[str, float],
    conditioned_zones: set[str],
) -> tuple[pd.DataFrame, str]:
    with tempfile.TemporaryDirectory(prefix="hk_v03_ep_") as temporary:
        work = Path(temporary)
        (work / "model.idf").write_text(idf, encoding="ascii")
        annual_schedules.to_csv(work / "schedules.csv", index=False, header=False)
        result = subprocess.run([str(executable), "-w", str(epw), "-d", str(work / "result"), str(work / "model.idf")], cwd=work, capture_output=True, text=True, encoding="utf-8", errors="replace")
        error_file = work / "result" / "eplusout.err"
        errors = error_file.read_text(encoding="utf-8", errors="replace") if error_file.exists() else ""
        if result.returncode != 0 or re.search(r"\*\*\s+Fatal\s+\*\*", errors):
            raise RuntimeError((result.stdout + result.stderr + errors)[-6000:])
        raw = pd.read_csv(work / "result" / "eplusout.csv")
        if len(raw) != len(weather):
            raise ValueError(f"EnergyPlus rows {len(raw)} != weather rows {len(weather)}")
        columns = list(raw)
        area_lookup = {name.lower(): area for name, area in zone_areas.items()}
        total_area = float(sum(zone_areas.values()))

        def matching(phrase: str) -> list[str]:
            return [column for column in columns if phrase.lower() in column.lower()]

        def summed(phrase: str) -> pd.Series:
            selected = matching(phrase)
            if not selected:
                return pd.Series(0.0, index=raw.index)
            return raw[selected].apply(numeric).sum(axis=1)

        def area_weighted(phrase: str, maximum: bool = False, conditioned_only: bool = False) -> pd.Series:
            selected = matching(phrase)
            if conditioned_only:
                selected = [column for column in selected if column.split(":", 1)[0].strip().lower() in {zone.lower() for zone in conditioned_zones}]
            if not selected:
                return pd.Series(math.nan, index=raw.index)
            values = raw[selected].apply(numeric)
            if maximum:
                return values.max(axis=1)
            weights = []
            for column in selected:
                key = column.split(":", 1)[0].strip().lower()
                weights.append(area_lookup.get(key, total_area / len(selected)))
            return values.mul(np.asarray(weights), axis=1).sum(axis=1) / float(sum(weights))

        temperature_columns = [column for column in matching("Zone Mean Air Temperature") if column.split(":", 1)[0].strip().lower() in {zone.lower() for zone in conditioned_zones}]
        temperature_by_zone = raw[temperature_columns].apply(numeric)
        canonical_zone = {zone.lower(): zone for zone in zone_areas}
        temperature_by_zone.columns = [canonical_zone[column.split(":", 1)[0].strip().lower()] for column in temperature_columns]
        occupied_temperatures = temperature_by_zone.copy()
        for zone in occupied_temperatures:
            occupied_temperatures.loc[event_schedules[f"{zone}_occupancy"].le(0.15), zone] = np.nan
        exposure_temperature = occupied_temperatures.max(axis=1)
        output = pd.DataFrame(
            {
                "timestamp_hkt": weather["timestamp_hkt"],
                "event_phase": weather["event_phase"],
                "outdoor_temperature_c": numeric(weather["temperature_c"]),
                "indoor_temperature_c": area_weighted("Zone Mean Air Temperature", conditioned_only=True),
                "exposure_temperature_c": exposure_temperature,
                "maximum_zone_temperature_c": area_weighted("Zone Mean Air Temperature", maximum=True, conditioned_only=True),
                "maximum_all_zone_temperature_c": area_weighted("Zone Mean Air Temperature", maximum=True),
                "indoor_rh_pct": area_weighted("Zone Air Relative Humidity", conditioned_only=True),
            }
        )
        occupancy_columns = [f"{zone}_occupancy" for zone in zone_areas if zone in conditioned_zones]
        occupancy_weights = np.asarray([zone_areas[column.removesuffix("_occupancy")] for column in occupancy_columns])
        output["occupied_fraction"] = event_schedules[occupancy_columns].mul(occupancy_weights, axis=1).sum(axis=1) / occupancy_weights.sum()
        ac_columns = [f"{zone}_ac" for zone in zone_areas if zone in conditioned_zones]
        ac_weights = np.asarray([zone_areas[column.removesuffix("_ac")] for column in ac_columns])
        ac_schedule = event_schedules[ac_columns].apply(numeric)
        output["active_cooling_fraction"] = ac_schedule.mul(ac_weights, axis=1).sum(axis=1) / ac_weights.sum()
        # Persist which conditioned zones start cooling in each hour.  This
        # identifies the single 15-minute start-up transient reported by
        # EnergyPlus without accepting any sustained temperature-control error.
        ac_started = ac_schedule.gt(0.0) & ac_schedule.shift(fill_value=0.0).le(0.0)
        output["cooling_start_transition_fraction"] = (
            ac_started.astype(float).mul(ac_weights, axis=1).sum(axis=1) / ac_weights.sum()
        )
        unmet_columns = matching("Zone Cooling Setpoint Not Met Time")
        unmet_weighted = pd.Series(0.0, index=raw.index)
        for column in unmet_columns:
            raw_zone = column.split(":", 1)[0].strip().lower()
            if raw_zone not in {zone.lower() for zone in conditioned_zones}:
                continue
            zone = canonical_zone[raw_zone]
            unmet_weighted += numeric(raw[column]) * event_schedules[f"{zone}_ac"].to_numpy() * zone_areas[zone]
        output["unmet_cooling_fraction"] = unmet_weighted / float(sum(zone_areas[zone] for zone in conditioned_zones))
        lighting_kwh_m2 = summed("Lights Electricity Energy") / 3_600_000 / total_area
        equipment_kwh_m2 = summed("Electric Equipment Electricity Energy") / 3_600_000 / total_area
        lighting_w_m2 = summed("Lights Electricity Rate") / total_area
        equipment_w_m2 = summed("Electric Equipment Electricity Rate") / total_area
        if free_running:
            output["cooling_kwh_m2"] = 0.0
            output["cooling_thermal_kwh_m2"] = 0.0
            output["latent_cooling_kwh_m2"] = 0.0
            output["cooling_electric_w_m2"] = 0.0
        else:
            total = summed("Total Cooling Energy") / 3_600_000 / total_area
            sensible_columns = matching("Sensible Cooling Energy")
            sensible = summed("Sensible Cooling Energy") / 3_600_000 / total_area if sensible_columns else total
            rate = summed("Total Cooling Rate") / total_area
            output["cooling_kwh_m2"] = total / cop
            output["cooling_thermal_kwh_m2"] = total
            output["latent_cooling_kwh_m2"] = (total - sensible).clip(lower=0) / cop
            output["cooling_electric_w_m2"] = rate / cop
        output["lighting_kwh_m2"] = lighting_kwh_m2
        output["equipment_kwh_m2"] = equipment_kwh_m2
        output["total_electricity_kwh_m2"] = output["cooling_kwh_m2"] + lighting_kwh_m2 + equipment_kwh_m2
        output["total_electric_w_m2"] = output["cooling_electric_w_m2"] + lighting_w_m2 + equipment_w_m2
        return output, errors


def run_annual_reference_models(
    executable: Path,
    parameters: pd.DataFrame,
    output: Path,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run one full observed-range-constrained year for each prototype."""
    weather = annual_calibration_weather_2023()
    weather_dir = output / "weather"
    weather_dir.mkdir(parents=True, exist_ok=True)
    epw = weather_dir / "annual_2023_hko_skg_anchored.epw"
    write_epw(weather, epw)
    contract_sha256 = simulation_contract_sha256()
    weather_sha256 = weather_frame_sha256(weather)
    cache_dir = output / "energyplus_run_cache" / "annual_2023"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    monthly_rows = []
    hourly_frames = []
    continuous = {"com_human_health", "com_data_centre", "com_accommodation"}
    for prototype in parameters.itertuples(index=False):
        cache = cache_dir / f"{prototype.prototype_id}_{MODEL_VERSION}.json"
        hourly_cache = cache.with_suffix(".parquet")
        if cache.exists() and hourly_cache.exists():
            payload = json.loads(cache.read_text(encoding="utf-8"))
            if (
                payload.get("energy_calibration_version") == ENERGY_CALIBRATION_VERSION
                and payload.get("simulation_contract_sha256") == contract_sha256
                and payload.get("weather_sha256") == weather_sha256
                and payload.get("monthly")
            ):
                rows.append(payload["summary"])
                monthly_rows.extend(payload["monthly"])
                hourly_frames.append(pd.read_parquet(hourly_cache))
                continue
        series = pd.Series(prototype._asdict())
        behaviour = (
            "working"
            if prototype.sector == "residential"
            else ("continuous" if prototype.prototype_id in continuous else "standard")
        )
        schedules, _, _, capacity_fraction = behaviour_schedules(
            weather,
            series,
            behaviour,
            0,
            1.0,
            "normal",
            seed + int(hashlib.sha256(prototype.prototype_id.encode()).hexdigest()[:8], 16),
        )
        annual_schedules = annualize_schedules(schedules, weather)
        idf, zone_areas, conditioned_zones = build_idf(
            series,
            0,
            "middle",
            25.0,
            annual_schedules,
            free_running=False,
            capacity_fraction=capacity_fraction,
            cooling_capacity_limited=False,
        )
        idf = add_run_period(idf, weather)
        hourly, warnings = run_energyplus(
            executable,
            idf,
            epw,
            annual_schedules,
            schedules,
            weather,
            float(prototype.baseline_ac_cop),
            free_running=False,
            zone_areas=zone_areas,
            conditioned_zones=conditioned_zones,
        )
        hourly["prototype_id"] = prototype.prototype_id
        hourly["sector"] = prototype.sector
        hourly["climate_case"] = "annual_observed"
        hourly["supply_stage"] = "normal"
        hourly["behavior_seed"] = int(seed)
        summary = {
            "prototype_id": prototype.prototype_id,
            "sector": prototype.sector,
            "official_energy_segment": official_energy_segment(prototype.prototype_id),
            "annual_total_eui_kwh_m2": float(hourly["total_electricity_kwh_m2"].sum()),
            "annual_cooling_eui_kwh_m2": float(hourly["cooling_kwh_m2"].sum()),
            "annual_non_cooling_eui_kwh_m2": float(
                hourly["lighting_kwh_m2"].sum() + hourly["equipment_kwh_m2"].sum()
            ),
            "annual_peak_w_m2": float(hourly["total_electric_w_m2"].max()),
            "severe_errors": int(warnings.count("** Severe  **")),
            "warnings": int(warnings.count("** Warning **")),
            "warning_messages": "\n".join(
                line.strip() for line in warnings.splitlines() if "** Warning **" in line
            ),
        }
        monthly = (
            hourly.assign(month=pd.to_datetime(hourly["timestamp_hkt"]).dt.month)
            .groupby("month", as_index=False)
            .agg(
                monthly_total_eui_kwh_m2=("total_electricity_kwh_m2", "sum"),
                monthly_cooling_eui_kwh_m2=("cooling_kwh_m2", "sum"),
                monthly_lighting_eui_kwh_m2=("lighting_kwh_m2", "sum"),
                monthly_equipment_eui_kwh_m2=("equipment_kwh_m2", "sum"),
            )
        )
        monthly["prototype_id"] = prototype.prototype_id
        monthly["sector"] = prototype.sector
        monthly_records = monthly.to_dict(orient="records")
        cache.write_text(
            json.dumps(
                {
                    "energy_calibration_version": ENERGY_CALIBRATION_VERSION,
                    "simulation_contract_sha256": contract_sha256,
                    "weather_sha256": weather_sha256,
                    "summary": summary,
                    "monthly": monthly_records,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        hourly.to_parquet(hourly_cache, index=False)
        rows.append(summary)
        monthly_rows.extend(monthly_records)
        hourly_frames.append(hourly)
        print(f"[annual calibration] {prototype.prototype_id} ready", flush=True)
    result = pd.DataFrame(rows)
    if len(result) != len(parameters) or result["severe_errors"].sum() != 0:
        raise RuntimeError("Annual EnergyPlus calibration ensemble is incomplete")
    result.to_csv(
        output / "annual_2023_prototype_energyplus_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )
    monthly_result = pd.DataFrame(monthly_rows)
    if len(monthly_result) != len(parameters) * 12:
        raise RuntimeError("Annual EnergyPlus monthly calibration output is incomplete")
    monthly_result.to_csv(
        output / "annual_2023_prototype_monthly_energyplus_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )
    hourly_result = pd.concat(hourly_frames, ignore_index=True)
    if len(hourly_result) != len(parameters) * len(weather):
        raise RuntimeError("Annual EnergyPlus hourly output is incomplete")
    return result, monthly_result, hourly_result


def component_area_table(inputs: gpd.GeoDataFrame) -> pd.DataFrame:
    stock = inputs.loc[inputs["pre_weather_simulation_candidate"]]
    areas = pd.concat(
        [
            stock.loc[
                stock["residential_component_present"],
                ["residential_prototype_id", "residential_gfa_m2"],
            ].rename(
                columns={
                    "residential_prototype_id": "prototype_id",
                    "residential_gfa_m2": "component_gfa_m2",
                }
            ),
            stock.loc[
                stock["commercial_component_present"],
                ["commercial_prototype_id", "commercial_gfa_m2"],
            ].rename(
                columns={
                    "commercial_prototype_id": "prototype_id",
                    "commercial_gfa_m2": "component_gfa_m2",
                }
            ),
        ],
        ignore_index=True,
    )
    areas["component_gfa_m2"] = numeric(areas["component_gfa_m2"], lower=0).fillna(0.0)
    return (
        areas.groupby("prototype_id", as_index=False, observed=True)["component_gfa_m2"]
        .sum()
    )


def build_official_energy_calibration(
    inputs: gpd.GeoDataFrame,
    annual_models: pd.DataFrame,
    annual_monthly: pd.DataFrame,
    output: Path,
) -> tuple[pd.DataFrame, dict]:
    targets = pd.read_csv(OFFICIAL_ENERGY_TARGETS)
    areas = component_area_table(inputs)
    table = annual_models.merge(areas, on="prototype_id", how="left", validate="one_to_one")
    table["official_energy_segment"] = table["prototype_id"].map(official_energy_segment)
    table["calibration_group"] = np.where(
        table["sector"].eq("residential"),
        table["official_energy_segment"],
        "commercial_sector_total",
    )
    residential_targets = targets.loc[targets["sector"].eq("residential")].copy()
    residential_targets["calibration_group"] = residential_targets["official_segment"]
    commercial_target = pd.DataFrame(
        [
            {
                "calibration_group": "commercial_sector_total",
                "electricity_tj": 108_524.0,
                "air_conditioning_tj": float(
                    targets.loc[targets["sector"].eq("commercial"), "air_conditioning_tj"].sum()
                ),
            }
        ]
    )
    group_targets = pd.concat(
        [
            residential_targets[
                ["calibration_group", "electricity_tj", "air_conditioning_tj"]
            ],
            commercial_target,
        ],
        ignore_index=True,
    )
    table["raw_stock_cooling_kwh"] = (
        table["annual_cooling_eui_kwh_m2"] * table["component_gfa_m2"]
    )
    table["raw_stock_non_cooling_kwh"] = (
        table["annual_non_cooling_eui_kwh_m2"] * table["component_gfa_m2"]
    )
    group_raw = table.groupby("calibration_group", as_index=False).agg(
        raw_stock_cooling_kwh=("raw_stock_cooling_kwh", "sum"),
        raw_stock_non_cooling_kwh=("raw_stock_non_cooling_kwh", "sum"),
        component_gfa_m2=("component_gfa_m2", "sum"),
        prototypes=("prototype_id", "nunique"),
    )
    group_raw = group_raw.merge(
        group_targets, on="calibration_group", how="left", validate="one_to_one"
    )
    group_raw["official_total_kwh"] = group_raw["electricity_tj"] * 1_000_000 / 3.6
    group_raw["official_cooling_kwh"] = (
        group_raw["air_conditioning_tj"] * 1_000_000 / 3.6
    )
    group_raw["official_non_cooling_kwh"] = (
        group_raw["official_total_kwh"] - group_raw["official_cooling_kwh"]
    )
    group_raw["cooling_calibration_factor"] = (
        group_raw["official_cooling_kwh"] / group_raw["raw_stock_cooling_kwh"]
    )
    group_raw["non_cooling_calibration_factor"] = (
        group_raw["official_non_cooling_kwh"] / group_raw["raw_stock_non_cooling_kwh"]
    )
    table = table.merge(
        group_raw[
            [
                "calibration_group",
                "cooling_calibration_factor",
                "non_cooling_calibration_factor",
            ]
        ],
        on="calibration_group",
        how="left",
        validate="many_to_one",
    )
    table["calibrated_stock_cooling_kwh"] = (
        table["raw_stock_cooling_kwh"] * table["cooling_calibration_factor"]
    )
    table["calibrated_stock_non_cooling_kwh"] = (
        table["raw_stock_non_cooling_kwh"] * table["non_cooling_calibration_factor"]
    )
    check = table.groupby("calibration_group", as_index=False).agg(
        calibrated_cooling_kwh=("calibrated_stock_cooling_kwh", "sum"),
        calibrated_non_cooling_kwh=("calibrated_stock_non_cooling_kwh", "sum"),
    )
    check = check.merge(
        group_raw[
            [
                "calibration_group",
                "official_total_kwh",
                "official_cooling_kwh",
                "official_non_cooling_kwh",
                "component_gfa_m2",
            ]
        ],
        on="calibration_group",
        how="left",
        validate="one_to_one",
    )
    check["calibrated_total_kwh"] = (
        check["calibrated_cooling_kwh"] + check["calibrated_non_cooling_kwh"]
    )
    check["total_relative_error"] = (
        check["calibrated_total_kwh"] / check["official_total_kwh"] - 1
    )
    check["cooling_relative_error"] = (
        check["calibrated_cooling_kwh"] / check["official_cooling_kwh"] - 1
    )
    table.to_csv(
        output / "official_energy_calibration_factors.csv",
        index=False,
        encoding="utf-8-sig",
    )
    check.to_csv(
        output / "official_energy_mass_balance_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    commercial_composition = targets.loc[targets["sector"].eq("commercial")].copy()
    commercial_composition["official_electricity_share"] = (
        commercial_composition["electricity_tj"]
        / commercial_composition["electricity_tj"].sum()
    )
    commercial_composition.to_csv(
        output / "official_commercial_composition_benchmark.csv",
        index=False,
        encoding="utf-8-sig",
    )
    monthly = annual_monthly.merge(
        table[
            [
                "prototype_id", "sector", "component_gfa_m2",
                "cooling_calibration_factor", "non_cooling_calibration_factor",
            ]
        ],
        on=["prototype_id", "sector"],
        how="left",
        validate="many_to_one",
    )
    monthly["calibrated_stock_kwh"] = monthly["component_gfa_m2"] * (
        monthly["monthly_cooling_eui_kwh_m2"] * monthly["cooling_calibration_factor"]
        + (
            monthly["monthly_lighting_eui_kwh_m2"]
            + monthly["monthly_equipment_eui_kwh_m2"]
        ) * monthly["non_cooling_calibration_factor"]
    )
    modeled_domestic = (
        monthly.loc[monthly["sector"].eq("residential")]
        .groupby("month", as_index=False)["calibrated_stock_kwh"]
        .sum()
        .rename(columns={"calibrated_stock_kwh": "modeled_domestic_kwh"})
    )
    official_monthly = pd.read_csv(OFFICIAL_MONTHLY_ENERGY)
    official_monthly = official_monthly.loc[official_monthly["year"].eq(2023), [
        "month", "domestic_electricity_gwh",
    ]].copy()
    monthly_check = official_monthly.merge(
        modeled_domestic, on="month", how="inner", validate="one_to_one"
    )
    if len(monthly_check) != 12:
        raise ValueError("C&SD 2023 domestic monthly electricity series is incomplete")
    monthly_check["official_domestic_kwh"] = (
        monthly_check["domestic_electricity_gwh"] * 1_000_000
    )
    monthly_check["modeled_share"] = (
        monthly_check["modeled_domestic_kwh"] / monthly_check["modeled_domestic_kwh"].sum()
    )
    monthly_check["official_share"] = (
        monthly_check["official_domestic_kwh"] / monthly_check["official_domestic_kwh"].sum()
    )
    monthly_check["share_error"] = monthly_check["modeled_share"] - monthly_check["official_share"]
    monthly_shape_cvrmse_pct = float(
        100
        * np.sqrt(np.mean(monthly_check["share_error"] ** 2))
        / monthly_check["official_share"].mean()
    )
    monthly_shape_correlation = float(
        monthly_check[["modeled_share", "official_share"]].corr().iloc[0, 1]
    )
    monthly_check.to_csv(
        output / "official_domestic_monthly_shape_holdout_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    bca_validation = validate_bca_hot_humid_eui(table, output)
    validation = {
        "version": ENERGY_CALIBRATION_VERSION,
        "official_reference_year": 2023,
        "residential_calibration_groups": 4,
        "commercial_calibration": "sector total; type composition retained as independent audit because HK-BFETD subclasses do not exhaust official segments",
        "official_residential_electricity_tj": float(
            targets.loc[targets["sector"].eq("residential"), "electricity_tj"].sum()
        ),
        "official_commercial_electricity_tj": 108_524.0,
        "commercial_segment_rounding_gap_tj": float(
            108_524.0
            - targets.loc[targets["sector"].eq("commercial"), "electricity_tj"].sum()
        ),
        "maximum_absolute_mass_balance_relative_error": float(
            check[["total_relative_error", "cooling_relative_error"]].abs().to_numpy().max()
        ),
        "annual_energy_mass_balance_exact": bool(
            check[["total_relative_error", "cooling_relative_error"]].abs().to_numpy().max()
            < 1e-10
        ),
        "monthly_domestic_shape_holdout": {
            "source": "C&SD monthly domestic electricity, 2023; not fitted month by month",
            "months": 12,
            "share_cvrmse_pct": monthly_shape_cvrmse_pct,
            "pearson_r": monthly_shape_correlation,
            "modeled_peak_month": int(
                monthly_check.loc[monthly_check["modeled_share"].idxmax(), "month"]
            ),
            "official_peak_month": int(
                monthly_check.loc[monthly_check["official_share"].idxmax(), "month"]
            ),
        },
        "bca_hot_humid_plausibility": bca_validation,
        "official_source": "EMSD Hong Kong Energy End-use Data 2025, Tables 23-26 and 37-44",
        "calibration_interpretation": "shared group factors preserve EnergyPlus prototype differences while matching official cooling and non-cooling stock totals",
    }
    (output / "official_energy_calibration_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return table[
        [
            "prototype_id",
            "calibration_group",
            "cooling_calibration_factor",
            "non_cooling_calibration_factor",
        ]
    ], validation


def validate_bca_hot_humid_eui(table: pd.DataFrame, output: Path) -> dict:
    """Compare commercial prototype EUI with an independent hot-humid disclosure."""
    if not BCA_BENCHMARK.exists():
        return {"available": False}
    bca = pd.read_excel(BCA_BENCHMARK, sheet_name="Dataset")
    bca["eui_kwh_m2_year"] = pd.to_numeric(bca["2024 EUI"], errors="coerce")
    mapping = {
        "com_office": "Office",
        "com_retail": "Retail",
        "com_accommodation": "Hotel",
        "com_education": "Autonomous University",
        "com_human_health": "Hospital or Specialist Clinic",
        "com_mixed_use_component": "Mixed Development",
        "com_other": "Mixed Development",
    }
    model = table.loc[table["prototype_id"].isin(mapping)].copy()
    model["bca_building_type"] = model["prototype_id"].map(mapping)
    model["modeled_calibrated_eui_kwh_m2_year"] = (
        model["annual_cooling_eui_kwh_m2"] * model["cooling_calibration_factor"]
        + model["annual_non_cooling_eui_kwh_m2"] * model["non_cooling_calibration_factor"]
    )
    distributions = (
        bca.dropna(subset=["eui_kwh_m2_year"])
        .groupby("Building Type", as_index=False)["eui_kwh_m2_year"]
        .agg(
            bca_buildings="count",
            bca_median_eui="median",
            bca_p05_eui=lambda value: value.quantile(0.05),
            bca_p95_eui=lambda value: value.quantile(0.95),
        )
        .rename(columns={"Building Type": "bca_building_type"})
    )
    comparison = model[[
        "prototype_id", "bca_building_type", "modeled_calibrated_eui_kwh_m2_year",
    ]].merge(distributions, on="bca_building_type", how="left", validate="many_to_one")
    comparison["inside_bca_p05_p95"] = comparison["modeled_calibrated_eui_kwh_m2_year"].between(
        comparison["bca_p05_eui"], comparison["bca_p95_eui"]
    )
    comparison.to_csv(
        output / "bca_2024_hot_humid_commercial_eui_plausibility.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return {
        "available": True,
        "mapped_prototypes": int(len(comparison)),
        "inside_bca_p05_p95": int(comparison["inside_bca_p05_p95"].sum()),
        "interpretation": "cross-city plausibility only; Singapore BCA data do not calibrate Hong Kong prototypes",
    }


def build_local_heat_sensitivity_concordance(
    annual_hourly: pd.DataFrame,
    polyu_validation: dict,
    output: Path,
) -> dict:
    """Compare local observations and simulations with one daily estimator.

    Each source is reduced to relative daily whole-building electricity and
    regressed on cooling degrees after weekday and month fixed effects.  PolyU
    retains its calibration-only balance point, while the comparison slopes
    come exclusively from held-out buildings. HKUST and the corresponding
    model comparisons use the pre-specified 24 C balance point.
    """
    required = {
        "prototype_id", "sector", "timestamp_hkt", "outdoor_temperature_c",
        "total_electricity_kwh_m2",
    }
    if missing := sorted(required - set(annual_hourly.columns)):
        raise ValueError(f"Annual EnergyPlus results lack local-validation fields: {missing}")

    model_hourly = annual_hourly.copy()
    model_hourly["timestamp_hkt"] = pd.to_datetime(
        model_hourly["timestamp_hkt"], errors="coerce"
    )
    model_hourly["date"] = model_hourly["timestamp_hkt"].dt.normalize()
    model_daily = model_hourly.groupby(
        ["prototype_id", "sector", "date"], as_index=False, observed=True
    ).agg(
        energy_kwh_m2=("total_electricity_kwh_m2", "sum"),
        mean_temperature_c=("outdoor_temperature_c", "mean"),
        valid_hours=("timestamp_hkt", "size"),
    )
    model_daily = model_daily.loc[model_daily["valid_hours"].eq(24)].copy()
    model_daily["relative_daily_load"] = model_daily["energy_kwh_m2"] / model_daily.groupby(
        "prototype_id", observed=True
    )["energy_kwh_m2"].transform("median").replace(0.0, np.nan)
    model_daily["day_of_week"] = model_daily["date"].dt.dayofweek
    model_daily["month"] = model_daily["date"].dt.month

    def slopes_by_group(
        daily: pd.DataFrame,
        group_field: str,
        balance_point_c: float,
    ) -> pd.DataFrame:
        rows = []
        for group_id, group in daily.groupby(group_field, observed=True):
            group = group.dropna(
                subset=["relative_daily_load", "mean_temperature_c", "day_of_week", "month"]
            ).copy()
            group["cooling_degree_c"] = (
                group["mean_temperature_c"] - balance_point_c
            ).clip(lower=0.0)
            if len(group) < 180 or group["cooling_degree_c"].var() <= 0:
                continue
            calendar = pd.get_dummies(
                group[["day_of_week", "month"]].astype(str),
                columns=["day_of_week", "month"],
                drop_first=True,
                dtype=float,
            )
            design = np.column_stack([
                np.ones(len(group)),
                group["cooling_degree_c"].to_numpy(float),
                calendar.to_numpy(float),
            ])
            coefficients, *_ = np.linalg.lstsq(
                design, group["relative_daily_load"].to_numpy(float), rcond=None
            )
            rows.append({
                group_field: str(group_id),
                "balance_point_c": float(balance_point_c),
                "daily_observations": int(len(group)),
                "relative_daily_load_slope_per_c": float(coefficients[1]),
            })
        return pd.DataFrame(rows)

    polyu = pd.read_csv(output / "polyu_campus_temperature_load_response.csv")
    polyu = polyu.loc[polyu["building_split"].eq("building_holdout")].drop_duplicates(
        "profile_equivalence_id"
    )
    polyu_slopes = numeric(
        polyu["calendar_adjusted_relative_daily_load_slope_per_c_above_balance_point"]
    ).dropna()
    polyu_balance_point = float(
        polyu_validation["selected_balance_point_c_from_calibration_only"]
    )

    hkust = pd.read_parquet(output / "hkust_whole_building_daily_validation.parquet")
    hkust = hkust.loc[hkust["building_split"].eq("building_holdout")].copy()
    hkust_slopes = slopes_by_group(hkust, "leaf_building", 24.0)
    scope_lookup = hkust[["leaf_building", "building_scope"]].drop_duplicates()
    hkust_slopes = hkust_slopes.merge(
        scope_lookup, on="leaf_building", how="left", validate="one_to_one"
    )

    model_by_balance_point = {
        polyu_balance_point: slopes_by_group(model_daily, "prototype_id", polyu_balance_point),
        24.0: slopes_by_group(model_daily, "prototype_id", 24.0),
    }
    prototype_sector = model_daily[["prototype_id", "sector"]].drop_duplicates()
    for frame in model_by_balance_point.values():
        frame["sector"] = frame["prototype_id"].map(
            prototype_sector.set_index("prototype_id")["sector"]
        )

    comparisons = [
        (
            "PolyU full campus", "nonresidential campus buildings", polyu_balance_point,
            polyu_slopes,
            ["com_office", "com_education", "com_data_centre", "com_other"],
        ),
        (
            "HKUST full archive", "campus residence proxy", 24.0,
            hkust_slopes.loc[
                hkust_slopes["building_scope"].eq("campus_residence"),
                "relative_daily_load_slope_per_c",
            ],
            sorted(prototype_sector.loc[
                prototype_sector["sector"].eq("residential"), "prototype_id"
            ]),
        ),
        (
            "HKUST full archive", "campus nonresidential", 24.0,
            hkust_slopes.loc[
                hkust_slopes["building_scope"].eq("campus_nonresidential"),
                "relative_daily_load_slope_per_c",
            ],
            ["com_office", "com_education", "com_data_centre", "com_other"],
        ),
    ]
    records = []
    for dataset, scope, balance_point, observed_values, model_ids in comparisons:
        observed_values = numeric(pd.Series(observed_values)).dropna()
        model_frame = model_by_balance_point[balance_point]
        model_values = numeric(model_frame.loc[
            model_frame["prototype_id"].isin(model_ids),
            "relative_daily_load_slope_per_c",
        ]).dropna()
        if observed_values.empty or model_values.empty:
            raise ValueError(f"No comparable daily heat-sensitivity values for {dataset}/{scope}")
        observed_p05 = float(observed_values.quantile(0.05))
        observed_p95 = float(observed_values.quantile(0.95))
        model_min = float(model_values.min())
        model_max = float(model_values.max())
        observed_median = float(observed_values.median())
        model_median = float(model_values.median())
        tolerance = 1e-12
        records.append({
            "dataset": dataset,
            "observed_scope": scope,
            "balance_point_c": float(balance_point),
            "estimator": "relative daily whole-building electricity ~ cooling degrees + weekday + month fixed effects",
            "observed_buildings": int(len(observed_values)),
            "observed_slope_p05": observed_p05,
            "observed_slope_median": observed_median,
            "observed_slope_p95": observed_p95,
            "model_prototypes": int(len(model_values)),
            "model_prototype_ids": "|".join(model_ids),
            "model_slope_min": model_min,
            "model_slope_median": model_median,
            "model_slope_max": model_max,
            "comparison_tolerance": tolerance,
            "observed_and_model_ranges_overlap": bool(
                observed_p05 <= model_max + tolerance
                and model_min <= observed_p95 + tolerance
            ),
            "observed_median_inside_model_type_range": bool(
                model_min - tolerance <= observed_median <= model_max + tolerance
            ),
            "model_median_inside_observed_p05_p95": bool(
                observed_p05 - tolerance <= model_median <= observed_p95 + tolerance
            ),
            "used_to_fit_event_dynamics": False,
        })
    comparison = pd.DataFrame(records)
    comparison["comparison_pass"] = comparison[[
        "observed_and_model_ranges_overlap",
        "observed_median_inside_model_type_range",
    ]].all(axis=1)
    comparison.to_csv(
        output / "local_observed_vs_energyplus_heat_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return {
        "comparisons": int(len(comparison)),
        "comparisons_passing_all_concordance_checks": int(comparison["comparison_pass"].sum()),
        "all_comparisons_pass": bool(comparison["comparison_pass"].all()),
        "estimator_matched_between_observation_and_simulation": True,
        "observed_building_split": "building_holdout",
        "pass_rule": "held-out observed p05-p95 overlaps model type range and held-out median lies inside model type range",
        "annual_model_weather_year": 2023,
        "used_for_parameter_fitting": False,
        "interpretation": "PolyU and HKUST held-out buildings independently test local whole-building temperature-load sensitivity after applying the same daily estimator to observations and EnergyPlus outputs",
    }


def apply_energy_calibration(
    hourly: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    result = hourly.merge(factors, on="prototype_id", how="left", validate="many_to_one")
    if result[["cooling_calibration_factor", "non_cooling_calibration_factor"]].isna().any().any():
        raise ValueError("Energy calibration factors do not cover all hourly prototypes")
    result["raw_total_electricity_kwh_m2"] = result["total_electricity_kwh_m2"]
    result["raw_total_electric_w_m2"] = result["total_electric_w_m2"]
    # EMSD factors calibrate electricity, not the EnergyPlus thermal load.
    # Keeping cooling_thermal_kwh_m2 unscaled is essential for the physical
    # required-delivered-unmet cooling balance used by the capacity score.
    for field in [
        "cooling_kwh_m2", "latent_cooling_kwh_m2", "cooling_electric_w_m2",
    ]:
        if field not in result:
            continue
        result[field] = result[field] * result["cooling_calibration_factor"]
    for field in ["lighting_kwh_m2", "equipment_kwh_m2"]:
        result[field] = result[field] * result["non_cooling_calibration_factor"]
    non_cooling_w_m2 = (
        result["raw_total_electric_w_m2"]
        - result["cooling_electric_w_m2"] / result["cooling_calibration_factor"]
    )
    result["total_electricity_kwh_m2"] = (
        result["cooling_kwh_m2"]
        + result["lighting_kwh_m2"]
        + result["equipment_kwh_m2"]
    )
    result["total_electric_w_m2"] = (
        result["cooling_electric_w_m2"]
        + non_cooling_w_m2 * result["non_cooling_calibration_factor"]
    )
    result["energy_calibration_version"] = ENERGY_CALIBRATION_VERSION
    return result


def add_cooling_thermal_energy(
    hourly: pd.DataFrame,
    parameters: pd.DataFrame,
) -> pd.DataFrame:
    """Recover delivered thermal cooling from electrical cooling and prototype COP.

    Existing resumable caches predate this explicit field.  The conversion is
    exact for the ideal-load implementation because electrical cooling is
    defined as delivered thermal cooling divided by the same prototype COP.
    """
    result = hourly.copy()
    if "cooling_thermal_kwh_m2" in result:
        return result
    cop = parameters.set_index("prototype_id")["baseline_ac_cop"]
    mapped = result["prototype_id"].map(cop)
    if mapped.isna().any():
        missing = sorted(result.loc[mapped.isna(), "prototype_id"].astype(str).unique())
        raise ValueError(f"Cooling COP missing for prototypes: {missing}")
    result["cooling_thermal_kwh_m2"] = numeric(
        result["cooling_kwh_m2"], lower=0
    ) * numeric(mapped, lower=0)
    return result


def refresh_energy_summary(
    results: pd.DataFrame,
    hourly: pd.DataFrame,
) -> pd.DataFrame:
    heat = hourly.loc[hourly["event_phase"].eq("heatwave")].copy()
    heat["heatwave_days"] = heat.groupby(CASE_KEYS, observed=True)["timestamp_hkt"].transform("size") / 24.0
    summary = heat.groupby(CASE_KEYS, observed=True).agg(
        heatwave_days=("heatwave_days", "first"),
        cooling_electricity_kwh_m2=("cooling_kwh_m2", "sum"),
        cooling_thermal_kwh_m2=("cooling_thermal_kwh_m2", "sum"),
        total_electricity_kwh_m2=("total_electricity_kwh_m2", "sum"),
        latent_cooling_electricity_kwh_m2=("latent_cooling_kwh_m2", "sum"),
        peak_electric_w_m2=("total_electric_w_m2", "max"),
    ).reset_index()
    summary["cooling_electricity_kwh_m2_per_heatwave_day"] = (
        summary["cooling_electricity_kwh_m2"] / summary["heatwave_days"]
    )
    summary["total_electricity_kwh_m2_per_heatwave_day"] = (
        summary["total_electricity_kwh_m2"] / summary["heatwave_days"]
    )
    summary["latent_share"] = (
        summary["latent_cooling_electricity_kwh_m2"]
        / summary["cooling_electricity_kwh_m2"].replace(0.0, np.nan)
    )
    energy_fields = [
        "heatwave_days",
        "cooling_electricity_kwh_m2",
        "cooling_electricity_kwh_m2_per_heatwave_day",
        "cooling_thermal_kwh_m2",
        "total_electricity_kwh_m2",
        "total_electricity_kwh_m2_per_heatwave_day",
        "latent_cooling_electricity_kwh_m2",
        "latent_share",
        "peak_electric_w_m2",
    ]
    result = results.drop(columns=energy_fields, errors="ignore").merge(
        summary, on=CASE_KEYS, how="left", validate="one_to_one"
    )
    return result


def maximum_consecutive_hours(mask: pd.Series) -> int:
    groups = mask.ne(mask.shift()).cumsum()
    lengths = mask.groupby(groups).sum()
    return int(lengths.max()) if len(lengths) else 0


def interpolated_threshold_crossing_hours(
    initial_temperature_c: float,
    hourly_temperature_c: pd.Series,
    threshold_c: float,
) -> float:
    """Return elapsed crossing time without collapsing the first hour to zero."""
    initial = float(initial_temperature_c)
    if np.isfinite(initial) and initial > threshold_c:
        return 0.0
    previous = initial
    previous_hour = 0.0
    for index, value in enumerate(numeric(hourly_temperature_c).to_numpy(dtype=float)):
        elapsed = float(index + 1)
        if not np.isfinite(value):
            continue
        if value > threshold_c:
            if np.isfinite(previous) and value > previous:
                fraction = (threshold_c - previous) / (value - previous)
                return float(np.clip(previous_hour + fraction * (elapsed - previous_hour), 0.0, elapsed))
            return previous_hour
        previous = value
        previous_hour = elapsed
    return float(len(hourly_temperature_c))


def summarize_response(
    result: pd.DataFrame,
    outage_start: pd.Timestamp | None,
    outage_end: pd.Timestamp | None,
) -> dict[str, float | None]:
    heat = result.loc[result["event_phase"].eq("heatwave")].copy()
    heatwave_days = len(heat) / 24.0
    if heatwave_days <= 0:
        raise ValueError("EnergyPlus result contains no heatwave hours")
    night = heat["timestamp_hkt"].dt.hour.ge(22) | heat["timestamp_hkt"].dt.hour.lt(7)
    cooling_energy = float(heat["cooling_kwh_m2"].sum())
    cooling_thermal_energy = float(heat["cooling_thermal_kwh_m2"].sum())
    total_energy = float(heat["total_electricity_kwh_m2"].sum())
    threshold_c = float(THERMAL_CONFIG["overheating_benchmark_c"])
    safety_screen_c = float(THERMAL_CONFIG["outage_safety_screen_c"])
    comfort_resistance = None
    if outage_start is not None:
        after = result.loc[result["timestamp_hkt"].ge(outage_start)]
        crossed = after.loc[after["exposure_temperature_c"].gt(threshold_c), "timestamp_hkt"]
        comfort_resistance = float((crossed.iloc[0] - outage_start).total_seconds() / 3600) if len(crossed) else float(len(after))
    recovery_to_threshold = None
    if outage_end is not None:
        assessment_hours = float(THERMAL_CONFIG["recovery_assessment_hours"])
        assessment_end = outage_end + pd.Timedelta(hours=assessment_hours)
        after_restoration = result.loc[
            result["timestamp_hkt"].ge(outage_end)
            & result["timestamp_hkt"].lt(assessment_end)
        ]
        if len(after_restoration) < int(assessment_hours):
            raise ValueError("Event weather does not contain the complete recovery assessment window")
        recovered = after_restoration.loc[after_restoration["exposure_temperature_c"].le(threshold_c), "timestamp_hkt"]
        recovery_to_threshold = float((recovered.iloc[0] - outage_end).total_seconds() / 3600) if len(recovered) else assessment_hours
    hot = heat["exposure_temperature_c"].gt(threshold_c)
    outage = pd.Series(False, index=heat.index)
    if outage_start is not None:
        outage = heat["timestamp_hkt"].ge(outage_start)
        if outage_end is not None:
            outage &= heat["timestamp_hkt"].lt(outage_end)
    occupied_lost = heat["occupied_fraction"] * (
        hot | heat["unmet_cooling_fraction"].gt(0) | outage
    ).astype(float)
    return {
        "maximum_indoor_temperature_c": float(heat["exposure_temperature_c"].max()),
        "night_hours_above_26c": int((heat.loc[night, "exposure_temperature_c"] > threshold_c).sum()),
        "night_degree_hours_above_26c": float((heat.loc[night, "exposure_temperature_c"] - threshold_c).clip(lower=0).sum()),
        "all_hour_degree_hours_above_26c": float((heat["exposure_temperature_c"] - threshold_c).clip(lower=0).sum()),
        "continuous_threshold_exceedance_hours": maximum_consecutive_hours(hot),
        "heatwave_days": heatwave_days,
        "overheating_degree_hours_per_heatwave_day": float((heat["exposure_temperature_c"] - threshold_c).clip(lower=0).sum() / heatwave_days),
        "cooling_electricity_kwh_m2": cooling_energy,
        "cooling_electricity_kwh_m2_per_heatwave_day": cooling_energy / heatwave_days,
        "cooling_thermal_kwh_m2": cooling_thermal_energy,
        "total_electricity_kwh_m2": total_energy,
        "total_electricity_kwh_m2_per_heatwave_day": total_energy / heatwave_days,
        "latent_cooling_electricity_kwh_m2": float(heat["latent_cooling_kwh_m2"].sum()),
        "latent_share": float(heat["latent_cooling_kwh_m2"].sum() / cooling_energy) if cooling_energy > 0 else None,
        "peak_electric_w_m2": float(heat["total_electric_w_m2"].max()),
        "unmet_cooling_hours": float(heat["unmet_cooling_fraction"].sum()),
        "cooling_start_transition_allowance_hours": float(
            heat["cooling_start_transition_fraction"].sum() / 4.0
        ),
        "occupied_service_hours_lost": float(occupied_lost.sum()),
        "comfort_resistance_hours_to_26c": comfort_resistance,
        "safety_resistance_hours_to_32c": (
            float(np.flatnonzero(
                heat.loc[outage, "exposure_temperature_c"].to_numpy() > safety_screen_c
            )[0])
            if outage.any() and np.any(
                heat.loc[outage, "exposure_temperature_c"].to_numpy() > safety_screen_c
            )
            else (float(outage.sum()) if outage.any() else None)
        ),
        "recovery_to_26c_hours": recovery_to_threshold,
    }


CASE_KEYS = [
    "event_id",
    "base_case_id",
    "prototype_id",
    "sector",
    "climate_case",
    "supply_stage",
]


def add_hourly_resilience_metrics(
    results: pd.DataFrame,
    hourly: pd.DataFrame,
) -> pd.DataFrame:
    """Add physical thermal response, occupied exposure and recovery metrics.

    These quantities are post-processing of the frozen EnergyPlus trajectories;
    they do not alter or rerun the physical model.
    """
    required = set(CASE_KEYS + [
        "timestamp_hkt", "event_phase", "exposure_temperature_c",
        "maximum_zone_temperature_c", "outdoor_temperature_c",
        "occupied_fraction", "active_cooling_fraction", "total_electricity_kwh_m2",
    ])
    missing = sorted(required - set(hourly.columns))
    if missing:
        raise ValueError(f"Hourly EnergyPlus results lack derived-metric fields: {missing}")
    work = hourly.copy()
    work["timestamp_hkt"] = pd.to_datetime(work["timestamp_hkt"], utc=True).dt.tz_convert("Asia/Hong_Kong")
    threshold = float(THERMAL_CONFIG["overheating_benchmark_c"])
    safety_screen = float(THERMAL_CONFIG["outage_safety_screen_c"])
    heat = work.loc[work["event_phase"].eq("heatwave")].copy()
    heat["occupied_overheating_hours"] = (
        heat["exposure_temperature_c"].gt(threshold).astype(float)
        * numeric(heat["occupied_fraction"], lower=0).fillna(0.0)
    )
    heat["occupied_overheating_degree_hours"] = (
        (numeric(heat["exposure_temperature_c"]) - threshold).clip(lower=0).fillna(0.0)
        * numeric(heat["occupied_fraction"], lower=0).fillna(0.0)
    )
    occupied = heat.groupby(CASE_KEYS, observed=True).agg(
        occupied_overheating_hours=("occupied_overheating_hours", "sum"),
        occupied_overheating_degree_hours=("occupied_overheating_degree_hours", "sum"),
    ).reset_index()

    rated_capacity_context = work.loc[
        work["climate_case"].eq("heatwave") & work["supply_stage"].eq("capacity_100pct"),
        [
            "event_id", "base_case_id", "timestamp_hkt", "event_phase",
            "outdoor_temperature_c", "active_cooling_fraction",
            "occupied_fraction",
            "exposure_temperature_c", "maximum_zone_temperature_c",
        ],
    ].sort_values("timestamp_hkt")
    rated_capacity_lookup = {
        (str(event_id), str(base_case_id)): group.reset_index(drop=True)
        for (event_id, base_case_id), group in rated_capacity_context.groupby(
            ["event_id", "base_case_id"], observed=True
        )
    }

    def physical_response(group: pd.DataFrame) -> pd.Series:
        group_key = dict(zip(CASE_KEYS, group.name))
        group = group.sort_values("timestamp_hkt").reset_index(drop=True)
        heat_group = group.loc[group["event_phase"].eq("heatwave")].copy()
        # Thermal safety follows the hottest occupied conditioned zone at each
        # hour.  Including an unoccupied bedroom or perimeter zone would turn
        # a room-use schedule into an apparent occupant-safety failure.
        temperature = numeric(heat_group["exposure_temperature_c"])
        hot = temperature.gt(threshold)
        night = heat_group["timestamp_hkt"].dt.hour.ge(22) | heat_group["timestamp_hkt"].dt.hour.lt(7)
        result = {
            "maximum_indoor_temperature_c": float(temperature.max()),
            "night_hours_above_26c": int(hot.loc[night].sum()),
            "night_degree_hours_above_26c": float((temperature.loc[night] - threshold).clip(lower=0).sum()),
            "all_hour_degree_hours_above_26c": float((temperature - threshold).clip(lower=0).sum()),
            "continuous_threshold_exceedance_hours": maximum_consecutive_hours(hot),
            "overheating_degree_hours_per_heatwave_day": float(
                (temperature - threshold).clip(lower=0).sum() / (len(heat_group) / 24.0)
            ),
            "outage_window_degree_hours_above_26c": np.nan,
            "outage_attributable_degree_hours_above_26c": np.nan,
            "outage_peak_temperature_increment_c": np.nan,
            "comfort_resistance_hours_to_26c": np.nan,
            "outage_window_degree_hours_above_32c": np.nan,
            "safety_resistance_hours_to_32c": np.nan,
            "outage_window_degree_hours_above_30c": np.nan,
            "safety_resistance_hours_to_30c": np.nan,
            "outage_window_degree_hours_above_34c": np.nan,
            "safety_resistance_hours_to_34c": np.nan,
            "recovery_to_26c_hours": np.nan,
        }
        stage = str(group_key["supply_stage"])
        if not stage.startswith("outage_"):
            return pd.Series(result)
        event_id = str(group_key["event_id"])
        base_case_id = str(group_key["base_case_id"])
        if stage == "outage_full_event":
            heat_positions = np.flatnonzero(group["event_phase"].eq("heatwave").to_numpy())
            start_index, end_index = int(heat_positions[0]), int(heat_positions[-1]) + 1
        elif stage == "outage_common_24h":
            context = rated_capacity_lookup[(event_id, base_case_id)]
            heat_positions = np.flatnonzero(context["event_phase"].eq("heatwave").to_numpy())
            start_index = int(
                heat_positions[
                    np.argmax(
                        numeric(context.iloc[heat_positions]["outdoor_temperature_c"]).to_numpy()
                    )
                ]
            )
            end_index = min(len(group), start_index + 24)
        elif stage in {"outage_peak_6h", "outage_peak_24h"}:
            hours = 6 if stage.endswith("6h") else 24
            context = rated_capacity_lookup[(event_id, base_case_id)]
            eligible = (
                context["event_phase"].eq("heatwave")
                & numeric(context["active_cooling_fraction"]).gt(0)
                & numeric(context["occupied_fraction"]).gt(0.15)
            )
            if not eligible.any():
                eligible = context["event_phase"].eq("heatwave") & numeric(
                    context["active_cooling_fraction"]
                ).gt(0)
            if not eligible.any():
                eligible = context["event_phase"].eq("heatwave")
            eligible_positions = np.flatnonzero(eligible.to_numpy())
            peak_index = int(
                eligible_positions[
                    np.argmax(numeric(context.iloc[eligible_positions]["outdoor_temperature_c"]).to_numpy())
                ]
            )
            start_index = peak_index
            end_index = min(len(group), start_index + hours)
        else:
            raise ValueError(f"Unknown outage stage {stage!r}")
        context = rated_capacity_lookup[(event_id, base_case_id)]
        outage_window = group.iloc[start_index:end_index]
        control_window = context.iloc[start_index:end_index]
        outage_temperature = numeric(outage_window["exposure_temperature_c"]).reset_index(drop=True)
        control_temperature = numeric(control_window["exposure_temperature_c"]).reset_index(drop=True)
        outage_stress = (outage_temperature - threshold).clip(lower=0)
        control_stress = (control_temperature - threshold).clip(lower=0)
        result["outage_window_degree_hours_above_26c"] = float(outage_stress.sum())
        result["outage_attributable_degree_hours_above_26c"] = float(
            (outage_stress - control_stress).clip(lower=0).sum()
        )
        result["outage_peak_temperature_increment_c"] = float(
            (outage_temperature - control_temperature).clip(lower=0).max()
        )
        result["outage_window_degree_hours_above_32c"] = float(
            (outage_temperature - safety_screen).clip(lower=0).sum()
        )
        result["safety_resistance_hours_to_32c"] = interpolated_threshold_crossing_hours(
            float(control_temperature.iloc[0]), outage_temperature, safety_screen
        )
        for threshold_c in (30.0, 34.0):
            suffix = str(int(threshold_c))
            result[f"outage_window_degree_hours_above_{suffix}c"] = float(
                (outage_temperature - threshold_c).clip(lower=0).sum()
            )
            result[f"safety_resistance_hours_to_{suffix}c"] = (
                interpolated_threshold_crossing_hours(
                    float(control_temperature.iloc[0]), outage_temperature, threshold_c
                )
            )
        outage_start = group.iloc[start_index]["timestamp_hkt"]
        outage_end = group.iloc[end_index]["timestamp_hkt"] if end_index < len(group) else None
        after = group.loc[group["timestamp_hkt"].ge(outage_start)]
        crossed = after.loc[
            numeric(after["exposure_temperature_c"]).gt(threshold), "timestamp_hkt"
        ]
        result["comfort_resistance_hours_to_26c"] = (
            float((crossed.iloc[0] - outage_start).total_seconds() / 3600)
            if len(crossed) else float(len(after))
        )
        if outage_end is not None:
            assessment_hours = float(THERMAL_CONFIG["recovery_assessment_hours"])
            assessment_end = outage_end + pd.Timedelta(hours=assessment_hours)
            after_restoration = group.loc[
                group["timestamp_hkt"].ge(outage_end)
                & group["timestamp_hkt"].lt(assessment_end)
            ]
            if len(after_restoration) < int(assessment_hours):
                raise ValueError(
                    f"Incomplete recovery window for {event_id}/{base_case_id}/{stage}"
                )
            recovered = after_restoration.loc[
                numeric(after_restoration["exposure_temperature_c"]).le(threshold),
                "timestamp_hkt",
            ]
            result["recovery_to_26c_hours"] = (
                float((recovered.iloc[0] - outage_end).total_seconds() / 3600)
                if len(recovered) else assessment_hours
            )
        return pd.Series(result)

    physical = work.groupby(CASE_KEYS, observed=True, group_keys=False).apply(
        physical_response, include_groups=False
    ).reset_index()

    heatwave_runs = work.loc[work["climate_case"].eq("heatwave")].copy()
    rated_capacity = heatwave_runs.loc[heatwave_runs["supply_stage"].eq("capacity_100pct"), [
        "event_id", "base_case_id", "timestamp_hkt", "total_electricity_kwh_m2"
    ]].rename(columns={"total_electricity_kwh_m2": "rated_capacity_electricity_kwh_m2"})
    if rated_capacity.duplicated(["event_id", "base_case_id", "timestamp_hkt"]).any():
        raise ValueError("Rated-capacity control trajectories are not unique")
    rebound = heatwave_runs.merge(
        rated_capacity,
        on=["event_id", "base_case_id", "timestamp_hkt"],
        how="left",
        validate="many_to_one",
    )
    rebound["positive_recovery_energy_kwh_m2"] = np.where(
        rebound["event_phase"].eq("recovery"),
        (
            numeric(rebound["total_electricity_kwh_m2"])
            - numeric(rebound["rated_capacity_electricity_kwh_m2"])
        ).clip(lower=0).fillna(0.0),
        0.0,
    )
    recovery = rebound.groupby(CASE_KEYS, observed=True)["positive_recovery_energy_kwh_m2"].sum().rename(
        "recovery_energy_kwh_m2"
    ).reset_index()

    physical_fields = [
        "maximum_indoor_temperature_c", "night_hours_above_26c",
        "night_degree_hours_above_26c", "all_hour_degree_hours_above_26c",
        "continuous_threshold_exceedance_hours", "overheating_degree_hours_per_heatwave_day",
        "outage_window_degree_hours_above_26c",
        "outage_attributable_degree_hours_above_26c",
        "outage_peak_temperature_increment_c",
        "comfort_resistance_hours_to_26c", "outage_window_degree_hours_above_32c",
        "safety_resistance_hours_to_32c", "recovery_to_26c_hours",
        "outage_window_degree_hours_above_30c", "safety_resistance_hours_to_30c",
        "outage_window_degree_hours_above_34c", "safety_resistance_hours_to_34c",
    ]
    refreshed_fields = physical_fields + [
        "occupied_overheating_hours", "occupied_overheating_degree_hours",
        "recovery_energy_kwh_m2",
    ]
    augmented = results.drop(columns=refreshed_fields, errors="ignore").merge(
        physical, on=CASE_KEYS, how="left", validate="one_to_one"
    )
    augmented = augmented.merge(occupied, on=CASE_KEYS, how="left", validate="one_to_one")
    augmented = augmented.merge(recovery, on=CASE_KEYS, how="left", validate="one_to_one")
    augmented[["occupied_overheating_hours", "occupied_overheating_degree_hours", "recovery_energy_kwh_m2"]] = (
        augmented[["occupied_overheating_hours", "occupied_overheating_degree_hours", "recovery_energy_kwh_m2"]]
        .fillna(0.0)
    )
    critical = augmented["prototype_id"].isin({"com_human_health", "com_data_centre"})
    outage_hours = augmented["supply_stage"].map(
        {
            "normal": 0.0,
            "capacity_100pct": 0.0,
            "capacity_70pct": 0.0,
            "outage_peak_6h": 6.0,
            "outage_peak_24h": 24.0,
            "outage_common_24h": 24.0,
        }
    )
    outage_hours = outage_hours.where(~augmented["supply_stage"].eq("outage_full_event"), augmented["heatwave_days"] * 24.0)
    augmented["critical_service_continuity_hours"] = np.where(
        critical,
        (augmented["heatwave_days"] * 24.0 - outage_hours.fillna(0.0)).clip(lower=0.0),
        np.nan,
    )
    return augmented


def audit_outage_temporal_contract(hourly: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Verify that every partial outage implements the pre-specified time contract.

    The audit is evaluated from the persisted EnergyPlus trajectories.  It
    therefore catches a mismatch between the schedule generator, EnergyPlus
    execution and the post-processing window rather than merely rechecking the
    schedule-construction function.
    """
    required = {
        "event_id", "base_case_id", "prototype_id", "climate_case",
        "supply_stage", "timestamp_hkt", "event_phase",
        "outdoor_temperature_c", "maximum_zone_temperature_c",
        "active_cooling_fraction", "occupied_fraction",
        "lighting_kwh_m2", "equipment_kwh_m2", "total_electricity_kwh_m2",
    }
    missing = sorted(required - set(hourly.columns))
    if missing:
        raise ValueError(f"Hourly results lack outage-audit fields: {missing}")
    work = hourly.loc[hourly["climate_case"].eq("heatwave")].copy()
    work["timestamp_hkt"] = pd.to_datetime(work["timestamp_hkt"], utc=True).dt.tz_convert(
        "Asia/Hong_Kong"
    )
    rated_capacity_lookup = {
        (str(event_id), str(base_case_id)): group.sort_values("timestamp_hkt").reset_index(drop=True)
        for (event_id, base_case_id), group in work.loc[
            work["supply_stage"].eq("capacity_100pct")
        ].groupby(["event_id", "base_case_id"], observed=True)
    }
    rows: list[dict[str, object]] = []
    for (event_id, base_case_id, stage), outage in work.loc[
        work["supply_stage"].isin(
            ["outage_peak_6h", "outage_peak_24h", "outage_common_24h"]
        )
    ].groupby(["event_id", "base_case_id", "supply_stage"], observed=True):
        outage = outage.sort_values("timestamp_hkt").reset_index(drop=True)
        normal = rated_capacity_lookup[(str(event_id), str(base_case_id))]
        timestamps_match = bool(
            len(outage) == len(normal)
            and outage["timestamp_hkt"].equals(normal["timestamp_hkt"])
        )
        if str(stage) == "outage_common_24h":
            eligible = normal["event_phase"].eq("heatwave")
            fallback_level = "common_hottest_outdoor_event_hour"
        else:
            eligible = (
                normal["event_phase"].eq("heatwave")
                & numeric(normal["active_cooling_fraction"]).gt(0)
                & numeric(normal["occupied_fraction"]).gt(0.15)
            )
            fallback_level = "none"
            if not bool(eligible.any()):
                eligible = normal["event_phase"].eq("heatwave") & numeric(
                    normal["active_cooling_fraction"]
                ).gt(0)
                fallback_level = "scheduled_cooling_without_occupied_interval"
            if not bool(eligible.any()):
                eligible = normal["event_phase"].eq("heatwave")
                fallback_level = "hottest_heatwave_hour"
        eligible_positions = np.flatnonzero(eligible.to_numpy())
        if not len(eligible_positions):
            raise ValueError(f"No heatwave interval in {event_id}/{base_case_id}")
        start_index = int(
            eligible_positions[
                np.argmax(
                    numeric(normal.iloc[eligible_positions]["outdoor_temperature_c"]).to_numpy()
                )
            ]
        )
        expected_hours = 6 if str(stage).endswith("6h") else 24
        end_index = min(len(outage), start_index + expected_hours)
        normal_window = normal.iloc[start_index:end_index]
        outage_window = outage.iloc[start_index:end_index]
        pre_normal = normal.iloc[:start_index]
        pre_outage = outage.iloc[:start_index]
        pre_temperature_difference = np.abs(
            numeric(pre_outage["maximum_zone_temperature_c"]).to_numpy()
            - numeric(pre_normal["maximum_zone_temperature_c"]).to_numpy()
        )
        pre_load_difference = np.abs(
            numeric(pre_outage["total_electricity_kwh_m2"]).to_numpy()
            - numeric(pre_normal["total_electricity_kwh_m2"]).to_numpy()
        )
        actual_hours = int(len(outage_window))
        row = {
            "event_id": str(event_id),
            "base_case_id": str(base_case_id),
            "prototype_id": str(outage["prototype_id"].iloc[0]),
            "supply_stage": str(stage),
            "outage_start_hkt": outage_window["timestamp_hkt"].iloc[0].isoformat(),
            "outage_end_exclusive_hkt": (
                outage["timestamp_hkt"].iloc[end_index].isoformat()
                if end_index < len(outage) else "end_of_simulation"
            ),
            "expected_hours": expected_hours,
            "actual_hours": actual_hours,
            "fallback_level": fallback_level,
            "timestamps_match_normal": timestamps_match,
            "starts_in_heatwave": bool(normal_window["event_phase"].iloc[0] == "heatwave"),
            "normal_cooling_active_at_start": bool(
                numeric(normal_window["active_cooling_fraction"]).iloc[0] > 0
                or fallback_level == "hottest_heatwave_hour"
            ),
            "occupied_service_at_start": bool(
                numeric(normal_window["occupied_fraction"]).iloc[0] > 0.15
                or fallback_level != "none"
            ),
            "all_grid_load_zero_during_outage": bool(
                numeric(outage_window["total_electricity_kwh_m2"]).abs().le(1e-10).all()
            ),
            "cooling_disabled_during_outage": bool(
                numeric(outage_window["active_cooling_fraction"]).abs().le(1e-10).all()
            ),
            "pre_outage_temperature_matches_normal": bool(
                len(pre_temperature_difference) == 0
                or float(np.nanmax(pre_temperature_difference)) <= 1e-8
            ),
            "pre_outage_load_matches_normal": bool(
                len(pre_load_difference) == 0
                or float(np.nanmax(pre_load_difference)) <= 1e-10
            ),
            "maximum_pre_outage_temperature_difference_c": (
                float(np.nanmax(pre_temperature_difference))
                if len(pre_temperature_difference) else 0.0
            ),
            "maximum_pre_outage_load_difference_kwh_m2": (
                float(np.nanmax(pre_load_difference)) if len(pre_load_difference) else 0.0
            ),
        }
        row["temporal_contract_pass"] = bool(
            actual_hours == expected_hours
            and row["timestamps_match_normal"]
            and row["starts_in_heatwave"]
            and (
                str(stage) == "outage_common_24h"
                or row["normal_cooling_active_at_start"]
            )
            and (
                str(stage) == "outage_common_24h"
                or row["occupied_service_at_start"]
            )
            and row["all_grid_load_zero_during_outage"]
            and row["cooling_disabled_during_outage"]
            and row["pre_outage_temperature_matches_normal"]
            and row["pre_outage_load_matches_normal"]
        )
        rows.append(row)
    audit = pd.DataFrame(rows)
    if audit.empty:
        raise ValueError("No 6-hour or 24-hour outage runs were available for temporal audit")
    summary = {
        "policy": CONFIG["resilience"]["peak_outage_start_policy"],
        "audited_cases": int(len(audit)),
        "six_hour_cases": int(audit["supply_stage"].eq("outage_peak_6h").sum()),
        "twenty_four_hour_cases": int(audit["supply_stage"].eq("outage_peak_24h").sum()),
        "common_clock_twenty_four_hour_cases": int(
            audit["supply_stage"].eq("outage_common_24h").sum()
        ),
        "fallback_cases": int(audit["fallback_level"].ne("none").sum()),
        "failed_cases": int((~audit["temporal_contract_pass"]).sum()),
        "maximum_pre_outage_temperature_difference_c": float(
            audit["maximum_pre_outage_temperature_difference_c"].max()
        ),
        "maximum_pre_outage_load_difference_kwh_m2": float(
            audit["maximum_pre_outage_load_difference_kwh_m2"].max()
        ),
        "pass": bool(audit["temporal_contract_pass"].all()),
    }
    if not summary["pass"]:
        failed = audit.loc[~audit["temporal_contract_pass"]].head(10)
        raise ValueError(f"Outage temporal contract failed:\n{failed.to_string(index=False)}")
    return audit, summary


def data_centre_outage_physics_audit(
    hourly: pd.DataFrame,
    parameters: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Trace data-centre electricity and temperature through 24-h outages."""
    prototype_id = "com_data_centre"
    parameter = parameters.loc[parameters["prototype_id"].eq(prototype_id)]
    if len(parameter) != 1:
        raise ValueError("Data-centre prototype parameter row is missing or duplicated")
    radiant_fraction = float(parameter["equipment_radiant_fraction"].iloc[0])
    equipment_w_m2 = float(parameter["equipment_w_m2"].iloc[0])
    work = hourly.loc[
        hourly["prototype_id"].eq(prototype_id)
        & hourly["climate_case"].eq("heatwave")
    ].copy()
    work["timestamp_hkt"] = pd.to_datetime(work["timestamp_hkt"], utc=True).dt.tz_convert(
        "Asia/Hong_Kong"
    )
    control = work.loc[work["supply_stage"].eq("capacity_100pct"), [
        "event_id", "base_case_id", "timestamp_hkt", "exposure_temperature_c",
    ]].rename(columns={"exposure_temperature_c": "control_exposure_temperature_c"})
    rows = []
    for keys, group in work.loc[work["supply_stage"].isin(
        ["outage_peak_24h", "outage_common_24h"]
    )].groupby(["event_id", "base_case_id", "supply_stage"], observed=True):
        event_id, base_case_id, stage = map(str, keys)
        group = group.sort_values("timestamp_hkt").merge(
            control.loc[
                control["event_id"].astype(str).eq(event_id)
                & control["base_case_id"].astype(str).eq(base_case_id)
            ],
            on=["event_id", "base_case_id", "timestamp_hkt"],
            how="left", validate="one_to_one",
        )
        outage = numeric(group["total_electricity_kwh_m2"]).abs().le(1e-10)
        window = group.loc[outage].copy()
        if len(window) != 24:
            raise ValueError(
                f"Data-centre {event_id}/{base_case_id}/{stage} has {len(window)} zero-grid hours"
            )
        rows.append({
            "event_id": event_id,
            "base_case_id": base_case_id,
            "supply_stage": stage,
            "outage_start_hkt": window["timestamp_hkt"].iloc[0].isoformat(),
            "outage_end_hkt_exclusive": (window["timestamp_hkt"].iloc[-1] + pd.Timedelta(hours=1)).isoformat(),
            "outage_hours": int(len(window)),
            "maximum_grid_electricity_kwh_m2_during_outage": float(
                numeric(window["total_electricity_kwh_m2"]).abs().max()
            ),
            "maximum_equipment_electricity_kwh_m2_during_outage": float(
                numeric(window["equipment_kwh_m2"]).abs().max()
            ),
            "maximum_lighting_electricity_kwh_m2_during_outage": float(
                numeric(window["lighting_kwh_m2"]).abs().max()
            ),
            "maximum_cooling_electricity_kwh_m2_during_outage": float(
                numeric(window["cooling_kwh_m2"]).abs().max()
            ),
            "temperature_at_outage_start_c": float(window["exposure_temperature_c"].iloc[0]),
            "maximum_outage_temperature_c": float(numeric(window["exposure_temperature_c"]).max()),
            "maximum_control_temperature_c": float(numeric(window["control_exposure_temperature_c"]).max()),
            "maximum_outage_temperature_increment_c": float(
                (
                    numeric(window["exposure_temperature_c"])
                    - numeric(window["control_exposure_temperature_c"])
                ).max()
            ),
            "equipment_process_power_w_m2": equipment_w_m2,
            "equipment_radiant_fraction": radiant_fraction,
            "equipment_radiant_gain_w_m2_when_operating": equipment_w_m2 * radiant_fraction,
            "equipment_convective_gain_w_m2_when_operating": equipment_w_m2 * (1.0 - radiant_fraction),
        })
    audit = pd.DataFrame(rows)
    checks = {
        "two_24h_outage_definitions_present": set(audit["supply_stage"].unique())
        == {"outage_peak_24h", "outage_common_24h"},
        "all_outage_windows_are_24h": bool(audit["outage_hours"].eq(24).all()),
        "grid_and_internal_electric_loads_zero_during_outage": bool(
            audit[[
                "maximum_grid_electricity_kwh_m2_during_outage",
                "maximum_equipment_electricity_kwh_m2_during_outage",
                "maximum_lighting_electricity_kwh_m2_during_outage",
                "maximum_cooling_electricity_kwh_m2_during_outage",
            ]].le(1e-10).all().all()
        ),
        "temperatures_finite": bool(np.isfinite(audit[[
            "temperature_at_outage_start_c", "maximum_outage_temperature_c",
            "maximum_control_temperature_c", "maximum_outage_temperature_increment_c",
        ]].to_numpy(dtype=float)).all()),
    }
    summary = {
        "rows": int(len(audit)),
        "events": int(audit["event_id"].nunique()),
        "base_cases": int(audit["base_case_id"].nunique()),
        "equipment_radiant_fraction": radiant_fraction,
        "maximum_outage_temperature_c": float(audit["maximum_outage_temperature_c"].max()),
        "maximum_outage_temperature_increment_c": float(
            audit["maximum_outage_temperature_increment_c"].max()
        ),
        "checks": checks,
        "pass": bool(all(checks.values())),
    }
    if not summary["pass"]:
        failures = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Data-centre outage physics audit failed: {failures}")
    return audit, summary


def audit_parameter_and_experiment_design(
    parameters: pd.DataFrame,
    design: pd.DataFrame,
    event_ids: list[str],
    expected_run_count: int,
    completed_run_count: int,
    smoke: bool = False,
) -> dict:
    """Validate parameter domains and the selected balanced experiment."""
    bounds = {
        "floor_area_m2": (1.0, 5000.0),
        "height_m": (2.0, 6.0),
        "u_wall_w_m2k": (0.1, 6.0),
        "u_roof_w_m2k": (0.1, 6.0),
        "window_to_wall_ratio": (0.01, 0.90),
        "window_shgc": (0.05, 0.95),
        "window_u_w_m2k": (0.5, 8.0),
        "infiltration_ach": (0.05, 3.0),
        "thermal_capacitance_kj_m2k": (50.0, 500.0),
        "baseline_ac_cop": (1.0, 8.0),
        "people_per_m2": (0.001, 0.5),
        "lighting_w_m2": (0.1, 40.0),
        "equipment_w_m2": (0.1, 1000.0),
        "equipment_radiant_fraction": (0.0, 1.0),
        "cooling_capacity_w_m2": (25.0, 1500.0),
    }
    violations: dict[str, int] = {}
    ranges: dict[str, dict[str, float]] = {}
    for field, (lower, upper) in bounds.items():
        values = numeric(parameters[field])
        violations[field] = int((values.isna() | values.lt(lower) | values.gt(upper)).sum())
        ranges[field] = {"minimum": float(values.min()), "maximum": float(values.max())}
    expected_prototypes = set(CONFIG["prototype_physics"]["values"])
    design_prototypes = set(design["prototype_id"].astype(str))
    expected_setpoints = set(map(float, THERMAL_CONFIG["setpoint_scenarios_c"]))
    observed_setpoints = set(map(float, numeric(design["setpoint_c"]).dropna().unique()))
    required_floors = {"bottom", "middle", "top", "single"}
    required_orientations = {0, 90, 180, 270}
    residential_behaviours = set(
        design.loc[design["sector"].eq("residential"), "behaviour"].astype(str)
    )
    noncritical_commercial = design.loc[
        design["sector"].eq("commercial")
        & ~design["prototype_id"].isin({"com_human_health", "com_data_centre", "com_accommodation"})
    ]
    residential_parameters = parameters.loc[parameters["sector"].eq("residential")]
    residential_evidence_bounds = CONFIG["prototype_physics"][
        "residential_evidence_bounds"
    ]
    residential_evidence_checks = {
        field: bool(
            numeric(residential_parameters[field]).between(
                float(interval[0]), float(interval[1])
            ).all()
        )
        for field, interval in residential_evidence_bounds.items()
        if field != "sources"
    }
    commercial_reference_checks = {}
    for prototype_id, reference in CONFIG["prototype_physics"][
        "commercial_internal_load_reference_ranges"
    ].items():
        row = parameters.loc[parameters["prototype_id"].eq(prototype_id)]
        if len(row) != 1:
            commercial_reference_checks[prototype_id] = False
            continue
        commercial_reference_checks[prototype_id] = bool(
            all(
                float(interval[0]) <= float(row.iloc[0][field]) <= float(interval[1])
                for field, interval in reference.items()
                if field != "sources"
            )
        )
    per_prototype_coverage = {}
    for prototype_id, group in design.groupby("prototype_id", observed=True):
        is_critical = prototype_id in {"com_human_health", "com_data_centre", "com_accommodation"}
        expected_behaviours = (
            {"continuous"}
            if is_critical
            else ({"working", "older_home", "wfh", "frugal"} if group["sector"].iloc[0] == "residential" else {"standard", "extended", "low_occupancy"})
        )
        expected_offsets = {0} if is_critical else set(map(int, THERMAL_CONFIG["start_offset_scenarios_h"]))
        expected_levels = {
            "floor_position": required_floors,
            "orientation_deg": required_orientations,
            "behaviour": expected_behaviours,
            "setpoint_c": expected_setpoints,
            "start_offset_h": expected_offsets,
        }
        pairwise_coverage = {}
        for left, right in itertools.combinations(expected_levels, 2):
            expected_pairs = {
                (str(left_value), str(right_value))
                for left_value in expected_levels[left]
                for right_value in expected_levels[right]
            }
            observed_pairs = set(
                map(tuple, group[[left, right]].astype(str).to_numpy())
            )
            pairwise_coverage[f"{left}__{right}"] = {
                "covered": int(len(observed_pairs & expected_pairs)),
                "expected": int(len(expected_pairs)),
                "fraction": float(len(observed_pairs & expected_pairs) / len(expected_pairs)),
            }
        pairwise_complete = all(
            item["covered"] == item["expected"] for item in pairwise_coverage.values()
        )
        per_prototype_coverage[str(prototype_id)] = {
            "floor_positions": sorted(group["floor_position"].astype(str).unique()),
            "orientations_deg": sorted(map(int, group["orientation_deg"].unique())),
            "behaviours": sorted(group["behaviour"].astype(str).unique()),
            "setpoints_c": sorted(map(float, numeric(group["setpoint_c"]).unique())),
            "start_offsets_h": sorted(map(int, numeric(group["start_offset_h"]).unique())),
            "pairwise_factor_coverage": pairwise_coverage,
            "pairwise_complete": pairwise_complete,
            "complete": bool(
                required_floors.issubset(set(group["floor_position"].astype(str)))
                and required_orientations.issubset(set(map(int, group["orientation_deg"].unique())))
                and expected_setpoints.issubset(set(map(float, numeric(group["setpoint_c"]).unique())))
                and expected_behaviours.issubset(set(group["behaviour"].astype(str)))
                and expected_offsets.issubset(set(map(int, numeric(group["start_offset_h"]).unique())))
                and pairwise_complete
            ),
        }
    full_design_checks = {
        "design_prototype_coverage_exact": design_prototypes == expected_prototypes,
        "floor_exposure_coverage": required_floors.issubset(set(design["floor_position"].astype(str))),
        "orientation_coverage": required_orientations.issubset(set(map(int, design["orientation_deg"].unique()))),
        "setpoint_coverage": expected_setpoints.issubset(observed_setpoints),
        "residential_behaviour_coverage": {"working", "older_home", "wfh", "frugal"}.issubset(residential_behaviours),
        "commercial_schedule_coverage": {"standard", "extended", "low_occupancy"}.issubset(
            set(noncritical_commercial["behaviour"].astype(str))
        ),
        "within_prototype_factor_level_coverage": all(
            item["complete"] for item in per_prototype_coverage.values()
        ),
        "within_prototype_pairwise_factor_coverage": all(
            item["pairwise_complete"] for item in per_prototype_coverage.values()
        ),
    }
    smoke_design_checks = {
        "smoke_subset_is_registered": bool(design_prototypes and design_prototypes.issubset(expected_prototypes)),
        "smoke_covers_both_sectors": set(design["sector"].astype(str)) == {"residential", "commercial"},
    }
    checks = {
        "parameter_domains": not any(violations.values()),
        "residential_parameters_within_hong_kong_evidence_bounds": all(
            residential_evidence_checks.values()
        ),
        "commercial_internal_loads_within_declared_reference_ranges": all(
            commercial_reference_checks.values()
        ),
        "conditioned_relative_humidity_range_valid": bool(
            len(THERMAL_CONFIG["conditioned_relative_humidity_range_pct"]) == 2
            and 0.0 < float(THERMAL_CONFIG["conditioned_relative_humidity_range_pct"][0])
            < float(THERMAL_CONFIG["conditioned_relative_humidity_range_pct"][1]) < 100.0
        ),
        "data_centre_70pct_capacity_exceeds_continuous_internal_power": bool(
            float(parameters.loc[
                parameters["prototype_id"].eq("com_data_centre"), "cooling_capacity_w_m2"
            ].iloc[0]) * 0.70
            > float(parameters.loc[
                parameters["prototype_id"].eq("com_data_centre"), "equipment_w_m2"
            ].iloc[0])
            + float(parameters.loc[
                parameters["prototype_id"].eq("com_data_centre"), "lighting_w_m2"
            ].iloc[0])
        ),
        "parameter_prototype_registry_exact": set(parameters["prototype_id"].astype(str)) == expected_prototypes,
        "design_rows_unique": not bool(design.drop(columns=["base_case_id"], errors="ignore").duplicated().any()),
        "base_case_ids_unique": not bool(design["base_case_id"].duplicated().any()),
        **(smoke_design_checks if smoke else full_design_checks),
        "available_ac_condition_explicit": bool(numeric(design["ac_use_probability"]).eq(1.0).all()),
        "historical_event_suite_size": len(event_ids) == (1 if smoke else int(THERMAL_CONFIG["max_events"])),
        "selected_design_execution_complete": completed_run_count == expected_run_count,
    }
    result = {
        "model_version": MODEL_VERSION,
        "parameter_status": str(parameters["parameter_status"].iloc[0]),
        "parameter_ranges": ranges,
        "parameter_bound_violations": violations,
        "residential_evidence_bounds": residential_evidence_bounds,
        "residential_evidence_checks": residential_evidence_checks,
        "commercial_internal_load_reference_ranges": CONFIG["prototype_physics"][
            "commercial_internal_load_reference_ranges"
        ],
        "commercial_internal_load_reference_checks": commercial_reference_checks,
        "prototype_count": int(parameters["prototype_id"].nunique()),
        "base_case_count": int(len(design)),
        "base_cases_per_prototype": {
            str(key): int(value)
            for key, value in design.groupby("prototype_id", observed=True).size().items()
        },
        "within_prototype_coverage": per_prototype_coverage,
        "event_ids": list(map(str, event_ids)),
        "supply_stages_per_event_case": int(
            expected_run_count / max(len(design) * len(event_ids), 1)
        ),
        "available_ac_condition": "installed AC available; schedules, rooms, setpoint and start offset vary",
        "conditioned_relative_humidity_range_pct": THERMAL_CONFIG[
            "conditioned_relative_humidity_range_pct"
        ],
        "conditioned_relative_humidity_source": THERMAL_CONFIG[
            "conditioned_relative_humidity_source"
        ],
        "commercial_outdoor_air_l_s_person": THERMAL_CONFIG["commercial_outdoor_air_l_s_person"],
        "checks": checks,
        "pass": bool(all(checks.values())),
    }
    if not result["pass"]:
        failures = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Parameter/experiment audit failed: {failures}")
    return result


def audit_building_physics_consistency(
    dynamic: pd.DataFrame,
    hourly: pd.DataFrame,
) -> dict:
    """Check cooling-energy balance, hourly control, and response bounds."""
    required = [
        "required_cooling_energy_kwh_th", "delivered_cooling_energy_kwh_th",
        "unmet_cooling_energy_kwh_th", "safety_resistance_hours_to_32c",
        "recovery_time_hours", "baseline_energy_kwh", "heatwave_energy_kwh",
        "baseline_peak_kw", "heatwave_peak_kw",
        "cooling_start_transition_allowance_hours",
    ]
    missing = sorted(set(required) - set(dynamic.columns))
    if missing:
        raise ValueError(f"Building dynamic results lack physics-audit fields: {missing}")
    required_hourly = [
        "supply_stage", "unmet_cooling_fraction", "cooling_start_transition_fraction",
    ]
    missing_hourly = sorted(set(required_hourly) - set(hourly.columns))
    if missing_hourly:
        raise ValueError(f"Hourly results lack physics-audit fields: {missing_hourly}")
    capacity = dynamic.loc[dynamic["supply_stage"].eq("capacity_70pct")].copy()
    normal = dynamic.loc[dynamic["supply_stage"].eq("normal")].copy()
    required_cooling = numeric(capacity["required_cooling_energy_kwh_th"])
    delivered = numeric(capacity["delivered_cooling_energy_kwh_th"])
    unmet = numeric(capacity["unmet_cooling_energy_kwh_th"])
    residual = (required_cooling - delivered - unmet).abs()
    outage = dynamic.loc[dynamic["supply_stage"].eq("outage_peak_24h")]
    safety = numeric(outage["safety_resistance_hours_to_32c"])
    recovery = numeric(outage["recovery_time_hours"])
    normal_unmet = numeric(normal["unmet_cooling_hours"]).fillna(0.0)
    startup_allowance = numeric(
        normal["cooling_start_transition_allowance_hours"]
    ).fillna(0.0)
    sustained_control_error = (normal_unmet - startup_allowance).clip(lower=0.0)
    normal_hourly = hourly.loc[hourly["supply_stage"].eq("normal")].copy()
    hourly_unmet = numeric(normal_hourly["unmet_cooling_fraction"]).fillna(0.0)
    hourly_startup_allowance = (
        numeric(normal_hourly["cooling_start_transition_fraction"]).fillna(0.0)
        * float(CONFIG["study_scope"]["simulation_timestep_minutes"]) / 60.0
    )
    hourly_unexplained_control_error = (
        hourly_unmet - hourly_startup_allowance
    ).clip(lower=0.0)
    finite_columns = [
        "baseline_energy_kwh", "heatwave_energy_kwh", "baseline_peak_kw",
        "heatwave_peak_kw", "required_cooling_energy_kwh_th",
        "delivered_cooling_energy_kwh_th", "unmet_cooling_energy_kwh_th",
    ]
    nonfinite = {
        field: int((~np.isfinite(numeric(dynamic[field]).to_numpy(dtype=float))).sum())
        for field in finite_columns
    }
    checks = {
        "capacity_rows_present": len(capacity) > 0,
        "normal_supply_unmet_is_resolved_startup_transient_only": bool(
            len(normal) > 0
            and len(normal_hourly) > 0
            and sustained_control_error.le(1e-9).all()
            and hourly_unexplained_control_error.le(1e-9).all()
        ),
        "cooling_energy_balance": bool(residual.le(1e-6).all()),
        "delivered_not_above_required": bool((delivered <= required_cooling + 1e-6).all()),
        "unmet_nonnegative": bool(unmet.ge(-1e-6).all()),
        "finite_energy_and_peak_outputs": not any(nonfinite.values()),
        "twenty_four_hour_safety_bounds": bool(safety.notna().all() and safety.between(0.0, 24.0).all()),
        "recovery_assessment_bounds": bool(
            recovery.notna().all()
            and recovery.between(0.0, float(THERMAL_CONFIG["recovery_assessment_hours"])).all()
        ),
    }
    summary = {
        "capacity_rows": int(len(capacity)),
        "normal_supply_rows": int(len(normal)),
        "maximum_normal_supply_unmet_cooling_hours": float(normal_unmet.max()),
        "maximum_normal_supply_startup_allowance_hours": float(startup_allowance.max()),
        "maximum_unexplained_normal_supply_control_error_hours": float(
            sustained_control_error.max()
        ),
        "normal_supply_hour_rows": int(len(normal_hourly)),
        "maximum_hourly_normal_supply_unmet_fraction": float(hourly_unmet.max()),
        "maximum_hourly_normal_supply_startup_allowance_fraction_hours": float(
            hourly_startup_allowance.max()
        ),
        "maximum_hourly_unexplained_normal_supply_control_error_hours": float(
            hourly_unexplained_control_error.max()
        ),
        "hourly_rows_with_unexplained_normal_supply_control_error": int(
            hourly_unexplained_control_error.gt(1e-9).sum()
        ),
        "normal_supply_control_tolerance_basis": (
            "Every non-zero Zone Cooling Setpoint Not Met Time must be no larger than "
            "one 15-minute model timestep multiplied by the conditioned floor-area "
            "fraction whose discrete cooling schedule starts in that hour; no sustained "
            "temperature-control error is accepted."
        ),
        "outage_24h_rows": int(len(outage)),
        "maximum_cooling_balance_residual_kwh_th": float(residual.max()),
        "delivered_above_required_rows": int((delivered > required_cooling + 1e-6).sum()),
        "negative_unmet_rows": int(unmet.lt(-1e-6).sum()),
        "nonfinite_counts": nonfinite,
        "safety_resistance_range_hours": [float(safety.min()), float(safety.max())],
        "recovery_time_range_hours": [float(recovery.min()), float(recovery.max())],
        "checks": checks,
        "pass": bool(all(checks.values())),
    }
    if not summary["pass"]:
        failures = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Building physics consistency audit failed: {failures}")
    return summary


def audit_paired_counterfactual_contract(hourly: pd.DataFrame) -> dict:
    """Verify the persisted normal-supply pair differs only through weather response."""
    keys = ["event_id", "base_case_id", "timestamp_hkt"]
    fields = [
        "prototype_id", "sector", "floor_position", "orientation_deg", "behaviour",
        "setpoint_c", "ac_use_probability", "start_offset_h", "behavior_seed",
        "event_phase", "occupied_fraction", "active_cooling_fraction",
        "lighting_kwh_m2", "equipment_kwh_m2", "outdoor_temperature_c",
    ]
    missing = sorted(set(keys + fields + ["climate_case", "supply_stage"]) - set(hourly))
    if missing:
        raise ValueError(f"Hourly results lack counterfactual-audit fields: {missing}")
    normal = hourly.loc[hourly["supply_stage"].eq("normal"), keys + fields + ["climate_case"]]
    heatwave = normal.loc[normal["climate_case"].eq("heatwave")].drop(columns="climate_case")
    baseline = normal.loc[normal["climate_case"].eq("counterfactual")].drop(columns="climate_case")
    paired = heatwave.merge(
        baseline, on=keys, how="outer", validate="one_to_one", suffixes=("_heatwave", "_counterfactual"),
        indicator=True,
    )
    identity_fields = [
        "prototype_id", "sector", "floor_position", "orientation_deg", "behaviour",
        "setpoint_c", "ac_use_probability", "start_offset_h", "behavior_seed", "event_phase",
    ]
    numeric_schedule_fields = [
        "occupied_fraction", "active_cooling_fraction", "lighting_kwh_m2", "equipment_kwh_m2",
    ]
    identity_mismatches = {
        field: int((paired[f"{field}_heatwave"].astype(str) != paired[f"{field}_counterfactual"].astype(str)).sum())
        for field in identity_fields
    }
    maximum_schedule_difference = {
        field: float(
            (
                numeric(paired[f"{field}_heatwave"])
                - numeric(paired[f"{field}_counterfactual"])
            ).abs().max()
        )
        for field in numeric_schedule_fields
    }
    outdoor_delta = (
        numeric(paired["outdoor_temperature_c_heatwave"])
        - numeric(paired["outdoor_temperature_c_counterfactual"])
    ).abs()
    checks = {
        "all_pairs_present": bool(paired["_merge"].eq("both").all()),
        "identifiers_and_event_phase_identical": not any(identity_mismatches.values()),
        "occupancy_and_grid_load_schedules_identical": bool(
            max(maximum_schedule_difference.values(), default=0.0) <= 1e-10
        ),
        "weather_perturbation_present": bool(outdoor_delta.gt(1e-6).any()),
    }
    result = {
        "paired_hour_rows": int(len(paired)),
        "identity_mismatches": identity_mismatches,
        "maximum_schedule_difference": maximum_schedule_difference,
        "maximum_outdoor_temperature_difference_c": float(outdoor_delta.max()),
        "checks": checks,
        "pass": bool(all(checks.values())),
    }
    if not result["pass"]:
        failures = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Paired counterfactual audit failed: {failures}")
    return result


def audit_energyplus_diagnostics(
    event_diagnostics: pd.DataFrame,
    annual_models: pd.DataFrame,
) -> dict:
    """Fail on severe errors or any unclassified EnergyPlus warning."""
    ground_temperature_warning = (
        "Site:GroundTemperature:BuildingSurface: Some values fall outside the range of 15-25C."
    )
    def is_saturation_limit_warning(line: str) -> bool:
        return (
            "ZoneHVAC:IdealLoadsAirSystem" in line
            and "exceeds saturation limit" in line
            and (
                "Supply humidity ratio =" in line
                or "Supply humidity ratio exceeds saturation limit warning continues" in line
            )
        )
    lines: list[str] = []
    for frame in [event_diagnostics, annual_models]:
        for value in frame.get("warning_messages", pd.Series(dtype="object")).dropna():
            lines.extend(line.strip() for line in str(value).splitlines() if line.strip())
    ground_temperature_warning_count = sum(
        ground_temperature_warning in line for line in lines
    )
    saturation_limit_warning_count = sum(
        is_saturation_limit_warning(line) for line in lines
    )
    unexpected = sorted(
        {
            line
            for line in lines
            if ground_temperature_warning not in line
            and not is_saturation_limit_warning(line)
        }
    )
    severe_errors = int(
        numeric(event_diagnostics.get("severe_errors", pd.Series(dtype=float))).fillna(0).sum()
        + numeric(annual_models.get("severe_errors", pd.Series(dtype=float))).fillna(0).sum()
    )
    reported_warnings = int(
        numeric(event_diagnostics.get("warnings", pd.Series(dtype=float))).fillna(0).sum()
        + numeric(annual_models.get("warnings", pd.Series(dtype=float))).fillna(0).sum()
    )
    checks = {
        "zero_severe_errors": severe_errors == 0,
        "all_warnings_classified": len(unexpected) == 0,
        "warning_lines_accounted_for": len(lines) == reported_warnings,
    }
    result = {
        "event_runs": int(len(event_diagnostics)),
        "annual_runs": int(len(annual_models)),
        "severe_errors": severe_errors,
        "reported_warnings": reported_warnings,
        "classified_warning_lines": int(len(lines)),
        "accepted_warning": ground_temperature_warning,
        "accepted_warning_classes": {
            "official_hko_ground_temperature_outside_generic_advisory_range": int(
                ground_temperature_warning_count
            ),
            "ideal_loads_supply_humidity_limited_to_saturation": int(
                saturation_limit_warning_count
            ),
        },
        "accepted_warning_basis": (
            "EnergyPlus advisory range is exceeded by the directly supplied HKO 1991-2020 "
            "0.5-m monthly soil normals (up to 29.8 C); the official values are retained. "
            "During a subset of high-humidity, capacity-derated stress cases, EnergyPlus "
            "limits the calculated ideal-loads supply humidity ratio to the thermodynamic "
            "saturation value. These corrections are retained, counted separately and are "
            "accepted only when no severe error or unclassified warning is present."
        ),
        "accepted_warning_source": THERMAL_CONFIG["soil_temperature_source"],
        "unexpected_warning_messages": unexpected,
        "checks": checks,
        "pass": bool(all(checks.values())),
    }
    if not result["pass"]:
        failures = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"EnergyPlus diagnostics audit failed: {failures}")
    return result


def response_quantiles(results: pd.DataFrame) -> pd.DataFrame:
    """Reduce the experiment while retaining simulated floor exposure."""
    if "floor_position" not in results:
        raise ValueError("EnergyPlus results lack floor_position needed for whole-building composition")
    baseline_fields = ["total_electricity_kwh_m2", "peak_electric_w_m2"]
    heatwave_fields = [
        "heatwave_days", "total_electricity_kwh_m2", "peak_electric_w_m2",
        "cooling_thermal_kwh_m2",
        "all_hour_degree_hours_above_26c", "night_degree_hours_above_26c",
        "continuous_threshold_exceedance_hours", "comfort_resistance_hours_to_26c",
        "unmet_cooling_hours", "cooling_start_transition_allowance_hours",
        "recovery_energy_kwh_m2", "recovery_to_26c_hours",
        "outage_window_degree_hours_above_26c",
        "outage_attributable_degree_hours_above_26c",
        "outage_peak_temperature_increment_c",
        "outage_window_degree_hours_above_32c", "safety_resistance_hours_to_32c",
        "outage_window_degree_hours_above_30c", "safety_resistance_hours_to_30c",
        "outage_window_degree_hours_above_34c", "safety_resistance_hours_to_34c",
        "occupied_service_hours_lost", "occupied_overheating_hours",
        "occupied_overheating_degree_hours", "critical_service_continuity_hours",
    ]

    def quantile_table(frame: pd.DataFrame, groups: list[str], fields: list[str], prefix: str) -> pd.DataFrame:
        pieces = []
        grouped = frame.groupby(groups, observed=True)
        for field in fields:
            quantiles = grouped[field].quantile([0.05, 0.50, 0.95]).unstack(-1).reset_index()
            quantiles = quantiles.rename(columns={
                0.05: f"{prefix}{field}_p05",
                0.50: f"{prefix}{field}_p50",
                0.95: f"{prefix}{field}_p95",
            })
            pieces.append(quantiles)
        result = pieces[0]
        for piece in pieces[1:]:
            result = result.merge(piece, on=groups, how="outer", validate="one_to_one")
        return result

    baseline = results.loc[
        results["climate_case"].eq("counterfactual") & results["supply_stage"].eq("normal")
    ]
    heatwave = results.loc[results["climate_case"].eq("heatwave")]
    floor_keys = ["prototype_id", "event_id", "floor_position"]
    baseline_q = quantile_table(baseline, floor_keys, baseline_fields, "baseline_")
    heatwave_q = quantile_table(
        heatwave, [*floor_keys, "supply_stage"], heatwave_fields, "heatwave_"
    )
    normal_service = heatwave_q.loc[
        heatwave_q["supply_stage"].eq("normal"),
        [*floor_keys, "heatwave_cooling_thermal_kwh_m2_p05",
         "heatwave_cooling_thermal_kwh_m2_p50", "heatwave_cooling_thermal_kwh_m2_p95"],
    ].rename(columns={
        "heatwave_cooling_thermal_kwh_m2_p05": "required_cooling_thermal_kwh_m2_p05",
        "heatwave_cooling_thermal_kwh_m2_p50": "required_cooling_thermal_kwh_m2_p50",
        "heatwave_cooling_thermal_kwh_m2_p95": "required_cooling_thermal_kwh_m2_p95",
    })
    response = heatwave_q.merge(
        baseline_q, on=floor_keys, how="left", validate="many_to_one"
    )
    response = response.merge(
        normal_service, on=floor_keys, how="left", validate="many_to_one"
    )
    if response[[
        "baseline_total_electricity_kwh_m2_p50", "heatwave_total_electricity_kwh_m2_p50"
    ]].isna().any().any():
        raise ValueError("Prototype response table lacks paired baseline or heatwave energy")
    return response


def build_building_dynamic_results(
    inputs: gpd.GeoDataFrame,
    results: pd.DataFrame,
    wrf_mapping: pd.DataFrame | None,
    seed: int,
) -> pd.DataFrame:
    """Propagate prototype ensembles to every valid stock footprint.

    Mixed-use components and physical floor positions are calculated
    independently, then conserved when aggregated to the physical building.
    """
    stock = inputs.loc[inputs["pre_weather_simulation_candidate"]].copy()
    population_field = "allocated_population__ssug_landuse_residential_gfa"
    static_fields = [
        "BUILDINGSTRUCTUREID", "main_class", "sub_class", "gfa_value_m2",
        "residential_gfa_m2", "commercial_gfa_m2", "floors_value",
        "height_value_m", "construction_year_value",
    ]
    static = stock[static_fields].drop_duplicates("BUILDINGSTRUCTUREID").set_index("BUILDINGSTRUCTUREID")
    if len(static) != stock["BUILDINGSTRUCTUREID"].nunique():
        raise ValueError("Building stock identifiers are not unique")

    components = []
    residential = stock.loc[
        stock["residential_component_present"] & stock["residential_gfa_m2"].gt(0),
        ["BUILDINGSTRUCTUREID", "residential_gfa_m2", "residential_prototype_id", population_field, "floors_value"],
    ].rename(columns={
        "residential_gfa_m2": "component_gfa_m2",
        "residential_prototype_id": "prototype_id",
        population_field: "resident_population",
    })
    residential["component"] = "residential"
    components.append(residential)
    commercial = stock.loc[
        stock["commercial_component_present"] & stock["commercial_gfa_m2"].gt(0),
        ["BUILDINGSTRUCTUREID", "commercial_gfa_m2", "commercial_prototype_id", "floors_value"],
    ].rename(columns={
        "commercial_gfa_m2": "component_gfa_m2",
        "commercial_prototype_id": "prototype_id",
    })
    commercial["resident_population"] = 0.0
    commercial["component"] = "commercial"
    components.append(commercial)
    component = pd.concat(components, ignore_index=True)
    component["component_gfa_m2"] = numeric(component["component_gfa_m2"], lower=0)
    component["resident_population"] = numeric(component["resident_population"], lower=0).fillna(0.0)
    if component[["prototype_id", "component_gfa_m2"]].isna().any().any():
        raise ValueError("A valid building component lacks a prototype or GFA")
    physics = prototype_parameter_table()[["prototype_id", "people_per_m2"]]
    component = component.merge(physics, on="prototype_id", how="left", validate="many_to_one")
    floor_weights = floor_role_weight_table(component["floors_value"])
    floor_parts = []
    for floor_position in FLOOR_ROLES:
        weight = floor_weights[floor_position]
        selected = weight.gt(0)
        part = component.loc[selected].copy()
        part["floor_position"] = floor_position
        part["floor_weight"] = weight.loc[selected].to_numpy()
        part["floor_component_gfa_m2"] = part["component_gfa_m2"] * part["floor_weight"]
        part["floor_resident_population"] = part["resident_population"] * part["floor_weight"]
        floor_parts.append(part)
    component_floors = pd.concat(floor_parts, ignore_index=True)
    response = response_quantiles(results)

    weighted_metrics = {
        "overheating_degree_hours": "heatwave_all_hour_degree_hours_above_26c_p50",
        "outage_window_degree_hours_above_26c": "heatwave_outage_window_degree_hours_above_26c_p50",
        "outage_attributable_degree_hours_above_26c": "heatwave_outage_attributable_degree_hours_above_26c_p50",
        "outage_peak_temperature_increment_c": "heatwave_outage_peak_temperature_increment_c_p50",
        "outage_window_degree_hours_above_32c": "heatwave_outage_window_degree_hours_above_32c_p50",
        "outage_window_degree_hours_above_30c": "heatwave_outage_window_degree_hours_above_30c_p50",
        "outage_window_degree_hours_above_34c": "heatwave_outage_window_degree_hours_above_34c_p50",
        "nighttime_overheating_degree_hours": "heatwave_night_degree_hours_above_26c_p50",
        "unmet_cooling_hours": "heatwave_unmet_cooling_hours_p50",
        "cooling_start_transition_allowance_hours": "heatwave_cooling_start_transition_allowance_hours_p50",
        "occupied_service_hours_lost": "heatwave_occupied_service_hours_lost_p50",
    }
    all_rows = []
    for (event_id, supply_stage), response_slice in response.groupby(["event_id", "supply_stage"], observed=True):
        joined = component_floors.merge(
            response_slice,
            on=["prototype_id", "floor_position"],
            how="left",
            validate="many_to_one",
        )
        if joined["heatwave_total_electricity_kwh_m2_p50"].isna().any():
            missing = sorted(joined.loc[joined["heatwave_total_electricity_kwh_m2_p50"].isna(), "prototype_id"].unique())
            raise ValueError(f"No experiment response for prototypes: {missing}")
        area = joined["floor_component_gfa_m2"]
        joined["baseline_energy_kwh"] = area * joined["baseline_total_electricity_kwh_m2_p50"]
        joined["heatwave_energy_kwh"] = area * joined["heatwave_total_electricity_kwh_m2_p50"]
        joined["baseline_peak_kw"] = area * joined["baseline_peak_electric_w_m2_p50"] / 1000.0
        joined["heatwave_peak_kw"] = area * joined["heatwave_peak_electric_w_m2_p50"] / 1000.0
        joined["balanced_design_energy_p05_kwh"] = area * joined["heatwave_total_electricity_kwh_m2_p05"]
        joined["balanced_design_energy_p95_kwh"] = area * joined["heatwave_total_electricity_kwh_m2_p95"]
        joined["recovery_energy_kwh"] = area * joined["heatwave_recovery_energy_kwh_m2_p50"]
        joined["required_cooling_energy_kwh_th"] = (
            area * joined["required_cooling_thermal_kwh_m2_p50"]
        )
        available_cooling_energy = (
            area * joined["heatwave_cooling_thermal_kwh_m2_p50"]
        )
        joined["delivered_cooling_energy_kwh_th"] = np.minimum(
            available_cooling_energy, joined["required_cooling_energy_kwh_th"]
        )
        joined["unmet_cooling_energy_kwh_th"] = (
            joined["required_cooling_energy_kwh_th"]
            - joined["delivered_cooling_energy_kwh_th"]
        )
        for output_field, source_field in weighted_metrics.items():
            joined[f"_{output_field}"] = area * joined[source_field]
        joined["population_weighted_overheating_person_hours"] = np.where(
            joined["component"].eq("residential"),
            joined["floor_resident_population"] * joined["heatwave_occupied_overheating_hours_p50"],
            0.0,
        )
        joined["residential_population_weighted_outage_degree_hours_above_32c"] = np.where(
            joined["component"].eq("residential"),
            joined["floor_resident_population"]
            * joined["heatwave_outage_window_degree_hours_above_32c_p50"],
            0.0,
        )
        joined["employee_visitor_weighted_overheating_person_hours"] = np.where(
            joined["component"].eq("commercial"),
            area * joined["people_per_m2"] * joined["heatwave_occupied_overheating_hours_p50"],
            0.0,
        )
        grouped = joined.groupby("BUILDINGSTRUCTUREID", sort=False, observed=True)
        extensive = grouped[[
            "baseline_energy_kwh", "heatwave_energy_kwh", "baseline_peak_kw",
            "heatwave_peak_kw", "balanced_design_energy_p05_kwh", "balanced_design_energy_p95_kwh",
            "recovery_energy_kwh", "population_weighted_overheating_person_hours",
            "residential_population_weighted_outage_degree_hours_above_32c",
            "employee_visitor_weighted_overheating_person_hours",
            "required_cooling_energy_kwh_th", "delivered_cooling_energy_kwh_th",
            "unmet_cooling_energy_kwh_th",
        ]].sum(min_count=1)
        total_component_area = grouped["floor_component_gfa_m2"].sum()
        for output_field in weighted_metrics:
            extensive[output_field] = grouped[f"_{output_field}"].sum(min_count=1) / total_component_area
        extensive["continuous_threshold_exceedance_hours"] = grouped[
            "heatwave_continuous_threshold_exceedance_hours_p50"
        ].max()
        extensive["time_to_thermal_threshold_hours"] = grouped[
            "heatwave_comfort_resistance_hours_to_26c_p50"
        ].min()
        extensive["safety_resistance_hours_to_32c"] = grouped[
            "heatwave_safety_resistance_hours_to_32c_p50"
        ].min()
        extensive["safety_resistance_hours_to_30c"] = grouped[
            "heatwave_safety_resistance_hours_to_30c_p50"
        ].min()
        extensive["safety_resistance_hours_to_34c"] = grouped[
            "heatwave_safety_resistance_hours_to_34c_p50"
        ].min()
        extensive["recovery_time_hours"] = grouped["heatwave_recovery_to_26c_hours_p50"].max()
        extensive["heatwave_days"] = grouped["heatwave_heatwave_days_p50"].max()
        critical_numerator = (
            joined["floor_component_gfa_m2"] * joined["heatwave_critical_service_continuity_hours_p50"]
        ).groupby(joined["BUILDINGSTRUCTUREID"], sort=False).sum(min_count=1)
        critical_area = joined["floor_component_gfa_m2"].where(
            joined["heatwave_critical_service_continuity_hours_p50"].notna()
        ).groupby(joined["BUILDINGSTRUCTUREID"], sort=False).sum(min_count=1)
        extensive["critical_service_continuity_hours"] = critical_numerator / critical_area
        extensive["component_count"] = grouped["component"].nunique()
        extensive = static.join(extensive, how="inner").reset_index()
        first_prototype = joined.drop_duplicates("BUILDINGSTRUCTUREID").set_index("BUILDINGSTRUCTUREID")["prototype_id"]
        extensive["prototype_id"] = np.where(
            extensive["component_count"].eq(1),
            extensive["BUILDINGSTRUCTUREID"].map(first_prototype),
            "mixed_residential_commercial",
        )
        extensive["event_id"] = str(event_id)
        extensive["climate_period"] = "observed_2018_2023"
        extensive["climate_scenario"] = "observed"
        extensive["supply_stage"] = str(supply_stage)
        extensive["behavior_seed"] = int(seed)
        extensive["calibration_dataset"] = np.select(
            [
                extensive["main_class"].eq("Commercial"),
                extensive["main_class"].eq("Residential"),
            ],
            [
                "EMSD_2023_plus_PolyU_and_HKUST_Hong_Kong",
                "EMSD_2023_plus_HKUST_campus_residence_proxy",
            ],
            default="PolyU_plus_EMSD_layered_evidence",
        )
        extensive["validation_status"] = np.select(
            [
                extensive["main_class"].eq("Commercial"),
                extensive["main_class"].eq("Residential"),
            ],
            [
                "annual_energy_mass_balanced_and_local_load_dynamics_holdout_validated",
                "annual_energy_mass_balanced_local_residential_indoor_temperature_pending",
            ],
            default="componentwise_layered_validation",
        )
        all_rows.append(extensive)
    dynamic = pd.concat(all_rows, ignore_index=True)
    if dynamic.duplicated([
        "BUILDINGSTRUCTUREID", "event_id", "climate_period", "climate_scenario",
        "supply_stage", "behavior_seed",
    ]).any():
        raise ValueError("Building dynamic result grain is not unique")
    if wrf_mapping is not None:
        # Hazard magnitudes remain in the canonical one-row-per-building hazard
        # table. Repeating all 22 hazard fields for every event and supply stage
        # would multiply static data about fifteen-fold and can exhaust memory.
        # The two grid indices are a lossless key back to that table.
        mapping_columns = [
            "BUILDINGSTRUCTUREID", "wrf_south_north", "wrf_west_east",
        ]
        missing_mapping_columns = set(mapping_columns).difference(wrf_mapping.columns)
        if missing_mapping_columns:
            raise ValueError(f"WRF building mapping lacks {sorted(missing_mapping_columns)}")
        mapping = wrf_mapping[mapping_columns].drop_duplicates("BUILDINGSTRUCTUREID")
        mapping = mapping.copy()
        mapping["BUILDINGSTRUCTUREID"] = pd.to_numeric(
            mapping["BUILDINGSTRUCTUREID"], errors="raise"
        ).astype(dynamic["BUILDINGSTRUCTUREID"].dtype)
        dynamic = dynamic.merge(mapping, on="BUILDINGSTRUCTUREID", how="left", validate="many_to_one")
    return dynamic


def build_citywide_coincident_load(
    inputs: gpd.GeoDataFrame,
    hourly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate prototype median trajectories before taking the city peak."""
    stock = inputs.loc[inputs["pre_weather_simulation_candidate"]]
    areas = pd.concat(
        [
            stock.loc[
                stock["residential_component_present"],
                ["residential_prototype_id", "residential_gfa_m2", "floors_value"],
            ]
            .rename(columns={"residential_prototype_id": "prototype_id", "residential_gfa_m2": "component_gfa_m2"}),
            stock.loc[
                stock["commercial_component_present"],
                ["commercial_prototype_id", "commercial_gfa_m2", "floors_value"],
            ]
            .rename(columns={"commercial_prototype_id": "prototype_id", "commercial_gfa_m2": "component_gfa_m2"}),
        ],
        ignore_index=True,
    )
    areas["component_gfa_m2"] = numeric(areas["component_gfa_m2"], lower=0).fillna(0.0)
    role_weights = floor_role_weight_table(areas["floors_value"])
    floor_areas = []
    for floor_position in FLOOR_ROLES:
        selected = role_weights[floor_position].gt(0)
        part = areas.loc[selected, ["prototype_id", "component_gfa_m2"]].copy()
        part["floor_position"] = floor_position
        part["component_gfa_m2"] *= role_weights.loc[selected, floor_position].to_numpy()
        floor_areas.append(part)
    areas = pd.concat(floor_areas, ignore_index=True).groupby(
        ["prototype_id", "floor_position"], observed=True
    )["component_gfa_m2"].sum().reset_index()
    profile_keys = [
        "event_id", "prototype_id", "floor_position", "climate_case", "supply_stage",
        "timestamp_hkt", "event_phase",
    ]
    prototype_hourly = hourly.groupby(profile_keys, observed=True)[
        "total_electricity_kwh_m2"
    ].median().rename("prototype_median_electricity_kwh_m2").reset_index()
    prototype_hourly = prototype_hourly.merge(
        areas,
        on=["prototype_id", "floor_position"],
        how="inner",
        validate="many_to_one",
    )
    prototype_hourly["city_electricity_kwh"] = (
        prototype_hourly["prototype_median_electricity_kwh_m2"]
        * prototype_hourly["component_gfa_m2"]
    )
    city_keys = ["event_id", "climate_case", "supply_stage", "timestamp_hkt", "event_phase"]
    city_hourly = prototype_hourly.groupby(city_keys, observed=True)["city_electricity_kwh"].sum().reset_index()
    city_hourly["city_average_load_kw"] = city_hourly["city_electricity_kwh"]

    baseline = city_hourly.loc[
        city_hourly["climate_case"].eq("counterfactual")
        & city_hourly["supply_stage"].eq("normal")
        & city_hourly["event_phase"].eq("heatwave")
    ].groupby("event_id", observed=True).agg(
        baseline_coincident_peak_kw=("city_average_load_kw", "max"),
        baseline_event_energy_kwh=("city_electricity_kwh", "sum"),
    ).reset_index()
    heatwave = city_hourly.loc[
        city_hourly["climate_case"].eq("heatwave")
        & city_hourly["event_phase"].eq("heatwave")
    ].groupby(["event_id", "supply_stage"], observed=True).agg(
        heatwave_coincident_peak_kw=("city_average_load_kw", "max"),
        heatwave_event_energy_kwh=("city_electricity_kwh", "sum"),
    ).reset_index()
    summary = heatwave.merge(baseline, on="event_id", how="left", validate="many_to_one")
    summary["coincident_peak_increment_kw"] = (
        summary["heatwave_coincident_peak_kw"] - summary["baseline_coincident_peak_kw"]
    )
    summary["coincident_peak_amplification_ratio"] = (
        summary["heatwave_coincident_peak_kw"]
        / summary["baseline_coincident_peak_kw"].replace(0.0, np.nan)
    )
    summary["aggregation_note"] = "prototype hourly medians multiplied by modeled component GFA, summed by timestamp before peak selection"
    return city_hourly, summary


def citywide_paired_trajectory_uncertainty(
    inputs: gpd.GeoDataFrame,
    hourly: pd.DataFrame,
    seed: int,
    draws: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recalculate event energy and coincident peaks within paired full-trajectory draws."""
    stock = inputs.loc[inputs["pre_weather_simulation_candidate"]]
    areas = pd.concat(
        [
            stock.loc[stock["residential_component_present"], [
                "residential_prototype_id", "residential_gfa_m2", "floors_value",
            ]].rename(columns={
                "residential_prototype_id": "prototype_id",
                "residential_gfa_m2": "component_gfa_m2",
            }),
            stock.loc[stock["commercial_component_present"], [
                "commercial_prototype_id", "commercial_gfa_m2", "floors_value",
            ]].rename(columns={
                "commercial_prototype_id": "prototype_id",
                "commercial_gfa_m2": "component_gfa_m2",
            }),
        ],
        ignore_index=True,
    )
    areas["component_gfa_m2"] = numeric(areas["component_gfa_m2"], lower=0).fillna(0.0)
    role_weights = floor_role_weight_table(areas["floors_value"])
    floor_areas = []
    for floor_position in FLOOR_ROLES:
        selected = role_weights[floor_position].gt(0)
        part = areas.loc[selected, ["prototype_id", "component_gfa_m2"]].copy()
        part["floor_position"] = floor_position
        part["component_gfa_m2"] *= role_weights.loc[selected, floor_position].to_numpy()
        floor_areas.append(part)
    weights = pd.concat(floor_areas, ignore_index=True).groupby(
        ["prototype_id", "floor_position"], observed=True
    )["component_gfa_m2"].sum()

    normal = hourly.loc[
        hourly["supply_stage"].eq("normal")
        & hourly["event_phase"].eq("heatwave")
        & hourly["climate_case"].isin(["counterfactual", "heatwave"]),
        [
            "event_id", "prototype_id", "floor_position", "base_case_id",
            "climate_case", "timestamp_hkt", "total_electricity_kwh_m2",
        ],
    ].copy()
    normal["timestamp_hkt"] = pd.to_datetime(normal["timestamp_hkt"], utc=True)
    rows = []
    events = sorted(normal["event_id"].astype(str).unique())
    for event_number, event_id in enumerate(events):
        event = normal.loc[normal["event_id"].astype(str).eq(event_id)].copy()
        timestamps = np.sort(event["timestamp_hkt"].unique())
        baseline_draws = np.zeros((draws, len(timestamps)), dtype=float)
        heatwave_draws = np.zeros((draws, len(timestamps)), dtype=float)
        rng = np.random.default_rng(seed + 1009 * (event_number + 1))
        unit_count = 0
        for (prototype_id, floor_position), area_m2 in weights.items():
            unit = event.loc[
                event["prototype_id"].eq(prototype_id)
                & event["floor_position"].eq(floor_position)
            ]
            if unit.empty or area_m2 <= 0:
                continue
            trajectories = []
            for _, case in unit.groupby("base_case_id", observed=True):
                pivot = case.pivot(
                    index="timestamp_hkt", columns="climate_case",
                    values="total_electricity_kwh_m2",
                ).reindex(timestamps)
                if list(pivot.columns) != ["counterfactual", "heatwave"]:
                    pivot = pivot.reindex(columns=["counterfactual", "heatwave"])
                if pivot.isna().any().any():
                    continue
                trajectories.append(pivot.to_numpy(dtype=float))
            if not trajectories:
                raise ValueError(
                    f"No complete paired trajectories for {event_id}/{prototype_id}/{floor_position}"
                )
            trajectories = np.stack(trajectories)
            selected = rng.integers(0, len(trajectories), size=draws)
            baseline_draws += trajectories[selected, :, 0] * float(area_m2)
            heatwave_draws += trajectories[selected, :, 1] * float(area_m2)
            unit_count += 1
        if unit_count != len(weights):
            raise ValueError(
                f"Paired city ensemble covered {unit_count} of {len(weights)} prototype-floor units"
            )
        baseline_energy = baseline_draws.sum(axis=1)
        heatwave_energy = heatwave_draws.sum(axis=1)
        baseline_peak = baseline_draws.max(axis=1)
        heatwave_peak = heatwave_draws.max(axis=1)
        for draw in range(draws):
            rows.append({
                "event_id": event_id,
                "draw": draw,
                "prototype_floor_units": unit_count,
                "baseline_event_energy_kwh": float(baseline_energy[draw]),
                "heatwave_event_energy_kwh": float(heatwave_energy[draw]),
                "heatwave_incremental_energy_kwh": float(
                    heatwave_energy[draw] - baseline_energy[draw]
                ),
                "baseline_coincident_peak_kw": float(baseline_peak[draw]),
                "heatwave_coincident_peak_kw": float(heatwave_peak[draw]),
                "coincident_peak_increment_kw": float(
                    heatwave_peak[draw] - baseline_peak[draw]
                ),
                "coincident_peak_change_pct": float(
                    100.0 * (heatwave_peak[draw] / baseline_peak[draw] - 1.0)
                ),
            })
    ensemble = pd.DataFrame(rows)
    summary_rows = []
    metrics = [
        "heatwave_incremental_energy_kwh", "coincident_peak_increment_kw",
        "coincident_peak_change_pct",
    ]
    for event_id, group in ensemble.groupby("event_id", observed=True):
        record = {"event_id": str(event_id), "draws": int(len(group))}
        for field in metrics:
            q10, median, q90 = numeric(group[field]).quantile([0.10, 0.50, 0.90])
            record[f"{field}_p10"] = float(q10)
            record[f"{field}_median"] = float(median)
            record[f"{field}_p90"] = float(q90)
        summary_rows.append(record)
    summary = pd.DataFrame(summary_rows)
    return ensemble, summary


def build_compact_hourly_interface(hourly: pd.DataFrame) -> pd.DataFrame:
    """Store balanced-design response ranges without expanding every building-hour."""
    keys = [
        "event_id", "prototype_id", "sector", "floor_position", "climate_case", "supply_stage",
        "timestamp_hkt", "event_phase",
    ]
    fields = [
        "outdoor_temperature_c", "exposure_temperature_c", "maximum_zone_temperature_c",
        "indoor_rh_pct", "occupied_fraction", "active_cooling_fraction",
        "unmet_cooling_fraction", "cooling_kwh_m2", "cooling_thermal_kwh_m2",
        "latent_cooling_kwh_m2",
        "lighting_kwh_m2", "equipment_kwh_m2", "total_electricity_kwh_m2",
        "total_electric_w_m2",
    ]
    missing = sorted(set(keys + fields) - set(hourly.columns))
    if missing:
        raise ValueError(f"Hourly result lacks compact-interface fields: {missing}")
    grouped = hourly.groupby(keys, observed=True)[fields]
    median = grouped.median().add_suffix("_p50")
    lower = grouped.quantile(0.05).add_suffix("_p05")
    upper = grouped.quantile(0.95).add_suffix("_p95")
    compact = pd.concat([median, lower, upper], axis=1).reset_index()
    for suffix in ["p05", "p50", "p95"]:
        total = compact[f"total_electricity_kwh_m2_{suffix}"].replace(0.0, np.nan)
        cooling = compact[f"cooling_kwh_m2_{suffix}"]
        compact[f"air_conditioning_share_{suffix}"] = cooling / total
        compact[f"lighting_share_{suffix}"] = compact[f"lighting_kwh_m2_{suffix}"] / total
        compact[f"equipment_share_{suffix}"] = compact[f"equipment_kwh_m2_{suffix}"] / total
        compact[f"latent_share_within_air_conditioning_{suffix}"] = (
            compact[f"latent_cooling_kwh_m2_{suffix}"] / cooling.replace(0.0, np.nan)
        )
    if compact.duplicated(keys).any():
        raise ValueError("Compact hourly result is not unique at prototype-hour grain")
    return compact


def build_citywide_hourly_end_use(
    building_inputs: pd.DataFrame,
    annual_hourly: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Aggregate calibrated prototype end uses using modeled component GFA."""
    residential = building_inputs.groupby("residential_prototype_id", observed=True)["residential_gfa_m2"].sum()
    commercial = building_inputs.groupby("commercial_prototype_id", observed=True)["commercial_gfa_m2"].sum()
    weights = pd.concat([residential, commercial]).groupby(level=0).sum().rename("component_gfa_m2")
    fields = [
        "cooling_kwh_m2", "latent_cooling_kwh_m2", "lighting_kwh_m2",
        "equipment_kwh_m2", "total_electricity_kwh_m2",
    ]
    profile = annual_hourly[["timestamp_hkt", "prototype_id", *fields]].copy()
    profile = profile.merge(weights, left_on="prototype_id", right_index=True, how="left", validate="many_to_one")
    for field in fields:
        profile[field.replace("_m2", "")] = profile[field] * profile["component_gfa_m2"]
    energy_fields = [field.replace("_m2", "") for field in fields]
    city = profile.groupby("timestamp_hkt", as_index=False, observed=True)[energy_fields].sum()
    total = city["total_electricity_kwh"].replace(0.0, np.nan)
    city["air_conditioning_share"] = city["cooling_kwh"] / total
    city["lighting_share"] = city["lighting_kwh"] / total
    city["equipment_share"] = city["equipment_kwh"] / total
    city["latent_share_within_air_conditioning"] = city["latent_cooling_kwh"] / city["cooling_kwh"].replace(0.0, np.nan)
    balance = city["cooling_kwh"] + city["lighting_kwh"] + city["equipment_kwh"]
    relative_error = (balance - city["total_electricity_kwh"]).abs() / total
    validation = {
        "hours": int(len(city)),
        "end_uses": ["air_conditioning_including_latent", "lighting", "equipment_and_other"],
        "latent_is_subset_of_air_conditioning": True,
        "maximum_hourly_end_use_balance_relative_error": float(relative_error.max()),
        "polyu_end_use_labels_used": False,
        "end_use_calibration_source": "EMSD 2023 cooling and non-cooling sector totals",
    }
    return city, validation


def write_normalized_building_hourly_interface(
    building_inputs: pd.DataFrame,
    wrf_mapping: pd.DataFrame | None,
    output: Path,
) -> dict:
    """Freeze an exact factorized interface for building-hour queries.

    The interface stores one static component row per building and one dynamic
    row per prototype-hour. This preserves hourly resolution while avoiding a
    dense 330,000-building Cartesian expansion.
    """
    fields = [
        "BUILDINGSTRUCTUREID", "main_class", "sub_class", "gfa_value_m2",
        "residential_gfa_m2", "commercial_gfa_m2", "floors_value", "residential_prototype_id",
        "commercial_prototype_id", "pre_weather_simulation_candidate",
    ]
    mapping = building_inputs[fields].copy()
    mapping["BUILDINGSTRUCTUREID"] = pd.to_numeric(mapping["BUILDINGSTRUCTUREID"], errors="raise").astype("int64")
    if wrf_mapping is not None:
        weather = wrf_mapping[["BUILDINGSTRUCTUREID", "wrf_south_north", "wrf_west_east"]].copy()
        weather["BUILDINGSTRUCTUREID"] = pd.to_numeric(weather["BUILDINGSTRUCTUREID"], errors="raise").astype("int64")
        weather = weather.drop_duplicates("BUILDINGSTRUCTUREID")
        mapping = mapping.merge(weather, on="BUILDINGSTRUCTUREID", how="left", validate="one_to_one")
    mapping = mapping.loc[mapping["pre_weather_simulation_candidate"]].copy()
    if mapping["BUILDINGSTRUCTUREID"].duplicated().any():
        raise ValueError("Compact building-hour component map has duplicate buildings")
    mapping.to_parquet(output / "building_hourly_component_map.parquet", index=False)
    manifest = {
        "storage_design": "normalized_factorized_building_component_by_prototype_hour",
        "building_rows": int(len(mapping)),
        "building_key": "BUILDINGSTRUCTUREID",
        "current_event_profile": "prototype_event_hourly_interface.parquet",
        "annual_profile": "annual_2023_prototype_hourly_results.parquet",
        "future_profile": "future_energyplus_hourly_profiles.parquet",
        "join_fields": {
            "residential": "residential_prototype_id -> prototype_id",
            "commercial": "commercial_prototype_id -> prototype_id",
            "floor_composition": "floors_value -> single/bottom/middle/top equal-floor-area weights",
            "time": "timestamp_hkt",
        },
        "floor_formula": "1 storey=single; 2 storeys=1/2 bottom+1/2 top; n>=3=1/n bottom+(n-2)/n middle+1/n top",
        "mixed_use_formula": "floor-conditioned residential and commercial profiles are weighted by component GFA",
        "dense_city_year_materialized": False,
        "query_result": "hourly building total, air-conditioning, lighting and equipment values at requested timestamps; latent cooling is a reported subset of air-conditioning; p05/p50/p95 fields summarize the deterministic balanced-design scenario ensemble",
    }
    (output / "building_hourly_interface_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def future_weather_node_design() -> pd.DataFrame:
    """Build three continuous-spatial interpolation nodes for each GCM case."""
    for path in [FUTURE_TMAX_FIELD, FUTURE_TMIN_FIELD, FUTURE_COMPOUND_FACTORS]:
        if not path.exists():
            raise FileNotFoundError(f"Future climate input missing: {path}")
    tmax = pd.read_parquet(FUTURE_TMAX_FIELD)
    tmin = pd.read_parquet(FUTURE_TMIN_FIELD)
    compound = pd.read_csv(FUTURE_COMPOUND_FACTORS)
    keys = ["source_id", "experiment_id", "gcm_future_period"]
    rows = []
    quantiles = [("spatial_p10", 0.10), ("spatial_p50", 0.50), ("spatial_p90", 0.90)]
    for case_keys, maximums in tmax.groupby(keys, observed=True):
        case = dict(zip(keys, case_keys))
        minimums = tmin.copy()
        for field, value in case.items():
            minimums = minimums.loc[minimums[field].eq(value)]
        factors = compound.loc[
            compound["source_id"].eq(case["source_id"])
            & compound["experiment_id"].eq(case["experiment_id"])
            & compound["future_period"].eq(case["gcm_future_period"])
        ]
        if minimums.empty or len(factors) != 1:
            raise ValueError(f"Future climate components do not align for {case}")
        factor = factors.iloc[0]
        for node, quantile in quantiles:
            rows.append(
                {
                    **case,
                    "spatial_node": node,
                    "spatial_quantile": quantile,
                    "delta_tmax_c": float(maximums["delta_p95_tmax_c"].quantile(quantile)),
                    "delta_tmin_c": float(minimums["delta_p95_tmin_c"].quantile(quantile)),
                    "huss_ratio": float(factor["huss_ratio"]),
                    "pressure_ratio": float(factor["psl_ratio"]),
                    "rsds_ratio": float(factor["rsds_ratio"]),
                    "wind_ratio": float(factor["sfcWind_ratio"]),
                }
            )
    design = pd.DataFrame(rows)
    expected = 5 * 2 * 2 * 3
    if len(design) != expected:
        raise ValueError(f"Expected {expected} future weather nodes; found {len(design)}")
    median_models = (
        design.loc[design["spatial_node"].eq("spatial_p50")]
        .assign(
            distance_to_case_median=lambda x: x.groupby(
                ["experiment_id", "gcm_future_period"], observed=True
            )["delta_tmax_c"].transform(lambda y: (y - y.median()).abs())
        )
        .sort_values(
            ["experiment_id", "gcm_future_period", "distance_to_case_median", "source_id"],
            kind="mergesort",
        )
        .drop_duplicates(["experiment_id", "gcm_future_period"])
        [["experiment_id", "gcm_future_period", "source_id"]]
        .rename(columns={"source_id": "resilience_reference_model"})
    )
    design = design.merge(
        median_models,
        on=["experiment_id", "gcm_future_period"],
        how="left",
        validate="many_to_one",
    )
    design["run_full_resilience_stages"] = design["source_id"].eq(
        design["resilience_reference_model"]
    )
    same_model_comparator = (
        design["source_id"].eq("BCC-CSM2-MR")
        & design["experiment_id"].eq("ssp585")
        & design["gcm_future_period"].eq("mid_century_2041_2060")
        & design["spatial_node"].eq("spatial_p50")
    )
    design.loc[same_model_comparator, "run_full_resilience_stages"] = True
    design["resilience_node_role"] = np.select(
        [
            design["source_id"].eq(design["resilience_reference_model"]),
            same_model_comparator,
        ],
        ["ensemble_median_reference", "same_model_comparator"],
        default="normal_supply_only",
    )
    return design


def run_future_energyplus_response(
    executable: Path,
    parameters: pd.DataFrame,
    weather_all: pd.DataFrame,
    calibration_factors: pd.DataFrame,
    output: Path,
    seed: int,
    max_events: int,
    smoke: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run future event analogues and retain prototype-hour trajectories."""
    output_tag = "_smoke" if smoke else ""
    available_events = set(weather_all["event_id"].astype(str).unique())
    reference_order = [
        str(event_id)
        for event_id in CONFIG["resilience"]["beri_reference_event_ids"]
        if str(event_id) in available_events
    ]
    event_ids = reference_order[:max_events]
    central = design_matrix(parameters, 2 * len(parameters), seed)
    central = central.loc[central["floor_position"].eq("middle")].copy()
    parameter_lookup = parameters.set_index("prototype_id", drop=False)
    nodes = future_weather_node_design()
    if smoke:
        central = central.loc[
            central["prototype_id"].isin(["res_public_rental", "com_office"])
        ].copy()
        nodes = nodes.head(1).copy()
        nodes["run_full_resilience_stages"] = True
    nodes.to_csv(output / f"future_weather_node_design{output_tag}.csv", index=False, encoding="utf-8-sig")
    future_weather_dir = output / "weather" / "future_analogues"
    future_weather_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output / "energyplus_run_cache" / "future_event_analogues_v1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    all_hourly = []
    all_summaries = []
    stage_names = [
        "normal", "capacity_100pct", "capacity_70pct",
        "outage_peak_6h", "outage_peak_24h", "outage_full_event",
    ]
    expected = 0
    for node in nodes.itertuples(index=False):
        expected += len(event_ids) * len(central) * (
            3 if smoke else (len(stage_names) if node.run_full_resilience_stages else 1)
        )
    completed = 0
    started = time.perf_counter()
    simulation_contract = simulation_contract_sha256()
    print(f"Future EnergyPlus analogues: {expected} resumable runs", flush=True)

    for node in nodes.itertuples(index=False):
        node_hourly = []
        node_summaries = []
        stages = (
            ["normal", "capacity_100pct", "outage_peak_6h"]
            if smoke
            else (stage_names if node.run_full_resilience_stages else ["normal"])
        )
        for event_id in event_ids:
            observed_event = validate_weather(weather_all.loc[weather_all["event_id"].eq(event_id)])
            future_weather = future_event_weather(
                observed_event,
                node.delta_tmax_c,
                node.delta_tmin_c,
                node.huss_ratio,
                node.rsds_ratio,
                node.wind_ratio,
                node.pressure_ratio,
            )
            weather_id = "_".join(
                [node.source_id, node.experiment_id, node.gcm_future_period, node.spatial_node, event_id]
            ).replace("/", "-")
            epw = future_weather_dir / f"{weather_id}.epw"
            write_epw(future_weather, epw)
            future_weather_sha256 = weather_frame_sha256(future_weather)
            for case in central.itertuples(index=False):
                prototype = parameter_lookup.loc[case.prototype_id]
                for supply_stage in stages:
                    behavior_seed = seed + sum(map(ord, f"{event_id}{case.base_case_id}"))
                    identifiers = {
                        "event_id": event_id,
                        "base_case_id": case.base_case_id,
                        "prototype_id": case.prototype_id,
                        "sector": case.sector,
                        "floor_position": case.floor_position,
                        "orientation_deg": int(case.orientation_deg),
                        "behaviour": case.behaviour,
                        "setpoint_c": float(case.setpoint_c),
                        "ac_use_probability": float(case.ac_use_probability),
                        "start_offset_h": int(case.start_offset_h),
                        "climate_case": "heatwave",
                        "supply_stage": supply_stage,
                        "behavior_seed": int(behavior_seed),
                    }
                    future_keys = {
                        "source_id": node.source_id,
                        "experiment_id": node.experiment_id,
                        "future_period": node.gcm_future_period,
                        "spatial_node": node.spatial_node,
                        "spatial_quantile": float(node.spatial_quantile),
                        "delta_tmax_c": float(node.delta_tmax_c),
                        "delta_tmin_c": float(node.delta_tmin_c),
                        "huss_ratio": float(node.huss_ratio),
                        "rsds_ratio": float(node.rsds_ratio),
                        "wind_ratio": float(node.wind_ratio),
                    }
                    cache_payload = {
                        "model_version": MODEL_VERSION,
                        "simulation_implementation_revision": SIMULATION_IMPLEMENTATION_REVISION,
                        "simulation_contract_sha256": simulation_contract,
                        "future_response_cache_revision": "diagnostics_v2",
                        "future_weather_version": "observed_event_morph_v3_solar_and_latent",
                        "weather_sha256": future_weather_sha256,
                        "identifiers": identifiers,
                        "future_keys": future_keys,
                        "prototype": prototype.to_dict(),
                    }
                    cache_contract_sha256 = payload_sha256(cache_payload)
                    cache_id = cache_contract_sha256[:20]
                    metadata_path = cache_dir / f"{cache_id}.json"
                    hourly_path = cache_dir / f"{cache_id}.parquet"
                    if metadata_path.exists() and hourly_path.exists():
                        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                        if payload.get("cache_contract_sha256") == cache_contract_sha256:
                            hourly = pd.read_parquet(hourly_path)
                            summary = payload["summary"]
                        else:
                            hourly = None
                            summary = None
                    else:
                        hourly = None
                        summary = None
                    if hourly is None:
                        schedules, outage_start, outage_end, capacity_fraction = behaviour_schedules(
                            future_weather,
                            prototype,
                            case.behaviour,
                            int(case.start_offset_h),
                            float(case.ac_use_probability),
                            supply_stage,
                            behavior_seed,
                        )
                        annual_schedules = annualize_schedules(schedules, future_weather)
                        idf, zone_areas, conditioned_zones = build_idf(
                            prototype,
                            int(case.orientation_deg),
                            case.floor_position,
                            float(case.setpoint_c),
                            annual_schedules,
                            free_running=False,
                            capacity_fraction=capacity_fraction,
                            cooling_capacity_limited=supply_stage != "normal",
                        )
                        idf = add_run_period(idf, future_weather)
                        hourly, warnings = run_energyplus(
                            executable,
                            idf,
                            epw,
                            annual_schedules,
                            schedules,
                            future_weather,
                            float(prototype["baseline_ac_cop"]),
                            free_running=False,
                            zone_areas=zone_areas,
                            conditioned_zones=conditioned_zones,
                        )
                        hourly = apply_energy_calibration(
                            hourly.assign(**identifiers, **future_keys), calibration_factors
                        )
                        summary = {
                            **identifiers,
                            **future_keys,
                            **summarize_response(hourly, outage_start, outage_end),
                            "severe_errors": int(warnings.count("** Severe  **")),
                            "warnings": int(warnings.count("** Warning **")),
                            "warning_messages": "\n".join(
                                line.strip()
                                for line in warnings.splitlines()
                                if "** Warning **" in line
                            ),
                            "severe_messages": "\n".join(
                                line.strip()
                                for line in warnings.splitlines()
                                if "** Severe  **" in line
                            ),
                        }
                        temporary = hourly_path.with_suffix(".parquet.part")
                        hourly.to_parquet(temporary, index=False)
                        temporary.replace(hourly_path)
                        metadata_path.write_text(
                            json.dumps(
                                {
                                    "cache_contract_sha256": cache_contract_sha256,
                                    "summary": summary,
                                },
                                ensure_ascii=False,
                                indent=2,
                                default=str,
                            ),
                            encoding="utf-8",
                        )
                    hourly = add_cooling_thermal_energy(
                        hourly, parameters.loc[parameters["prototype_id"].eq(case.prototype_id)]
                    )
                    node_hourly.append(hourly)
                    node_summaries.append(summary)
                    completed += 1
                    if completed <= 3 or completed % 50 == 0 or completed == expected:
                        elapsed = time.perf_counter() - started
                        rate = completed / elapsed if elapsed else 0.0
                        eta = (expected - completed) / rate / 60 if rate else math.nan
                        print(
                            f"[future EnergyPlus {completed:04d}/{expected}] "
                            f"elapsed={elapsed / 60:.1f} min, eta={eta:.1f} min",
                            flush=True,
                        )
        node_hourly_frame = pd.concat(node_hourly, ignore_index=True)
        node_summary_frame = pd.DataFrame(node_summaries)
        node_summary_frame = add_hourly_resilience_metrics(node_summary_frame, node_hourly_frame)
        all_hourly.append(node_hourly_frame)
        all_summaries.append(node_summary_frame)

    hourly_output = pd.concat(all_hourly, ignore_index=True)
    summary_output = pd.DataFrame.from_records(
        itertools.chain.from_iterable(frame.to_dict("records") for frame in all_summaries)
    )
    current = pd.read_csv(output / "energyplus_multizone_results.csv")
    baseline = current.loc[
        current["climate_case"].eq("counterfactual")
        & current["supply_stage"].eq("normal")
        & current["base_case_id"].isin(set(central["base_case_id"])),
        ["event_id", "base_case_id", "prototype_id", "total_electricity_kwh_m2", "peak_electric_w_m2"],
    ].rename(
        columns={
            "total_electricity_kwh_m2": "observed_counterfactual_energy_kwh_m2",
            "peak_electric_w_m2": "observed_counterfactual_peak_w_m2",
        }
    )
    summary_output = summary_output.merge(
        baseline,
        on=["event_id", "base_case_id", "prototype_id"],
        how="left",
        validate="many_to_one",
    )
    summary_output["future_incremental_energy_kwh_m2"] = (
        summary_output["total_electricity_kwh_m2"]
        - summary_output["observed_counterfactual_energy_kwh_m2"]
    )
    summary_output["future_peak_amplification_ratio"] = (
        summary_output["peak_electric_w_m2"]
        / summary_output["observed_counterfactual_peak_w_m2"].replace(0.0, np.nan)
    )
    diagnostics_audit = audit_energyplus_diagnostics(summary_output, pd.DataFrame())
    (output / f"future_energyplus_diagnostics_audit{output_tag}.json").write_text(
        json.dumps(diagnostics_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    finite_fields = [
        "total_electricity_kwh_m2",
        "peak_electric_w_m2",
        "maximum_indoor_temperature_c",
        "cooling_electricity_kwh_m2",
        "cooling_thermal_kwh_m2",
        "future_incremental_energy_kwh_m2",
        "future_peak_amplification_ratio",
    ]
    nonfinite_counts = {
        field: int((~np.isfinite(numeric(summary_output[field]))).sum())
        for field in finite_fields
    }
    validation_checks = {
        "all_expected_runs_present": len(summary_output) == expected,
        "zero_severe_errors": int(numeric(summary_output["severe_errors"]).sum()) == 0,
        "all_warnings_classified": bool(diagnostics_audit["pass"]),
        "finite_primary_outputs": not any(nonfinite_counts.values()),
    }
    hourly_output.to_parquet(output / f"future_energyplus_hourly_profiles{output_tag}.parquet", index=False)
    summary_output.to_csv(output / f"future_energyplus_results{output_tag}.csv", index=False, encoding="utf-8-sig")
    validation = {
        "runs": int(len(summary_output)),
        "hourly_rows": int(len(hourly_output)),
        "events": int(summary_output["event_id"].nunique()),
        "prototypes": int(summary_output["prototype_id"].nunique()),
        "gcm_models": int(summary_output["source_id"].nunique()),
        "scenarios": sorted(summary_output["experiment_id"].unique()),
        "periods": sorted(summary_output["future_period"].unique()),
        "spatial_nodes": sorted(summary_output["spatial_node"].unique()),
        "severe_errors": int(numeric(summary_output["severe_errors"]).sum()),
        "reported_warnings": int(numeric(summary_output["warnings"]).sum()),
        "diagnostics_audit_file": f"future_energyplus_diagnostics_audit{output_tag}.json",
        "nonfinite_counts": nonfinite_counts,
        "checks": validation_checks,
        "pass": bool(all(validation_checks.values())),
        "simulation_contract_sha256": simulation_contract,
        "weather_method": "HKO-constrained observed hourly event analogue morphed by five-GCM QDM magnitude, 1-km WRF spatial-change nodes and compound humidity-radiation-wind factors",
        "hourly_product_is_forecast": False,
    }
    (output / f"future_energyplus_validation{output_tag}.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not validation["pass"]:
        failures = [name for name, passed in validation_checks.items() if not passed]
        raise ValueError(f"Future EnergyPlus validation failed: {failures}")
    return summary_output, hourly_output


def design_matrix(parameters: pd.DataFrame, max_base_cases: int, seed: int) -> pd.DataFrame:
    if max_base_cases < 1:
        raise ValueError("max_base_cases must be positive")
    minimum = len(parameters)
    if max_base_cases < minimum:
        raise ValueError(
            f"max_base_cases must be at least {minimum} to cover every prototype"
        )
    grids = []
    continuous_critical = {"com_human_health", "com_data_centre", "com_accommodation"}
    for prototype in parameters.itertuples(index=False):
        if prototype.sector == "residential":
            behaviours = ["working", "older_home", "wfh", "frugal"]
            # The primary experiment conditions on an available AC system.
            # Household non-adoption/non-use must be coupled to a calibrated
            # window-opening and natural-ventilation model, so it is not
            # represented by randomly disabling cooling for an entire day.
            use_probabilities = [1.0]
            start_offsets = THERMAL_CONFIG["start_offset_scenarios_h"]
        elif prototype.prototype_id in continuous_critical:
            behaviours = ["continuous"]
            use_probabilities = [1.0]
            start_offsets = [0]
        else:
            behaviours = ["standard", "extended", "low_occupancy"]
            # Commercial HVAC availability is not a household adoption/use
            # probability. Operating uncertainty is represented by business
            # hours, occupancy profile, start offset and setpoint instead.
            use_probabilities = [1.0]
            start_offsets = THERMAL_CONFIG["start_offset_scenarios_h"]
        grids.extend(itertools.product([prototype.prototype_id], [prototype.sector], list(FLOOR_ROLES), [0, 90, 180, 270], behaviours, THERMAL_CONFIG["setpoint_scenarios_c"], use_probabilities, start_offsets))
    columns = ["prototype_id", "sector", "floor_position", "orientation_deg", "behaviour", "setpoint_c", "ac_use_probability", "start_offset_h"]
    candidates = pd.DataFrame(grids, columns=columns)
    rng = np.random.default_rng(seed)
    central = parameters[["prototype_id", "sector"]].copy()
    central = central.assign(floor_position="middle", orientation_deg=0, behaviour="standard", setpoint_c=float(THERMAL_CONFIG["central_setpoint_c"]), ac_use_probability=1.0, start_offset_h=0)
    central.loc[central["sector"].eq("residential"), "behaviour"] = "working"
    central.loc[central["prototype_id"].isin(continuous_critical), "behaviour"] = "continuous"
    remaining = max(max_base_cases - len(central), 0)
    prototype_ids = parameters["prototype_id"].tolist()
    per_prototype, extra = divmod(remaining, len(prototype_ids))
    selected_parts = []
    for position, prototype_id in enumerate(prototype_ids):
        take = per_prototype + int(position < extra)
        if take == 0:
            continue
        pool = candidates.loc[candidates["prototype_id"].eq(prototype_id)].copy()
        central_key = tuple(
            central.loc[central["prototype_id"].eq(prototype_id), columns].iloc[0]
        )
        pool_keys = pool[columns].apply(tuple, axis=1)
        pool = pool.loc[~pool_keys.map(lambda value: value == central_key)]
        if take > len(pool):
            raise ValueError(f"Insufficient legal design combinations for {prototype_id}")
        factor_columns = [
            "floor_position", "orientation_deg", "behaviour", "setpoint_c", "start_offset_h"
        ]
        covered_single = {
            (field, central_key[columns.index(field)]) for field in factor_columns
        }
        covered_pairs = {
            (left, central_key[columns.index(left)], right, central_key[columns.index(right)])
            for left_index, left in enumerate(factor_columns)
            for right in factor_columns[left_index + 1 :]
        }
        chosen = []
        available = pool.copy()
        for _ in range(take):
            scores = []
            for candidate in available.itertuples(index=True):
                values = {field: getattr(candidate, field) for field in factor_columns}
                new_single = sum((field, values[field]) not in covered_single for field in factor_columns)
                new_pairs = sum(
                    (left, values[left], right, values[right]) not in covered_pairs
                    for left_index, left in enumerate(factor_columns)
                    for right in factor_columns[left_index + 1 :]
                )
                scores.append((1000 * new_single + new_pairs, candidate.Index))
            best_score = max(score for score, _ in scores)
            tied = [index for score, index in scores if score == best_score]
            selected_index = int(rng.choice(tied))
            row = available.loc[selected_index]
            chosen.append(row)
            values = {field: row[field] for field in factor_columns}
            covered_single.update((field, values[field]) for field in factor_columns)
            covered_pairs.update(
                (left, values[left], right, values[right])
                for left_index, left in enumerate(factor_columns)
                for right in factor_columns[left_index + 1 :]
            )
            available = available.drop(index=selected_index)
        selected_parts.append(pd.DataFrame(chosen, columns=pool.columns))
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else candidates.iloc[0:0]
    design = pd.concat([central, selected], ignore_index=True).reset_index(drop=True)
    if len(design) != max_base_cases:
        raise ValueError(f"Balanced design has {len(design)} rows, expected {max_base_cases}")
    design["base_case_id"] = [f"B{i:04d}" for i in range(1, len(design) + 1)]
    design["behaviour_probability_status"] = "ac_system_available_primary_experiment; nonuse_requires_calibrated_window_opening_sensitivity"
    return design


def experiment_setting_provenance(design: pd.DataFrame, event_ids: list[str]) -> pd.DataFrame:
    """Freeze the role, value and evidence basis of every primary experiment setting."""
    records = [
        {
            "setting": "simulation_timestep",
            "value": f"{CONFIG['study_scope']['simulation_timestep_minutes']} min",
            "role": "EnergyPlus heat-balance and control timestep",
            "basis": "pre-specified numerical resolution",
            "source": "EnergyPlus 26.1 Input Output Reference",
        },
        {
            "setting": "weather_timestep",
            "value": f"{CONFIG['study_scope']['weather_input_timestep_minutes']} min",
            "role": "HKO-constrained event forcing",
            "basis": "native auditable hourly weather interface",
            "source": "HKO daily observations plus NASA POWER hourly fields",
        },
        {
            "setting": "historical_reference_events",
            "value": "|".join(map(str, event_ids)),
            "role": "frozen multi-event stress suite",
            "basis": "three highest-severity complete hourly events in the 2016-2025 event catalogue",
            "source": "output/heatwave_day3/weather_event_catalog.csv",
        },
        {
            "setting": "spatial_hazard_and_vulnerability_separation",
            "value": "common HKO-constrained event weather for stock stress test; independently validated 1-km hazard surface",
            "role": "common exposure isolates building-stock susceptibility while the 1-km layer quantifies spatial hazard",
            "basis": "pre-specified experimental contrast",
            "source": "01_hazard/wrf_distributional_spatial_bias_validation.json",
        },
        {
            "setting": "paired_counterfactual",
            "value": "same timestamp, weekday, behaviour seed, occupancy and internal-load schedules; temperature and humidity replaced",
            "role": "identify heatwave-attributable energy and peak changes",
            "basis": "weather-only paired contrast",
            "source": "paired_counterfactual_contract_audit.json",
        },
        {
            "setting": "cooling_setpoint_service_envelope",
            "value": "/".join(f"{value:g} C" for value in THERMAL_CONFIG["setpoint_scenarios_c"]),
            "role": "summer indoor service-level sensitivity",
            "basis": "Hong Kong summer comfort range",
            "source": "https://www.iaq.gov.hk/wp-content/uploads/2021/04/gn_officeandpublicplace_eng-2019.pdf",
        },
        {
            "setting": "residential_occupancy_and_room_cooling_profiles",
            "value": "working/older-home/work-from-home/frugal; weekday/weekend and living-room/bedroom schedules",
            "role": "represent observed Hong Kong occupancy and intermittent room-air-conditioner timing heterogeneity",
            "basis": "pre-specified behaviour scenarios with daytime presence retained for older-home and work-from-home households",
            "source": "https://doi.org/10.1016/j.enbuild.2019.109373|https://doi.org/10.3390/buildings12091414",
        },
        {
            "setting": "commercial_operation_profiles",
            "value": "use-specific standard/extended/low-occupancy schedules; continuous healthcare/data-centre/accommodation service",
            "role": "separate daytime, evening and continuous non-residential demand timing",
            "basis": "Hong Kong default building-use schedules and locally held-out campus load-shape evidence",
            "source": "https://www.emsd.gov.hk/filemanager/en/content_724/pb-bec_2007.pdf|PolyU full campus|HKUST full archive",
        },
        {
            "setting": "normal_supply_capacity",
            "value": "unlimited ideal-load delivery during scheduled cooling",
            "role": "hold indoor service constant when estimating heatwave-attributable electricity",
            "basis": "service-matched counterfactual definition",
            "source": "EnergyPlus ZoneHVAC:IdealLoadsAirSystem NoLimit cooling mode",
        },
        {
            "setting": "capacity_stress",
            "value": "70% of prototype rated cooling capacity",
            "role": "standardized cooling-service adequacy stress test",
            "basis": "pre-specified derating scenario",
            "source": "project protocol; sensitivity setting rather than an observed outage probability",
        },
        {
            "setting": "rated_capacity_outage_control",
            "value": "100% of prototype rated cooling capacity",
            "role": "matched no-outage trajectory for outage-attributable thermal and recovery effects",
            "basis": "identical pre-outage capacity, schedule and weather",
            "source": "outage_temporal_contract_audit.csv",
        },
        {
            "setting": "outage_stress",
            "value": "/".join(f"{value:g} h" for value in CONFIG["resilience"]["outage_stress_test_durations_hours"]),
            "role": "grid-loss consequence tests starting at the hottest occupied scheduled-cooling hour",
            "basis": "pre-specified stress durations",
            "source": "project protocol",
        },
        {
            "setting": "recovery_assessment",
            "value": f"{THERMAL_CONFIG['recovery_assessment_hours']:g} h after restoration",
            "role": "common recovery-time denominator and rebound window",
            "basis": "complete post-restoration window available for every reference event",
            "source": "event weather window audit",
        },
        {
            "setting": "thermal_safety_screen",
            "value": f"{THERMAL_CONFIG['outage_safety_screen_c']:g} C",
            "role": "hours-of-safety threshold for the 24-hour outage score",
            "basis": str(THERMAL_CONFIG["outage_safety_screen_role"]),
            "source": str(THERMAL_CONFIG["outage_safety_screen_source"]),
        },
        {
            "setting": "hours_of_safety_metric",
            "value": "time from outage onset to the 32 C screen, capped at 24 h",
            "role": "outage thermal-endurance component of BERI",
            "basis": "published hours-of-safety resilience metric",
            "source": str(THERMAL_CONFIG["hours_of_safety_metric_source"]),
        },
        {
            "setting": "passive_outage_adaptation",
            "value": "occupied residential zones attempt window opening subject to temperature-difference, outdoor-temperature and wind limits; commercial windows remain closed",
            "role": "represent non-electric household adaptation during grid loss",
            "basis": "type-specific operational prior",
            "source": str(THERMAL_CONFIG["natural_ventilation_source"]),
        },
        {
            "setting": "balanced_design",
            "value": f"{len(design)} cases; {design['prototype_id'].nunique()} prototypes; seed {RANDOM_SEED}",
            "role": "cover floor, orientation, operation, setpoint and start-time heterogeneity",
            "basis": "all within-prototype factor levels and all legal factor pairs covered",
            "source": "parameter_and_experiment_audit.json",
        },
    ]
    return pd.DataFrame(records)


def fixed_effect_heat_response(
    frame: pd.DataFrame,
    timestamp: str,
    outdoor: str,
    cooling: str,
    indoor: str,
    minimum_outdoor_c: float,
) -> dict | None:
    """Estimate weather response after removing weekday/weekend hourly schedules."""
    data = frame[[timestamp, outdoor, cooling, indoor]].copy()
    data[timestamp] = pd.to_datetime(data[timestamp], errors="coerce")
    for column in [outdoor, cooling, indoor]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna().loc[
        data[outdoor].between(minimum_outdoor_c, 60.0)
        & data[indoor].between(0.0, 50.0)
        & data[cooling].ge(0.0)
    ].copy()
    if not data.empty:
        data = data.loc[data[cooling].le(data[cooling].quantile(0.999))].copy()
    if len(data) < 96 or float(data[cooling].mean()) <= 0:
        return None
    data["schedule_slot"] = np.where(data[timestamp].dt.dayofweek.lt(5), 0, 1) * 24 + data[timestamp].dt.hour
    residuals = {}
    for column in [outdoor, cooling, indoor]:
        residuals[column] = data[column] - data.groupby("schedule_slot", observed=True)[column].transform("mean")
    x = residuals[outdoor].to_numpy(dtype=float)
    denominator = float(np.dot(x, x))
    if denominator <= 0:
        return None
    relative_cooling = residuals[cooling].to_numpy(dtype=float) / float(data[cooling].mean())
    indoor_residual = residuals[indoor].to_numpy(dtype=float)
    return {
        "hours": int(len(data)),
        "normalized_cooling_slope_per_c": float(np.dot(x, relative_cooling) / denominator),
        "indoor_temperature_attenuation": float(np.dot(x, indoor_residual) / denominator),
        "temperature_cooling_residual_r": float(np.corrcoef(x, relative_cooling)[0, 1]),
    }


def validate_external_dynamic_structure(hourly_results: pd.DataFrame, output: Path) -> dict:
    """Compare simulated response structure with independent hot-humid and residential records."""
    rows: list[dict] = []
    if CU_BEMS_ARCHIVE.exists() and CU_BEMS_WEATHER.exists():
        weather = pd.read_csv(CU_BEMS_WEATHER, skiprows=11)
        weather["timestamp"] = pd.to_datetime(
            dict(year=weather["YEAR"], month=weather["MO"], day=weather["DY"], hour=weather["HR"]),
            errors="coerce",
        )
        weather = weather[["timestamp", "T2M"]].rename(columns={"T2M": "outdoor_temperature_c"})
        with zipfile.ZipFile(CU_BEMS_ARCHIVE) as archive:
            for member in sorted(name for name in archive.namelist() if name.lower().endswith(".csv")):
                with archive.open(member) as stream:
                    header = pd.read_csv(stream, nrows=0).columns.tolist()
                ac_columns = [name for name in header if "AC" in name and name.endswith("(kW)")]
                temperature_columns = [name for name in header if name.endswith("(degC)")]
                if not ac_columns or not temperature_columns:
                    continue
                with archive.open(member) as stream:
                    raw = pd.read_csv(stream, usecols=["Date", *ac_columns, *temperature_columns])
                raw["timestamp"] = pd.to_datetime(raw.pop("Date"), errors="coerce")
                raw["cooling_power"] = raw[ac_columns].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
                raw["indoor_temperature_c"] = raw[temperature_columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                observed = (
                    raw.set_index("timestamp")[["cooling_power", "indoor_temperature_c"]]
                    .resample("h").mean().reset_index().merge(weather, on="timestamp", how="inner")
                )
                metrics = fixed_effect_heat_response(
                    observed, "timestamp", "outdoor_temperature_c", "cooling_power",
                    "indoor_temperature_c", 25.0,
                )
                if metrics:
                    rows.append({"dataset": "CU_BEMS", "group": Path(member).stem, "sector": "commercial", **metrics})

    for year in ["y1", "y2"]:
        paths = {
            "hvac": NIST_DIR / f"{year}_HVAC_hour.csv",
            "indoor": NIST_DIR / f"{year}_IndEnv_hour.csv",
            "outdoor": NIST_DIR / f"{year}_OutEnv_hour.csv",
        }
        if not all(path.exists() for path in paths.values()):
            continue
        hvac = pd.read_csv(paths["hvac"])
        indoor = pd.read_csv(paths["indoor"])
        outdoor = pd.read_csv(paths["outdoor"])
        power_columns = [name for name in hvac if name.endswith("Power")]
        temperature_columns = [name for name in indoor if "RoomTemp" in name]
        hvac["timestamp"] = pd.to_datetime(hvac.pop("Timestamp"), errors="coerce", utc=True).dt.tz_localize(None)
        indoor["timestamp"] = pd.to_datetime(indoor.pop("Timestamp"), errors="coerce", utc=True).dt.tz_localize(None)
        outdoor["timestamp"] = pd.to_datetime(outdoor.pop("Timestamp"), errors="coerce", utc=True).dt.tz_localize(None)
        hvac["cooling_power"] = hvac[power_columns].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
        indoor["indoor_temperature_c"] = indoor[temperature_columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        observed = (
            hvac[["timestamp", "cooling_power"]]
            .merge(indoor[["timestamp", "indoor_temperature_c"]], on="timestamp", how="inner")
            .merge(
                outdoor[["timestamp", "OutEnv_OutdoorAmbTemp"]].rename(
                    columns={"OutEnv_OutdoorAmbTemp": "outdoor_temperature_c"}
                ),
                on="timestamp", how="inner",
            )
        )
        metrics = fixed_effect_heat_response(
            observed, "timestamp", "outdoor_temperature_c", "cooling_power",
            "indoor_temperature_c", 20.0,
        )
        if metrics:
            rows.append({"dataset": "NIST_NZERTF", "group": year, "sector": "residential", **metrics})

    model = hourly_results.loc[
        hourly_results["climate_case"].eq("heatwave")
        & hourly_results["supply_stage"].eq("normal")
        & hourly_results["event_phase"].eq("heatwave")
    ].copy()
    for (sector, base_case_id), group in model.groupby(
        ["sector", "base_case_id"], observed=True
    ):
        metrics = fixed_effect_heat_response(
            group, "timestamp_hkt", "outdoor_temperature_c", "cooling_electric_w_m2",
            "indoor_temperature_c", 25.0 if sector == "commercial" else 20.0,
        )
        if metrics:
            rows.append({
                "dataset": "EnergyPlus", "group": str(base_case_id),
                "sector": sector, **metrics,
            })
    detailed = pd.DataFrame(rows)
    detailed.to_csv(output / "external_dynamic_structure_validation.csv", index=False, encoding="utf-8-sig")
    summaries = {}
    comparisons = [("CU_BEMS", "commercial"), ("NIST_NZERTF", "residential")]
    for dataset, sector in comparisons:
        observed = detailed.loc[detailed["dataset"].eq(dataset)]
        simulated = detailed.loc[detailed["dataset"].eq("EnergyPlus") & detailed["sector"].eq(sector)]
        item = {"observed_groups": int(len(observed)), "simulated_groups": int(len(simulated)), "metrics": {}}
        for metric in ["normalized_cooling_slope_per_c", "indoor_temperature_attenuation"]:
            observed_values = observed[metric].replace([np.inf, -np.inf], np.nan).dropna()
            simulated_values = simulated[metric].replace([np.inf, -np.inf], np.nan).dropna()
            if observed_values.empty or simulated_values.empty:
                continue
            lower = float(observed_values.quantile(0.05)) if len(observed_values) >= 5 else float(observed_values.min())
            upper = float(observed_values.quantile(0.95)) if len(observed_values) >= 5 else float(observed_values.max())
            model_median = float(simulated_values.median())
            model_lower = float(simulated_values.quantile(0.05))
            model_upper = float(simulated_values.quantile(0.95))
            item["metrics"][metric] = {
                "observed_interval_lower": lower,
                "observed_median": float(observed_values.median()),
                "observed_interval_upper": upper,
                "simulated_p05": model_lower,
                "simulated_median": model_median,
                "simulated_p95": model_upper,
                "observed_and_simulated_intervals_overlap": bool(
                    max(lower, model_lower) <= min(upper, model_upper)
                ),
                "observed_and_simulated_same_sign": bool(np.sign(observed_values.median()) == np.sign(model_median)),
            }
        item["metrics_with_overlapping_intervals"] = int(sum(
            metric["observed_and_simulated_intervals_overlap"] for metric in item["metrics"].values()
        ))
        item["metrics_compared"] = int(len(item["metrics"]))
        summaries[dataset] = item
    payload = {
        "method": "weekday_weekend-hour fixed-effect residual response; independent datasets test dynamics, not Hong Kong absolute calibration",
        "comparisons": summaries,
        "smeter30_role": "audited but excluded from heat-response calibration because UK household total electricity is not a cooling-end-use measure",
    }
    (output / "external_dynamic_structure_validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def write_validation_evidence_matrix(
    population_validation: dict,
    polyu_validation: dict,
    hkust_validation: dict,
    official_energy_validation: dict,
    dynamic_validation: dict,
    output: Path,
) -> None:
    """Write one auditable table separating calibration, validation and gaps."""
    wrf_path = (
        ROOT / CONFIG["output_root"] / "01_hazard"
        / "wrf_distributional_spatial_bias_validation.json"
    )
    wrf = json.loads(wrf_path.read_text(encoding="utf-8")) if wrf_path.exists() else {}
    wrf_pass = bool(wrf.get("validation")) and all(
        item.get("all_three_distribution_metrics_improved", False)
        for item in wrf.get("validation", [])
    )
    polyu_building = polyu_validation["building_holdout_validation"]
    polyu_time = polyu_validation["final_28_day_time_holdout_validation"]
    monthly = official_energy_validation["monthly_domestic_shape_holdout"]
    bca = official_energy_validation["bca_hot_humid_plausibility"]
    local_sensitivity_path = output / "local_observed_vs_energyplus_heat_sensitivity.csv"
    local_sensitivity = (
        pd.read_csv(local_sensitivity_path)
        if local_sensitivity_path.exists()
        else pd.DataFrame()
    )
    local_comparison_pass = (
        local_sensitivity["comparison_pass"].astype(str).str.lower().eq("true")
        if not local_sensitivity.empty
        else pd.Series(dtype=bool)
    )
    external_path = output / "external_dynamic_structure_validation.json"
    external = json.loads(external_path.read_text(encoding="utf-8")) if external_path.exists() else {}
    external_rows = []
    for dataset, label, role in [
        ("CU_BEMS", "CU-BEMS hot-humid office", "commercial cooling and indoor-temperature response structure"),
        ("NIST_NZERTF", "NIST NZERTF", "residential HVAC and indoor-temperature response structure"),
    ]:
        item = external.get("comparisons", {}).get(dataset)
        if not item:
            continue
        compared = int(item["metrics_compared"])
        inside = int(item["metrics_with_overlapping_intervals"])
        same_sign = sum(
            int(metric["observed_and_simulated_same_sign"])
            for metric in item["metrics"].values()
        )
        external_rows.append([
            "cross-dataset dynamic structure", label, role,
            "quantified_structural_concordance",
            "observed and simulated response intervals overlap; same-sign metrics",
            f"{inside}/{compared} overlap; {same_sign}/{compared} same sign",
            "tests response structure only; geography and building type are not Hong Kong stock calibration",
        ])
    rows = [
        [
            "high-resolution meteorological hazard", "HKO stations plus WRF 1 km",
            "spatial and temporal holdout", "pass" if wrf_pass else "not_passed",
            "corrected tmax/tmin distribution MAE",
            "; ".join(
                f"{item['variable']}={item['corrected_quantile_mae_c']:.3f} C"
                for item in wrf.get("validation", [])
            ),
            "warm-season distributions, not day-to-day WRF chronology",
        ],
        [
            "building population", "C&SD SSUG plus major housing estates",
            "mass conservation and estate holdout", "partial",
            "estate relative MAE/RMSE",
            f"{population_validation['primary_major_housing_estate_relative_mae_pct']:.2f}%/"
            f"{population_validation['primary_major_housing_estate_relative_rmse_pct']:.2f}%",
            "building-level population is allocated, not directly observed",
        ],
        [
            "annual absolute energy", "EMSD 2023 end-use totals",
            "fitted calibration mass balance", "pass",
            "maximum absolute mass-balance error",
            f"{official_energy_validation['maximum_absolute_mass_balance_relative_error']:.3g}",
            "commercial subtypes are calibrated only at sector total",
        ],
        [
            "residential seasonal load shape", "C&SD monthly domestic electricity 2023",
            "independent monthly holdout", "quantified", "Pearson r; share CVRMSE",
            f"r={monthly['pearson_r']:.3f}; CVRMSE={monthly['share_cvrmse_pct']:.2f}%",
            "territory seasonal shape, not individual dwellings",
        ],
        [
            "local observed nonresidential benchmark", "PolyU full campus",
            "leakage-safe building and final-28-day prediction holdouts",
            "pass_reference_band" if (
                abs(polyu_building["nmbe_pct"]) <= 10
                and polyu_building["cvrmse_pct"] <= 30
                and abs(polyu_time["nmbe_pct"]) <= 10
                and polyu_time["cvrmse_pct"] <= 30
            ) else "outside_reference_band",
            "building/time NMBE and CVRMSE",
            f"building {polyu_building['nmbe_pct']:.2f}%/{polyu_building['cvrmse_pct']:.2f}%; "
            f"time {polyu_time['nmbe_pct']:.2f}%/{polyu_time['cvrmse_pct']:.2f}%",
            "establishes a reliable local whole-load response benchmark for the separate EnergyPlus concordance test",
        ],
        [
            "local observed mixed-campus benchmark", "HKUST complete archive",
            "leakage-safe held-out-building prediction",
            "pass_reference_band" if (
                abs(hkust_validation["held_out_nmbe_pct"]) <= 10
                and hkust_validation["held_out_cvrmse_pct"] <= 30
            ) else "outside_reference_band",
            "held-out NMBE and CVRMSE",
            f"{hkust_validation['held_out_nmbe_pct']:.2f}%/"
            f"{hkust_validation['held_out_cvrmse_pct']:.2f}%",
            "supplies independent residential-proxy and nonresidential heat-response distributions",
        ],
        *([
            [
                "local EnergyPlus heat-sensitivity concordance", "PolyU and HKUST",
                "held-out buildings; same daily relative-load estimator in observations and annual EnergyPlus outputs",
                "pass" if local_comparison_pass.all() else "not_passed",
                "held-out observed interval overlap and median coverage by model type range",
                f"{int(local_comparison_pass.sum())}/{len(local_sensitivity)} scopes",
                "temperature-load response is compared by matched campus scope; balance points are fixed without using holdout outcomes",
            ]
        ] if not local_sensitivity.empty else []),
        [
            "hot-humid commercial annual EUI", "Singapore BCA 2024",
            "cross-city plausibility", "partial", "prototype inside matched BCA p05-p95",
            f"{bca['inside_bca_p05_p95']}/{bca['mapped_prototypes']}",
            "not a Hong Kong calibration target",
        ],
        *external_rows,
        [
            "Hong Kong residential indoor thermal response", "none available locally",
            "required joint meter-temperature holdout", "missing",
            "electricity plus indoor temperature/RH", "not available",
            "absolute residential overheating and outage safety remain unvalidated locally",
        ],
        [
            "territory simultaneous hourly peak", "CLP/HK Electric public annual maxima",
            "upper-bound plausibility only", "partial",
            "normal peaks below sum of individual utility maxima",
            str(dynamic_validation[
                "normal_supply_heatwave_peaks_below_sum_of_utility_maxima"
            ]),
            "public simultaneous territory hourly demand was not found",
        ],
    ]
    pd.DataFrame(
        rows,
        columns=[
            "validation_layer", "dataset", "role", "status", "primary_metric",
            "value", "claim_boundary",
        ],
    ).to_csv(output / "validation_evidence_matrix.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--building-stock", type=Path, default=BUILDING_STOCK)
    parser.add_argument("--census-context", type=Path, default=CENSUS_CONTEXT)
    parser.add_argument("--archetype-assignments", type=Path, default=OLD_ASSIGNMENTS)
    parser.add_argument("--archetype-catalog", type=Path, default=OLD_CATALOG)
    parser.add_argument("--energyplus-exe", type=Path, default=Path(os.environ.get("ENERGYPLUS_EXE", ENGINE)))
    parser.add_argument("--max-base-cases", type=int, default=int(THERMAL_CONFIG["max_base_cases"]))
    parser.add_argument("--max-events", type=int, default=int(THERMAL_CONFIG["max_events"]))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--external-structure-only",
        action="store_true",
        help="Refresh CU-BEMS and NIST structural validation from the existing EnergyPlus hourly result.",
    )
    parser.add_argument(
        "--future-response-only",
        action="store_true",
        help="Run resumable future hourly EnergyPlus event analogues using frozen 01 climate changes and existing official calibration factors.",
    )
    parser.add_argument(
        "--prepare-inputs-only",
        action="store_true",
        help="Prepare the audited residential/commercial component table, population fields and prototype coverage without using heatwave weather.",
    )
    parser.add_argument(
        "--population-only",
        action="store_true",
        help="Backward-compatible alias for --prepare-inputs-only.",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.external_structure_only:
        hourly_path = output / "energyplus_multizone_hourly_results.parquet"
        validation_path = output / "result2_validation.json"
        if not hourly_path.exists() or not validation_path.exists():
            raise FileNotFoundError("Run the complete 02 pipeline before --external-structure-only")
        external = validate_external_dynamic_structure(pd.read_parquet(hourly_path), output)
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        validation["external_dynamic_structure"] = external
        write_validation_evidence_matrix(
            validation,
            validation["polyu_campus"],
            validation["hkust_whole_building"],
            validation["energy_calibration"],
            validation["building_dynamic_results"],
            output,
        )
        validation["validation_evidence_matrix"] = "validation_evidence_matrix.csv"
        validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"02 external dynamic structure validation complete: {output}")
        return

    context = gpd.read_parquet(args.census_context.resolve())
    stock_columns = [
        "BUILDINGSTRUCTUREID",
        "hkbfetd_main_class",
        "hkbfetd_sub_class",
        "hkbfetd_residential_value",
        "ZONE_LABEL",
        "floors_value",
        "height_value_m",
        "construction_year_value",
        "gfa_value_m2",
        "residential_gfa_estimate_m2",
        "nonresidential_gfa_estimate_m2",
        "footprint_area_m2",
        "perimeter_m",
        "geometry",
    ]
    stock = gpd.read_parquet(args.building_stock.resolve(), columns=stock_columns)
    assignments = pd.read_csv(args.archetype_assignments.resolve())
    old_catalog = pd.read_csv(args.archetype_catalog.resolve())
    groups = pd.read_parquet(BG_MEMBERSHIP)
    als = pd.read_parquet(ALS_BUILDINGS)
    hma_boundaries = gpd.read_parquet(HMA_BOUNDARIES)
    ssug_boundaries = gpd.read_parquet(SSUG_BOUNDARIES)
    exposure, catalog, population_validation, bg_linkage, bg_linkage_audit, mhe_evaluation = build_population_and_archetypes(
        context, assignments, old_catalog, groups, als, hma_boundaries, ssug_boundaries
    )
    exposure.to_parquet(output / "building_population_and_archetypes.parquet", index=False)
    catalog.to_csv(output / "archetype_catalog.csv", index=False, encoding="utf-8-sig")
    bg_linkage.to_parquet(output / "building_group_major_building_linkage.parquet", index=False)
    bg_linkage_audit.to_csv(output / "building_group_major_building_linkage_audit.csv", index=False, encoding="utf-8-sig")
    mhe_evaluation.to_csv(output / "major_housing_estate_population_validation.csv", index=False, encoding="utf-8-sig")
    (output / "population_validation.json").write_text(json.dumps(population_validation, ensure_ascii=False, indent=2), encoding="utf-8")
    all_sector_inputs, prototype_coverage = build_all_sector_model_inputs(stock, exposure)
    all_sector_inputs.to_parquet(output / "all_sector_building_model_inputs.parquet", index=False)
    prototype_coverage.to_csv(output / "prototype_coverage.csv", index=False, encoding="utf-8-sig")
    final_sample_audit = write_final_sample_audit(all_sector_inputs, output)
    polyu_validation = audit_polyu_campus_response(output, args.seed)
    hkust_validation = audit_hkust_hourly_response(output, args.seed)
    implementation_audit = pd.DataFrame(
        [
            {
                "model_element": "city_building_scope",
                "current_implementation": "330761 quality-controlled residential commercial and mixed-use buildings from the 341153-footprint inventory",
                "evidence_status": "complete",
                "prepared_now": True,
                "requires_future_weather": False,
                "requires_additional_data": False,
            },
            {
                "model_element": "population",
                "current_implementation": "SSUG-conserving residential-land-use and GFA allocation tested against 540 major housing estates",
                "evidence_status": "validated_area_allocation_not_direct_building_census",
                "prepared_now": True,
                "requires_future_weather": False,
                "requires_additional_data": False,
            },
            {
                "model_element": "commercial_presence",
                "current_implementation": "use-specific occupancy density and hourly schedules kept separate from residential population",
                "evidence_status": "schedule_based_service_exposure",
                "prepared_now": True,
                "requires_future_weather": False,
                "requires_additional_data": False,
            },
            {
                "model_element": "thermal_topology",
                "current_implementation": "three-zone residential and five-zone commercial core-perimeter prototypes",
                "evidence_status": "implemented_energyplus_multizone",
                "prepared_now": True,
                "requires_future_weather": False,
                "requires_additional_data": False,
            },
            {
                "model_element": "dynamic_behavior",
                "current_implementation": "use-specific occupancy AC equipment lighting ventilation latent-load and recovery schedules with fixed paired behavior profiles",
                "evidence_status": "official_shape_and_local_load_holdout_constrained",
                "prepared_now": True,
                "requires_future_weather": False,
                "requires_additional_data": False,
            },
            {
                "model_element": "extreme_event_counterfactual",
                "current_implementation": "paired baseline and heatwave with identical calendar position and behavior specification",
                "evidence_status": "complete",
                "prepared_now": True,
                "requires_future_weather": False,
                "requires_additional_data": False,
            },
            {
                "model_element": "dynamic_resilience",
                "current_implementation": "normal constrained outage and recovery phases with direct energy peak thermal and service metrics",
                "evidence_status": "complete_for_scenario_response",
                "prepared_now": True,
                "requires_future_weather": False,
                "requires_additional_data": False,
            },
        ]
    )
    implementation_audit.to_csv(output / "model_implementation_audit.csv", index=False, encoding="utf-8-sig")
    if args.population_only or args.prepare_inputs_only:
        print(
            "02 pre-weather inputs complete: "
            f"{len(all_sector_inputs):,} target-scope buildings; "
            f"{int(all_sector_inputs['pre_weather_simulation_candidate'].sum()):,} pre-weather candidates; "
            f"{output}"
        )
        return

    engine = validate_engine(args.energyplus_exe.resolve())
    weather_all = pd.read_csv(WEATHER)
    weather_all["timestamp_hkt"] = pd.to_datetime(weather_all["timestamp_hkt"], utc=True).dt.tz_convert("Asia/Hong_Kong")
    solar_decomposition_audit = audit_solar_decomposition(weather_all)
    (output / "solar_decomposition_audit.json").write_text(
        json.dumps(solar_decomposition_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    available_events = set(weather_all["event_id"].astype(str).unique())
    reference_order = [
        str(event_id)
        for event_id in CONFIG["resilience"]["beri_reference_event_ids"]
        if str(event_id) in available_events
    ]
    event_ids = reference_order[: args.max_events]
    if len(event_ids) != min(args.max_events, len(available_events)):
        raise ValueError("Event weather does not match the frozen reference-event registry")
    parameters = prototype_parameter_table()
    parameters.to_csv(output / "prototype_physics_parameters.csv", index=False, encoding="utf-8-sig")
    parameter_provenance = prototype_parameter_provenance(parameters)
    parameter_provenance.to_csv(
        output / "prototype_parameter_provenance.csv", index=False, encoding="utf-8-sig"
    )
    if args.future_response_only:
        factors_path = output / "official_energy_calibration_factors.csv"
        current_result_path = output / "energyplus_multizone_results.csv"
        for required in [factors_path, current_result_path]:
            if not required.exists():
                raise FileNotFoundError(
                    f"Future response requires frozen observed calibration output: {required}. "
                    "Run 02 once without --future-response-only first."
                )
        future_summary, future_hourly = run_future_energyplus_response(
            args.energyplus_exe.resolve(),
            parameters,
            weather_all,
            pd.read_csv(factors_path)[[
                "prototype_id", "calibration_group", "cooling_calibration_factor",
                "non_cooling_calibration_factor",
            ]],
            output,
            args.seed,
            1 if args.smoke else args.max_events,
            args.smoke,
        )
        print(
            f"02 future response complete: {len(future_summary):,} cases, "
            f"{len(future_hourly):,} prototype-hours; {output}",
            flush=True,
        )
        return
    if args.smoke:
        smoke_rows = []
        for prototype_id, sector, behaviour in [
            ("res_public_rental", "residential", "working"),
            ("com_office", "commercial", "standard"),
            ("com_data_centre", "commercial", "continuous"),
        ]:
            for floor_position in FLOOR_ROLES:
                smoke_rows.append(
                    {
                        "prototype_id": prototype_id,
                        "sector": sector,
                        "floor_position": floor_position,
                        "orientation_deg": 0,
                        "behaviour": behaviour,
                        "setpoint_c": 25.0,
                        "ac_use_probability": 1.0,
                        "start_offset_h": 0,
                        "base_case_id": f"B{len(smoke_rows) + 1:04d}",
                        "behaviour_probability_status": "fixed_smoke_test",
                    }
                )
        design = pd.DataFrame(smoke_rows)
        event_ids = event_ids[:1]
        heatwave_supply_stages = [
            "normal", "capacity_100pct", "capacity_70pct", "outage_peak_6h",
            "outage_peak_24h", "outage_common_24h", "outage_full_event",
        ]
    else:
        design = design_matrix(parameters, args.max_base_cases, args.seed)
        heatwave_supply_stages = [
            "normal", "capacity_100pct", "capacity_70pct",
            "outage_peak_6h", "outage_peak_24h", "outage_common_24h", "outage_full_event",
        ]
    design.to_csv(output / "thermal_experiment_design.csv", index=False, encoding="utf-8-sig")
    setting_provenance = experiment_setting_provenance(design, event_ids)
    setting_provenance.to_csv(
        output / "experiment_setting_provenance.csv", index=False, encoding="utf-8-sig"
    )

    parameter_lookup = parameters.set_index("prototype_id", drop=False)
    rows, diagnostics, hourly_frames, event_temperature_rows = [], [], [], []
    epw_dir = output / "weather"
    epw_dir.mkdir(exist_ok=True)
    run_cache_dir = output / "energyplus_run_cache" / MODEL_VERSION
    run_cache_dir.mkdir(parents=True, exist_ok=True)
    expected_run_count = len(event_ids) * len(design) * (1 + len(heatwave_supply_stages))
    completed_runs = 0
    run_started = time.perf_counter()
    print(
        f"EnergyPlus {MODEL_VERSION}: {expected_run_count} runs "
        f"({len(design)} cases, {len(event_ids)} events); resumable cache enabled",
        flush=True,
    )
    simulation_contract = simulation_contract_sha256()

    def report_progress() -> None:
        if completed_runs <= 3 or completed_runs % 25 == 0 or completed_runs == expected_run_count:
            elapsed = time.perf_counter() - run_started
            rate = completed_runs / elapsed if elapsed > 0 else 0.0
            remaining_minutes = (expected_run_count - completed_runs) / rate / 60 if rate > 0 else math.nan
            print(
                f"[EnergyPlus {completed_runs:04d}/{expected_run_count}] "
                f"elapsed={elapsed / 60:.1f} min, eta={remaining_minutes:.1f} min",
                flush=True,
            )
    for event_id in event_ids:
        event_weather = validate_weather(weather_all.loc[weather_all["event_id"].eq(event_id)])
        paired_weather = {case: climate_weather(event_weather, case) for case in ["counterfactual", "heatwave"]}
        paired_weather_sha256 = {
            case: weather_frame_sha256(case_weather)
            for case, case_weather in paired_weather.items()
        }
        heat_mask = event_weather["event_phase"].eq("heatwave").to_numpy()
        event_temperature_rows.append(
            {
                "event_id": event_id,
                "mean_heatwave_temperature_delta_c": float(
                    paired_weather["heatwave"].loc[heat_mask, "temperature_c"].mean()
                    - paired_weather["counterfactual"].loc[heat_mask, "temperature_c"].mean()
                ),
            }
        )
        epw_files = {}
        for climate_case, weather in paired_weather.items():
            epw_files[climate_case] = epw_dir / f"{event_id}_{climate_case}.epw"
            write_epw(weather, epw_files[climate_case])
        for case in design.itertuples(index=False):
            prototype = parameter_lookup.loc[case.prototype_id]
            for climate_case in ["counterfactual", "heatwave"]:
                supply_stages = ["normal"] if climate_case == "counterfactual" else heatwave_supply_stages
                active_weather = paired_weather[climate_case]
                for supply_stage in supply_stages:
                    behavior_seed = args.seed + sum(map(ord, f"{event_id}{case.base_case_id}"))
                    identifiers = {
                        "event_id": event_id,
                        "base_case_id": case.base_case_id,
                        "prototype_id": case.prototype_id,
                        "sector": case.sector,
                        "floor_position": case.floor_position,
                        "orientation_deg": int(case.orientation_deg),
                        "behaviour": case.behaviour,
                        "setpoint_c": float(case.setpoint_c),
                        "ac_use_probability": float(case.ac_use_probability),
                        "start_offset_h": int(case.start_offset_h),
                        "climate_case": climate_case,
                        "supply_stage": supply_stage,
                        "behavior_seed": int(behavior_seed),
                    }
                    cache_payload = {
                        "model_version": MODEL_VERSION,
                        "simulation_implementation_revision": SIMULATION_IMPLEMENTATION_REVISION,
                        "simulation_contract_sha256": simulation_contract,
                        "weather_sha256": paired_weather_sha256[climate_case],
                        "identifiers": identifiers,
                        "prototype": prototype.to_dict(),
                    }
                    cache_contract_sha256 = payload_sha256(cache_payload)
                    cache_id = cache_contract_sha256[:20]
                    cache_metadata = run_cache_dir / f"{cache_id}.json"
                    cache_hourly = run_cache_dir / f"{cache_id}.parquet"
                    if cache_metadata.exists() and cache_hourly.exists():
                        cached = json.loads(cache_metadata.read_text(encoding="utf-8"))
                        if cached.get("cache_contract_sha256") == cache_contract_sha256:
                            rows.append(cached["summary"])
                            diagnostics.append(cached["diagnostic"])
                            hourly_frames.append(pd.read_parquet(cache_hourly))
                            completed_runs += 1
                            report_progress()
                            continue
                    event_schedules, outage_start, outage_end, capacity_fraction = behaviour_schedules(
                        active_weather,
                        prototype,
                        case.behaviour,
                        int(case.start_offset_h),
                        float(case.ac_use_probability),
                        supply_stage,
                        behavior_seed,
                    )
                    annual_schedules = annualize_schedules(event_schedules, active_weather)
                    idf, zone_areas, conditioned_zones = build_idf(
                        prototype,
                        int(case.orientation_deg),
                        case.floor_position,
                        float(case.setpoint_c),
                        annual_schedules,
                        free_running=False,
                        capacity_fraction=capacity_fraction,
                        cooling_capacity_limited=supply_stage != "normal",
                    )
                    idf = add_run_period(idf, active_weather)
                    result, warnings = run_energyplus(
                        args.energyplus_exe.resolve(),
                        idf,
                        epw_files[climate_case],
                        annual_schedules,
                        event_schedules,
                        active_weather,
                        float(prototype["baseline_ac_cop"]),
                        free_running=False,
                        zone_areas=zone_areas,
                        conditioned_zones=conditioned_zones,
                    )
                    summary_row = {**identifiers, **summarize_response(result, outage_start, outage_end)}
                    warning_messages = "\n".join(
                        line.strip() for line in warnings.splitlines() if "** Warning **" in line
                    )
                    severe_messages = "\n".join(
                        line.strip() for line in warnings.splitlines() if "** Severe  **" in line
                    )
                    diagnostic_row = {
                        **identifiers,
                        "severe_errors": warnings.count("** Severe  **"),
                        "warnings": warnings.count("** Warning **"),
                        "warning_messages": warning_messages,
                        "severe_messages": severe_messages,
                    }
                    hourly_frame = result.assign(**identifiers)
                    summary_temporary = cache_metadata.with_suffix(".json.part")
                    hourly_temporary = cache_hourly.with_suffix(".parquet.part")
                    hourly_frame.to_parquet(hourly_temporary, index=False)
                    summary_temporary.write_text(
                        json.dumps(
                            {
                                "model_version": MODEL_VERSION,
                                "cache_contract_sha256": cache_contract_sha256,
                                "summary": summary_row,
                                "diagnostic": diagnostic_row,
                            },
                            ensure_ascii=False,
                            indent=2,
                            default=lambda value: value.item() if hasattr(value, "item") else str(value),
                        ),
                        encoding="utf-8",
                    )
                    hourly_temporary.replace(cache_hourly)
                    summary_temporary.replace(cache_metadata)
                    rows.append(summary_row)
                    hourly_frames.append(hourly_frame)
                    diagnostics.append(diagnostic_row)
                    completed_runs += 1
                    report_progress()
    results = pd.DataFrame(rows)
    diagnostics_frame = pd.DataFrame(diagnostics)
    hourly_results = pd.concat(hourly_frames, ignore_index=True)
    annual_models, annual_monthly, annual_hourly_raw = run_annual_reference_models(
        args.energyplus_exe.resolve(), parameters, output, args.seed
    )
    diagnostics_audit = audit_energyplus_diagnostics(diagnostics_frame, annual_models)
    (output / "energyplus_diagnostics_audit.json").write_text(
        json.dumps(diagnostics_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    calibration_factors, official_energy_validation = build_official_energy_calibration(
        all_sector_inputs, annual_models, annual_monthly, output
    )
    hourly_results = add_cooling_thermal_energy(hourly_results, parameters)
    annual_hourly_raw = add_cooling_thermal_energy(annual_hourly_raw, parameters)
    hourly_results = apply_energy_calibration(hourly_results, calibration_factors)
    annual_hourly = apply_energy_calibration(annual_hourly_raw, calibration_factors)
    paired_counterfactual_audit = audit_paired_counterfactual_contract(hourly_results)
    (output / "paired_counterfactual_contract_audit.json").write_text(
        json.dumps(paired_counterfactual_audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    annual_hourly.to_parquet(
        output / "annual_2023_prototype_hourly_results.parquet", index=False
    )
    annual_city_end_use, annual_end_use_validation = build_citywide_hourly_end_use(
        all_sector_inputs, annual_hourly
    )
    annual_city_end_use.to_parquet(
        output / "annual_2023_citywide_hourly_end_use.parquet", index=False
    )
    compact_hourly = build_compact_hourly_interface(hourly_results)
    compact_hourly.to_parquet(
        output / "prototype_event_hourly_interface.parquet", index=False
    )
    results = refresh_energy_summary(results, hourly_results)
    results = add_hourly_resilience_metrics(results, hourly_results)
    outage_timing_audit, outage_timing_summary = audit_outage_temporal_contract(hourly_results)
    outage_timing_audit.to_csv(
        output / "outage_timing_contract_audit.csv", index=False, encoding="utf-8-sig"
    )
    data_centre_audit, data_centre_audit_summary = data_centre_outage_physics_audit(
        hourly_results, parameters
    )
    data_centre_audit.to_csv(
        output / "data_centre_outage_physics_audit.csv", index=False, encoding="utf-8-sig"
    )
    selected_design_execution_complete = bool(
        len(results) == expected_run_count
        and len(diagnostics_frame) == expected_run_count
        and int(pd.to_numeric(diagnostics_frame["severe_errors"], errors="coerce").fillna(1).sum()) == 0
    )
    results.to_csv(output / "energyplus_multizone_results.csv", index=False, encoding="utf-8-sig")
    hourly_results.to_parquet(output / "energyplus_multizone_hourly_results.parquet", index=False)
    diagnostics_frame.to_csv(output / "energyplus_diagnostics.csv", index=False, encoding="utf-8-sig")
    external_structure_validation = validate_external_dynamic_structure(hourly_results, output)

    wrf_mapping = (
        pd.read_parquet(WRF_BUILDING_HAZARD)
        if WRF_BUILDING_HAZARD.exists()
        else (pd.read_parquet(WRF_BUILDING_MAPPING) if WRF_BUILDING_MAPPING.exists() else None)
    )
    hourly_interface_manifest = write_normalized_building_hourly_interface(
        all_sector_inputs, wrf_mapping, output
    )
    propagation_inputs = all_sector_inputs
    if args.smoke:
        selected_prototypes = set(design["prototype_id"].astype(str))
        propagation_inputs = all_sector_inputs.loc[
            all_sector_inputs["residential_prototype_id"].astype(str).isin(selected_prototypes)
            | all_sector_inputs["commercial_prototype_id"].astype(str).isin(selected_prototypes)
        ].copy()
        propagation_inputs["residential_component_present"] &= propagation_inputs[
            "residential_prototype_id"
        ].astype(str).isin(selected_prototypes)
        propagation_inputs["commercial_component_present"] &= propagation_inputs[
            "commercial_prototype_id"
        ].astype(str).isin(selected_prototypes)
        propagation_inputs.loc[
            ~propagation_inputs["residential_component_present"], "residential_gfa_m2"
        ] = 0.0
        propagation_inputs.loc[
            ~propagation_inputs["commercial_component_present"], "commercial_gfa_m2"
        ] = 0.0
        propagation_inputs["pre_weather_simulation_candidate"] = (
            propagation_inputs["residential_component_present"]
            | propagation_inputs["commercial_component_present"]
        )
    dynamic = build_building_dynamic_results(propagation_inputs, results, wrf_mapping, args.seed)
    physical_consistency_audit = audit_building_physics_consistency(dynamic, hourly_results)
    (output / "building_physics_consistency_audit.json").write_text(
        json.dumps(physical_consistency_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    dynamic.to_parquet(output / "building_dynamic_results.parquet", index=False)
    city_hourly, city_peak_summary = build_citywide_coincident_load(propagation_inputs, hourly_results)
    official_peak_upper_kw = float(
        pd.read_csv(OFFICIAL_SYSTEM_PEAK)["maximum_demand_mw"].sum() * 1000
    )
    city_peak_summary["sum_of_utility_individual_maxima_upper_bound_kw"] = official_peak_upper_kw
    city_peak_summary["below_official_system_upper_bound"] = (
        city_peak_summary["heatwave_coincident_peak_kw"] <= official_peak_upper_kw
    )
    city_peak_summary["peak_bound_interpretation"] = (
        "CLP plus HK Electric individual 2023 maxima; an upper bound, not a coincident territory target"
    )
    paired_city_ensemble, paired_city_ensemble_summary = citywide_paired_trajectory_uncertainty(
        propagation_inputs, hourly_results, args.seed
    )
    paired_city_ensemble.to_csv(
        output / "citywide_paired_trajectory_uncertainty.csv",
        index=False, encoding="utf-8-sig",
    )
    paired_city_ensemble_summary.to_csv(
        output / "citywide_paired_trajectory_uncertainty_summary.csv",
        index=False, encoding="utf-8-sig",
    )
    city_peak_summary = city_peak_summary.merge(
        paired_city_ensemble_summary,
        on="event_id", how="left", validate="many_to_one",
    )
    city_hourly.to_parquet(output / "citywide_coincident_hourly_response.parquet", index=False)
    city_peak_summary.to_csv(
        output / "citywide_coincident_peak_summary.csv", index=False, encoding="utf-8-sig"
    )
    dynamic_validation = {
        "rows": int(len(dynamic)),
        "buildings": int(dynamic["BUILDINGSTRUCTUREID"].nunique()),
        "events": int(dynamic["event_id"].nunique()),
        "supply_stages": sorted(dynamic["supply_stage"].unique()),
        "mixed_use_components_conserved_before_building_aggregation": True,
        "experiment_aggregation": "prototype_event_supply_stage_balanced_design_median",
        "balanced_design_range": "prototype balanced-design p05-p95 for heatwave event electricity; not a probabilistic confidence or prediction interval",
        "wrf_grid_mapping_available": bool(wrf_mapping is not None),
        "wrf_hazard_join_policy": (
            "dynamic table stores WRF row/column keys; static bias-corrected hazard values remain in "
            "01_hazard/building_heat_hazard_wrf1km.parquet"
        ),
        "citywide_peak_uses_hourly_coincidence": True,
        "official_energy_calibration": official_energy_validation,
        "normal_supply_heatwave_peaks_below_sum_of_utility_maxima": bool(
            city_peak_summary.loc[
                city_peak_summary["supply_stage"].eq("normal"),
                "below_official_system_upper_bound",
            ].all()
        ),
        "outage_recovery_cases_above_sum_of_utility_maxima": int(
            (
                city_peak_summary["supply_stage"].str.startswith("outage_peak")
                & ~city_peak_summary["below_official_system_upper_bound"]
            ).sum()
        ),
        "citywide_coincident_peak_file": "citywide_coincident_peak_summary.csv",
        "building_hourly_interface": hourly_interface_manifest,
        "annual_hourly_end_use": annual_end_use_validation,
        "model_version": MODEL_VERSION,
        "random_seed": int(args.seed),
        "outage_timing_contract": outage_timing_summary,
        "data_centre_outage_physics": data_centre_audit_summary,
        "paired_city_trajectory_uncertainty": {
            "draws_per_event": 500,
            "sampling_unit": "complete paired trajectory within prototype-floor unit",
            "summary_file": "citywide_paired_trajectory_uncertainty_summary.csv",
        },
        "building_physics_consistency": physical_consistency_audit,
        "solar_decomposition": solar_decomposition_audit,
        "energyplus_diagnostics": diagnostics_audit,
        "paired_counterfactual_contract": paired_counterfactual_audit,
    }
    (output / "building_dynamic_results_validation.json").write_text(
        json.dumps(dynamic_validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    normal = results.loc[results["supply_stage"].eq("normal")]
    paired = normal.pivot(index=["event_id", "base_case_id", "prototype_id", "sector"], columns="climate_case", values=["total_electricity_kwh_m2", "peak_electric_w_m2", "all_hour_degree_hours_above_26c"]).reset_index()
    paired.columns = ["_".join(part for part in column if part).rstrip("_") if isinstance(column, tuple) else column for column in paired.columns]
    paired["heatwave_incremental_electricity_kwh_m2"] = paired["total_electricity_kwh_m2_heatwave"] - paired["total_electricity_kwh_m2_counterfactual"]
    paired["heatwave_relative_electricity_increase"] = paired["heatwave_incremental_electricity_kwh_m2"] / paired["total_electricity_kwh_m2_counterfactual"].replace(0.0, np.nan)
    paired["heatwave_incremental_peak_w_m2"] = paired["peak_electric_w_m2_heatwave"] - paired["peak_electric_w_m2_counterfactual"]
    paired["heatwave_peak_amplification_ratio"] = paired["peak_electric_w_m2_heatwave"] / paired["peak_electric_w_m2_counterfactual"].replace(0.0, np.nan)
    paired = paired.merge(pd.DataFrame(event_temperature_rows), on="event_id", how="left", validate="many_to_one")
    paired["model_relative_energy_slope_per_c"] = paired["heatwave_relative_electricity_increase"] / paired["mean_heatwave_temperature_delta_c"].replace(0.0, np.nan)
    paired.to_csv(output / "paired_heatwave_effects.csv", index=False, encoding="utf-8-sig")
    local_sensitivity_validation = build_local_heat_sensitivity_concordance(
        annual_hourly, polyu_validation, output
    )

    summary = results.groupby(["prototype_id", "climate_case", "supply_stage"]).agg(
        response_cases=("base_case_id", "size"),
        overheating_degree_hours_per_day_p05=("overheating_degree_hours_per_heatwave_day", lambda x: x.quantile(0.05)),
        overheating_degree_hours_per_day_median=("overheating_degree_hours_per_heatwave_day", "median"),
        overheating_degree_hours_per_day_p95=("overheating_degree_hours_per_heatwave_day", lambda x: x.quantile(0.95)),
        cooling_kwh_m2_per_day_p05=("cooling_electricity_kwh_m2_per_heatwave_day", lambda x: x.quantile(0.05)),
        cooling_kwh_m2_per_day_median=("cooling_electricity_kwh_m2_per_heatwave_day", "median"),
        cooling_kwh_m2_per_day_p95=("cooling_electricity_kwh_m2_per_heatwave_day", lambda x: x.quantile(0.95)),
        peak_w_m2_p05=("peak_electric_w_m2", lambda x: x.quantile(0.05)),
        peak_w_m2_median=("peak_electric_w_m2", "median"),
        peak_w_m2_p95=("peak_electric_w_m2", lambda x: x.quantile(0.95)),
    ).reset_index()
    summary.to_csv(output / "prototype_response_ranges.csv", index=False, encoding="utf-8-sig")
    readiness = pd.DataFrame([
        {
            "requirement": "Hong Kong residential stock archetypes",
            "current_status": "partial",
            "evidence_now": "five residential stock prototypes expanded by floor, orientation, setpoint, room and behaviour scenarios",
            "required_to_pass": "locally calibrate public, subsidized, private and village-house envelope and system parameters with withheld Hong Kong buildings",
            "benchmark": "https://doi.org/10.1016/j.energy.2026.142082",
            "blocks_claim": "absolute indoor temperature, cooling energy and peak",
        },
        {
            "requirement": "whole-building thermal topology",
            "current_status": "implemented_pending_calibration",
            "evidence_now": "three-zone dwelling and five-zone commercial core-perimeter prototypes with inter-zone air exchange and zone-specific loads",
            "required_to_pass": "complete cross-type calibration and held-out validation without changing the frozen topology",
            "benchmark": "https://doi.org/10.1016/j.enbuild.2026.117663",
            "blocks_claim": "all-building UBEM and whole-building peak coincidence",
        },
        {
            "requirement": "Hong Kong annual energy calibration",
            "current_status": "implemented_sector_mass_balance",
            "evidence_now": "2023 EMSD residential tenure and commercial-sector electricity/cooling totals are matched by component-wise calibration",
            "required_to_pass": "obtain complete commercial subtype floor-area/use attribution before making subtype-specific absolute totals",
            "benchmark": "https://doi.org/10.1016/j.enbuild.2026.117663",
            "blocks_claim": "absolute citywide energy",
        },
        {
            "requirement": "Hong Kong heatwave-hour validation",
            "current_status": "implemented_local_holdout_validation",
            "evidence_now": "full-year PolyU phase/building series and complete HKUST archive are evaluated with building and temporal holdouts",
            "required_to_pass": "add end-use-labelled cooling meters for cooling-only slope validation and retain current whole-building holdouts",
            "benchmark": "https://doi.org/10.1038/s41597-024-04106-1",
            "blocks_claim": "heatwave peak magnitude and temporal response",
        },
        {
            "requirement": "joint meter and indoor-temperature calibration",
            "current_status": "not_complete",
            "evidence_now": "non-Hong Kong datasets test structure only; no representative Hong Kong residential joint observations",
            "required_to_pass": "calibrate and test electricity, indoor temperature and humidity jointly, with uncertainty intervals and held-out homes",
            "benchmark": "https://doi.org/10.1016/j.enbuild.2016.10.050",
            "blocks_claim": "absolute overheating burden and outage consequence",
        },
        {
            "requirement": "full building-use coverage",
            "current_status": "implemented_pending_validation",
            "evidence_now": "residential plus office retail restaurant accommodation education healthcare data-centre mixed and other commercial prototypes are executable",
            "required_to_pass": "validate material use classes and fit employee visitor and critical-service quantities before final all-building claims",
            "benchmark": "https://doi.org/10.1016/j.enbuild.2026.117663",
            "blocks_claim": "Hong Kong all-building simulation",
        },
    ])
    readiness.to_csv(output / "result2_model_readiness.csv", index=False, encoding="utf-8-sig")
    parameter_experiment_audit = audit_parameter_and_experiment_design(
        parameters, design, event_ids, expected_run_count, len(results), smoke=args.smoke
    )
    (output / "parameter_and_experiment_audit.json").write_text(
        json.dumps(parameter_experiment_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    validation = {
        **population_validation,
        "polyu_campus": polyu_validation,
        "hkust_whole_building": hkust_validation,
        "energyplus": engine,
        "thermal_model": MODEL_VERSION,
        "prototype_floor_areas_m2": parameters.set_index("prototype_id")["floor_area_m2"].to_dict(),
        "events": event_ids,
        "base_cases": len(design),
        "climate_cases": ["counterfactual", "heatwave"],
        "supply_stages": heatwave_supply_stages,
        "dimensions_present": sorted(set(["paired_weather_only_counterfactual", "weekday_weekend", "residential_and_commercial_operation", "zone_level_split_ac", "setpoint_24_25_26", "installed_ac_available_condition", "staggered_start", "bottom_middle_top", "four_orientations", "latent_load_from_people_and_infiltration", "capacity_outage_recovery"])),
        "prototype_parameter_provenance_file": "prototype_parameter_provenance.csv",
        "experiment_setting_provenance_file": "experiment_setting_provenance.csv",
        "selected_design_expected_runs": expected_run_count,
        "selected_design_completed_runs": len(results),
        "selected_design_execution_complete": selected_design_execution_complete,
        "parameter_and_experiment_audit": parameter_experiment_audit,
        "outage_timing_contract": outage_timing_summary,
        "data_centre_outage_physics": data_centre_audit_summary,
        "paired_city_trajectory_uncertainty": {
            "draws_per_event": 500,
            "sampling_unit": "complete paired trajectory within prototype-floor unit",
            "draw_file": "citywide_paired_trajectory_uncertainty.csv",
            "summary_file": "citywide_paired_trajectory_uncertainty_summary.csv",
        },
        "building_physics_consistency": physical_consistency_audit,
        "solar_decomposition": solar_decomposition_audit,
        "energyplus_diagnostics": diagnostics_audit,
        "paired_counterfactual_contract": paired_counterfactual_audit,
        "building_dynamic_results": dynamic_validation,
        "final_building_sample": final_sample_audit,
        "absolute_energy_is_empirically_calibrated": bool(
            official_energy_validation["annual_energy_mass_balance_exact"]
        ),
        "hong_kong_archetype_parameter_calibration_complete": False,
        "hong_kong_residential_joint_external_calibration_complete": False,
        "full_city_building_type_coverage_complete": set(parameters["prototype_id"]).issubset(set(design["prototype_id"])) and not args.smoke,
        "energy_calibration": official_energy_validation,
        "local_heat_sensitivity_external_validation": local_sensitivity_validation,
        "external_dynamic_structure": external_structure_validation,
        "thermal_model_scope": "dynamic residential three-zone and commercial five-zone prototype ensemble; absolute annual stock totals are component-calibrated to EMSD 2023 while event dynamics remain EnergyPlus-derived",
        "direct_hong_kong_citybem_benchmark_met": False,
        "model_readiness_file": "result2_model_readiness.csv",
        "outage_probability_estimated": False,
        "pressure_test_consequences": ["paired heatwave incremental total electricity", "paired peak amplification", "required, delivered and unmet cooling energy", "comfort resistance time to 26C", "night and all-hour degree-hours above 26C", "safety resistance time to 32C during a 24-hour outage", "outage-window degree-hours above 32C", "continuous threshold exceedance", "unmet cooling hours", "occupied service hours lost", "recovery to 26C", "recovery electricity rebound", "cooling electricity", "latent share", "total peak electric intensity"],
        "threshold_note": "The 24-26C Hong Kong summer comfort guidance supports the setpoint scenarios; 26C is retained as the overheating comparison benchmark, while 32C is the pre-specified comparative screen used to implement the hot-humid hours-of-safety concept for the standardized 24-hour outage score.",
        "sources": {
            "hong_kong_prh_energyplus_2026": "https://doi.org/10.1016/j.energy.2026.142082",
            "hong_kong_ac_behaviour": "https://doi.org/10.1016/j.habitatint.2016.12.009",
            "hong_kong_bedroom_survey": "https://doi.org/10.1016/j.enbuild.2006.04.004",
            "hong_kong_citybem_direct_benchmark": "https://doi.org/10.1016/j.enbuild.2026.117663",
            "cibse_tm59": "https://www.cibse.org/knowledge-research/knowledge-portal/technical-memorandum-59-design-methodology-for-the-assessment-of-overheating-risk-in-homes/",
            "hong_kong_summer_comfort_guidance": "https://www.iaq.gov.hk/wp-content/uploads/2021/04/gn_officeandpublicplace_eng-2019.pdf",
            "hot_humid_hours_of_safety_screen": "https://doi.org/10.1016/j.xcrp.2024.101986",
        },
    }
    write_validation_evidence_matrix(
        population_validation,
        polyu_validation,
        hkust_validation,
        official_energy_validation,
        dynamic_validation,
        output,
    )
    validation["validation_evidence_matrix"] = "validation_evidence_matrix.csv"
    (output / "result2_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"02 complete: {output}")


if __name__ == "__main__":
    main()
