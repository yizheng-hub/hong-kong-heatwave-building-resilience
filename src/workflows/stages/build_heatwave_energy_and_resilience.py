"""Result 3: quantify heatwave energy consequences and dynamic resilience.

This entry reads finalized building-level outputs from Result 2. It reports
direct physical quantities by residential and commercial type, supply stage,
event and climate scenario. Ranking concentration is retained only as a
secondary descriptive analysis; no ranking reversal is assumed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
CONFIG = json.loads((ROOT / "config" / "foundation_analysis.json").read_text(encoding="utf-8"))
PIPELINE = ROOT / CONFIG["output_root"]
RESPONSE_DIR = PIPELINE / "02_building_response"
OUT = PIPELINE / "03_energy_resilience"
DEFAULT_RESULTS = RESPONSE_DIR / "building_dynamic_results.parquet"
DEFAULT_INPUTS = RESPONSE_DIR / "all_sector_building_model_inputs.parquet"
COINCIDENT_PEAKS = RESPONSE_DIR / "citywide_coincident_peak_summary.csv"
COINCIDENT_HOURLY = RESPONSE_DIR / "citywide_coincident_hourly_response.parquet"
PROTOTYPE_HOURLY_INTERFACE = RESPONSE_DIR / "prototype_event_hourly_interface.parquet"
PROTOTYPE_RESULTS = RESPONSE_DIR / "energyplus_multizone_results.csv"
PROTOTYPE_PARAMETERS = RESPONSE_DIR / "prototype_physics_parameters.csv"
PROTOTYPE_HOURLY_RESULTS = RESPONSE_DIR / "energyplus_multizone_hourly_results.parquet"
OUTAGE_TIMING_AUDIT = RESPONSE_DIR / "outage_timing_contract_audit.csv"
FUTURE_RESULTS = RESPONSE_DIR / "future_energyplus_results.csv"

IDENTIFIERS = [
    "BUILDINGSTRUCTUREID",
    "main_class",
    "sub_class",
    "prototype_id",
    "event_id",
    "climate_period",
    "climate_scenario",
    "supply_stage",
    "behavior_seed",
]
ENERGY_FIELDS = [
    "baseline_energy_kwh",
    "heatwave_energy_kwh",
    "baseline_peak_kw",
    "heatwave_peak_kw",
]
RESILIENCE_FIELDS = [
    "overheating_degree_hours",
    "outage_window_degree_hours_above_26c",
    "outage_attributable_degree_hours_above_26c",
    "outage_peak_temperature_increment_c",
    "nighttime_overheating_degree_hours",
    "continuous_threshold_exceedance_hours",
    "time_to_thermal_threshold_hours",
    "safety_resistance_hours_to_32c",
    "outage_window_degree_hours_above_32c",
    "unmet_cooling_hours",
    "required_cooling_energy_kwh_th",
    "delivered_cooling_energy_kwh_th",
    "unmet_cooling_energy_kwh_th",
    "recovery_energy_kwh",
    "recovery_time_hours",
    "occupied_service_hours_lost",
    "population_weighted_overheating_person_hours",
    "residential_population_weighted_outage_degree_hours_above_32c",
    "employee_visitor_weighted_overheating_person_hours",
    "critical_service_continuity_hours",
]
DIRECT_OUTCOMES = [
    "heatwave_incremental_energy_kwh",
    "heatwave_relative_energy_increase",
    "incremental_peak_kw",
    "peak_amplification_ratio",
    *RESILIENCE_FIELDS,
]
ADDITIVE_OUTCOMES = {
    "heatwave_incremental_energy_kwh",
    "recovery_energy_kwh",
    "population_weighted_overheating_person_hours",
    "residential_population_weighted_outage_degree_hours_above_32c",
    "employee_visitor_weighted_overheating_person_hours",
    "occupied_service_hours_lost",
    "required_cooling_energy_kwh_th",
    "delivered_cooling_energy_kwh_th",
    "unmet_cooling_energy_kwh_th",
}

BERI_COMPONENTS = ["score_normal", "score_capacity", "score_outage", "score_recovery"]
REGION_BOUNDS_2326 = {
    "Ho Man Tin": (836126.18034143, 818910.80931283, 837345.03745461, 820422.94111362),
    "Mong Kok": (835132.28, 819514.11, 836016.10, 820302.95),
    "Tsim Sha Tsui": (835001.12, 817109.65, 836317.21, 818280.89),
}


def _component_floor_area_weights(building_inputs: pd.DataFrame) -> pd.DataFrame:
    """Return modeled component GFA by stock class, prototype and floor role."""
    stock = building_inputs.loc[building_inputs["pre_weather_simulation_candidate"]].copy()
    residential = stock.loc[
        stock["residential_component_present"],
        ["main_class", "residential_prototype_id", "residential_gfa_m2", "floors_value"],
    ].rename(columns={
        "residential_prototype_id": "prototype_id",
        "residential_gfa_m2": "component_gfa_m2",
    })
    commercial = stock.loc[
        stock["commercial_component_present"],
        ["main_class", "commercial_prototype_id", "commercial_gfa_m2", "floors_value"],
    ].rename(columns={
        "commercial_prototype_id": "prototype_id",
        "commercial_gfa_m2": "component_gfa_m2",
    })
    components = pd.concat([residential, commercial], ignore_index=True)
    components["component_gfa_m2"] = numeric(components["component_gfa_m2"]).fillna(0.0)
    components["floors"] = numeric(components["floors_value"]).round().clip(lower=1).fillna(1)
    parts = []
    for floor_position in ["single", "bottom", "middle", "top"]:
        if floor_position == "single":
            weight = components["floors"].eq(1).astype(float)
        elif floor_position == "bottom":
            weight = np.where(components["floors"].eq(2), 0.5, np.where(components["floors"].ge(3), 1.0 / components["floors"], 0.0))
        elif floor_position == "middle":
            weight = np.where(components["floors"].ge(3), (components["floors"] - 2.0) / components["floors"], 0.0)
        else:
            weight = np.where(components["floors"].eq(2), 0.5, np.where(components["floors"].ge(3), 1.0 / components["floors"], 0.0))
        selected = np.asarray(weight) > 0
        part = components.loc[selected, ["main_class", "prototype_id", "component_gfa_m2"]].copy()
        part["floor_position"] = floor_position
        part["component_gfa_m2"] *= np.asarray(weight)[selected]
        parts.append(part)
    return pd.concat(parts, ignore_index=True).groupby(
        ["main_class", "prototype_id", "floor_position"], as_index=False, observed=True
    )["component_gfa_m2"].sum()


def build_class_hourly_attribution(
    building_inputs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Resolve event electricity and peak timing by stock class and end use."""
    if not PROTOTYPE_HOURLY_INTERFACE.exists():
        raise FileNotFoundError(f"Prototype hourly interface missing: {PROTOTYPE_HOURLY_INTERFACE}")
    if not COINCIDENT_HOURLY.exists():
        raise FileNotFoundError(f"City coincident hourly response missing: {COINCIDENT_HOURLY}")
    hourly = pd.read_parquet(PROTOTYPE_HOURLY_INTERFACE)
    fields = ["cooling", "lighting", "equipment", "total_electricity"]
    required = {
        "event_id", "prototype_id", "floor_position", "climate_case",
        "supply_stage", "timestamp_hkt", "event_phase",
        *[f"{field}_kwh_m2_p50" for field in fields],
    }
    missing = sorted(required.difference(hourly.columns))
    if missing:
        raise ValueError(f"Prototype hourly interface lacks {missing}")
    hourly = hourly.loc[
        hourly["supply_stage"].eq("normal")
        & hourly["event_phase"].eq("heatwave")
        & hourly["climate_case"].isin(["counterfactual", "heatwave"]),
        [
            "event_id", "prototype_id", "floor_position", "climate_case",
            "timestamp_hkt", *[f"{field}_kwh_m2_p50" for field in fields],
        ],
    ].copy()
    hourly["timestamp_hkt"] = pd.to_datetime(
        hourly["timestamp_hkt"], utc=True
    ).dt.tz_convert("Asia/Hong_Kong")
    weights = _component_floor_area_weights(building_inputs)
    weighted = hourly.merge(
        weights,
        on=["prototype_id", "floor_position"],
        how="inner",
        validate="many_to_many",
    )
    for field in fields:
        weighted[f"{field}_kwh"] = (
            numeric(weighted[f"{field}_kwh_m2_p50"]) * weighted["component_gfa_m2"]
        )
    keys = ["event_id", "main_class", "climate_case", "timestamp_hkt"]
    class_hourly = weighted.groupby(keys, as_index=False, observed=True)[
        [f"{field}_kwh" for field in fields]
    ].sum()
    index = ["event_id", "main_class", "timestamp_hkt"]
    pieces = []
    for field in fields:
        wide = class_hourly.pivot(
            index=index, columns="climate_case", values=f"{field}_kwh"
        ).reindex(columns=["counterfactual", "heatwave"])
        wide.columns = [f"{field}_{case}_kwh" for case in wide.columns]
        wide[f"{field}_increment_kwh"] = (
            wide[f"{field}_heatwave_kwh"] - wide[f"{field}_counterfactual_kwh"]
        )
        pieces.append(wide)
    resolved = pd.concat(pieces, axis=1).reset_index()

    event_rows = []
    class_rows = []
    for event_id, event in resolved.groupby("event_id", observed=True):
        energy_columns = [column for column in resolved.columns if column.endswith("_kwh")]
        city = event.groupby("timestamp_hkt", as_index=False, observed=True)[energy_columns].sum()
        heatwave_peak_row = city.loc[city["total_electricity_heatwave_kwh"].idxmax()]
        counterfactual_peak_row = city.loc[city["total_electricity_counterfactual_kwh"].idxmax()]
        heatwave_peak_time = heatwave_peak_row["timestamp_hkt"]
        counterfactual_peak_time = counterfactual_peak_row["timestamp_hkt"]
        exact_peak_increment = (
            heatwave_peak_row["total_electricity_heatwave_kwh"]
            - counterfactual_peak_row["total_electricity_counterfactual_kwh"]
        )
        weather_increment_at_peak = heatwave_peak_row["total_electricity_increment_kwh"]
        baseline_timing_term = (
            heatwave_peak_row["total_electricity_counterfactual_kwh"]
            - counterfactual_peak_row["total_electricity_counterfactual_kwh"]
        )
        event_record = {
            "event_id": event_id,
            "heatwave_peak_timestamp_hkt": heatwave_peak_time,
            "counterfactual_peak_timestamp_hkt": counterfactual_peak_time,
            "peak_time_shift_hours": float((heatwave_peak_time - counterfactual_peak_time).total_seconds() / 3600.0),
            "heatwave_coincident_peak_kw": float(heatwave_peak_row["total_electricity_heatwave_kwh"]),
            "counterfactual_coincident_peak_kw": float(counterfactual_peak_row["total_electricity_counterfactual_kwh"]),
            "coincident_peak_increment_kw": float(exact_peak_increment),
            "weather_increment_at_heatwave_peak_kw": float(weather_increment_at_peak),
            "counterfactual_timing_term_kw": float(baseline_timing_term),
            "peak_decomposition_residual_kw": float(exact_peak_increment - weather_increment_at_peak - baseline_timing_term),
        }
        for field in fields:
            event_record[f"event_{field}_increment_gwh"] = float(event[f"{field}_increment_kwh"].sum() / 1e6)
        total_increment = event_record["event_total_electricity_increment_gwh"]
        end_use_sum = sum(
            event_record[f"event_{field}_increment_gwh"]
            for field in ["cooling", "lighting", "equipment"]
        )
        event_record["end_use_median_aggregation_residual_gwh"] = total_increment - end_use_sum
        event_record["end_use_median_aggregation_residual_fraction"] = (
            (total_increment - end_use_sum) / total_increment if total_increment else np.nan
        )
        event_record["cooling_share_of_event_increment"] = (
            event_record["event_cooling_increment_gwh"] / total_increment if total_increment else np.nan
        )
        event_rows.append(event_record)

        at_heatwave_peak = event.loc[event["timestamp_hkt"].eq(heatwave_peak_time)]
        for main_class, group in event.groupby("main_class", observed=True):
            peak_class = at_heatwave_peak.loc[at_heatwave_peak["main_class"].eq(main_class)].iloc[0]
            class_total = group["total_electricity_increment_kwh"].sum()
            class_record = {
                "event_id": event_id,
                "main_class": main_class,
                "heatwave_peak_timestamp_hkt": heatwave_peak_time,
                "event_incremental_energy_gwh": float(class_total / 1e6),
                "weather_increment_at_heatwave_peak_kw": float(peak_class["total_electricity_increment_kwh"]),
            }
            for field in ["cooling", "lighting", "equipment"]:
                class_record[f"event_{field}_increment_gwh"] = float(group[f"{field}_increment_kwh"].sum() / 1e6)
                class_record[f"{field}_increment_at_heatwave_peak_kw"] = float(peak_class[f"{field}_increment_kwh"])
            class_record["share_of_city_event_increment"] = (
                class_record["event_incremental_energy_gwh"] / total_increment if total_increment else np.nan
            )
            class_record["share_of_weather_increment_at_heatwave_peak"] = (
                class_record["weather_increment_at_heatwave_peak_kw"] / weather_increment_at_peak if weather_increment_at_peak else np.nan
            )
            local_hours = group["timestamp_hkt"].dt.tz_convert("Asia/Hong_Kong").dt.hour
            evening = local_hours.ge(18) & local_hours.le(23)
            business = local_hours.ge(8) & local_hours.lt(18)
            class_record["event_increment_during_18_23_fraction"] = float(
                group.loc[evening, "total_electricity_increment_kwh"].sum() / class_total
            )
            class_record["event_increment_during_08_18_fraction"] = float(
                group.loc[business, "total_electricity_increment_kwh"].sum() / class_total
            )
            class_rows.append(class_record)

    event_summary = pd.DataFrame(event_rows)
    class_summary = pd.DataFrame(class_rows)
    frozen_city = pd.read_parquet(COINCIDENT_HOURLY)
    frozen_city["timestamp_hkt"] = pd.to_datetime(
        frozen_city["timestamp_hkt"], utc=True
    ).dt.tz_convert("Asia/Hong_Kong")
    frozen_city = frozen_city.loc[
        frozen_city["supply_stage"].eq("normal")
        & frozen_city["event_phase"].eq("heatwave")
        & frozen_city["climate_case"].isin(["counterfactual", "heatwave"]),
        ["event_id", "climate_case", "timestamp_hkt", "city_electricity_kwh"],
    ]
    reconstructed = class_hourly.groupby(
        ["event_id", "climate_case", "timestamp_hkt"], as_index=False, observed=True
    )["total_electricity_kwh"].sum()
    check = frozen_city.merge(
        reconstructed,
        on=["event_id", "climate_case", "timestamp_hkt"],
        how="outer",
        validate="one_to_one",
    )
    denominator = numeric(check["city_electricity_kwh"]).abs().replace(0.0, np.nan)
    relative_error = (
        numeric(check["city_electricity_kwh"]) - numeric(check["total_electricity_kwh"])
    ).abs() / denominator
    audit = {
        "events": int(event_summary["event_id"].nunique()),
        "classes": sorted(class_summary["main_class"].astype(str).unique().tolist()),
        "end_uses": ["air_conditioning", "lighting", "equipment_and_other"],
        "maximum_relative_error_against_frozen_city_hourly": float(relative_error.fillna(0.0).max()),
        "maximum_peak_decomposition_residual_kw": float(numeric(event_summary["peak_decomposition_residual_kw"]).abs().max()),
        "maximum_absolute_end_use_median_aggregation_residual_fraction": float(
            numeric(event_summary["end_use_median_aggregation_residual_fraction"]).abs().max()
        ),
        "end_use_residual_explanation": "componentwise pointwise medians need not add exactly to the pointwise median of total electricity",
        "passed": bool(relative_error.fillna(0.0).max() < 1e-10),
    }
    if not audit["passed"]:
        raise ValueError(f"Class-hourly reconstruction does not reproduce frozen city trajectory: {audit}")
    return resolved, event_summary, class_summary, audit


def build_district_spatial_summary(
    building_inputs: gpd.GeoDataFrame,
    effects: pd.DataFrame,
    beri: pd.DataFrame,
) -> pd.DataFrame:
    """Quantify the same building sets displayed in the three district maps."""
    candidates = building_inputs.loc[
        building_inputs["pre_weather_simulation_candidate"]
    ].copy()
    records = []
    for region_name, (xmin, ymin, xmax, ymax) in REGION_BOUNDS_2326.items():
        local = candidates.cx[xmin:xmax, ymin:ymax]
        identifiers = set(local["BUILDINGSTRUCTUREID"].astype(str))
        if not identifiers:
            raise ValueError(f"No modeled buildings in frozen region {region_name}")
        local_effects = effects.loc[
            effects["BUILDINGSTRUCTUREID"].astype(str).isin(identifiers)
        ]
        local_beri = beri.loc[
            beri["BUILDINGSTRUCTUREID"].astype(str).isin(identifiers),
            ["BUILDINGSTRUCTUREID", "building_energy_resilience_index"],
        ]
        for event_id in sorted(local_effects["event_id"].astype(str).unique()):
            normal = local_effects.loc[
                local_effects["event_id"].astype(str).eq(event_id)
                & local_effects["supply_stage"].eq("normal")
            ]
            outage = local_effects.loc[
                local_effects["event_id"].astype(str).eq(event_id)
                & local_effects["supply_stage"].eq("outage_peak_24h")
            ]
            for class_label in ["All", "Residential", "Mixed-use", "Commercial"]:
                class_normal = normal if class_label == "All" else normal.loc[normal["main_class"].eq(class_label)]
                class_outage = outage if class_label == "All" else outage.loc[outage["main_class"].eq(class_label)]
                if class_normal.empty:
                    continue
                class_ids = set(class_normal["BUILDINGSTRUCTUREID"].astype(str))
                class_stock = local.loc[local["BUILDINGSTRUCTUREID"].astype(str).isin(class_ids)]
                class_beri = local_beri.loc[local_beri["BUILDINGSTRUCTUREID"].astype(str).isin(class_ids)]
                baseline_energy = numeric(class_normal["baseline_energy_kwh"]).sum()
                incremental_energy = numeric(class_normal["heatwave_incremental_energy_kwh"]).sum()
                records.append({
                    "region": region_name,
                    "event_id": event_id,
                    "main_class": class_label,
                    "buildings": int(len(class_normal)),
                    "gross_floor_area_million_m2": float(numeric(class_stock["gfa_value_m2"]).sum() / 1e6),
                    "heatwave_event_electricity_gwh": float(numeric(class_normal["heatwave_energy_kwh"]).sum() / 1e6),
                    "heatwave_incremental_electricity_gwh": float(incremental_energy / 1e6),
                    "aggregate_relative_electricity_increase_pct": float(
                        100.0 * incremental_energy / baseline_energy if baseline_energy else np.nan
                    ),
                    "building_median_relative_electricity_increase_pct": float(
                        100.0 * numeric(class_normal["heatwave_relative_energy_increase"]).median()
                    ),
                    "building_median_time_to_32c_hours": float(
                        numeric(class_outage["safety_resistance_hours_to_32c"]).median()
                    ),
                    "building_median_outage_attributable_degree_hours_c_h": float(
                        numeric(class_outage["outage_attributable_degree_hours_above_26c"]).median()
                    ),
                    "building_median_recovery_time_hours": float(
                        numeric(class_outage["recovery_time_hours"]).median()
                    ),
                    "building_median_reference_beri": float(
                        numeric(class_beri["building_energy_resilience_index"]).median()
                    ),
                })
    return pd.DataFrame(records)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def write_analysis_contract(output: Path) -> None:
    field_rows = []
    for field in IDENTIFIERS:
        field_rows.append({"field": field, "role": "identifier", "required_column": True, "missing_values_permitted": False})
    for field in ENERGY_FIELDS:
        field_rows.append({"field": field, "role": "paired_energy_or_peak", "required_column": True, "missing_values_permitted": False})
    for field in RESILIENCE_FIELDS:
        field_rows.append(
            {
                "field": field,
                "role": "direct_resilience_outcome",
                "required_column": True,
                "missing_values_permitted": True,
            }
        )
    pd.DataFrame(field_rows).to_csv(
        output / "dynamic_result_field_contract.csv", index=False, encoding="utf-8-sig"
    )
    plan = pd.DataFrame(
        [
            {
                "result_block": "model_performance",
                "primary_outputs": "NMBE|CVRMSE|MAE|peak_magnitude_error|peak_timing_error|indoor_temperature_RMSE|temperature_load_sensitivity",
                "grouping": "dataset|building_or_holdout_block|prototype",
            },
            {
                "result_block": "heatwave_energy",
                "primary_outputs": "heatwave_incremental_energy_kwh|heatwave_relative_energy_increase",
                "grouping": "main_class|prototype|event|climate_period|climate_scenario",
            },
            {
                "result_block": "heatwave_peak",
                "primary_outputs": "incremental_peak_kw|peak_amplification_ratio",
                "grouping": "main_class|prototype|event|climate_period|climate_scenario",
            },
            {
                "result_block": "dynamic_resilience",
                "primary_outputs": "|".join(RESILIENCE_FIELDS),
                "grouping": "supply_stage|main_class|prototype|event|climate_period|climate_scenario",
            },
            {
                "result_block": "driver_decomposition",
                "primary_outputs": "main_effects|interactions|out_of_sample_explanatory_performance",
                "grouping": "physical|system|behavioral_driver",
            },
            {
                "result_block": "secondary_concentration",
                "primary_outputs": "top_fraction_capture_and_overlap_if_scientifically_informative",
                "grouping": "outcome|scenario",
            },
        ]
    )
    plan.to_csv(output / "result3_analysis_plan.csv", index=False, encoding="utf-8-sig")


def validate_results(frame: pd.DataFrame) -> dict:
    required = set(IDENTIFIERS + ENERGY_FIELDS + RESILIENCE_FIELDS)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Result 2 building dynamic table lacks required columns: {missing}")
    if frame[IDENTIFIERS].isna().any().any():
        bad = frame[IDENTIFIERS].isna().sum()
        raise ValueError(f"Missing result identifiers: {bad[bad.gt(0)].to_dict()}")
    if frame.duplicated(IDENTIFIERS).any():
        raise ValueError("Building dynamic results are not unique at the declared result grain")
    negative_fields = []
    for field in ENERGY_FIELDS + RESILIENCE_FIELDS:
        values = numeric(frame[field])
        if bool(values.dropna().lt(0).any()):
            negative_fields.append(field)
    if negative_fields:
        raise ValueError(f"Physically non-negative fields contain negative values: {negative_fields}")
    return {
        "rows": int(len(frame)),
        "buildings": int(frame["BUILDINGSTRUCTUREID"].nunique()),
        "main_classes": sorted(frame["main_class"].astype(str).unique()),
        "prototypes": int(frame["prototype_id"].nunique()),
        "events": int(frame["event_id"].nunique()),
        "climate_periods": sorted(frame["climate_period"].astype(str).unique()),
        "climate_scenarios": sorted(frame["climate_scenario"].astype(str).unique()),
        "supply_stages": sorted(frame["supply_stage"].astype(str).unique()),
        "result_grain_is_unique": True,
    }


def add_paired_effects(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for field in ENERGY_FIELDS + RESILIENCE_FIELDS:
        result[field] = numeric(result[field])
    result["heatwave_incremental_energy_kwh"] = (
        result["heatwave_energy_kwh"] - result["baseline_energy_kwh"]
    )
    result["heatwave_relative_energy_increase"] = (
        result["heatwave_incremental_energy_kwh"]
        / result["baseline_energy_kwh"].replace(0.0, np.nan)
    )
    result["incremental_peak_kw"] = result["heatwave_peak_kw"] - result["baseline_peak_kw"]
    result["peak_amplification_ratio"] = (
        result["heatwave_peak_kw"] / result["baseline_peak_kw"].replace(0.0, np.nan)
    )
    return result


def build_event_response_score(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score each building under each observed-event stress test.

    This table is event conditioned and is therefore not the pre-event BERI.
    It is retained to audit how the four stage scores change between forcing
    events before they are aggregated into the frozen reference rating.
    """
    keys = [
        "BUILDINGSTRUCTUREID", "main_class", "sub_class", "prototype_id",
        "event_id", "climate_period", "climate_scenario", "behavior_seed",
    ]

    def one_stage(stage: str, fields: list[str]) -> pd.DataFrame:
        selected = frame.loc[frame["supply_stage"].eq(stage), keys + fields].copy()
        if selected.duplicated(keys).any():
            raise ValueError(f"Event-response stage {stage} is not unique at building-event grain")
        return selected

    normal = one_stage("normal", ["heatwave_relative_energy_increase"])
    capacity = one_stage(
        "capacity_70pct",
        ["required_cooling_energy_kwh_th", "delivered_cooling_energy_kwh_th",
         "unmet_cooling_energy_kwh_th"],
    )
    outage_fields = [
        "safety_resistance_hours_to_32c", "recovery_time_hours",
        "outage_window_degree_hours_above_32c",
        "outage_attributable_degree_hours_above_26c", "recovery_energy_kwh",
    ]
    for threshold_c in (30, 34):
        for field in (
            f"safety_resistance_hours_to_{threshold_c}c",
            f"outage_window_degree_hours_above_{threshold_c}c",
        ):
            if field in frame.columns:
                outage_fields.append(field)
    outage = one_stage("outage_peak_24h", outage_fields)
    result = normal.merge(capacity, on=keys, how="inner", validate="one_to_one")
    result = result.merge(outage, on=keys, how="inner", validate="one_to_one")
    expected = frame["BUILDINGSTRUCTUREID"].nunique() * frame["event_id"].nunique()
    if len(result) != expected:
        raise ValueError(f"Event-response rows {len(result):,} != expected {expected:,}")

    relative = numeric(result["heatwave_relative_energy_increase"]).clip(lower=0.0)
    required = numeric(result["required_cooling_energy_kwh_th"]).clip(lower=0.0)
    unmet = numeric(result["unmet_cooling_energy_kwh_th"]).clip(lower=0.0)
    outage_hours = float(CONFIG["thermal"]["outage_safety_assessment_hours"])
    recovery_hours = float(CONFIG["thermal"]["recovery_assessment_hours"])
    result["score_normal"] = 1.0 / (1.0 + relative)
    result["score_capacity"] = np.where(
        required.gt(0.0), 1.0 - unmet / required.replace(0.0, np.nan), 1.0
    )
    result["score_outage"] = (
        numeric(result["safety_resistance_hours_to_32c"]) / outage_hours
    )
    result["score_recovery"] = (
        1.0 - numeric(result["recovery_time_hours"]) / recovery_hours
    )
    result[BERI_COMPONENTS] = result[BERI_COMPONENTS].clip(0.0, 1.0)
    if result[BERI_COMPONENTS].isna().any().any():
        bad = result[BERI_COMPONENTS].isna().sum()
        raise ValueError(f"BERI components contain missing values: {bad[bad.gt(0)].to_dict()}")

    weight_scenarios = CONFIG["resilience"]["beri_weight_scenarios"]
    sensitivity_rows = []
    for scenario, weights in weight_scenarios.items():
        weight = np.asarray(weights, dtype=float)
        if len(weight) != len(BERI_COMPONENTS) or not np.isclose(weight.sum(), 1.0):
            raise ValueError(f"Invalid event-response weights for {scenario}: {weights}")
        score = np.ones(len(result), dtype=float)
        for field, exponent in zip(BERI_COMPONENTS, weight):
            score *= np.power(result[field].to_numpy(dtype=float), exponent)
        column = "event_response_score" if scenario == "equal" else f"event_response_score_{scenario}"
        result[column] = 100.0 * score
        if scenario != "equal":
            for event_id, group in result.groupby("event_id", observed=True):
                sensitivity_rows.append({
                    "event_id": str(event_id),
                    "weight_scenario": str(scenario),
                    "spearman_rank_correlation_with_equal": float(
                        group["event_response_score"].corr(group[column], method="spearman")
                    ),
                    "median_absolute_index_change": float(
                        (group[column] - group["event_response_score"]).abs().median()
                    ),
                    "p95_absolute_index_change": float(
                        (group[column] - group["event_response_score"]).abs().quantile(0.95)
                    ),
                })
    return result, pd.DataFrame(sensitivity_rows)


def build_reference_beri(event_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one pre-event BERI per building from the frozen reference suite.

    The componentwise median prevents one selected event from defining the
    rating while preserving the four physically interpretable stage scores.
    The same reference BERI is used for every historical-event map and video.
    """
    reference_events = [str(value) for value in CONFIG["resilience"]["beri_reference_event_ids"]]
    selected = event_scores.loc[event_scores["event_id"].isin(reference_events)].copy()
    present = sorted(selected["event_id"].astype(str).unique())
    if sorted(reference_events) != present:
        missing = sorted(set(reference_events) - set(present))
        raise ValueError(f"Reference BERI suite is incomplete; missing events: {missing}")

    keys = ["BUILDINGSTRUCTUREID", "main_class", "sub_class", "prototype_id"]
    counts = selected.groupby("BUILDINGSTRUCTUREID", observed=True)["event_id"].nunique()
    if not counts.eq(len(reference_events)).all():
        raise ValueError("Every building must be covered by every reference BERI event")

    reference = selected.groupby(keys, as_index=False, observed=True)[BERI_COMPONENTS].median()
    reference["reference_event_count"] = len(reference_events)
    reference["reference_event_ids"] = "|".join(reference_events)
    reference["beri_rating_vintage"] = str(CONFIG["resilience"]["beri_rating_vintage"])
    reference["reference_event_aggregation"] = str(
        CONFIG["resilience"]["beri_reference_event_aggregation"]
    )
    stage_labels = {
        "score_normal": "normal_supply_adaptation",
        "score_capacity": "cooling_service_adequacy",
        "score_outage": "outage_thermal_safety",
        "score_recovery": "post_outage_recovery",
    }
    reference["limiting_resilience_stage"] = (
        reference[BERI_COMPONENTS].idxmin(axis=1).map(stage_labels)
    )

    sensitivity_rows = []
    for scenario, weights in CONFIG["resilience"]["beri_weight_scenarios"].items():
        weight = np.asarray(weights, dtype=float)
        if len(weight) != len(BERI_COMPONENTS) or not np.isclose(weight.sum(), 1.0):
            raise ValueError(f"Invalid reference BERI weights for {scenario}: {weights}")
        score = np.ones(len(reference), dtype=float)
        for field, exponent in zip(BERI_COMPONENTS, weight):
            score *= np.power(reference[field].to_numpy(dtype=float), exponent)
        column = "building_energy_resilience_index" if scenario == "equal" else f"beri_{scenario}"
        reference[column] = 100.0 * score
        if scenario != "equal":
            sensitivity_rows.append(
                {
                    "rating_vintage": str(CONFIG["resilience"]["beri_rating_vintage"]),
                    "weight_scenario": str(scenario),
                    "spearman_rank_correlation_with_equal": float(
                        reference["building_energy_resilience_index"].corr(
                            reference[column], method="spearman"
                        )
                    ),
                    "median_absolute_index_change": float(
                        (reference[column] - reference["building_energy_resilience_index"]).abs().median()
                    ),
                    "p95_absolute_index_change": float(
                        (reference[column] - reference["building_energy_resilience_index"]).abs().quantile(0.95)
                    ),
                }
            )
    return reference, pd.DataFrame(sensitivity_rows)


def beri_definition_sensitivity(
    event_scores: pd.DataFrame,
    reference: pd.DataFrame,
    top_fraction: float,
) -> pd.DataFrame:
    """Test whether BERI ranks depend on threshold, score map or aggregator.

    All alternatives use the same simulated 24-hour outage trajectories.  A
    12-hour normalization is therefore right-censored at 12 hours; no result
    is extrapolated beyond the simulated window.
    """
    required = {
        "BUILDINGSTRUCTUREID", "heatwave_relative_energy_increase",
        "score_capacity", "recovery_time_hours",
        "safety_resistance_hours_to_30c", "safety_resistance_hours_to_32c",
        "safety_resistance_hours_to_34c",
    }
    missing = sorted(required - set(event_scores.columns))
    if missing:
        raise ValueError(f"BERI definition sensitivity lacks fields: {missing}")
    scenarios = [
        ("primary_32c_geometric", 32, 24.0, 24.0, "reciprocal", "geometric"),
        ("threshold_30c", 30, 24.0, 24.0, "reciprocal", "geometric"),
        ("threshold_34c", 34, 24.0, 24.0, "reciprocal", "geometric"),
        ("outage_horizon_12h", 32, 12.0, 24.0, "reciprocal", "geometric"),
        ("recovery_horizon_12h", 32, 24.0, 12.0, "reciprocal", "geometric"),
        ("normal_exponential_map", 32, 24.0, 24.0, "exponential", "geometric"),
        ("arithmetic_aggregation", 32, 24.0, 24.0, "reciprocal", "arithmetic"),
    ]
    primary = reference[["BUILDINGSTRUCTUREID", "building_energy_resilience_index"]].copy()
    primary["_id"] = primary["BUILDINGSTRUCTUREID"].astype(str)
    count = max(1, int(np.ceil(len(primary) * top_fraction)))

    def cohort_membership(frame: pd.DataFrame, score_field: str) -> dict[str, object]:
        ranked = frame.sort_values([score_field, "_id"], kind="mergesort")
        selected = ranked.head(count)
        cutoff = float(selected[score_field].max())
        strict = set(ranked.loc[ranked[score_field].lt(cutoff), "_id"])
        tied = set(ranked.loc[ranked[score_field].eq(cutoff), "_id"])
        tie_slots = count - len(strict)
        return {
            "selected": set(selected["_id"]),
            "strict": strict,
            "tied": tied,
            "tie_slots": tie_slots,
            "tie_probability": tie_slots / len(tied),
            "cutoff": cutoff,
        }

    def expected_overlap(left: dict[str, object], right: dict[str, object]) -> float:
        expected_intersection = (
            len(left["strict"] & right["strict"])
            + float(right["tie_probability"]) * len(left["strict"] & right["tied"])
            + float(left["tie_probability"]) * len(left["tied"] & right["strict"])
            + float(left["tie_probability"])
            * float(right["tie_probability"])
            * len(left["tied"] & right["tied"])
        )
        return float(expected_intersection / count)

    primary_membership = cohort_membership(primary, "building_energy_resilience_index")
    primary_low = primary_membership["selected"]
    rows = []
    primary_reconstruction: pd.DataFrame | None = None
    for name, threshold_c, outage_horizon, recovery_horizon, normal_map, aggregator in scenarios:
        work = event_scores[[
            "BUILDINGSTRUCTUREID", "event_id", "heatwave_relative_energy_increase",
            "score_capacity", "recovery_time_hours",
            f"safety_resistance_hours_to_{threshold_c}c",
        ]].copy()
        relative = numeric(work["heatwave_relative_energy_increase"]).clip(lower=0.0)
        work["score_normal_test"] = (
            np.exp(-relative) if normal_map == "exponential" else 1.0 / (1.0 + relative)
        )
        work["score_capacity_test"] = numeric(work["score_capacity"]).clip(0.0, 1.0)
        work["score_outage_test"] = (
            numeric(work[f"safety_resistance_hours_to_{threshold_c}c"])
            .clip(upper=outage_horizon) / outage_horizon
        ).clip(0.0, 1.0)
        work["score_recovery_test"] = (
            1.0 - numeric(work["recovery_time_hours"]) / recovery_horizon
        ).clip(0.0, 1.0)
        components = [
            "score_normal_test", "score_capacity_test",
            "score_outage_test", "score_recovery_test",
        ]
        building = work.groupby("BUILDINGSTRUCTUREID", as_index=False, observed=True)[components].median()
        if aggregator == "arithmetic":
            building["beri_test"] = 100.0 * building[components].mean(axis=1)
        else:
            building["beri_test"] = 100.0 * building[components].prod(axis=1) ** 0.25
        if name == "primary_32c_geometric":
            frozen_primary = primary.set_index("BUILDINGSTRUCTUREID")[
                "building_energy_resilience_index"
            ]
            building["beri_test"] = building["BUILDINGSTRUCTUREID"].map(frozen_primary)
            primary_reconstruction = building[["BUILDINGSTRUCTUREID", "beri_test"]].copy()
        building["_id"] = building["BUILDINGSTRUCTUREID"].astype(str)
        merged = primary.merge(
            building[["BUILDINGSTRUCTUREID", "beri_test"]],
            on="BUILDINGSTRUCTUREID", how="inner", validate="one_to_one",
        )
        membership = cohort_membership(building, "beri_test")
        selected = membership["selected"]
        limiting = building[components].idxmin(axis=1).value_counts(normalize=True)
        rows.append({
            "scenario": name,
            "temperature_threshold_c": float(threshold_c),
            "outage_normalization_hours": outage_horizon,
            "recovery_normalization_hours": recovery_horizon,
            "normal_score_map": normal_map,
            "component_aggregation": aggregator,
            "buildings": int(len(building)),
            "beri_median": float(building["beri_test"].median()),
            "beri_p05": float(building["beri_test"].quantile(0.05)),
            "beri_p95": float(building["beri_test"].quantile(0.95)),
            "spearman_with_primary": float(merged["building_energy_resilience_index"].corr(
                merged["beri_test"], method="spearman"
            )),
            "deterministic_lowest_decile_overlap_with_primary": float(
                len(selected & primary_low) / count
            ),
            "tie_aware_expected_lowest_decile_overlap_with_primary": (
                1.0 if name == "primary_32c_geometric"
                else expected_overlap(membership, primary_membership)
            ),
            "buildings_strictly_below_cutoff": int(len(membership["strict"])),
            "buildings_tied_at_cutoff": int(len(membership["tied"])),
            "tie_slots_selected": int(membership["tie_slots"]),
            "tie_selection_probability": float(membership["tie_probability"]),
            "normal_is_limiting_fraction": float(limiting.get("score_normal_test", 0.0)),
            "capacity_is_limiting_fraction": float(limiting.get("score_capacity_test", 0.0)),
            "outage_is_limiting_fraction": float(limiting.get("score_outage_test", 0.0)),
            "recovery_is_limiting_fraction": float(limiting.get("score_recovery_test", 0.0)),
        })
    result = pd.DataFrame(rows)
    if primary_reconstruction is None:
        raise RuntimeError("Primary BERI sensitivity scenario was not constructed")
    primary_comparison = primary.merge(
        primary_reconstruction,
        on="BUILDINGSTRUCTUREID", how="inner", validate="one_to_one",
    )
    maximum_primary_difference = float(
        (
            primary_comparison["building_energy_resilience_index"]
            - primary_comparison["beri_test"]
        ).abs().max()
    )
    result["primary_reconstruction_maximum_absolute_difference"] = maximum_primary_difference
    if maximum_primary_difference >= 1e-10:
        raise ValueError("Primary BERI sensitivity reconstruction does not match the frozen rating")
    return result


def cross_stage_ranking_audit(
    reference: pd.DataFrame,
    fraction: float,
) -> pd.DataFrame:
    """Quantify whether the four energy-resilience stages rank buildings alike.

    Spearman correlation uses every building.  Low-score cohort overlap is
    reported both with deterministic building-ID tie resolution and as the
    expected overlap when buildings tied at either cutoff are exchangeable.
    """
    count = max(1, int(math.ceil(len(reference) * fraction)))
    stage_labels = {
        "score_normal": "normal_supply_relative_demand_robustness",
        "score_capacity": "cooling_service_adequacy",
        "score_outage": "outage_thermal_tolerance",
        "score_recovery": "post_outage_recovery",
    }
    cohorts: dict[str, dict[str, object]] = {}
    for field in BERI_COMPONENTS:
        ranking = reference[["BUILDINGSTRUCTUREID", field]].copy()
        ranking["_id"] = ranking["BUILDINGSTRUCTUREID"].astype(str)
        ranking = ranking.sort_values([field, "_id"], kind="mergesort")
        cutoff = float(ranking.iloc[count - 1][field])
        strict = set(ranking.loc[ranking[field].lt(cutoff), "_id"])
        tied = set(ranking.loc[ranking[field].eq(cutoff), "_id"])
        needed_from_ties = count - len(strict)
        cohorts[field] = {
            "selected": set(ranking.head(count)["_id"]),
            "strict": strict,
            "tied": tied,
            "tie_selection_probability": needed_from_ties / len(tied),
            "cutoff": cutoff,
        }
    rows = []
    for left_index, left in enumerate(BERI_COMPONENTS):
        for right in BERI_COMPONENTS[left_index + 1:]:
            a, b = cohorts[left], cohorts[right]
            strict_a, tie_a = a["strict"], a["tied"]
            strict_b, tie_b = b["strict"], b["tied"]
            p_a = float(a["tie_selection_probability"])
            p_b = float(b["tie_selection_probability"])
            expected_intersection = (
                len(strict_a & strict_b)
                + p_b * len(strict_a & tie_b)
                + p_a * len(tie_a & strict_b)
                + p_a * p_b * len(tie_a & tie_b)
            )
            rows.append({
                "left_component": left,
                "left_stage": stage_labels[left],
                "right_component": right,
                "right_stage": stage_labels[right],
                "buildings": int(len(reference)),
                "selected_buildings_per_stage": count,
                "spearman_rank_correlation": float(
                    numeric(reference[left]).corr(numeric(reference[right]), method="spearman")
                ),
                "deterministic_low_fraction_overlap": float(
                    len(a["selected"] & b["selected"]) / count
                ),
                "tie_aware_expected_low_fraction_overlap": float(
                    expected_intersection / count
                ),
                "left_cutoff": float(a["cutoff"]),
                "right_cutoff": float(b["cutoff"]),
                "left_buildings_tied_at_cutoff": int(len(tie_a)),
                "right_buildings_tied_at_cutoff": int(len(tie_b)),
            })
    return pd.DataFrame(rows)


def cross_stage_cutoff_sensitivity(
    reference: pd.DataFrame,
    fractions: tuple[float, ...] = (0.05, 0.10, 0.20),
) -> pd.DataFrame:
    """Repeat the stage-ranking audit at predeclared screening fractions."""
    unique_fractions = sorted({float(value) for value in fractions})
    if not unique_fractions or any(not 0 < value < 1 for value in unique_fractions):
        raise ValueError("Every screening fraction must be between zero and one")
    audits = []
    for fraction in unique_fractions:
        audit = cross_stage_ranking_audit(reference, fraction).copy()
        audit.insert(0, "selected_fraction", fraction)
        audits.append(audit)
    return pd.concat(audits, ignore_index=True)


def audit_beri_contract(
    effects: pd.DataFrame,
    event_scores: pd.DataFrame,
    reference: pd.DataFrame,
) -> dict:
    """Verify stage completeness, physical score identities and reference aggregation."""
    reference_events = [str(value) for value in CONFIG["resilience"]["beri_reference_event_ids"]]
    required_stages = {
        "normal", "capacity_100pct", "capacity_70pct", "outage_peak_24h"
    }
    observed_stages = set(effects["supply_stage"].astype(str).unique())
    event_counts = event_scores.groupby("BUILDINGSTRUCTUREID", observed=True)["event_id"].nunique()
    component_range = event_scores[BERI_COMPONENTS].apply(pd.to_numeric, errors="coerce")
    expected_event_score = 100.0 * component_range.prod(axis=1) ** (1.0 / len(BERI_COMPONENTS))
    event_formula_error = (
        expected_event_score - numeric(event_scores["event_response_score"])
    ).abs()

    reference_components = reference[BERI_COMPONENTS].apply(pd.to_numeric, errors="coerce")
    expected_reference_score = 100.0 * reference_components.prod(axis=1) ** (
        1.0 / len(BERI_COMPONENTS)
    )
    reference_formula_error = (
        expected_reference_score - numeric(reference["building_energy_resilience_index"])
    ).abs()
    medians = (
        event_scores.loc[event_scores["event_id"].isin(reference_events)]
        .groupby("BUILDINGSTRUCTUREID", as_index=False, observed=True)[BERI_COMPONENTS]
        .median()
    )
    component_check = reference[["BUILDINGSTRUCTUREID", *BERI_COMPONENTS]].merge(
        medians,
        on="BUILDINGSTRUCTUREID",
        how="outer",
        validate="one_to_one",
        suffixes=("_reference", "_event_median"),
        indicator=True,
    )
    component_errors = {
        field: float(
            (
                numeric(component_check[f"{field}_reference"])
                - numeric(component_check[f"{field}_event_median"])
            ).abs().max()
        )
        for field in BERI_COMPONENTS
    }
    stage_labels = {
        "score_normal": "normal_supply_adaptation",
        "score_capacity": "cooling_service_adequacy",
        "score_outage": "outage_thermal_safety",
        "score_recovery": "post_outage_recovery",
    }
    expected_limiting = reference_components.idxmin(axis=1).map(stage_labels)
    limiting_mismatches = int(
        (expected_limiting.astype(str) != reference["limiting_resilience_stage"].astype(str)).sum()
    )
    capacity = effects.loc[effects["supply_stage"].eq("capacity_70pct")]
    cooling_balance_error = (
        numeric(capacity["required_cooling_energy_kwh_th"])
        - numeric(capacity["delivered_cooling_energy_kwh_th"])
        - numeric(capacity["unmet_cooling_energy_kwh_th"])
    ).abs()
    weights_valid = all(
        len(weights) == len(BERI_COMPONENTS)
        and np.isclose(np.asarray(weights, dtype=float).sum(), 1.0)
        and np.asarray(weights, dtype=float).min() >= 0.0
        for weights in CONFIG["resilience"]["beri_weight_scenarios"].values()
    )
    checks = {
        "all_physical_control_stages_present": required_stages.issubset(observed_stages),
        "reference_event_set_exact": sorted(event_scores["event_id"].astype(str).unique()) == sorted(reference_events),
        "every_building_has_every_reference_event": bool(event_counts.eq(len(reference_events)).all()),
        "event_components_finite_and_bounded": bool(
            np.isfinite(component_range.to_numpy()).all()
            and component_range.ge(0.0).all().all()
            and component_range.le(1.0).all().all()
        ),
        "event_geometric_mean_identity": bool(event_formula_error.le(1e-10).all()),
        "reference_unique_by_building": not bool(reference["BUILDINGSTRUCTUREID"].duplicated().any()),
        "reference_components_are_event_medians": bool(
            component_check["_merge"].eq("both").all()
            and max(component_errors.values(), default=0.0) <= 1e-12
        ),
        "reference_geometric_mean_identity": bool(reference_formula_error.le(1e-10).all()),
        "limiting_stage_identity": limiting_mismatches == 0,
        "capacity_cooling_balance": bool(cooling_balance_error.le(1e-6).all()),
        "weight_scenarios_valid": bool(weights_valid),
    }
    audit = {
        "buildings": int(reference["BUILDINGSTRUCTUREID"].nunique()),
        "reference_events": reference_events,
        "observed_supply_stages": sorted(observed_stages),
        "maximum_event_formula_error": float(event_formula_error.max()),
        "maximum_reference_formula_error": float(reference_formula_error.max()),
        "maximum_reference_component_median_error": component_errors,
        "limiting_stage_mismatches": limiting_mismatches,
        "maximum_capacity_cooling_balance_error_kwh_th": float(cooling_balance_error.max()),
        "checks": checks,
        "pass": bool(all(checks.values())),
    }
    if not audit["pass"]:
        failures = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"BERI contract audit failed: {failures}")
    return audit


def summarize_outcomes(frame: pd.DataFrame, grouping: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(grouping, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        prefix = dict(zip(grouping, keys))
        for outcome in DIRECT_OUTCOMES:
            values = numeric(group[outcome]).dropna()
            if values.empty:
                continue
            rows.append(
                {
                    **prefix,
                    "outcome": outcome,
                    "aggregation_type": "additive" if outcome in ADDITIVE_OUTCOMES else "building_distribution_only",
                    "building_results": int(len(values)),
                    "sum": float(values.sum()) if outcome in ADDITIVE_OUTCOMES else np.nan,
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "p05": float(values.quantile(0.05)),
                    "p95": float(values.quantile(0.95)),
                }
            )
    return pd.DataFrame(rows)


def top_fraction_capture(frame: pd.DataFrame, fraction: float) -> pd.DataFrame:
    grouping = ["event_id", "climate_period", "climate_scenario", "supply_stage"]
    rows = []
    outcomes = [
        "heatwave_incremental_energy_kwh",
        "incremental_peak_kw",
        "overheating_degree_hours",
        "population_weighted_overheating_person_hours",
        "occupied_service_hours_lost",
    ]
    for keys, group in frame.groupby(grouping, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        count = max(1, int(math.ceil(len(group) * fraction)))
        for selector in outcomes:
            selector_values = numeric(group[selector]).clip(lower=0).fillna(0.0)
            if float(selector_values.sum()) <= 0:
                continue
            selected_index = selector_values.sort_values(ascending=False, kind="mergesort").head(count).index
            for captured in outcomes:
                captured_values = numeric(group[captured]).clip(lower=0).fillna(0.0)
                total = float(captured_values.sum())
                rows.append(
                    {
                        **dict(zip(grouping, keys)),
                        "selection_outcome": selector,
                        "captured_outcome": captured,
                        "selected_buildings": count,
                        "selected_fraction": fraction,
                        "captured_share": float(captured_values.loc[selected_index].sum() / total) if total > 0 else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def beri_risk_top_fraction_capture(
    frame: pd.DataFrame,
    beri: pd.DataFrame,
    building_inputs: pd.DataFrame,
    fraction: float,
) -> pd.DataFrame:
    """Quantify burdens covered by the lowest-BERI building fraction.

    Individual building peak demands are additive only as a capacity-burden
    inventory. They are deliberately not labelled as the city coincident peak,
    which is calculated from the separate hourly aggregation.
    """
    population_field = (
        "allocated_population__" + CONFIG["population"]["primary_allocation"]
    )
    required_inputs = {"BUILDINGSTRUCTUREID", population_field}
    missing_inputs = sorted(required_inputs - set(building_inputs.columns))
    if missing_inputs:
        raise ValueError(
            f"Building inputs lack BERI capture fields: {missing_inputs}"
        )
    population = building_inputs[
        ["BUILDINGSTRUCTUREID", population_field]
    ].copy()
    population["BUILDINGSTRUCTUREID"] = population["BUILDINGSTRUCTUREID"].astype(str)
    population[population_field] = numeric(population[population_field]).clip(lower=0).fillna(0.0)
    if population.duplicated("BUILDINGSTRUCTUREID").any():
        conflicts = population.groupby("BUILDINGSTRUCTUREID")[population_field].nunique()
        if bool(conflicts.gt(1).any()):
            raise ValueError("Building inputs contain conflicting population values")
        population = population.drop_duplicates("BUILDINGSTRUCTUREID")
    population = population.set_index("BUILDINGSTRUCTUREID")[population_field]

    effects = frame.copy()
    effects["BUILDINGSTRUCTUREID"] = effects["BUILDINGSTRUCTUREID"].astype(str)
    risk = beri.copy()
    risk["BUILDINGSTRUCTUREID"] = risk["BUILDINGSTRUCTUREID"].astype(str)
    if risk["BUILDINGSTRUCTUREID"].duplicated().any():
        raise ValueError("Pre-event reference BERI must be unique by building")
    risk = risk.sort_values(
        ["building_energy_resilience_index", "BUILDINGSTRUCTUREID"],
        ascending=[True, True],
        kind="mergesort",
    )
    count = max(1, int(math.ceil(len(risk) * fraction)))
    selected_ids = set(risk.head(count)["BUILDINGSTRUCTUREID"])
    score_cutoff = float(risk.head(count)["building_energy_resilience_index"].max())
    strict_ids = set(
        risk.loc[
            risk["building_energy_resilience_index"].lt(score_cutoff),
            "BUILDINGSTRUCTUREID",
        ]
    )
    tied_ids = set(
        risk.loc[
            risk["building_energy_resilience_index"].eq(score_cutoff),
            "BUILDINGSTRUCTUREID",
        ]
    )
    tied_slots = count - len(strict_ids)
    tie_selection_probability = tied_slots / len(tied_ids)
    rows = []
    for event_id in sorted(effects["event_id"].astype(str).unique()):
        event_effects = effects.loc[effects["event_id"].eq(event_id)]

        def stage_values(stage: str, column: str) -> pd.Series:
            subset = event_effects.loc[
                event_effects["supply_stage"].eq(stage),
                ["BUILDINGSTRUCTUREID", column],
            ].copy()
            if subset["BUILDINGSTRUCTUREID"].duplicated().any():
                raise ValueError(
                    f"{event_id}/{stage}/{column} is not unique by building"
                )
            values = numeric(subset[column]).clip(lower=0).fillna(0.0)
            return pd.Series(
                values.to_numpy(),
                index=subset["BUILDINGSTRUCTUREID"],
                dtype=float,
            )

        normal_incremental_energy = stage_values(
            "normal", "heatwave_incremental_energy_kwh"
        )
        normal_heatwave_peak = stage_values("normal", "heatwave_peak_kw")
        normal_incremental_peak = stage_values("normal", "incremental_peak_kw")
        capacity_overheating = stage_values(
            "capacity_70pct", "overheating_degree_hours"
        )
        capacity_population_overheating = stage_values(
            "capacity_70pct", "population_weighted_overheating_person_hours"
        )
        capacity_service_hours_lost = stage_values(
            "capacity_70pct", "occupied_service_hours_lost"
        )
        outage_above_32 = stage_values(
            "outage_peak_24h", "outage_window_degree_hours_above_32c"
        )
        aligned_population = population.reindex(risk["BUILDINGSTRUCTUREID"]).fillna(0.0)
        outage_population_degree_hours = stage_values(
            "outage_peak_24h",
            "residential_population_weighted_outage_degree_hours_above_32c",
        )

        measures = {
            "allocated_residential_population": (
                aligned_population,
                "2021 SSUG-conserved residential population assigned to modeled residential components",
            ),
            "normal_heatwave_incremental_energy_kwh": (
                normal_incremental_energy,
                "positive building heatwave-attributable electricity increment under normal supply",
            ),
            "normal_sum_of_individual_heatwave_peak_kw": (
                normal_heatwave_peak,
                "sum of individual-building heatwave peaks; not the city coincident peak",
            ),
            "normal_positive_individual_peak_increment_kw": (
                normal_incremental_peak,
                "sum of positive individual-building peak increments; not the city coincident peak",
            ),
            "capacity_overheating_degree_hours": (
                capacity_overheating,
                "sum of building-level degree-hours above 26 C during the 70% cooling-capacity case",
            ),
            "capacity_population_weighted_overheating_person_hours": (
                capacity_population_overheating,
                "residential population-weighted occupied overheating person-hours in the 70% capacity case",
            ),
            "capacity_occupied_service_hours_lost": (
                capacity_service_hours_lost,
                "sum of occupied building-hours with cooling service lost in the 70% capacity case",
            ),
            "outage_24h_degree_hours_above_32c": (
                outage_above_32,
                "sum of building-level degree-hours above 32 C during the 24-hour peak outage",
            ),
            "outage_24h_population_degree_hours_above_32c": (
                outage_population_degree_hours,
                "residential-component population-weighted degree-hours above 32 C during the 24-hour peak outage",
            ),
        }
        for metric, (values, definition) in measures.items():
            values = values.groupby(level=0).sum().clip(lower=0).fillna(0.0)
            total = float(values.sum())
            captured = float(values.reindex(list(selected_ids)).fillna(0.0).sum())
            strict_total = float(values.reindex(list(strict_ids)).fillna(0.0).sum())
            tied_values = values.reindex(list(tied_ids)).fillna(0.0)
            tie_adjusted_expected = float(
                strict_total + tie_selection_probability * tied_values.sum()
            )
            tie_adjusted_minimum = float(
                strict_total + tied_values.nsmallest(tied_slots).sum()
            )
            tie_adjusted_maximum = float(
                strict_total + tied_values.nlargest(tied_slots).sum()
            )
            rows.append(
                {
                    "event_id": event_id,
                    "selection_rule": "lowest_pre_event_reference_BERI",
                    "selected_buildings": count,
                    "selected_fraction": fraction,
                    "beri_cutoff_max": score_cutoff,
                    "buildings_strictly_below_cutoff": len(strict_ids),
                    "buildings_tied_at_cutoff": len(tied_ids),
                    "tie_slots_selected": tied_slots,
                    "tie_selection_probability": tie_selection_probability,
                    "captured_metric": metric,
                    "metric_definition": definition,
                    "captured_total": captured,
                    "citywide_total": total,
                    "captured_share": captured / total if total > 0 else np.nan,
                    "tie_adjusted_expected_captured_total": tie_adjusted_expected,
                    "tie_adjusted_minimum_captured_total": tie_adjusted_minimum,
                    "tie_adjusted_maximum_captured_total": tie_adjusted_maximum,
                    "tie_adjusted_expected_captured_share": (
                        tie_adjusted_expected / total if total > 0 else np.nan
                    ),
                    "tie_adjusted_minimum_captured_share": (
                        tie_adjusted_minimum / total if total > 0 else np.nan
                    ),
                    "tie_adjusted_maximum_captured_share": (
                        tie_adjusted_maximum / total if total > 0 else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def beri_leave_one_event_out_validation(
    effects: pd.DataFrame,
    event_scores: pd.DataFrame,
    reference_beri: pd.DataFrame,
    building_inputs: pd.DataFrame,
    fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a rating built from two events against the excluded event.

    The low-score cohort is frozen before the held-out event is inspected.
    Physical burden capture is reported beside single-component, building-type
    and construction-age selection rules. Deterministic and exchangeable-tie
    estimands are both retained at every low-score cutoff.
    """
    components = list(BERI_COMPONENTS)
    population_field = "allocated_population__" + CONFIG["population"]["primary_allocation"]
    input_fields = [
        "BUILDINGSTRUCTUREID", "construction_year_value", "floors_value",
        population_field,
    ]
    missing = sorted(set(input_fields) - set(building_inputs))
    if missing:
        raise ValueError(f"Building inputs lack leave-one-event-out fields: {missing}")
    stock = building_inputs[input_fields].copy()
    stock["BUILDINGSTRUCTUREID"] = stock["BUILDINGSTRUCTUREID"].astype(str)
    if stock.duplicated("BUILDINGSTRUCTUREID").any():
        conflicts = stock.groupby("BUILDINGSTRUCTUREID")[input_fields[1:]].nunique(dropna=False)
        if bool(conflicts.gt(1).any().any()):
            raise ValueError("Building inputs contain conflicting LOEO attributes")
        stock = stock.drop_duplicates("BUILDINGSTRUCTUREID")
    stock = stock.set_index("BUILDINGSTRUCTUREID")
    population = numeric(stock[population_field]).clip(lower=0).fillna(0.0)
    construction_year = numeric(stock["construction_year_value"])
    storeys = numeric(stock["floors_value"]).clip(lower=1.0)
    floor_exposure_ratio = (2.0 / storeys).clip(upper=1.0)

    scores = event_scores.copy()
    scores["BUILDINGSTRUCTUREID"] = scores["BUILDINGSTRUCTUREID"].astype(str)
    physical = effects.copy()
    physical["BUILDINGSTRUCTUREID"] = physical["BUILDINGSTRUCTUREID"].astype(str)
    full_risk = reference_beri[["BUILDINGSTRUCTUREID", "building_energy_resilience_index"]].copy()
    full_risk["BUILDINGSTRUCTUREID"] = full_risk["BUILDINGSTRUCTUREID"].astype(str)
    full_risk = full_risk.sort_values(
        ["building_energy_resilience_index", "BUILDINGSTRUCTUREID"], kind="mergesort"
    )
    selected_count = max(1, int(math.ceil(len(full_risk) * fraction)))

    def cohort_membership(ranking: pd.DataFrame) -> dict[str, object]:
        selected = ranking.head(selected_count)
        cutoff = float(selected["selection_score"].max())
        strict = set(
            ranking.loc[ranking["selection_score"].lt(cutoff), "BUILDINGSTRUCTUREID"]
        )
        tied = set(
            ranking.loc[ranking["selection_score"].eq(cutoff), "BUILDINGSTRUCTUREID"]
        )
        tie_slots = selected_count - len(strict)
        return {
            "selected": set(selected["BUILDINGSTRUCTUREID"]),
            "strict": strict,
            "tied": tied,
            "tie_slots": tie_slots,
            "tie_probability": tie_slots / len(tied),
            "cutoff": cutoff,
        }

    def expected_overlap(left: dict[str, object], right: dict[str, object]) -> float:
        strict_left, tied_left = left["strict"], left["tied"]
        strict_right, tied_right = right["strict"], right["tied"]
        p_left = float(left["tie_probability"])
        p_right = float(right["tie_probability"])
        expected_intersection = (
            len(strict_left & strict_right)
            + p_right * len(strict_left & tied_right)
            + p_left * len(tied_left & strict_right)
            + p_left * p_right * len(tied_left & tied_right)
        )
        return float(expected_intersection / selected_count)

    full_membership = cohort_membership(
        full_risk.rename(columns={"building_energy_resilience_index": "selection_score"})
    )
    full_low_ids = full_membership["selected"]

    capture_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for held_out in sorted(scores["event_id"].astype(str).unique()):
        training_events = sorted(
            scores.loc[scores["event_id"].ne(held_out), "event_id"].astype(str).unique()
        )
        training = (
            scores.loc[scores["event_id"].isin(training_events)]
            .groupby(
                ["BUILDINGSTRUCTUREID", "main_class", "sub_class", "prototype_id"],
                as_index=False,
                observed=True,
            )[components]
            .median()
        )
        training["training_beri"] = 100.0 * training[components].prod(axis=1) ** 0.25
        training = training.set_index("BUILDINGSTRUCTUREID", drop=False)
        test_score = scores.loc[
            scores["event_id"].eq(held_out),
            ["BUILDINGSTRUCTUREID", "event_response_score"],
        ].set_index("BUILDINGSTRUCTUREID")["event_response_score"]
        if len(training) != len(test_score):
            raise ValueError(f"Incomplete BERI LOEO building coverage for {held_out}")
        held_out_ranking = pd.DataFrame({
            "BUILDINGSTRUCTUREID": test_score.index.astype(str),
            "event_response_score": numeric(test_score).to_numpy(),
        }).sort_values(
            ["event_response_score", "BUILDINGSTRUCTUREID"], kind="mergesort"
        )
        held_out_membership = cohort_membership(
            held_out_ranking.rename(columns={"event_response_score": "selection_score"})
        )
        held_out_low_ids = held_out_membership["selected"]

        type_score = training.groupby("main_class", observed=True)["training_beri"].transform("median")
        detailed_use_score = training.groupby("prototype_id", observed=True)[
            "training_beri"
        ].transform("median")
        training_floor_exposure = floor_exposure_ratio.reindex(training.index).fillna(
            floor_exposure_ratio.median()
        )
        detailed_floor_key = pd.DataFrame(
            {
                "prototype_id": training["prototype_id"].astype(str),
                "floor_exposure_ratio": training_floor_exposure.round(6),
            },
            index=training.index,
        )
        detailed_floor_score = training["training_beri"].groupby(
            [
                detailed_floor_key["prototype_id"],
                detailed_floor_key["floor_exposure_ratio"],
            ],
            observed=True,
        ).transform("median")
        selection_scores = {
            "lowest_two_event_BERI": training["training_beri"],
            "lowest_two_event_normal_score": training["score_normal"],
            "lowest_two_event_capacity_score": training["score_capacity"],
            "lowest_two_event_outage_score": training["score_outage"],
            "lowest_two_event_recovery_score": training["score_recovery"],
            "lowest_training_class_median_BERI": type_score,
            "lowest_training_detailed_use_median_BERI": detailed_use_score,
            "lowest_training_detailed_use_floor_exposure_median_BERI": detailed_floor_score,
            "oldest_construction_year": construction_year.reindex(training.index).fillna(
                construction_year.median()
            ),
        }

        event_effects = physical.loc[physical["event_id"].eq(held_out)]

        def stage_values(stage: str, field: str) -> pd.Series:
            subset = event_effects.loc[
                event_effects["supply_stage"].eq(stage), ["BUILDINGSTRUCTUREID", field]
            ]
            if subset["BUILDINGSTRUCTUREID"].duplicated().any():
                raise ValueError(f"Duplicate LOEO field {held_out}/{stage}/{field}")
            return pd.Series(
                numeric(subset[field]).clip(lower=0).fillna(0).to_numpy(),
                index=subset["BUILDINGSTRUCTUREID"],
                dtype=float,
            )

        outage_degree = stage_values(
            "outage_peak_24h", "outage_window_degree_hours_above_32c"
        )
        residential_outage_burden = stage_values(
            "outage_peak_24h",
            "residential_population_weighted_outage_degree_hours_above_32c",
        )
        measures = {
            "normal_incremental_electricity_kwh": stage_values(
                "normal", "heatwave_incremental_energy_kwh"
            ),
            "capacity_unmet_cooling_kwh_th": stage_values(
                "capacity_70pct", "unmet_cooling_energy_kwh_th"
            ),
            "capacity_overheating_degree_hours": stage_values(
                "capacity_70pct", "overheating_degree_hours"
            ),
            "outage_degree_hours_above_32c": outage_degree,
            "outage_attributable_degree_hours_above_26c": stage_values(
                "outage_peak_24h", "outage_attributable_degree_hours_above_26c"
            ),
            "outage_population_degree_hours_above_32c": residential_outage_burden,
            "recovery_time_hours": stage_values(
                "outage_peak_24h", "recovery_time_hours"
            ),
        }
        beri_selected_ids: set[str] | None = None
        beri_membership: dict[str, object] | None = None
        for rule, rule_scores in selection_scores.items():
            ranking = pd.DataFrame(
                {
                    "BUILDINGSTRUCTUREID": training.index.astype(str),
                    "selection_score": numeric(rule_scores).to_numpy(),
                }
            ).sort_values(
                ["selection_score", "BUILDINGSTRUCTUREID"], kind="mergesort"
            )
            membership = cohort_membership(ranking)
            selected_ids = membership["selected"]
            if rule == "lowest_two_event_BERI":
                beri_selected_ids = selected_ids
                beri_membership = membership
            cutoff = float(membership["cutoff"])
            tied_ids = membership["tied"]
            strict_ids = membership["strict"]
            tie_slots = int(membership["tie_slots"])
            tie_probability = float(membership["tie_probability"])
            cutoff_ties = int(len(tied_ids))
            held_out_low_overlap = float(
                len(selected_ids & held_out_low_ids) / selected_count
            )
            tie_aware_held_out_overlap = expected_overlap(
                membership, held_out_membership
            )
            for metric, values in measures.items():
                values = values.groupby(level=0).sum().clip(lower=0).fillna(0)
                total = float(values.sum())
                captured = float(values.reindex(list(selected_ids)).fillna(0).sum())
                strict_total = float(values.reindex(list(strict_ids)).fillna(0).sum())
                tied_values = values.reindex(list(tied_ids)).fillna(0)
                tie_adjusted_expected = float(
                    strict_total + tie_probability * tied_values.sum()
                )
                tie_adjusted_minimum = float(
                    strict_total + tied_values.nsmallest(tie_slots).sum()
                )
                tie_adjusted_maximum = float(
                    strict_total + tied_values.nlargest(tie_slots).sum()
                )
                capture_rows.append(
                    {
                        "held_out_event": held_out,
                        "training_events": "|".join(training_events),
                        "selection_rule": rule,
                        "selected_buildings": selected_count,
                        "selected_fraction": fraction,
                        "selection_cutoff": cutoff,
                        "buildings_tied_at_cutoff": cutoff_ties,
                        "tie_slots_selected": tie_slots,
                        "tie_selection_probability": tie_probability,
                        "lowest_fraction_overlap_with_held_out_event_response": held_out_low_overlap,
                        "tie_aware_expected_overlap_with_held_out_event_response": tie_aware_held_out_overlap,
                        "captured_metric": metric,
                        "captured_total": captured,
                        "citywide_total": total,
                        "captured_share": captured / total if total > 0 else np.nan,
                        "capture_multiple_of_random": (
                            captured / total / fraction if total > 0 else np.nan
                        ),
                        "tie_adjusted_expected_captured_share": (
                            tie_adjusted_expected / total if total > 0 else np.nan
                        ),
                        "tie_adjusted_minimum_captured_share": (
                            tie_adjusted_minimum / total if total > 0 else np.nan
                        ),
                        "tie_adjusted_maximum_captured_share": (
                            tie_adjusted_maximum / total if total > 0 else np.nan
                        ),
                    }
                )
        if beri_selected_ids is None or beri_membership is None:
            raise RuntimeError("BERI LOEO selection was not constructed")
        summary_rows.append(
            {
                "held_out_event": held_out,
                "training_events": "|".join(training_events),
                "chronological_test": held_out == max(scores["event_id"].astype(str).unique()),
                "buildings": int(len(training)),
                "selected_buildings": selected_count,
                "spearman_training_beri_vs_held_out_event_score": float(
                    training["training_beri"].corr(test_score.reindex(training.index), method="spearman")
                ),
                "lowest_decile_overlap_with_full_reference": float(
                    len(beri_selected_ids & full_low_ids) / selected_count
                ),
                "lowest_decile_overlap_with_held_out_event_response": float(
                    len(beri_selected_ids & held_out_low_ids) / selected_count
                ),
                "tie_aware_expected_lowest_decile_overlap_with_full_reference": expected_overlap(
                    beri_membership, full_membership
                ),
                "tie_aware_expected_lowest_decile_overlap_with_held_out_event_response": expected_overlap(
                    beri_membership, held_out_membership
                ),
            }
        )
    return pd.DataFrame(capture_rows), pd.DataFrame(summary_rows)


def recovery_elapsed_and_active_cooling_audit(output: Path) -> tuple[pd.DataFrame, dict]:
    """Separate elapsed restoration time from scheduled cooling opportunity.

    BERI uses elapsed time until the first hourly record at or below 26 C.  This
    audit quantifies how much of that elapsed period actually had scheduled
    cooling available, so night or closure periods are not mistaken for slow
    post-restoration equipment response.
    """
    hourly = pd.read_parquet(
        PROTOTYPE_HOURLY_RESULTS,
        columns=[
            "event_id", "base_case_id", "prototype_id", "climate_case",
            "supply_stage", "timestamp_hkt", "exposure_temperature_c",
            "active_cooling_fraction", "cooling_kwh_m2",
        ],
        filters=[
            ("climate_case", "==", "heatwave"),
            ("supply_stage", "==", "outage_peak_24h"),
        ],
    )
    timing = pd.read_csv(OUTAGE_TIMING_AUDIT)
    timing = timing.loc[timing["supply_stage"].eq("outage_peak_24h"), [
        "event_id", "base_case_id", "prototype_id", "outage_end_exclusive_hkt",
    ]].copy()
    hourly["timestamp_hkt"] = pd.to_datetime(hourly["timestamp_hkt"])
    timing["outage_end_exclusive_hkt"] = pd.to_datetime(timing["outage_end_exclusive_hkt"])
    hourly = hourly.merge(
        timing,
        on=["event_id", "base_case_id", "prototype_id"],
        how="inner",
        validate="many_to_one",
    )
    assessment_hours = 24.0
    comfort_threshold_c = 26.0
    rows: list[dict[str, object]] = []
    for keys, group in hourly.groupby(
        ["event_id", "base_case_id", "prototype_id"], observed=True
    ):
        group = group.sort_values("timestamp_hkt")
        restoration = group["outage_end_exclusive_hkt"].iloc[0]
        recovery_window = group.loc[
            group["timestamp_hkt"].ge(restoration)
            & group["timestamp_hkt"].lt(restoration + pd.Timedelta(hours=assessment_hours))
        ].copy()
        recovered = recovery_window.loc[
            numeric(recovery_window["exposure_temperature_c"]).le(comfort_threshold_c)
        ]
        if recovered.empty:
            elapsed = assessment_hours
            pre_recovery = recovery_window
            censored = True
        else:
            recovered_at = recovered["timestamp_hkt"].iloc[0]
            elapsed = float((recovered_at - restoration).total_seconds() / 3600.0)
            pre_recovery = recovery_window.loc[recovery_window["timestamp_hkt"].lt(recovered_at)]
            censored = False
        rows.append(
            {
                "event_id": str(keys[0]),
                "base_case_id": str(keys[1]),
                "prototype_id": str(keys[2]),
                "elapsed_hours_to_26c": elapsed,
                "scheduled_cooling_hours_before_26c": float(
                    numeric(pre_recovery["active_cooling_fraction"]).clip(0.0, 1.0).sum()
                ),
                "cooling_electricity_before_26c_kwh_m2": float(
                    numeric(pre_recovery["cooling_kwh_m2"]).clip(lower=0.0).sum()
                ),
                "recovery_right_censored_at_24h": censored,
            }
        )
    audit = pd.DataFrame(rows)
    stored = pd.read_csv(PROTOTYPE_RESULTS)
    stored = stored.loc[
        stored["climate_case"].eq("heatwave")
        & stored["supply_stage"].eq("outage_peak_24h"),
        ["event_id", "base_case_id", "prototype_id", "recovery_to_26c_hours"],
    ]
    audit = audit.merge(
        stored,
        on=["event_id", "base_case_id", "prototype_id"],
        how="left",
        validate="one_to_one",
    )
    audit["elapsed_minus_stored_hours"] = (
        audit["elapsed_hours_to_26c"] - numeric(audit["recovery_to_26c_hours"])
    )
    audit.to_csv(
        output / "recovery_elapsed_vs_active_cooling_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        "cases": int(len(audit)),
        "elapsed_hours_median": float(audit["elapsed_hours_to_26c"].median()),
        "scheduled_cooling_hours_before_recovery_median": float(
            audit["scheduled_cooling_hours_before_26c"].median()
        ),
        "right_censored_cases": int(audit["recovery_right_censored_at_24h"].sum()),
        "maximum_absolute_difference_from_stored_elapsed_hours": float(
            audit["elapsed_minus_stored_hours"].abs().max()
        ),
        "beri_recovery_metric": "elapsed hours from restoration to first hourly record at or below 26 C",
        "diagnostic_metric": "scheduled cooling availability accumulated before that record",
        "pass": bool(audit["elapsed_minus_stored_hours"].abs().max() < 1e-9),
    }
    if not summary["pass"]:
        raise ValueError("Recovery elapsed-time audit does not reproduce stored prototype results")
    return audit, summary


def historical_future_recovery_comparability_audit(
    effects: pd.DataFrame,
    output: Path,
) -> dict:
    """Compare historical and future recovery at the same central prototype grain."""
    historical = pd.read_csv(PROTOTYPE_RESULTS)
    future = pd.read_csv(FUTURE_RESULTS)
    central_ids = [f"B{number:04d}" for number in range(1, 15)]
    historical = historical.loc[
        historical["climate_case"].eq("heatwave")
        & historical["supply_stage"].eq("outage_peak_24h")
        & historical["base_case_id"].isin(central_ids)
    ].copy()
    future = future.loc[
        future["supply_stage"].eq("outage_peak_24h")
        & future["base_case_id"].isin(central_ids)
    ].copy()
    expected_prototypes = set(historical["prototype_id"].astype(str))
    if len(expected_prototypes) != 14:
        raise ValueError("Historical central recovery audit does not contain 14 prototypes")
    for keys, group in future.groupby(
        ["experiment_id", "future_period", "source_id"], observed=True
    ):
        if set(group["prototype_id"].astype(str)) != expected_prototypes:
            raise ValueError(f"Future recovery node {keys} lacks the common 14 prototypes")
    rows = []
    for event_id, group in historical.groupby("event_id", observed=True):
        values = numeric(group["recovery_to_26c_hours"])
        rows.append({
            "climate_scope": "observed_weather_central_14_prototypes",
            "event_or_node": str(event_id),
            "experiment_id": "observed",
            "future_period": "observed",
            "source_id": "observed",
            "cases": int(len(group)),
            "recovery_time_median_hours": float(values.median()),
            "recovery_time_minimum_hours": float(values.min()),
            "recovery_time_maximum_hours": float(values.max()),
        })
    for keys, group in future.groupby(
        ["experiment_id", "future_period", "source_id"], observed=True
    ):
        values = numeric(group["recovery_to_26c_hours"])
        rows.append({
            "climate_scope": "future_representative_model_central_14_prototypes",
            "event_or_node": f"{keys[0]}|{keys[1]}|{keys[2]}",
            "experiment_id": str(keys[0]),
            "future_period": str(keys[1]),
            "source_id": str(keys[2]),
            "cases": int(len(group)),
            "recovery_time_median_hours": float(values.median()),
            "recovery_time_minimum_hours": float(values.min()),
            "recovery_time_maximum_hours": float(values.max()),
        })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        output / "historical_future_recovery_comparability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stock_2022 = effects.loc[
        effects["event_id"].eq("HW_20220722_20220729")
        & effects["supply_stage"].eq("outage_peak_24h"),
        "recovery_time_hours",
    ]
    central_2022 = historical.loc[
        historical["event_id"].eq("HW_20220722_20220729"),
        "recovery_to_26c_hours",
    ]
    audit = {
        "shared_prototypes": 14,
        "shared_base_case_ids": central_ids,
        "shared_recovery_definition": "elapsed hours after restoration until hottest occupied conditioned zone first returns to 26 C; right-censored at 24 h",
        "historical_2022_stock_weighted_building_median_hours": float(numeric(stock_2022).median()),
        "historical_2022_equal_prototype_central_case_median_hours": float(numeric(central_2022).median()),
        "future_representative_node_median_range_hours": [
            float(comparison.loc[comparison["climate_scope"].str.startswith("future"), "recovery_time_median_hours"].min()),
            float(comparison.loc[comparison["climate_scope"].str.startswith("future"), "recovery_time_median_hours"].max()),
        ],
        "interpretation": "the 8 h historical value is a stock-weighted building median; matched central-prototype medians are 1 h historically and 1-2 h at future representative nodes",
        "passed": True,
    }
    (output / "historical_future_recovery_comparability_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def future_ensemble_statistics_audit(output: Path) -> pd.DataFrame:
    """Freeze pooled-case and model-median future estimands side by side."""
    if not FUTURE_RESULTS.exists():
        return pd.DataFrame()
    future = pd.read_csv(FUTURE_RESULTS)
    rows = []
    for (scenario, period), group in future.loc[
        future["supply_stage"].eq("normal")
    ].groupby(["experiment_id", "future_period"], observed=True):
        raw = numeric(group["future_incremental_energy_kwh_m2"])
        model_medians = group.groupby("source_id", observed=True)[
            "future_incremental_energy_kwh_m2"
        ].median()
        outage = future.loc[
            future["experiment_id"].eq(scenario)
            & future["future_period"].eq(period)
            & future["supply_stage"].eq("outage_peak_24h")
        ]
        rows.append(
            {
                "scenario": str(scenario),
                "period": str(period),
                "normal_case_rows": int(len(group)),
                "climate_models": int(group["source_id"].nunique()),
                "spatial_nodes": int(group["spatial_node"].nunique()),
                "prototypes": int(group["prototype_id"].nunique()),
                "historical_event_analogues": int(group["event_id"].nunique()),
                "pooled_case_p10_kwh_m2": float(raw.quantile(0.10)),
                "pooled_case_median_kwh_m2": float(raw.median()),
                "pooled_case_p90_kwh_m2": float(raw.quantile(0.90)),
                "model_median_ensemble_p10_kwh_m2": float(model_medians.quantile(0.10)),
                "model_median_ensemble_median_kwh_m2": float(model_medians.median()),
                "model_median_ensemble_p90_kwh_m2": float(model_medians.quantile(0.90)),
                "outage_representative_models": int(outage["source_id"].nunique()),
                "outage_representative_source_ids": "|".join(
                    sorted(outage["source_id"].astype(str).unique())
                ),
                "outage_attributable_degree_hours_c_h_median": float(
                    numeric(outage["outage_attributable_degree_hours_above_26c"]).median()
                ),
                "outage_recovery_to_26c_hours_median": float(
                    numeric(outage["recovery_to_26c_hours"]).median()
                ),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(
        output / "future_ensemble_statistics_audit.csv", index=False, encoding="utf-8-sig"
    )
    return result


def paired_prototype_driver_table() -> pd.DataFrame:
    """Create one physically simulated row per design, event and target."""
    for path in [PROTOTYPE_RESULTS, PROTOTYPE_PARAMETERS]:
        if not path.exists():
            raise FileNotFoundError(f"Driver-analysis input missing: {path}")
    results = pd.read_csv(PROTOTYPE_RESULTS)
    parameters = pd.read_csv(PROTOTYPE_PARAMETERS)
    keys = [
        "event_id", "base_case_id", "prototype_id", "sector", "floor_position",
        "orientation_deg", "behaviour", "setpoint_c", "start_offset_h",
    ]
    baseline = results.loc[
        results["climate_case"].eq("counterfactual")
        & results["supply_stage"].eq("normal")
    ][keys + ["total_electricity_kwh_m2", "peak_electric_w_m2"]].rename(
        columns={
            "total_electricity_kwh_m2": "baseline_energy_kwh_m2",
            "peak_electric_w_m2": "baseline_peak_w_m2",
        }
    )
    targets = []
    stage_targets = {
        "normal": ["total_electricity_kwh_m2", "peak_electric_w_m2"],
        "capacity_70pct": ["all_hour_degree_hours_above_26c", "unmet_cooling_hours"],
        "outage_peak_24h": [
            "outage_attributable_degree_hours_above_26c",
            "outage_peak_temperature_increment_c", "comfort_resistance_hours_to_26c",
            "recovery_to_26c_hours",
        ],
    }
    for stage, fields in stage_targets.items():
        subset = results.loc[
            results["climate_case"].eq("heatwave")
            & results["supply_stage"].eq(stage),
            keys + fields,
        ].copy()
        subset = subset.merge(baseline, on=keys, how="left", validate="one_to_one")
        if stage == "normal":
            subset["relative_energy_increase"] = (
                subset["total_electricity_kwh_m2"] / subset["baseline_energy_kwh_m2"] - 1.0
            )
            subset["peak_amplification_ratio"] = (
                subset["peak_electric_w_m2"] / subset["baseline_peak_w_m2"]
            )
            fields = ["relative_energy_increase", "peak_amplification_ratio"]
        for field in fields:
            part = subset[keys + [field]].rename(columns={field: "target_value"})
            part["target"] = f"{stage}__{field}"
            targets.append(part)
    table = pd.concat(targets, ignore_index=True)
    physics = [
        "u_wall_w_m2k", "u_roof_w_m2k", "window_to_wall_ratio", "window_shgc",
        "window_u_w_m2k",
        "infiltration_ach", "thermal_capacitance_kj_m2k", "baseline_ac_cop",
        "people_per_m2", "lighting_w_m2", "equipment_w_m2",
        "equipment_radiant_fraction", "cooling_capacity_w_m2",
    ]
    table = table.merge(
        parameters[["prototype_id", *physics]],
        on="prototype_id",
        how="left",
        validate="many_to_one",
    )
    table["orientation_sin"] = np.sin(np.deg2rad(numeric(table["orientation_deg"])))
    table["orientation_cos"] = np.cos(np.deg2rad(numeric(table["orientation_deg"])))
    table = table.rename(columns={"target_value": "target_value_raw"})
    table["target_value"] = table.groupby(
        ["target", "event_id"], observed=True
    )["target_value_raw"].rank(method="average", pct=True)
    return table


def summarize_driver_stability(performance: pd.DataFrame) -> pd.DataFrame:
    """Classify cross-event generalization without hiding unsuccessful targets."""
    summary = performance.groupby("target", as_index=False, observed=True).agg(
        held_out_events=("held_out_event", "nunique"),
        median_test_r2=("r2", "median"),
        minimum_test_r2=("r2", "min"),
        maximum_test_r2=("r2", "max"),
        median_test_mae=("mae", "median"),
    )
    summary["cross_event_reliability"] = np.select(
        [
            summary["minimum_test_r2"].ge(0.50),
            summary["minimum_test_r2"].ge(0.0) & summary["median_test_r2"].ge(0.50),
        ],
        ["strong_in_every_event", "consistent_but_variable"],
        default="unstable_across_events",
    )
    summary["main_text_driver_eligible"] = (
        summary["minimum_test_r2"].ge(0.0) & summary["median_test_r2"].ge(0.50)
    )
    summary["classification_rule"] = (
        "eligible when every whole-event holdout R2 is non-negative and median holdout R2 is at least 0.50"
    )
    return summary


def driver_decomposition(output: Path, seed: int) -> dict:
    """Estimate driver importance with whole-event holdout folds."""
    table = paired_prototype_driver_table()
    categorical = ["sector", "floor_position", "behaviour"]
    numeric_features = [
        "setpoint_c", "start_offset_h", "orientation_sin", "orientation_cos",
        "u_wall_w_m2k", "u_roof_w_m2k", "window_to_wall_ratio", "window_shgc",
        "window_u_w_m2k",
        "infiltration_ach", "thermal_capacitance_kj_m2k", "baseline_ac_cop",
        "people_per_m2", "lighting_w_m2", "equipment_w_m2",
        "equipment_radiant_fraction", "cooling_capacity_w_m2",
    ]
    features = categorical + numeric_features
    performance_rows = []
    importance_rows = []
    splitter = GroupKFold(n_splits=3)
    for target, target_frame in table.groupby("target", observed=True):
        target_frame = target_frame.dropna(subset=["target_value", *features]).reset_index(drop=True)
        if target_frame["event_id"].nunique() < 3 or target_frame["target_value"].nunique() < 2:
            continue
        x = target_frame[features]
        y = numeric(target_frame["target_value"]).to_numpy()
        groups = target_frame["event_id"].astype(str).to_numpy()
        for fold, (train, test) in enumerate(splitter.split(x, y, groups), start=1):
            transformer = ColumnTransformer(
                [
                    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
                    ("numeric", "passthrough", numeric_features),
                ]
            )
            model = Pipeline(
                [
                    ("features", transformer),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=300,
                            min_samples_leaf=5,
                            max_features=0.75,
                            random_state=seed + fold,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
            model.fit(x.iloc[train], y[train])
            predicted = model.predict(x.iloc[test])
            held_out_event = str(target_frame.iloc[test]["event_id"].iloc[0])
            performance_rows.append(
                {
                    "target": target,
                    "fold": fold,
                    "held_out_event": held_out_event,
                    "test_rows": int(len(test)),
                    "r2": float(r2_score(y[test], predicted)),
                    "mae": float(mean_absolute_error(y[test], predicted)),
                }
            )
            permuted = permutation_importance(
                model,
                x.iloc[test],
                y[test],
                scoring="r2",
                n_repeats=20,
                random_state=seed + 100 + fold,
                n_jobs=-1,
            )
            for feature, mean, spread in zip(
                features, permuted.importances_mean, permuted.importances_std
            ):
                importance_rows.append(
                    {
                        "target": target,
                        "fold": fold,
                        "held_out_event": held_out_event,
                        "driver": feature,
                        "permutation_r2_drop": float(mean),
                        "permutation_r2_drop_sd": float(spread),
                    }
                )
    performance = pd.DataFrame(performance_rows)
    importance = pd.DataFrame(importance_rows)
    performance.to_csv(output / "driver_event_holdout_performance.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(output / "driver_permutation_importance_by_fold.csv", index=False, encoding="utf-8-sig")
    stability = summarize_driver_stability(performance)
    stability.to_csv(output / "driver_stability_summary.csv", index=False, encoding="utf-8-sig")
    importance_summary = (
        importance.groupby(["target", "driver"], as_index=False, observed=True)
        .agg(
            mean_permutation_r2_drop=("permutation_r2_drop", "mean"),
            min_permutation_r2_drop=("permutation_r2_drop", "min"),
            max_permutation_r2_drop=("permutation_r2_drop", "max"),
        )
        .sort_values(["target", "mean_permutation_r2_drop"], ascending=[True, False])
    )
    importance_summary = importance_summary.merge(
        stability[[
            "target", "median_test_r2", "minimum_test_r2",
            "cross_event_reliability", "main_text_driver_eligible",
        ]],
        on="target",
        how="left",
        validate="many_to_one",
    )
    importance_summary["driver_positive_in_every_event"] = (
        importance_summary["min_permutation_r2_drop"] > 0
    )
    importance_summary.to_csv(output / "driver_decomposition.csv", index=False, encoding="utf-8-sig")
    finding_candidates = importance_summary.loc[
        importance_summary["main_text_driver_eligible"]
        & importance_summary["driver_positive_in_every_event"]
    ].copy()
    finding_candidates["within_target_rank"] = finding_candidates.groupby(
        "target", observed=True
    )["mean_permutation_r2_drop"].rank(method="first", ascending=False).astype(int)
    finding_candidates.sort_values(
        ["target", "within_target_rank"], kind="mergesort"
    ).to_csv(output / "driver_finding_candidates.csv", index=False, encoding="utf-8-sig")
    return {
        "targets": int(performance["target"].nunique()) if not performance.empty else 0,
        "folds_per_target": 3,
        "holdout_unit": "entire_heatwave_event",
        "prediction_target": "within-event percentile of the physical outcome",
        "interpretation": "cross-event stability of building susceptibility; absolute outcome magnitudes are reported separately",
        "median_test_r2": float(performance["r2"].median()) if not performance.empty else None,
        "minimum_test_r2": float(performance["r2"].min()) if not performance.empty else None,
        "main_text_eligible_targets": sorted(
            stability.loc[stability["main_text_driver_eligible"], "target"].tolist()
        ),
        "unstable_targets": sorted(
            stability.loc[~stability["main_text_driver_eligible"], "target"].tolist()
        ),
        "stability_table": "driver_stability_summary.csv",
        "finding_candidate_table": "driver_finding_candidates.csv",
    }


def summarize_future_prototype_response(output: Path) -> dict:
    if not FUTURE_RESULTS.exists():
        return {"available": False}
    future = pd.read_csv(FUTURE_RESULTS)
    grouping = [
        "experiment_id", "future_period", "source_id", "spatial_node",
        "prototype_id", "sector", "supply_stage",
    ]
    fields = [
        "future_incremental_energy_kwh_m2", "future_peak_amplification_ratio",
        "all_hour_degree_hours_above_26c", "night_degree_hours_above_26c",
        "outage_window_degree_hours_above_26c",
        "outage_window_degree_hours_above_32c", "safety_resistance_hours_to_32c",
        "outage_attributable_degree_hours_above_26c",
        "outage_peak_temperature_increment_c",
        "unmet_cooling_hours", "comfort_resistance_hours_to_26c",
        "recovery_to_26c_hours", "recovery_energy_kwh_m2", "cooling_thermal_kwh_m2",
    ]
    summary = future.groupby(grouping, as_index=False, observed=True)[fields].agg(
        ["median", lambda values: values.quantile(0.05), lambda values: values.quantile(0.95)]
    )
    summary.columns = [
        "_".join(str(part) for part in column if str(part))
        .replace("_<lambda_0>", "_p05")
        .replace("_<lambda_1>", "_p95")
        if isinstance(column, tuple) else str(column)
        for column in summary.columns
    ]
    summary.to_csv(output / "future_prototype_stage_summary.csv", index=False, encoding="utf-8-sig")
    return {
        "available": True,
        "rows": int(len(future)),
        "models": int(future["source_id"].nunique()),
        "scenarios": sorted(future["experiment_id"].unique()),
        "periods": sorted(future["future_period"].unique()),
    }


def matched_physical_contrasts(output: Path) -> pd.DataFrame:
    """Extract exact one-factor contrasts already present in the 297-case design.

    Each pair shares prototype, event and every experimental setting except the
    named factor.  The table therefore supports controlled interpretation of
    floor exposure, orientation and cooling setpoint without treating random-
    forest importance as a causal effect.
    """
    raw = pd.read_csv(PROTOTYPE_RESULTS)
    design_fields = [
        "base_case_id", "prototype_id", "sector", "floor_position",
        "orientation_deg", "behaviour", "setpoint_c", "ac_use_probability",
        "start_offset_h",
    ]
    design = raw[design_fields].drop_duplicates("base_case_id")
    if design["base_case_id"].duplicated().any():
        raise ValueError("Controlled contrast design is not unique by base case")

    normal = raw.loc[raw["supply_stage"].eq("normal")].pivot(
        index=["event_id", "base_case_id"],
        columns="climate_case",
        values="total_electricity_kwh_m2",
    )
    if not {"heatwave", "counterfactual"}.issubset(normal.columns):
        raise ValueError("Controlled contrasts require paired normal-supply cases")
    normal["relative_electricity_increase"] = (
        (normal["heatwave"] - normal["counterfactual"])
        / normal["counterfactual"].replace(0.0, np.nan)
    )
    normal = normal[["relative_electricity_increase"]].reset_index()
    capacity = raw.loc[
        raw["climate_case"].eq("heatwave")
        & raw["supply_stage"].eq("capacity_70pct"),
        ["event_id", "base_case_id", "unmet_cooling_hours"],
    ]
    outage = raw.loc[
        raw["climate_case"].eq("heatwave")
        & raw["supply_stage"].eq("outage_peak_24h"),
        [
            "event_id", "base_case_id", "safety_resistance_hours_to_32c",
            "outage_attributable_degree_hours_above_26c",
            "recovery_to_26c_hours",
        ],
    ]
    case = normal.merge(
        capacity, on=["event_id", "base_case_id"], how="inner", validate="one_to_one"
    ).merge(
        outage, on=["event_id", "base_case_id"], how="inner", validate="one_to_one"
    ).merge(design, on="base_case_id", how="left", validate="many_to_one")
    outcomes = [
        "relative_electricity_increase", "unmet_cooling_hours",
        "safety_resistance_hours_to_32c",
        "outage_attributable_degree_hours_above_26c", "recovery_to_26c_hours",
    ]
    factors = ["floor_position", "orientation_deg", "setpoint_c"]
    setting_fields = [
        "floor_position", "orientation_deg", "behaviour", "setpoint_c",
        "ac_use_probability", "start_offset_h",
    ]
    order = {
        "floor_position": {"bottom": 0, "middle": 1, "top": 2, "single": 3},
    }
    rows: list[dict[str, object]] = []
    for factor in factors:
        fixed = ["prototype_id", "event_id"] + [
            field for field in setting_fields if field != factor
        ]
        for _, group in case.groupby(fixed, dropna=False, observed=True):
            if group[factor].nunique(dropna=False) < 2:
                continue
            records = list(group.itertuples(index=False))
            records.sort(
                key=lambda record: order.get(factor, {}).get(
                    str(getattr(record, factor)), getattr(record, factor)
                )
            )
            for lower, upper in itertools.combinations(records, 2):
                for outcome in outcomes:
                    low_value = float(getattr(lower, outcome))
                    high_value = float(getattr(upper, outcome))
                    rows.append(
                        {
                            "factor": factor,
                            "level_from": getattr(lower, factor),
                            "level_to": getattr(upper, factor),
                            "prototype_id": str(lower.prototype_id),
                            "sector": str(lower.sector),
                            "event_id": str(lower.event_id),
                            "base_case_from": str(lower.base_case_id),
                            "base_case_to": str(upper.base_case_id),
                            "outcome": outcome,
                            "value_from": low_value,
                            "value_to": high_value,
                            "difference_to_minus_from": high_value - low_value,
                        }
                    )
    contrasts = pd.DataFrame(rows)
    if contrasts.empty:
        raise ValueError("The frozen design contains no exact one-factor contrasts")
    contrasts.to_csv(
        output / "matched_physical_contrasts.csv", index=False, encoding="utf-8-sig"
    )
    summary = contrasts.groupby(
        ["factor", "level_from", "level_to", "outcome"],
        as_index=False,
        observed=True,
    ).agg(
        matched_pairs=("difference_to_minus_from", "size"),
        median_difference=("difference_to_minus_from", "median"),
        p10_difference=("difference_to_minus_from", lambda values: values.quantile(0.10)),
        p90_difference=("difference_to_minus_from", lambda values: values.quantile(0.90)),
        positive_difference_fraction=(
            "difference_to_minus_from", lambda values: float((values > 0).mean())
        ),
    )
    summary.to_csv(
        output / "matched_physical_contrast_summary.csv", index=False, encoding="utf-8-sig"
    )
    return summary


def write_model_resolution_audit(
    output: Path,
    building_inputs: pd.DataFrame,
    event_scores: pd.DataFrame,
    reference_beri: pd.DataFrame,
) -> dict:
    """Record the spatial inventory and independent thermal-response resolution."""
    stock_rows = int(
        len(pd.read_parquet(
            ROOT / "output" / "heatwave_stock_foundation_v0_3"
            / "building_stock_repaired.geoparquet",
            columns=["BUILDINGSTRUCTUREID"],
        ))
    )
    simulated = building_inputs.loc[
        building_inputs["pre_weather_simulation_candidate"]
    ]
    response_units = pd.read_csv(PROTOTYPE_RESULTS, usecols=[
        "prototype_id", "floor_position",
    ]).drop_duplicates()
    score_fields = [*BERI_COMPONENTS, "event_response_score"]
    audit = {
        "catalogued_hkbfetd_footprints": stock_rows,
        "target_scope_buildings_before_final_qc": int(
            json.loads(
                (RESPONSE_DIR / "final_building_sample_audit.json").read_text(
                    encoding="utf-8"
                )
            )["target_scope_buildings_before_qc"]
        ),
        "simulated_buildings": int(len(simulated)),
        "building_prototypes": int(
            pd.concat([
                simulated["residential_prototype_id"],
                simulated["commercial_prototype_id"],
            ]).dropna().nunique()
        ),
        "prototype_floor_response_units": int(len(response_units)),
        "event_level_unique_stage_score_vectors": int(
            event_scores[score_fields].round(12).drop_duplicates().shape[0]
        ),
        "reference_level_unique_stage_score_vectors": int(
            reference_beri[BERI_COMPONENTS].round(12).drop_duplicates().shape[0]
        ),
        "reference_level_unique_exact_beri_values": int(
            reference_beri["building_energy_resilience_index"].nunique()
        ),
        "interpretation": (
            "Building footprints are individually mapped and area weighted; thermal "
            "response is propagated from prototype-floor ensembles and is not an "
            "independent calibration of every physical building."
        ),
    }
    (output / "model_resolution_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def write_manuscript_key_findings(
    output: Path,
    type_summary: pd.DataFrame,
    class_summary: pd.DataFrame,
    city_summary: pd.DataFrame,
    coincident_peaks: pd.DataFrame,
    event_scores: pd.DataFrame,
    beri: pd.DataFrame,
    beri_sensitivity: pd.DataFrame,
    beri_capture: pd.DataFrame,
    beri_loeo_capture: pd.DataFrame,
    beri_loeo_summary: pd.DataFrame,
    beri_definition_audit: pd.DataFrame,
    cross_stage_audit: pd.DataFrame,
    cross_stage_cutoff_audit: pd.DataFrame,
    future_audit: pd.DataFrame,
) -> None:
    """Freeze the exact aggregations quoted in the manuscript."""
    normal_city = city_summary.loc[city_summary["supply_stage"].eq("normal")]
    normal_peaks = coincident_peaks.loc[
        coincident_peaks["supply_stage"].eq("normal")
    ].set_index("event_id")
    observed_events = []
    for event_id in sorted(normal_city["event_id"].unique()):
        event = normal_city.loc[normal_city["event_id"].eq(event_id)]
        energy = event.loc[event["outcome"].eq("heatwave_incremental_energy_kwh")].iloc[0]
        relative = event.loc[event["outcome"].eq("heatwave_relative_energy_increase")].iloc[0]
        peak = normal_peaks.loc[event_id]
        observed_events.append(
            {
                "event_id": event_id,
                "incremental_energy_gwh": float(energy["sum"]) / 1e6,
                "paired_trajectory_incremental_energy_median_gwh": float(
                    peak["heatwave_incremental_energy_kwh_median"]
                ) / 1e6,
                "paired_trajectory_incremental_energy_p10_gwh": float(
                    peak["heatwave_incremental_energy_kwh_p10"]
                ) / 1e6,
                "paired_trajectory_incremental_energy_p90_gwh": float(
                    peak["heatwave_incremental_energy_kwh_p90"]
                ) / 1e6,
                "building_relative_energy_median_pct": 100.0 * float(relative["median"]),
                "building_relative_energy_p05_pct": 100.0 * float(relative["p05"]),
                "building_relative_energy_p95_pct": 100.0 * float(relative["p95"]),
                "city_coincident_peak_change_mw": float(peak["coincident_peak_increment_kw"]) / 1e3,
                "city_coincident_peak_change_pct": 100.0
                * (float(peak["coincident_peak_amplification_ratio"]) - 1.0),
                "paired_trajectory_coincident_peak_change_pct_median": float(
                    peak["coincident_peak_change_pct_median"]
                ),
                "paired_trajectory_coincident_peak_change_pct_p10": float(
                    peak["coincident_peak_change_pct_p10"]
                ),
                "paired_trajectory_coincident_peak_change_pct_p90": float(
                    peak["coincident_peak_change_pct_p90"]
                ),
            }
        )

    class_energy = event_scores.groupby(
        ["main_class", "event_id"], as_index=False, observed=True
    )["heatwave_relative_energy_increase"].median()
    class_energy["building_median_energy_increase_pct"] = 100.0 * class_energy.pop(
        "heatwave_relative_energy_increase"
    )

    outage = class_summary.loc[
        class_summary["supply_stage"].eq("outage_peak_24h")
        & class_summary["outcome"].eq("outage_attributable_degree_hours_above_26c")
    ]
    outage = outage[["main_class", "event_id", "median", "p05", "p95"]].rename(
        columns={"median": "building_median_outage_attributable_degree_hours_c_h"}
    )
    outage_across_event = outage.groupby(
        "main_class", as_index=False, observed=True
    )["building_median_outage_attributable_degree_hours_c_h"].median()

    driver = pd.read_csv(output / "driver_stability_summary.csv")
    driver_records = driver[
        [
            "target", "median_test_r2", "minimum_test_r2",
            "cross_event_reliability", "main_text_driver_eligible",
        ]
    ].to_dict(orient="records")

    future_records = future_audit.to_dict(orient="records") if len(future_audit) else []

    event_score_city = event_scores.groupby("event_id", as_index=False, observed=True).agg(
        event_response_score_median=("event_response_score", "median"),
        event_response_score_p05=("event_response_score", lambda values: values.quantile(0.05)),
        event_response_score_p95=("event_response_score", lambda values: values.quantile(0.95)),
    )
    beri_ranked = beri.assign(
        _id=beri["BUILDINGSTRUCTUREID"].astype(str)
    ).sort_values(["building_energy_resilience_index", "_id"], kind="mergesort")
    beri_low_count = max(1, int(math.ceil(len(beri_ranked) * float(CONFIG["risk"]["top_fraction"]))))
    beri_low_cutoff = float(
        beri_ranked.iloc[beri_low_count - 1]["building_energy_resilience_index"]
    )
    beri_city = {
        "buildings": int(len(beri)),
        "beri_median": float(numeric(beri["building_energy_resilience_index"]).median()),
        "beri_p05": float(numeric(beri["building_energy_resilience_index"]).quantile(0.05)),
        "beri_p95": float(numeric(beri["building_energy_resilience_index"]).quantile(0.95)),
        "normal_score_median": float(numeric(beri["score_normal"]).median()),
        "capacity_score_median": float(numeric(beri["score_capacity"]).median()),
        "outage_score_median": float(numeric(beri["score_outage"]).median()),
        "recovery_score_median": float(numeric(beri["score_recovery"]).median()),
        "unique_exact_beri_values": int(beri["building_energy_resilience_index"].nunique()),
        "lowest_fraction_buildings": beri_low_count,
        "lowest_fraction_cutoff": beri_low_cutoff,
        "buildings_strictly_below_lowest_fraction_cutoff": int(
            beri["building_energy_resilience_index"].lt(beri_low_cutoff).sum()
        ),
        "buildings_tied_at_lowest_fraction_cutoff": int(
            beri["building_energy_resilience_index"].eq(beri_low_cutoff).sum()
        ),
    }
    beri_class = beri.groupby(
        ["main_class"], as_index=False, observed=True
    ).agg(
        buildings=("BUILDINGSTRUCTUREID", "size"),
        beri_median=("building_energy_resilience_index", "median"),
        beri_p05=("building_energy_resilience_index", lambda values: values.quantile(0.05)),
        beri_p95=("building_energy_resilience_index", lambda values: values.quantile(0.95)),
    )
    payload = {
        "source_tables": {
            "city": "city_event_stage_summary.csv",
            "coincident_peak": "city_coincident_peak_summary.csv",
            "prototype_type": "type_event_stage_summary.csv",
            "driver": "driver_stability_summary.csv",
            "future": "../02_building_response/future_energyplus_results.csv",
            "pre_event_beri": "building_reference_beri.parquet",
            "event_response_score": "building_event_response_score.parquet",
            "beri_weight_sensitivity": "beri_weight_sensitivity_summary.csv",
            "beri_leave_one_event_out_capture": "beri_leave_one_event_out_capture.csv",
            "beri_leave_one_event_out_summary": "beri_leave_one_event_out_summary.csv",
            "beri_definition_sensitivity": "beri_definition_sensitivity.csv",
            "cross_stage_ranking_audit": "cross_stage_ranking_audit.csv",
            "cross_stage_cutoff_sensitivity": "cross_stage_cutoff_sensitivity.csv",
            "event_timing_decomposition": "event_timing_decomposition.csv",
            "class_event_timing_decomposition": "class_event_timing_decomposition.csv",
            "district_spatial_summary": "district_spatial_summary.csv",
            "historical_future_recovery_comparability": "historical_future_recovery_comparability.csv",
            "future_ensemble_statistics_audit": "future_ensemble_statistics_audit.csv",
        },
        "observed_events": observed_events,
        "class_energy": {
            "aggregation": "median across all modelled buildings within class and event",
            "values": class_energy.to_dict(orient="records"),
        },
        "outage_class": {
            "aggregation": "building-level class median reported separately for every event",
            "values_by_event": outage.to_dict(orient="records"),
            "across_event_median_of_building_class_medians": outage_across_event.to_dict(orient="records"),
        },
        "driver_transfer": driver_records,
        "event_timing_and_end_use_attribution": {
            "aggregation": "pointwise prototype-design medians multiplied by component GFA and summed by stock class and timestamp",
            "event": pd.read_csv(output / "event_timing_decomposition.csv").to_dict(orient="records"),
            "class": pd.read_csv(output / "class_event_timing_decomposition.csv").to_dict(orient="records"),
        },
        "district_spatial_results": {
            "selection": "the exact EPSG:2326 map windows used for Ho Man Tin, Mong Kok and Tsim Sha Tsui",
            "values": pd.read_csv(output / "district_spatial_summary.csv").to_dict(orient="records"),
        },
        "historical_future_recovery_comparability": json.loads(
            (output / "historical_future_recovery_comparability_audit.json").read_text(encoding="utf-8")
        ),
        "model_resolution": json.loads(
            (output / "model_resolution_audit.json").read_text(encoding="utf-8")
        ),
        "matched_physical_contrasts": pd.read_csv(
            output / "matched_physical_contrast_summary.csv"
        ).to_dict(orient="records"),
        "building_energy_resilience_index": {
            "role": CONFIG["resilience"]["beri_role"],
            "rating_vintage": CONFIG["resilience"]["beri_rating_vintage"],
            "reference_events": CONFIG["resilience"]["beri_reference_event_ids"],
            "reference_aggregation": CONFIG["resilience"]["beri_reference_event_aggregation"],
            "equal_weight_geometric_mean": True,
            "city_reference": beri_city,
            "class_reference": beri_class.to_dict(orient="records"),
            "minimum_weight_sensitivity_spearman": float(
                beri_sensitivity["spearman_rank_correlation_with_equal"].min()
            ),
            "lowest_beri_top_fraction_capture": beri_capture.to_dict(orient="records"),
            "leave_one_event_out_summary": beri_loeo_summary.to_dict(orient="records"),
            "leave_one_event_out_capture": beri_loeo_capture.to_dict(orient="records"),
            "definition_sensitivity": beri_definition_audit.to_dict(orient="records"),
            "cross_stage_ranking": cross_stage_audit.to_dict(orient="records"),
            "cross_stage_cutoff_sensitivity": cross_stage_cutoff_audit.to_dict(orient="records"),
        },
        "event_response_score": {
            "role": "diagnostic composite for an individual observed heatwave; not the pre-event BERI",
            "city_event": event_score_city.to_dict(orient="records"),
        },
        "future": {
            "primary_energy_estimand": "median across the five climate-model medians; each model median pools its prototype, spatial-node and event-analogue cases",
            "secondary_energy_estimand": "pooled case distribution across prototype, spatial-node, event-analogue and climate-model cases",
            "outage_estimand": "one pre-specified representative climate model per scenario-period node",
            "values": future_records,
        },
    }
    (output / "manuscript_key_findings.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--building-inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--write-contract-only", action="store_true")
    parser.add_argument("--top-fraction", type=float, default=float(CONFIG["risk"]["top_fraction"]))
    parser.add_argument("--seed", type=int, default=int(CONFIG["random_seed"]))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_analysis_contract(output)
    if args.write_contract_only:
        print(f"03 analysis contract complete: {output}")
        return
    if not 0 < args.top_fraction < 1:
        raise ValueError("top_fraction must be between zero and one")
    results_path = args.results.resolve()
    if not results_path.exists():
        raise FileNotFoundError(
            f"Final Result 2 building dynamics not found: {results_path}. "
            "Run 03 --write-contract-only until the calibrated 02 model produces this table."
        )
    inputs_path = args.building_inputs.resolve()
    if not inputs_path.exists():
        raise FileNotFoundError(f"Building input table not found: {inputs_path}")

    frame = pd.read_parquet(results_path)
    validation = validate_results(frame)
    effects = add_paired_effects(frame)
    effects.to_parquet(output / "building_heatwave_energy_resilience.parquet", index=False)
    event_scores, event_score_sensitivity = build_event_response_score(effects)
    event_scores.to_parquet(output / "building_event_response_score.parquet", index=False)
    beri, beri_sensitivity = build_reference_beri(event_scores)
    beri.to_parquet(output / "building_reference_beri.parquet", index=False)
    beri_definition_audit = beri_definition_sensitivity(
        event_scores, beri, args.top_fraction
    )
    beri_definition_audit.to_csv(
        output / "beri_definition_sensitivity.csv", index=False, encoding="utf-8-sig"
    )
    cross_stage_audit = cross_stage_ranking_audit(beri, args.top_fraction)
    cross_stage_audit.to_csv(
        output / "cross_stage_ranking_audit.csv", index=False, encoding="utf-8-sig"
    )
    cross_stage_cutoff_audit = cross_stage_cutoff_sensitivity(
        beri, (0.05, args.top_fraction, 0.20)
    )
    cross_stage_cutoff_audit.to_csv(
        output / "cross_stage_cutoff_sensitivity.csv", index=False, encoding="utf-8-sig"
    )
    beri_contract = audit_beri_contract(effects, event_scores, beri)
    (output / "beri_contract_audit.json").write_text(
        json.dumps(beri_contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    beri_summary = beri.groupby(
        ["main_class", "prototype_id"], as_index=False, observed=True
    ).agg(
        buildings=("BUILDINGSTRUCTUREID", "size"),
        beri_median=("building_energy_resilience_index", "median"),
        beri_p05=("building_energy_resilience_index", lambda values: values.quantile(0.05)),
        beri_p95=("building_energy_resilience_index", lambda values: values.quantile(0.95)),
        normal_score_median=("score_normal", "median"),
        capacity_score_median=("score_capacity", "median"),
        outage_score_median=("score_outage", "median"),
        recovery_score_median=("score_recovery", "median"),
    )
    beri_summary.to_csv(output / "building_energy_resilience_index_summary.csv", index=False, encoding="utf-8-sig")
    beri_sensitivity.to_csv(output / "beri_weight_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    event_score_summary = event_scores.groupby(
        ["main_class", "prototype_id", "event_id"], as_index=False, observed=True
    ).agg(
        buildings=("BUILDINGSTRUCTUREID", "size"),
        event_response_score_median=("event_response_score", "median"),
        event_response_score_p05=("event_response_score", lambda values: values.quantile(0.05)),
        event_response_score_p95=("event_response_score", lambda values: values.quantile(0.95)),
    )
    event_score_summary.to_csv(
        output / "building_event_response_score_summary.csv", index=False, encoding="utf-8-sig"
    )
    event_score_sensitivity.to_csv(
        output / "event_response_weight_sensitivity_summary.csv", index=False, encoding="utf-8-sig"
    )
    type_summary = summarize_outcomes(
        effects,
        ["main_class", "prototype_id", "event_id", "climate_period", "climate_scenario", "supply_stage"],
    )
    type_summary.to_csv(output / "type_event_stage_summary.csv", index=False, encoding="utf-8-sig")
    class_summary = summarize_outcomes(
        effects,
        ["main_class", "event_id", "climate_period", "climate_scenario", "supply_stage"],
    )
    class_summary.to_csv(
        output / "class_event_stage_summary.csv", index=False, encoding="utf-8-sig"
    )
    city_summary = summarize_outcomes(
        effects,
        ["event_id", "climate_period", "climate_scenario", "supply_stage"],
    )
    city_summary.to_csv(output / "city_event_stage_summary.csv", index=False, encoding="utf-8-sig")
    if not COINCIDENT_PEAKS.exists():
        raise FileNotFoundError(f"Coincident city peak summary missing: {COINCIDENT_PEAKS}")
    coincident_peaks = pd.read_csv(COINCIDENT_PEAKS)
    coincident_peaks.to_csv(
        output / "city_coincident_peak_summary.csv", index=False, encoding="utf-8-sig"
    )
    capture = top_fraction_capture(effects, args.top_fraction)
    capture.to_csv(output / "secondary_top_fraction_capture.csv", index=False, encoding="utf-8-sig")
    building_inputs = gpd.read_parquet(inputs_path)
    matched_contrasts = matched_physical_contrasts(output)
    model_resolution = write_model_resolution_audit(
        output, building_inputs, event_scores, beri
    )
    class_hourly, event_timing, class_timing, timing_audit = build_class_hourly_attribution(
        building_inputs
    )
    class_hourly.to_parquet(output / "class_hourly_end_use_attribution.parquet", index=False)
    event_timing.to_csv(
        output / "event_timing_decomposition.csv", index=False, encoding="utf-8-sig"
    )
    class_timing.to_csv(
        output / "class_event_timing_decomposition.csv", index=False, encoding="utf-8-sig"
    )
    (output / "class_hourly_attribution_audit.json").write_text(
        json.dumps(timing_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    district_summary = build_district_spatial_summary(
        building_inputs, effects, beri
    )
    district_summary.to_csv(
        output / "district_spatial_summary.csv", index=False, encoding="utf-8-sig"
    )
    beri_capture = beri_risk_top_fraction_capture(
        effects, beri, building_inputs, args.top_fraction
    )
    beri_capture.to_csv(
        output / "beri_lowest_top_fraction_capture.csv",
        index=False,
        encoding="utf-8-sig",
    )
    beri_loeo_capture, beri_loeo_summary = beri_leave_one_event_out_validation(
        effects, event_scores, beri, building_inputs, args.top_fraction
    )
    beri_loeo_capture.to_csv(
        output / "beri_leave_one_event_out_capture.csv",
        index=False,
        encoding="utf-8-sig",
    )
    beri_loeo_summary.to_csv(
        output / "beri_leave_one_event_out_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    driver_validation = driver_decomposition(output, args.seed)
    future_validation = summarize_future_prototype_response(output)
    future_audit = future_ensemble_statistics_audit(output)
    _, recovery_schedule_audit = recovery_elapsed_and_active_cooling_audit(output)
    recovery_comparability_audit = historical_future_recovery_comparability_audit(
        effects, output
    )
    write_manuscript_key_findings(
        output, type_summary, class_summary, city_summary, coincident_peaks, event_scores,
        beri, beri_sensitivity, beri_capture, beri_loeo_capture, beri_loeo_summary,
        beri_definition_audit, cross_stage_audit, cross_stage_cutoff_audit, future_audit,
    )
    validation.update(
        {
            "primary_result_is_direct_physical_metrics": True,
            "building_energy_resilience_index_role": CONFIG["resilience"]["beri_role"],
            "building_energy_resilience_index_is_pre_event_reference_rating": True,
            "building_energy_resilience_index_rating_vintage": CONFIG["resilience"]["beri_rating_vintage"],
            "building_energy_resilience_index_reference_events": CONFIG["resilience"]["beri_reference_event_ids"],
            "building_energy_resilience_index_rows": int(len(beri)),
            "building_energy_resilience_index_range": [
                float(beri["building_energy_resilience_index"].min()),
                float(beri["building_energy_resilience_index"].max()),
            ],
            "beri_weight_sensitivity_min_spearman": float(
                beri_sensitivity["spearman_rank_correlation_with_equal"].min()
            ),
            "beri_contract": beri_contract,
            "beri_leave_one_event_out_summary": beri_loeo_summary.to_dict(orient="records"),
            "beri_definition_sensitivity": beri_definition_audit.to_dict(orient="records"),
            "cross_stage_ranking": cross_stage_audit.to_dict(orient="records"),
            "cross_stage_cutoff_sensitivity": cross_stage_cutoff_audit.to_dict(orient="records"),
            "class_hourly_attribution": timing_audit,
            "district_spatial_summary_rows": int(len(district_summary)),
            "ranking_reversal_assumed": False,
            "secondary_top_fraction": args.top_fraction,
            "driver_decomposition": driver_validation,
            "future_prototype_response": future_validation,
            "recovery_elapsed_vs_active_cooling": recovery_schedule_audit,
            "historical_future_recovery_comparability": recovery_comparability_audit,
            "matched_physical_contrasts": matched_contrasts.to_dict(orient="records"),
            "model_resolution": model_resolution,
            "outputs": {
                "building": "building_heatwave_energy_resilience.parquet",
                "building_reference_beri": "building_reference_beri.parquet",
                "building_event_response_score": "building_event_response_score.parquet",
                "beri_summary": "building_energy_resilience_index_summary.csv",
                "beri_weight_sensitivity": "beri_weight_sensitivity_summary.csv",
                "beri_contract": "beri_contract_audit.json",
                "beri_leave_one_event_out_capture": "beri_leave_one_event_out_capture.csv",
                "beri_leave_one_event_out_summary": "beri_leave_one_event_out_summary.csv",
                "beri_definition_sensitivity": "beri_definition_sensitivity.csv",
                "cross_stage_ranking": "cross_stage_ranking_audit.csv",
                "cross_stage_cutoff_sensitivity": "cross_stage_cutoff_sensitivity.csv",
                "class_hourly_end_use_attribution": "class_hourly_end_use_attribution.parquet",
                "event_timing_decomposition": "event_timing_decomposition.csv",
                "class_event_timing_decomposition": "class_event_timing_decomposition.csv",
                "class_hourly_attribution_audit": "class_hourly_attribution_audit.json",
                "district_spatial_summary": "district_spatial_summary.csv",
                "future_ensemble_statistics_audit": "future_ensemble_statistics_audit.csv",
                "recovery_elapsed_vs_active_cooling": "recovery_elapsed_vs_active_cooling_audit.csv",
                "historical_future_recovery_comparability": "historical_future_recovery_comparability.csv",
                "historical_future_recovery_comparability_audit": "historical_future_recovery_comparability_audit.json",
                "matched_physical_contrasts": "matched_physical_contrasts.csv",
                "matched_physical_contrast_summary": "matched_physical_contrast_summary.csv",
                "model_resolution_audit": "model_resolution_audit.json",
                "event_response_summary": "building_event_response_score_summary.csv",
                "event_response_weight_sensitivity": "event_response_weight_sensitivity_summary.csv",
                "type": "type_event_stage_summary.csv",
                "class": "class_event_stage_summary.csv",
                "city": "city_event_stage_summary.csv",
                "city_coincident_peak": "city_coincident_peak_summary.csv",
                "secondary_concentration": "secondary_top_fraction_capture.csv",
                "lowest_beri_concentration": "beri_lowest_top_fraction_capture.csv",
                "driver_stability": "driver_stability_summary.csv",
                "driver_findings": "driver_finding_candidates.csv",
                "manuscript_key_findings": "manuscript_key_findings.json",
            },
        }
    )
    validation["citywide_peak_uses_hourly_coincidence"] = True
    validation["sum_of_individual_building_peaks_used_as_city_peak"] = False
    (output / "result3_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"03 complete: {validation['buildings']:,} buildings, "
        f"{validation['events']} events, {output}"
    )


if __name__ == "__main__":
    main()
