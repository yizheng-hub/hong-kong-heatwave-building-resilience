"""Generate the corrected non-solver evidence required before the E8 G3 gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from utils.e8_g3_runner import (  # noqa: E402
    HKUST_CALIBRATION,
    HKUST_TEST,
    load_hkust_validation_weather,
    load_production,
)
from utils.e8_preflight import sha256_file, stable_sha256  # noqa: E402


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def counterfactual_audit(root: Path, work: Path) -> None:
    manuscript = root / "MANUSCRIPT_DRAFT.md"
    source = root / "src" / "00_PREPARE_AND_AUDIT_INPUTS.py"
    legacy_cache = (
        root / "output" / "paper_pipeline_v0_3" / "02_building_response"
        / "energyplus_run_cache" / "energyplus_multizone_counterfactual_resilience_v0_23"
    )
    rows = [
        {
            "layer": "manuscript_v0.3.3",
            "claimed_baseline": "1980-2025 calendar-day climatology",
            "actual_source": str(manuscript),
            "actual_years": "1980-2025 (text claim)",
            "status": "conflicts_with_production",
            "G3_use": "none",
            "cache_effect": "text cannot establish cache compatibility",
        },
        {
            "layer": "production_source",
            "claimed_baseline": "REFERENCE_DAILY_WEATHER",
            "actual_source": str(root / "intermediate" / "source_derived" / "weather" / "hko_kp_daily_2016_2025.csv"),
            "actual_years": "2016-2025",
            "status": "verified_from_source_and_metadata",
            "G3_use": "integration_only",
            "cache_effect": "source used by legacy v0.23 counterfactual construction",
        },
        {
            "layer": "legacy_cache",
            "claimed_baseline": "v0.23 production outputs",
            "actual_source": str(legacy_cache),
            "actual_years": "2016-2025 production reference",
            "status": "not_E8_reusable_for_other_reasons",
            "G3_use": "comparison_only",
            "cache_effect": "subset windows and aggregate schema already prevent E8 reuse",
        },
        {
            "layer": "corrected_G3_candidate",
            "claimed_baseline": "unchanged production algorithm",
            "actual_source": str(root / "intermediate" / "source_derived" / "weather" / "hko_kp_daily_2016_2025.csv"),
            "actual_years": "2016-2025",
            "status": "integration_only_not_main_E8_approved",
            "G3_use": "CF_UNL interface and pairing check only",
            "cache_effect": "new EPW/IDF/schedule bytes have new physics hashes",
        },
        {
            "layer": "E8_main_counterfactual",
            "claimed_baseline": "pending scientific reconciliation",
            "actual_source": "not frozen",
            "actual_years": "not frozen",
            "status": "blocked_for_G4_G5",
            "G3_use": "not inferred from integration result",
            "cache_effect": "a changed reference requires new CF_UNL and dependent demand metrics",
        },
    ]
    frame = pd.DataFrame(rows)
    frame["source_sha256"] = frame["actual_source"].map(
        lambda value: sha256_file(Path(value)) if Path(value).is_file() else ""
    )
    write_csv(frame, work / "00_protocol" / "counterfactual_version_reconciliation.csv")


def weather_audit(root: Path, work: Path) -> None:
    assignment = pd.read_csv(work / "01_inputs" / "E8_event_assignment.csv")
    records = []
    residuals = []
    for event_id in assignment["parent_event_id"]:
        path = next((work / "01_inputs" / "hourly_weather").glob(f"{event_id}_*.parquet"))
        weather = pd.read_parquet(path)
        weather["timestamp_hkt"] = pd.to_datetime(weather["timestamp_hkt"], utc=True).dt.tz_convert(
            "Asia/Hong_Kong"
        )
        weather["date"] = weather["timestamp_hkt"].dt.tz_localize(None).dt.normalize()
        daily = weather.groupby("date").agg(
            original_tmin_c=("T2M", "min"),
            original_tmean_c=("T2M", "mean"),
            original_tmax_c=("T2M", "max"),
            original_rh_mean_pct=("RH2M", "mean"),
            anchored_tmin_c=("temperature_c", "min"),
            anchored_tmean_c=("temperature_c", "mean"),
            anchored_tmax_c=("temperature_c", "max"),
            anchored_rh_mean_pct=("relative_humidity_pct", "mean"),
        ).reset_index()
        hko = pd.read_csv(
            root / "data" / "weather" / "hko_kp_daily_1992_2025.csv",
            parse_dates=["date"],
        )[["date", "tmin_c", "tmean_c", "tmax_c", "rh_mean_pct"]]
        hko = hko.rename(columns={column: f"hko_{column}" for column in hko if column != "date"})
        daily = daily.merge(hko, on="date", how="left", validate="one_to_one")
        for target in ("tmin_c", "tmean_c", "tmax_c", "rh_mean_pct"):
            daily[f"original_minus_hko_{target}"] = daily[f"original_{target}"] - daily[f"hko_{target}"]
            daily[f"anchored_minus_hko_{target}"] = daily[f"anchored_{target}"] - daily[f"hko_{target}"]
        daily.insert(0, "parent_event_id", event_id)
        residuals.append(daily)
        records.append(
            {
                "parent_event_id": event_id,
                "catalog_station_code": "HKO",
                "catalog_station_role": "event definition",
                "hourly_shape_source": "NASA POWER UTC point 22.311,114.172",
                "temperature_rh_anchor_station": "King's Park (KP)",
                "timezone_rule": "UTC converted to Asia/Hong_Kong before local-date anchoring",
                "radiation_source": "NASA POWER ALLSKY_SFC_SW_DWN",
                "wind_source": "NASA POWER WS10M/WD10M",
                "pressure_source": "NASA POWER PS",
                "precipitation_source": "NASA POWER PRECTOTCORR",
                "duplicate_hours": int(weather["timestamp_hkt"].duplicated().sum()),
                "missing_required_values": int(
                    weather[["temperature_c", "relative_humidity_pct", "ALLSKY_SFC_SW_DWN", "WS10M", "WD10M", "PS", "PRECTOTCORR"]].isna().sum().sum()
                ),
                "max_abs_anchored_daily_temperature_residual_c": float(
                    daily[[
                        "anchored_minus_hko_tmin_c", "anchored_minus_hko_tmean_c",
                        "anchored_minus_hko_tmax_c",
                    ]].abs().to_numpy().max()
                ),
                "max_abs_anchored_daily_rh_residual_pct": float(
                    daily["anchored_minus_hko_rh_mean_pct"].abs().max()
                ),
            }
        )
    write_csv(pd.DataFrame(records), work / "01_inputs" / "weather_source_and_boundary_audit.csv")
    write_csv(pd.concat(residuals, ignore_index=True), work / "01_inputs" / "weather_anchor_residuals.csv")


def hkust_audit(root: Path, work: Path) -> None:
    observed_path = (
        root / "output" / "paper_pipeline_v0_3" / "02_building_response"
        / "hkust_whole_building_hourly_calibration.parquet"
    )
    observed = pd.read_parquet(observed_path)
    observed["timestamp_hkt"] = pd.to_datetime(observed["timestamp_hkt"], format="mixed")
    held = observed.loc[observed["building_split"].eq("building_holdout")].copy()
    rows = []
    split_rows = []
    windows = (
        ("calibration", HKUST_CALIBRATION[0], HKUST_CALIBRATION[1]),
        ("test", HKUST_TEST[0], HKUST_TEST[1]),
    )
    for (name, scope), group in held.groupby(["leaf_building", "building_scope"], sort=True):
        prototype = "res_private_highrise" if scope == "campus_residence" else "com_education"
        for window_name, start, end in windows:
            subset = group.loc[group["timestamp_hkt"].between(start, end)].copy()
            expected = pd.date_range(start, end, freq="h")
            rows.append(
                {
                    "observed_object": name,
                    "observed_scope": scope,
                    "prototype_id": prototype,
                    "window": window_name,
                    "start_hkt": start,
                    "end_hkt": end,
                    "expected_hours": len(expected),
                    "rows": len(subset),
                    "unique_hours": subset["timestamp_hkt"].nunique(),
                    "duplicate_rows": int(subset.duplicated("timestamp_hkt").sum()),
                    "nonmissing_energy_hours": int(subset["energy_kwh"].notna().sum()),
                    "complete": bool(
                        len(subset) == len(expected)
                        and subset["timestamp_hkt"].nunique() == len(expected)
                        and subset["energy_kwh"].notna().all()
                    ),
                    "object_identity": "observed_meter_object_not_asserted_independent_physical_building",
                }
            )
        split_rows.append(
            {
                "case_id": "HKUST_" + str(name),
                "observed_object": name,
                "observed_scope": scope,
                "prototype_id": prototype,
                "calibration_start_hkt": HKUST_CALIBRATION[0],
                "calibration_end_hkt": HKUST_CALIBRATION[1],
                "test_start_hkt": HKUST_TEST[0],
                "test_end_hkt": HKUST_TEST[1],
                "selection_rule": "common complete 14-day calibration followed by non-overlapping complete high-temperature 14-day test; selected before model evaluation",
                "absolute_energy_target": False,
                "indoor_temperature_or_outage_target": False,
                "status": "eligible_relative_shape_peak_timing_and_temperature_sensitivity",
            }
        )
    write_csv(pd.DataFrame(rows), work / "03_validation" / "hkust_window_completeness_audit.csv")
    write_csv(pd.DataFrame(split_rows), work / "03_validation" / "validation_split_manifest_revised.csv")
    weather = load_hkust_validation_weather(root)
    weather_summary = pd.DataFrame(
        [
            {
                "hours": len(weather),
                "start_hkt": weather["timestamp_hkt"].min(),
                "end_hkt": weather["timestamp_hkt"].max(),
                "temperature_rh_pressure_wind_rain": "HKO Co-WIN station 6136",
                "solar": "NASA POWER Sai Kung LST",
                "station_6136_wind_direction_G_hours": 562,
                "station_6136_wind_direction_M_hours": 110,
                "all_numeric_fields_finite": bool(
                    np.isfinite(weather.select_dtypes(include=[np.number]).to_numpy()).all()
                ),
            }
        ]
    )
    write_csv(weather_summary, work / "03_validation" / "hkust_weather_mapping_audit.csv")


def mechanism_and_vintage_audit(root: Path, work: Path) -> None:
    production = load_production(root)
    params = production.prototype_parameter_table()
    design = pd.read_csv(
        root / "output" / "paper_pipeline_v0_3" / "02_building_response"
        / "thermal_experiment_design.csv"
    )
    residential = params.loc[params["sector"].eq("residential")].copy()
    reference_rows = []
    for row in residential.itertuples(index=False):
        candidate = design.loc[
            design["prototype_id"].eq(row.prototype_id)
            & design["orientation_deg"].eq(0)
            & design["floor_position"].eq("middle")
            & design["setpoint_c"].eq(25)
            & design["ac_use_probability"].eq(1)
            & design["start_offset_h"].eq(0)
        ].copy()
        reference_rows.append(
            {
                "prototype_id": row.prototype_id,
                "thermal_capacitance_kj_m2k": row.thermal_capacitance_kj_m2k,
                "cooling_capacity_w_m2": row.cooling_capacity_w_m2,
                "reference_start_offset_h": 0,
                "reference_design_ids": "|".join(candidate["base_case_id"].astype(str)),
                "subsidised_sale_165_source": "prototype_parameter_table baseline" if row.prototype_id == "res_subsidised_sale" else "not_applicable",
                "village_capacity_155_source": "prototype_parameter_table baseline" if row.prototype_id == "res_village_house" else "not_applicable",
                "middle_floor_legal_in_frozen_design": bool(
                    design.loc[design["prototype_id"].eq(row.prototype_id), "floor_position"].eq("middle").any()
                ),
                "mechanism_science_approval": "pending; no mechanism runs authorized",
            }
        )
    write_csv(pd.DataFrame(reference_rows), work / "04_mechanisms" / "mechanism_reference_design_audit.csv")

    final = pd.read_parquet(
        root / "output" / "paper_pipeline_v0_3" / "02_building_response"
        / "all_sector_building_model_inputs.parquet",
        columns=["BUILDINGSTRUCTUREID", "main_class", "gfa_value_m2"],
    )
    final_ids = pd.read_parquet(
        root / "output" / "paper_pipeline_v0_3" / "03_energy_resilience"
        / "building_reference_beri.parquet",
        columns=["BUILDINGSTRUCTUREID"],
    )
    final = final.merge(final_ids.drop_duplicates(), on="BUILDINGSTRUCTUREID", how="inner", validate="one_to_one")
    if len(final) != 330_761:
        raise ValueError(f"expected final 330761-building cohort, found {len(final)}")
    repaired = pd.read_parquet(
        root / "output" / "heatwave_stock_foundation_v0_3"
        / "building_stock_repaired.geoparquet",
        columns=[
            "BUILDINGSTRUCTUREID", "occupation_permit_year", "completion_date_year",
            "compliance_certificate_year", "construction_year_value",
            "construction_year_lower", "construction_year_upper", "construction_year_source",
            "construction_year_quality", "construction_year_conflict",
        ],
    )
    merged = final.merge(repaired, on="BUILDINGSTRUCTUREID", how="left", validate="one_to_one")
    merged["year_value_present"] = merged["construction_year_value"].notna()
    merged["direct_year_evidence"] = merged[
        ["occupation_permit_year", "completion_date_year", "compliance_certificate_year"]
    ].notna().any(axis=1)
    coverage = (
        merged.groupby("main_class", dropna=False)
        .agg(
            buildings=("BUILDINGSTRUCTUREID", "size"),
            gfa_m2=("gfa_value_m2", "sum"),
            year_value_buildings=("year_value_present", "sum"),
            direct_year_evidence_buildings=("direct_year_evidence", "sum"),
            year_conflict_buildings=("construction_year_conflict", lambda x: x.fillna(False).astype(bool).sum()),
        )
        .reset_index()
    )
    coverage["year_value_building_coverage"] = coverage["year_value_buildings"] / coverage["buildings"]
    coverage["direct_year_building_coverage"] = coverage["direct_year_evidence_buildings"] / coverage["buildings"]
    write_csv(coverage, work / "01_inputs" / "construction_year_coverage_by_use.csv")
    source = (
        merged.groupby(["main_class", "construction_year_source", "construction_year_quality"], dropna=False)
        .agg(buildings=("BUILDINGSTRUCTUREID", "size"), gfa_m2=("gfa_value_m2", "sum"))
        .reset_index()
    )
    source["historical_interpretation"] = "existence_sensitivity_only_not_verified_in_use_date"
    write_csv(source, work / "01_inputs" / "construction_year_source_audit.csv")


def approval_and_index(root: Path, work: Path) -> None:
    manifest = pd.read_csv(work / "02_runs" / "g3_manifest_revised.csv")
    approvals = pd.DataFrame(
        [
            ("G3_SCOPE", "approved", "26 unique inputs; maximum 30 EnergyPlus calls; no lockbox/PolyU/G4/G5"),
            ("COUNTERFACTUAL_MAIN_E8", "not_approved", "2016-2025 only integration_only in G3; 1980-2025 manuscript conflict unresolved"),
            ("C70_CONTROL_INTERVAL", "passed_G3", "all four real G3 groups used mutually exclusive dual IdealLoads systems: rated outside the formal event and 70% during the event; zero phase violations and all declared capacity checks passed"),
            ("RECOVERY_PRIMARY", "approved_for_G3", "first return to <=26C; <=26.2C sustained 1h diagnostic only"),
            ("PREHISTORY_QC", "approved_for_G3", "report raw differences against 0.1C, 1%, one 15-min interval"),
            ("HKUST_SCOPE", "approved_for_G3", "two physical inputs; four observed objects; relative shape/peak timing/temperature sensitivity only"),
            ("MECHANISM_LEVELS", "not_approved", "evidence audit only; zero mechanism runs"),
            ("HISTORICAL_STOCK", "not_required_for_fixed_state_G3", "year fields support sensitivity only; no historical city reconstruction"),
        ],
        columns=["approval_id", "status", "decision"],
    )
    approvals["evidence_version"] = stable_sha256(
        manifest[["run_id", "physics_hash", "role", "call_order"]].to_dict(orient="records")
    )
    write_csv(approvals, work / "00_protocol" / "approval_items_revised.csv")
    index = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "g3_manifest_rows": len(manifest),
        "g3_manifest_hash": manifest["g3_manifest_hash"].iloc[0],
        "lockbox_runs": 0,
        "energyplus_calls_at_audit_time": 0,
        "files": {},
    }
    for path in sorted(work.rglob("*.csv")):
        if path.name in {"g3_call_ledger.csv"}:
            continue
        index["files"][str(path.relative_to(work))] = sha256_file(path)
    target = work / "00_protocol" / "g2_correction_evidence_index.json"
    target.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    root = PROJECT_ROOT
    work = root / "intermediate/simulation_outputs/upstream"
    counterfactual_audit(root, work)
    weather_audit(root, work)
    hkust_audit(root, work)
    mechanism_and_vintage_audit(root, work)
    approval_and_index(root, work)


if __name__ == "__main__":
    main()
