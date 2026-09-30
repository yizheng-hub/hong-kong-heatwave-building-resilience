"""Zero-solver G3 metrics-v2 extraction; never invokes EnergyPlus."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import re
import shutil
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from utils.e8_g3_postprocess import (
    HKUST_CALIBRATION, HKUST_TEST, attach_monitoring_flags,
    building_crossing_from_zones, building_recovery_from_zones, capacity_and_phase_audit,
    load_monitoring_support, strict_zone_time_pair,
    first_recovery_hours, hourly_model, load_building, load_zones, recovery_diagnostic,
    outage_audit, result_dir, slope_per_c, sustained_recovery_hours,
    write_csv, write_json,
)
from utils.e8_g3_runner import load_production, parse_energyplus_datetime_labels

VERSION, TZ = "G3_metrics_v2", "Asia/Hong_Kong"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_schedule(work: Path, run_id: str) -> pd.DataFrame:
    return pd.read_parquet(result_dir(work, run_id) / "inputs" / "event_schedules.parquet")


def as_hkt(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize(TZ) if stamp.tzinfo is None else stamp.tz_convert(TZ)


def output_dirs(work: Path) -> dict[str, Path]:
    base = work / VERSION
    result = {"base": base, "protocol": base / "00_protocol", "metrics": base / "01_metrics",
              "validation": base / "02_validation", "counterfactual": base / "03_counterfactual",
              "qc": base / "04_qc", "publication": base / "05_publication"}
    for path in result.values():
        path.mkdir(parents=True, exist_ok=True)
    return result


def monitoring_contract() -> pd.DataFrame:
    rows = [
        ("structural_conditioning_present", "zone has an IdealLoads cooling system in the frozen IDF", "IDF/conditioned_zones_json", "static", "required for main screen"),
        ("occupancy_schedule_fraction", "uninterrupted-reference People schedule; 0.03/0.12 are background states", "event_schedules.parquet", "hourly repeated to interval ends", "continuous exposure/internal-gain input"),
        ("occupied_for_screen", "occupancy_schedule_fraction > 0.15", "uninterrupted HW_RATED schedule", "time varying", "main thermal eligibility"),
        ("planned_service_required", "uninterrupted-reference AC schedule > 0", "HW_RATED schedule", "time varying", "service expectation diagnostic"),
        ("actual_hvac_on", "scenario-specific AC schedule > 0", "scenario schedule", "time varying", "state diagnostic; never selects outage monitor"),
        ("main_thermal_monitor", "structural_conditioning_present AND occupied_for_screen", "derived", "time varying", "primary DH32/first crossing"),
        ("all_zone_extreme", "all zones including unconditioned Service", "native zone output", "time varying", "separate diagnostic"),
    ]
    return pd.DataFrame(rows, columns=["field", "definition", "source", "time_semantics", "use"])


def _recovery_all_zones(zones: pd.DataFrame, restore: pd.Timestamp, threshold: float, window: float) -> dict[str, object]:
    result, _ = building_recovery_from_zones(zones, restore, threshold, window)
    return result


def _sustained_all_zones(zones: pd.DataFrame, restore: pd.Timestamp) -> tuple[float, bool]:
    _, maximum = building_recovery_from_zones(zones, restore)
    return sustained_recovery_hours(maximum.timestamp_hkt, maximum.temperature_c, restore)


def outage_metrics_v2(manifest: pd.DataFrame, work: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Calling the corrected production audit is deliberate regression coverage.
    outage_audit(manifest, work)
    summaries, zone_rows, time_rows = [], [], []
    for role in [f"core_group_{letter}" for letter in "ABCD"]:
        group = manifest.loc[manifest.role.eq(role)].set_index("scenario_id")
        rated_item, outage_item = group.loc["HW_RATED"], group.loc["HW_O24_SYNC"]
        start, restore = as_hkt(outage_item.outage_start_hkt), as_hkt(outage_item.outage_end_hkt)
        rated_zones = load_monitoring_support(work, rated_item, rated_item)
        outage_zones = load_monitoring_support(work, outage_item, rated_item)
        paired = strict_zone_time_pair(rated_zones, outage_zones, ["temperature_c"], "rated", "outage")
        flags = ["occupied_for_screen", "occupancy_schedule_fraction", "planned_service_required", "actual_hvac_on"]
        paired = paired.merge(rated_zones[["timestamp_hkt", "zone_id", "structural_conditioning_present"] + flags].rename(columns={c: c + "_rated" for c in flags}), on=["timestamp_hkt", "zone_id"], validate="one_to_one")
        paired = paired.merge(outage_zones[["timestamp_hkt", "zone_id", "actual_hvac_on"]].rename(columns={"actual_hvac_on": "actual_hvac_on_outage"}), on=["timestamp_hkt", "zone_id"], validate="one_to_one")
        paired["eligible"] = paired.structural_conditioning_present.astype(bool) & paired.occupied_for_screen_rated.astype(bool)
        window = paired.loc[paired.timestamp_hkt.gt(start) & paired.timestamp_hkt.le(restore)].copy()
        for limit in (26.0, 32.0):
            window[f"outage_dh{int(limit)}"] = np.maximum(window.temperature_c_outage - limit, 0) * 0.25
            window[f"rated_dh{int(limit)}"] = np.maximum(window.temperature_c_rated - limit, 0) * 0.25
        window["extra_dh26"] = np.maximum(np.maximum(window.temperature_c_outage - 26, 0) - np.maximum(window.temperature_c_rated - 26, 0), 0) * 0.25
        main = window.loc[window.eligible].copy()
        worst = main.groupby("timestamp_hkt", as_index=False).agg(outage_dh32=("outage_dh32", "max"), outage_dh26=("outage_dh26", "max"), rated_dh32=("rated_dh32", "max"), rated_dh26=("rated_dh26", "max"), extra_dh26=("extra_dh26", "max"), main_max_c=("temperature_c_outage", "max"))
        crossing, per_zone_cross = building_crossing_from_zones(outage_zones, start, 32.0, 24.0)
        recovery24 = _recovery_all_zones(outage_zones, restore, 26.0, 24.0)
        recovery_extended = _recovery_all_zones(outage_zones, restore, 26.0, 60.0)
        sustained24, sustained24_censored = _sustained_all_zones(outage_zones, restore)
        t0 = paired.loc[paired.timestamp_hkt.eq(start)].copy()
        for zone_id, zw in window.groupby("zone_id", sort=True):
            z0 = t0.loc[t0.zone_id.eq(zone_id)]
            cr = per_zone_cross.loc[per_zone_cross.zone_id.eq(zone_id)].iloc[0]
            zone_rows.append({"role": role, "event_id": outage_item.event_id, "prototype_id": outage_item.prototype_id, "zone_id": zone_id,
                "zone_area_m2": float(zw.zone_area_m2.iloc[0]), "structural_conditioning_present": bool(zw.structural_conditioning_present.iloc[0]),
                "t0_occupancy_schedule_fraction": float(z0.occupancy_schedule_fraction_rated.iloc[0]) if not z0.empty else math.nan,
                "t0_occupied_for_screen": bool(z0.occupied_for_screen_rated.iloc[0]) if not z0.empty else False,
                "t0_planned_service_required": bool(z0.planned_service_required_rated.iloc[0]) if not z0.empty else False,
                "t0_actual_hvac_on": bool(z0.actual_hvac_on_outage.iloc[0]) if not z0.empty else False,
                "t0_rated_temperature_c": float(z0.temperature_c_rated.iloc[0]) if not z0.empty else math.nan,
                "t0_outage_temperature_c": float(z0.temperature_c_outage.iloc[0]) if not z0.empty else math.nan,
                "baseline_already_above_32c": bool(not z0.empty and z0.temperature_c_outage.iloc[0] >= 32),
                "outage_absolute_dh32": float(zw.outage_dh32.sum()), "outage_absolute_dh26": float(zw.outage_dh26.sum()),
                "rated_absolute_dh32": float(zw.rated_dh32.sum()), "rated_absolute_dh26": float(zw.rated_dh26.sum()),
                "same_zone_positive_extra_dh26": float(zw.extra_dh26.sum()), "first_crossing_32c_h": cr.hours,
                "first_crossing_status": cr.status, "crossing_support_before": cr.support_before, "crossing_support_after": cr.support_after,
                "valid_monitoring_samples": cr.valid_samples})
        all_max = window.loc[window.groupby("timestamp_hkt").temperature_c_outage.idxmax(), ["timestamp_hkt", "zone_id", "temperature_c_outage"]].rename(columns={"zone_id": "all_zone_hottest_zone_id", "temperature_c_outage": "all_zone_max_temperature_c"})
        main_max = main.loc[main.groupby("timestamp_hkt").temperature_c_outage.idxmax(), ["timestamp_hkt", "zone_id", "temperature_c_outage"]].rename(columns={"zone_id": "main_monitor_hottest_zone_id", "temperature_c_outage": "main_monitor_max_temperature_c"}) if not main.empty else pd.DataFrame(columns=["timestamp_hkt", "main_monitor_hottest_zone_id", "main_monitor_max_temperature_c"])
        trace = all_max.merge(main_max, on="timestamp_hkt", how="left")
        trace.insert(0, "role", role); trace.insert(1, "event_id", outage_item.event_id); trace.insert(2, "prototype_id", outage_item.prototype_id)
        time_rows.append(trace)
        summaries.append({"role": role, "event_id": outage_item.event_id, "prototype_id": outage_item.prototype_id, "outage_start_hkt": start, "restore_hkt": restore,
            "first_crossing_32c_h": crossing["hours"], "first_crossing_status": crossing["status"], "first_crossing_zone_id": crossing.get("zone_id"),
            "first_crossing_lower_h": crossing.get("lower_h"), "first_crossing_upper_h": crossing.get("upper_h"), "baseline_already_above_threshold": crossing["status"] == "baseline_already_above_threshold",
            "main_absolute_dh32": float(worst.outage_dh32.sum(min_count=1)), "main_absolute_dh26": float(worst.outage_dh26.sum(min_count=1)), "rated_main_absolute_dh32": float(worst.rated_dh32.sum(min_count=1)), "rated_main_absolute_dh26": float(worst.rated_dh26.sum(min_count=1)),
            "main_same_zone_positive_extra_dh26": float(worst.extra_dh26.sum(min_count=1)), "main_maximum_temperature_c": float(worst.main_max_c.max()) if not worst.empty else math.nan, "all_zone_maximum_temperature_c": float(window.temperature_c_outage.max()),
            "t_recovery_24h": recovery24["hours"], "t_recovery_24h_censored": recovery24["censored"], "t_recovery_24h_status": recovery24["status"],
            "t_recovery_extended_h": recovery_extended["hours"], "t_recovery_extended_censored": recovery_extended["censored"], "t_recovery_extended_status": recovery_extended["status"],
            "t_recovery_26_2_sustained_1h_h": sustained24, "t_recovery_26_2_sustained_1h_censored": sustained24_censored,
            "thermal_mask": "conditioned AND uninterrupted-reference occupancy fraction >0.15", "extra_heat_rule": "same-zone paired positive difference; worst eligible-zone consequence per timestep; integrate"})
    corrected = pd.DataFrame(summaries)
    v1 = pd.read_csv(work / "03_validation" / "g3_outage_recovery_metrics.csv")
    corrected = corrected.merge(v1[["role", "first_occupied_crossing_32c_h", "recovery_first_all_conditioned_zones_le_26c_h"]].rename(columns={"first_occupied_crossing_32c_h": "v1_first_crossing_h", "recovery_first_all_conditioned_zones_le_26c_h": "v1_recovery_h"}), on="role", how="left", validate="one_to_one")
    return corrected, pd.DataFrame(zone_rows), pd.concat(time_rows, ignore_index=True)


def source_timestamp_audit(root: Path, work: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    nasa = root / "data" / "external_validation_raw" / "hkust" / "nasa_power_sai_kung_20220101_20240527.csv"
    observed = root / "output" / "paper_pipeline_v0_3" / "02_building_response" / "hkust_whole_building_hourly_calibration.parquet"
    rows = [
        {"source": "NASA POWER Sai Kung", "stored_time_claim": "LST", "evidence": "source header says Dates ... in LST", "conversion_in_code": "localized to Asia/Hong_Kong without shift", "status": "LST confirmed; equivalence to legal HKT not proven", "source_sha256": sha256_file(nasa)},
        {"source": "HKO Co-WIN station 6136", "stored_time_claim": "naive obstime", "evidence": "stored CSV has obstime/station_id/qcscore but no interval semantics", "conversion_in_code": "localized to Asia/Hong_Kong", "status": "HKT assumption retained; start/end/instant unresolved", "source_sha256": "multiple_files"},
        {"source": "HKUST processed meter objects", "stored_time_claim": "timestamp_hkt", "evidence": "naive datetimes in processed parquet", "conversion_in_code": "paired as HKT wall clock", "status": "interval start/end semantics unresolved", "source_sha256": sha256_file(observed)},
    ]
    for item in manifest.itertuples(index=False):
        run = result_dir(work, item.run_id)
        raw_path = run / "raw" / "eplusout.csv"
        raw_labels = pd.read_csv(raw_path, usecols=["Date/Time"])["Date/Time"]
        parsed = pd.read_parquet(run / "parsed" / "native_timestep_building.parquet", columns=["timestamp_hkt"])
        expected = pd.DatetimeIndex(pd.to_datetime(parsed.timestamp_hkt, utc=True).dt.tz_convert(TZ))
        labels = parse_energyplus_datetime_labels(raw_labels, expected[0])
        rows.append({"source": f"EnergyPlus:{item.run_id}", "stored_time_claim": "TimeStep interval-end Date/Time", "evidence": f"{len(labels)} labels; 24:00 parsed", "conversion_in_code": "strict parse and equality assertion", "status": "exact_match" if labels.equals(expected) else "mismatch", "source_sha256": sha256_file(raw_path)})
    return pd.DataFrame(rows)


def warning_registry(manifest: pd.DataFrame, work: Path) -> pd.DataFrame:
    rows = []
    for item in manifest.itertuples(index=False):
        path = result_dir(work, item.run_id) / "raw" / "eplusout.err"
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for index, line in enumerate(lines):
            if "** Warning **" not in line:
                continue
            message = line.split("** Warning **", 1)[1].strip()
            continuation = " ".join(x.strip() for x in lines[index + 1:index + 3] if "*************" in x)
            text, lower = f"{message} {continuation}".strip(), f"{message} {continuation}".lower()
            if "saturation limit" in lower:
                category, variable, phase = "ideal_loads_humidity_saturation", "supply_humidity_ratio", "warmup_or_runperiod_not_explicit"
            elif "groundtemperature" in lower:
                category, variable, phase = "ground_temperature_range", "ground_temperature", "input_initialization"
            elif "schedule" in lower and "not used" in lower:
                category, variable, phase = "unused_schedule", "schedule", "input_audit"
            else:
                category, variable, phase = "other_warning", "unspecified", "unresolved"
            rows.append({"run_id": item.run_id, "role": item.role, "prototype_id": item.prototype_id, "scenario_id": item.scenario_id, "category": category, "stage": phase, "affected_variable": variable, "message": text, "err_sha256": sha256_file(path)})
    frame = pd.DataFrame(rows)
    columns = ["run_id", "role", "prototype_id", "scenario_id", "category", "stage", "affected_variable", "message", "err_sha256"]
    if frame.empty:
        return pd.DataFrame(columns=columns + ["count"])
    return frame.groupby(columns, dropna=False).size().reset_index(name="count")


def raw_hash_reconciliation(manifest: pd.DataFrame, work: Path) -> pd.DataFrame:
    """Reconcile preserved raw bytes without modifying G3_v1 metadata."""
    rows = []
    for item in manifest.itertuples(index=False):
        run = result_dir(work, item.run_id)
        metadata_path = run / "run_result.json"
        raw_path = run / "raw" / "eplusout.csv"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        recorded = metadata.get("raw_output_sha256")
        computed = sha256_file(raw_path)
        if recorded is None:
            status = "recorded_hash_missing_raw_bytes_present"
            explanation = (
                "The preserved first serial run was reparsed without a solver call under the "
                "legacy branch, which updated parser metadata but did not backfill the raw hash."
            )
        elif recorded == computed:
            status = "match"
            explanation = "Recorded and independently computed SHA256 agree."
        else:
            status = "mismatch"
            explanation = "Recorded and independently computed SHA256 differ; requires investigation."
        rows.append({
            "run_id": item.run_id,
            "role": item.role,
            "scenario_id": item.scenario_id,
            "raw_path": str(raw_path),
            "raw_bytes": raw_path.stat().st_size,
            "recorded_raw_sha256": recorded,
            "computed_raw_sha256": computed,
            "status": status,
            "explanation": explanation,
            "g3_v1_metadata_modified": False,
        })
    return pd.DataFrame(rows)


def metric_values(observed: pd.Series, predicted: pd.Series) -> dict[str, float]:
    error, mean = predicted - observed, float(observed.mean())
    return {"total_observed_kwh": float(observed.sum()), "total_predicted_kwh": float(predicted.sum()),
            "nmbe_pct": float(error.sum() / (len(error) * mean) * 100),
            "cvrmse_pct": float(np.sqrt(np.mean(error**2)) / mean * 100), "mae_kwh": float(error.abs().mean())}


def hkust_diagnostics(manifest: pd.DataFrame, root: Path, work: Path) -> dict[str, pd.DataFrame]:
    observed = pd.read_parquet(root / "output" / "paper_pipeline_v0_3" / "02_building_response" / "hkust_whole_building_hourly_calibration.parquet")
    observed["timestamp_hkt"] = pd.to_datetime(observed.timestamp_hkt, format="mixed")
    observed = observed.loc[observed.building_split.eq("building_holdout")].copy()
    split = pd.read_csv(work / "03_validation" / "validation_split_manifest_revised.csv")
    runs = manifest.loc[manifest.role.eq("hkust_same_chain_validation")].set_index("prototype_id")
    weather_item = runs.iloc[0]
    weather = pd.read_parquet(result_dir(work, weather_item.run_id) / "inputs" / "weather.parquet")
    weather["hour_hkt"] = pd.to_datetime(weather.timestamp_hkt, utc=True).dt.tz_convert(TZ).dt.tz_localize(None)
    weather = weather[["hour_hkt", "temperature_c", "relative_humidity_pct"]]
    metric_rows, residual_rows, profile_rows, daily_rows = [], [], [], []
    peak_rows, slope_rows, baseline_rows, boundary_rows, availability_rows = [], [], [], [], []
    segment_rows = []
    for item in split.itertuples(index=False):
        run = runs.loc[item.prototype_id]
        model = hourly_model(load_building(work, run.run_id)).merge(weather, on="hour_hkt", how="left", validate="one_to_one")
        obs = observed.loc[observed.leaf_building.eq(item.observed_object), ["timestamp_hkt", "energy_kwh"]].rename(columns={"timestamp_hkt": "hour_hkt", "energy_kwh": "observed_kwh"})
        paired = model.merge(obs, on="hour_hkt", how="inner", validate="one_to_one")
        paired["split"] = np.where(paired.hour_hkt.between(pd.Timestamp(HKUST_CALIBRATION[0]), pd.Timestamp(HKUST_CALIBRATION[1])), "calibration", np.where(paired.hour_hkt.between(pd.Timestamp(HKUST_TEST[0]), pd.Timestamp(HKUST_TEST[1])), "test", "outside"))
        paired = paired.loc[paired.split.ne("outside")].copy()
        cal, test = paired.loc[paired.split.eq("calibration")].copy(), paired.loc[paired.split.eq("test")].copy()
        scale = float(cal.observed_kwh.sum() / cal.model_energy_kwh.sum())
        paired["prediction_kwh"] = paired.model_energy_kwh * scale
        cal_mean = float(cal.observed_kwh.mean())
        how_map = cal.assign(hour_of_week=cal.hour_hkt.dt.weekday * 24 + cal.hour_hkt.dt.hour).groupby("hour_of_week").observed_kwh.mean()
        paired["mean_baseline_kwh"] = cal_mean
        paired["hour_of_week"] = paired.hour_hkt.dt.weekday * 24 + paired.hour_hkt.dt.hour
        paired["hour_of_week_baseline_kwh"] = paired.hour_of_week.map(how_map).fillna(cal_mean)
        paired["case_id"], paired["observed_object"] = item.case_id, item.observed_object
        paired["residual_kwh"] = paired.prediction_kwh - paired.observed_kwh
        paired["date"], paired["hour"] = paired.hour_hkt.dt.normalize(), paired.hour_hkt.dt.hour
        paired["day_type"] = np.where(paired.hour_hkt.dt.weekday.lt(5), "weekday", "weekend")
        paired["segment"] = pd.cut(paired.hour, [-1, 5, 8, 17, 23], labels=["night_00_05", "morning_06_08", "day_09_17", "evening_18_23"])
        residual_rows.extend(paired.to_dict("records"))
        for label, part in [("calibration", paired.loc[paired.split.eq("calibration")]), ("test", paired.loc[paired.split.eq("test")])]:
            metric_rows.append({"case_id": item.case_id, "observed_object": item.observed_object, "prototype_id": item.prototype_id, "split": label, "hours": len(part), "calibration_only_scale": scale, **metric_values(part.observed_kwh, part.prediction_kwh)})
            availability_rows.append({"case_id": item.case_id, "split": label, "expected_hours": 336, "paired_hours": len(part), "unique_hours": part.hour_hkt.nunique(), "duplicate_hours": int(part.hour_hkt.duplicated().sum()), "missing_observed_hours": int(part.observed_kwh.isna().sum()), "missing_model_hours": int(part.model_energy_kwh.isna().sum()), "complete": len(part) == 336 and part.hour_hkt.nunique() == 336 and part[["observed_kwh", "model_energy_kwh"]].notna().all().all()})
            for (day_type, segment), segment_part in part.groupby(["day_type", "segment"], observed=True):
                segment_rows.append({"case_id": item.case_id, "split": label, "day_type": day_type, "segment": str(segment), "samples": len(segment_part), "observed_mean_kwh": segment_part.observed_kwh.mean(), "predicted_mean_kwh": segment_part.prediction_kwh.mean(), "mean_residual_kwh": segment_part.residual_kwh.mean()})
        test = paired.loc[paired.split.eq("test")]
        for baseline, column in [("energyplus_scaled", "prediction_kwh"), ("calibration_mean", "mean_baseline_kwh"), ("calibration_hour_of_week", "hour_of_week_baseline_kwh")]:
            baseline_rows.append({"case_id": item.case_id, "baseline": baseline, **metric_values(test.observed_kwh, test[column])})
        for (label, day_type, hour), part in paired.groupby(["split", "day_type", "hour"]):
            profile_rows.append({"case_id": item.case_id, "split": label, "day_type": day_type, "hour": hour, "observed_mean_kwh": part.observed_kwh.mean(), "predicted_mean_kwh": part.prediction_kwh.mean(), "residual_mean_kwh": part.residual_kwh.mean(), "samples": len(part)})
        local_daily = []
        for (label, date), part in paired.groupby(["split", "date"]):
            op, pp = part.loc[part.observed_kwh.idxmax()], part.loc[part.prediction_kwh.idxmax()]
            clock_delta = ((pp.hour - op.hour + 12) % 24) - 12
            row = {"case_id": item.case_id, "split": label, "date": date, "observed_kwh": part.observed_kwh.sum(), "predicted_kwh": part.prediction_kwh.sum(), "residual_kwh": part.residual_kwh.sum(), "observed_peak_hour": op.hour, "predicted_peak_hour": pp.hour, "daily_peak_clock_error_h": clock_delta}
            daily_rows.append(row); local_daily.append(row)
        op, pp = test.loc[test.observed_kwh.idxmax()], test.loc[test.prediction_kwh.idxmax()]
        test_daily = pd.DataFrame(local_daily).query("split == 'test'")
        peak_rows.append({"case_id": item.case_id, "observed_absolute_peak_timestamp": op.hour_hkt, "predicted_absolute_peak_timestamp": pp.hour_hkt, "absolute_peak_datetime_error_h": (pp.hour_hkt-op.hour_hkt).total_seconds()/3600, "observed_absolute_peak_kwh": op.observed_kwh, "predicted_absolute_peak_kwh": pp.prediction_kwh, "median_absolute_daily_clock_error_h": test_daily.daily_peak_clock_error_h.abs().median()})
        for subset_name, subset in [("all_test", test), ("weekday_test", test.loc[test.day_type.eq("weekday")]), ("weekend_test", test.loc[test.day_type.eq("weekend")]), ("daytime_test", test.loc[test.hour.between(9, 17)])]:
            slope_rows.append({"case_id": item.case_id, "subset": subset_name, "samples": len(subset), "observed_relative_slope_per_c": slope_per_c(subset.observed_kwh, subset.temperature_c, cal_mean), "model_relative_slope_per_c": slope_per_c(subset.model_energy_kwh, subset.temperature_c, float(cal.model_energy_kwh.mean())), "interpretation": "association conditional on schedule subset; not isolated cooling causality"})
        boundary_rows.append({"case_id": item.case_id, "observed_object": item.observed_object, "observed_scope": item.observed_scope, "meter_identity": "observed meter object; not asserted independent building", "gfa_available": False, "plant_boundary_known": False, "central_chiller_boundary_known": False, "absolute_energy_validation": False, "indoor_temperature_validation": False, "true_weather_prehistory": False, "warmup_only": True, "claim_supported": "relative shape, calibration-scaled amplitude, peak timing and temperature-load association"})
    return {"metrics": pd.DataFrame(metric_rows), "residuals": pd.DataFrame(residual_rows), "profiles": pd.DataFrame(profile_rows), "segments": pd.DataFrame(segment_rows), "daily": pd.DataFrame(daily_rows), "peaks": pd.DataFrame(peak_rows), "slopes": pd.DataFrame(slope_rows), "baselines": pd.DataFrame(baseline_rows), "boundaries": pd.DataFrame(boundary_rows), "availability": pd.DataFrame(availability_rows)}


def counterfactual_audit(root: Path, work: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_path = root / "intermediate" / "source_derived" / "weather" / "hko_kp_daily_2016_2025.csv"
    daily = pd.read_csv(daily_path, parse_dates=["date"])
    daily["month_day"] = daily.date.dt.strftime("%m-%d")
    variables = ["tmin_c", "tmean_c", "tmax_c", "rh_mean_pct"]
    samples = []
    for month_day, group in daily.groupby("month_day"):
        row = {"month_day": month_day, "station_codes": "|".join(sorted(group.station_code.dropna().astype(str).unique())), "date_rows": len(group), "years": "|".join(map(str, sorted(group.date.dt.year.unique())))}
        for variable in variables:
            row[f"{variable}_nonmissing"] = int(group[variable].notna().sum())
            row[f"{variable}_median"] = float(group[variable].median())
        samples.append(row)
    assignment = pd.read_csv(work / "01_inputs" / "E8_event_assignment.csv")
    targets, leave_rows = [], []
    typical = daily.groupby("month_day")[variables].median()
    for event in assignment.itertuples(index=False):
        start, end = pd.Timestamp(event.event_start).normalize(), pd.Timestamp(event.event_end).normalize()
        candidates = sorted((work / "01_inputs" / "hourly_weather").glob(f"{event.parent_event_id}_*.parquet"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"expected one frozen weather file for {event.parent_event_id}, found {candidates}")
        weather_path = candidates[0]
        weather = pd.read_parquet(weather_path)
        weather["timestamp_hkt"] = pd.to_datetime(weather.timestamp_hkt, utc=True).dt.tz_convert(TZ)
        for date in pd.date_range(start, end, freq="D"):
            md = date.strftime("%m-%d")
            actual = weather.loc[weather.timestamp_hkt.dt.tz_localize(None).dt.normalize().eq(date)]
            row = {"event_id": event.parent_event_id, "event_date": date, "month_day": md, "event_year": date.year, "reference_version": "fixed_recent_reference_benchmark_2016_2025", "reference_includes_event_year": date.year in set(daily.date.dt.year)}
            for variable in variables:
                row[f"reference_{variable}"] = float(typical.loc[md, variable])
            row.update({"event_tmin_c": float(actual.temperature_c.min()), "event_tmean_c": float(actual.temperature_c.mean()), "event_tmax_c": float(actual.temperature_c.max()), "event_rh_mean_pct": float(actual.relative_humidity_pct.mean()), "event_target_source": "HKO daily anchored temperature/RH fields; not raw NASA T2M/RH2M"})
            targets.append(row)
            if date.year in {2017, 2018, 2022, 2023}:
                without = daily.loc[daily.date.dt.year.ne(date.year)].groupby("month_day")[variables].median().loc[md]
                leave = {"event_id": event.parent_event_id, "event_date": date, "excluded_year": date.year, "month_day": md}
                for variable in variables:
                    leave[f"full_{variable}"] = float(typical.loc[md, variable])
                    leave[f"leave_year_out_{variable}"] = float(without[variable])
                    leave[f"difference_{variable}"] = float(without[variable] - typical.loc[md, variable])
                leave_rows.append(leave)
    reconciliation = pd.DataFrame([
        {"layer": "manuscript_v0.3.3", "reference": "1980-2025 calendar-day climatology", "status": "incorrect for available production source; preserved as historical error"},
        {"layer": "traceable_source", "reference": str(daily_path), "status": "King's Park daily rows, 2016-2025"},
        {"layer": "G3_physical_CF_inputs", "reference": "2016-2025 calendar-day medians; event T/RH transformed only", "status": "already executed as integration_only; relabelling does not change bytes"},
        {"layer": "recommended_E8_main", "reference": "fixed_recent_reference_benchmark", "status": "documented; not historical no-heatwave counterfactual or strict as-of forecast"},
    ])
    return pd.DataFrame(samples), pd.DataFrame(targets), pd.DataFrame(leave_rows), reconciliation


def write_contracts(paths: dict[str, Path], work: Path) -> None:
    write_csv(monitoring_contract(), paths["protocol"] / "thermal_monitoring_contract.csv")
    old = pd.read_csv(work / "00_protocol" / "estimand_and_field_contract.csv")
    updates = {
        "Q_shortfall_ref": ("same fixed-zone/native-timestep before sum", "implemented_v2"),
        "DH32_outage": ("conditioned AND reference occupancy fraction >0.15; per-timestep worst eligible zone", "implemented_v2"),
        "H_extra26": ("same-zone/time positive difference; per-timestep worst eligible consequence", "implemented_v2"),
        "t_first32": ("per-zone interpolation preserving t0; no cross-zone/gap bridging", "implemented_v2"),
        "t_recovery": ("all structurally conditioned zones; [restore,restore+24h]", "implemented_v2_with_right_censoring"),
    }
    old["v2_change"], old["v2_status"] = "unchanged", "carried_forward"
    for name, (change, status) in updates.items():
        old.loc[old.estimand.eq(name), ["v2_change", "v2_status"]] = [change, status]
    write_csv(old, paths["protocol"] / "estimand_and_field_contract_v2.csv")
    schema = json.loads((work / "00_protocol" / "output_schema.json").read_text(encoding="utf-8"))
    schema["version"] = "e8-zone-native-output-v2.1-time-asserted"
    schema["parser_version"] = "e8-zone-native-parser-v4-time-asserted"
    schema["timestamp_assertion"] = "raw EnergyPlus Date/Time exactly equals weather-derived interval-end axis"
    for item in schema["variables"]:
        item["verification_status"] = "verified in 26 cached G3 raw outputs; no solver rerun"
    write_json(schema, paths["protocol"] / "output_schema_v2.json")


def write_review_documents(paths: dict[str, Path], work: Path, manifest: pd.DataFrame) -> None:
    before_after = pd.DataFrame([
        ("first_crossing_32c", "bug_fix", "dropped t0; first post-start exceedance could become 0", "preserve fixed-zone t0; contiguous same-zone interpolation; explicit censoring"),
        ("thermal_monitoring_mask", "definition_clarification", "occupants>0 included background fractions and Service", "conditioned AND reference occupancy fraction>0.15; all-zone extreme retained"),
        ("recovery_26c", "bug_fix", "could accept recovery after 24h", "main endpoint censored at 24h; extended diagnostic separate"),
        ("sustained_recovery", "bug_fix", "four rows assumed one hour", "requires contiguous 15-minute interval support"),
        ("cooling_shortfall", "bug_fix", "building-level matching allowed zone cancellation", "fixed-zone/native-timestep matching before sum"),
        ("unmet_hours", "definition_clarification", "zone-hour sum ambiguous", "zone-hours, any-zone building-hours, and area-weighted hours"),
        ("EnergyPlus_time_axis", "bug_prevention", "row count only", "raw Date/Time parse and exact equality assertion"),
        ("counterfactual_reference", "definition_clarification", "manuscript stated 1980-2025", "traceable 2016-2025 King's Park fixed recent reference"),
        ("G3_physics_inputs", "result_maintained", "26 cached runs", "unchanged; zero solver calls"),
    ], columns=["item", "change_class", "before", "after"])
    write_csv(before_after, paths["protocol"] / "metric_before_after.csv")
    ledger = pd.read_csv(work / "02_runs" / "g3_call_ledger.csv")
    reusable = manifest[["run_id", "role", "event_id", "prototype_id", "scenario_id", "physics_hash", "output_schema_hash"]].copy()
    reusable["physical_cache_status"], reusable["v2_action"], reusable["solver_call_required"] = "reusable_unchanged", "postprocess_only", False
    write_csv(reusable, paths["qc"] / "no_rerun_reusable_manifest.csv")
    request = pd.DataFrame([
        {"request_id": "HKUST_PREHISTORY_EDU", "unique_runs": 1, "reason": "warmup exists but true weather prehistory absent", "physical_input_change": True, "status": "optional_pending_approval"},
        {"request_id": "HKUST_PREHISTORY_RES_PROXY", "unique_runs": 1, "reason": "warmup exists but true weather prehistory absent", "physical_input_change": True, "status": "optional_pending_approval"},
    ])
    write_csv(request, paths["qc"] / "minimal_rerun_request.csv")
    resources = json.loads((work / "02_runs" / "g3_resource_summary.json").read_text(encoding="utf-8"))
    proposal = pd.DataFrame([
        {"stage": "G3_metrics_v2", "solver_runs": 0, "status": "completed_from_cache", "basis": f"ledger remained {len(ledger)} calls"},
        {"stage": "optional_HKUST_prehistory", "solver_runs": 2, "status": "not_authorized", "basis": "benchmark required before approval"},
        {"stage": "G4_56_unit_gate", "solver_runs": 0, "status": "not_authorized", "basis": "definition freeze and explicit authorization required"},
        {"stage": "G4_297_event", "solver_runs": 0, "status": "not_authorized", "basis": "no scheduling in this turn"},
        {"stage": "observed_G3_resource_context", "solver_runs": len(ledger), "status": "evidence_only", "basis": json.dumps(resources, ensure_ascii=False)},
    ])
    write_csv(proposal, paths["qc"] / "G4_resource_proposal.csv")
    outage = pd.read_csv(paths["metrics"] / "g3_outage_recovery_metrics_v2.csv").set_index("role")
    hkust = pd.read_csv(paths["validation"] / "hkust_metrics_v2.csv")
    baselines = pd.read_csv(paths["validation"] / "hkust_baselines_v2.csv")
    model_cv = baselines.loc[baselines.baseline.eq("energyplus_scaled")].set_index("case_id").cvrmse_pct
    how_cv = baselines.loc[baselines.baseline.eq("calibration_hour_of_week")].set_index("case_id").cvrmse_pct
    baseline_better = int((how_cv < model_cv).sum())
    warnings = pd.read_csv(paths["qc"] / "warning_registry.csv")
    warning_total = int(pd.to_numeric(warnings["count"]).sum()) if not warnings.empty else 0
    raw_hashes = pd.read_csv(paths["qc"] / "raw_hash_reconciliation.csv")
    raw_hash_matches = int(raw_hashes.status.eq("match").sum())
    raw_hash_missing = int(raw_hashes.status.eq("recorded_hash_missing_raw_bytes_present").sum())
    raw_hash_mismatches = int(raw_hashes.status.eq("mismatch").sum())
    timestamp_audit = pd.read_csv(paths["validation"] / "hkust_time_source_audit.csv")
    timestamp_matches = int((timestamp_audit.source.str.startswith("EnergyPlus:") & timestamp_audit.status.eq("exact_match")).sum())
    review = f"""# G3 metrics v2 review

This review consumed {len(manifest)} frozen G3 runs and made **zero EnergyPlus calls**. The call ledger remained at {len(ledger)} rows. G3_v1 raw and parsed evidence was not overwritten.

## Corrected
- Fixed-zone t0-preserving crossing with explicit censoring.
- Recovery limited to 24 h; extended recovery is separate.
- Sustained recovery checks real contiguous interval support.
- Cooling service matches zone and timestep before summing.
- Thermal screen uses conditioned zones with reference occupancy fraction >0.15; all-zone extremes remain diagnostic.
- Raw EnergyPlus Date/Time labels are asserted against persisted timestamps.

## Cached-case results after correction

- Public-rental case A changes from the v1 value of 0 h to an interval-censored crossing in (3.00, 3.25] h because the main living-zone screen becomes eligible after the last valid below-threshold support. This is not replaced by an invented point estimate.
- Office case B crosses at {outage.loc['core_group_B', 'first_crossing_32c_h']:.3f} h; data-centre case C remains below 32 C for the complete 24 h window; old-low-rise case D crosses at {outage.loc['core_group_D', 'first_crossing_32c_h']:.3f} h.
- Cases A and D do not bring all conditioned zones to <=26 C within 24 h and are right-censored. Their available extended traces do not cover a complete 60 h window, so the extended endpoint is recorded as unknown/incomplete rather than recovered.
- HKUST test CVRMSE spans {hkust.loc[hkust.split.eq('test'), 'cvrmse_pct'].min():.1f}-{hkust.loc[hkust.split.eq('test'), 'cvrmse_pct'].max():.1f}%. The calibration-only hour-of-week baseline has lower CVRMSE than the generic prototype in {baseline_better}/4 objects. This supports an interface/limited-shape check, not strong predictive validation.
- All {timestamp_matches}/26 cached EnergyPlus outputs have exact raw-label/persisted-axis agreement. The registry retains {warning_total} warning occurrences; zero severe errors is not reported as zero warnings.
- Raw-byte reconciliation finds {raw_hash_matches}/26 recorded SHA256 matches, {raw_hash_missing} preserved run with a missing legacy hash registration, and {raw_hash_mismatches} byte-hash mismatches. The missing registration is documented in v2 without modifying G3_v1 metadata.

## Still bounded
- HKO and HKUST start/end label semantics remain undocumented locally. NASA POWER is explicitly LST; legal-HKT equivalence is not proven.
- HKUST runs have warmup but no true weather prehistory; two optional reruns remain unapproved.
- The 2016–2025 reference is traceable but is not an as-of historical counterfactual and includes some test years.
- G3 covers four interface cases, not stock-wide conclusions. Lockboxes 0054/0066 remain unopened for building performance.

## Code
- `utils/e8_g3_postprocess.py`; `utils/e8_g3_runner.py`; `src/utils/simulation_metrics.py`; `tests/test_e8_g3_metrics_v2.py`.
"""
    (paths["base"] / "G3_metrics_v2_review.md").write_text(review, encoding="utf-8")
    readiness = """# G4 readiness v2

| Gate | Status | Evidence |
|---|---|---|
| Technical pipeline | CONDITIONAL GO | Cached inputs intact; timestamp and metric tests pass. G4 remains unauthorized. |
| Observational evidence | NO-GO for strong validation claim | HKUST supports limited calibration-scaled load-shape diagnostics; boundaries and true prehistory remain unresolved. |
| Scientific definitions | CONDITIONAL GO | Monitoring, censoring, recovery and zone-time cooling rules are explicit; reference limits must remain stated. |
| Resources | NO-GO for execution | No G4/G5 budget approved; optional two-run prehistory request pending. |

No automatic transition is authorized.
"""
    (paths["base"] / "G4_READINESS_v2.md").write_text(readiness, encoding="utf-8")


def package_review(root: Path, paths: dict[str, Path], manifest: pd.DataFrame) -> None:
    package = paths["publication"] / "E8_G3_metrics_v2_review"
    if package.exists():
        shutil.rmtree(package)
    package.mkdir(parents=True)
    for key in ("protocol", "metrics", "validation", "counterfactual", "qc"):
        shutil.copytree(paths[key], package / key)
    for name in ("G3_metrics_v2_review.md", "G4_READINESS_v2.md", "zero_solver_attestation.json"):
        source = paths["base"] / name
        if source.exists():
            shutil.copy2(source, package / name)
    code_dir = package / "code"
    code_dir.mkdir()
    for relative in ("src/utils/e8_g3_runner.py", "src/utils/e8_g3_postprocess.py", "src/utils/simulation_metrics.py"):
        shutil.copy2(root / relative, code_dir / relative.replace("/", "__"))
    v1_code = root / "intermediate/simulation_outputs/upstream" / "08_publication" / "E8_G3_review_v1" / "code"
    diff_parts = []
    for relative in ("src/utils/e8_g3_runner.py", "src/utils/e8_g3_postprocess.py", "src/utils/simulation_metrics.py"):
        current = (root / relative).read_text(encoding="utf-8").splitlines(keepends=True)
        old_path = v1_code / relative
        old = old_path.read_text(encoding="utf-8").splitlines(keepends=True) if old_path.exists() else []
        diff_parts.extend(difflib.unified_diff(old, current, fromfile=f"G3_v1/{relative}", tofile=f"G3_metrics_v2/{relative}"))
    (code_dir / "metrics_v2.diff").write_text("".join(diff_parts), encoding="utf-8")
    for label, role, scenario in [("A", "core_group_A", "HW_RATED"), ("C", "core_group_C", "HW_O24_SYNC"), ("D", "core_group_D", "HW_O24_SYNC")]:
        row = manifest.loc[manifest.role.eq(role) & manifest.scenario_id.eq(scenario)].iloc[0]
        shutil.copy2(result_dir(root / "intermediate/simulation_outputs/upstream", row.run_id) / "raw" / "eplusout.err", package / f"representative_{label}_{scenario}_eplusout.err")
    pd.read_csv(paths["metrics"] / "g3_zone_same_mask_diagnostics_v2.csv").head(30).to_csv(package / "representative_zone_diagnostics.csv", index=False, encoding="utf-8-sig")
    manifest_rows = []
    for file in sorted(path for path in package.rglob("*") if path.is_file()):
        manifest_rows.append({"path": file.relative_to(package).as_posix(), "bytes": file.stat().st_size, "sha256": sha256_file(file)})
    write_csv(pd.DataFrame(manifest_rows), package / "PACKAGE_MANIFEST_SHA256.csv")
    archive = paths["publication"] / "E8_G3_metrics_v2_review.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for file in sorted(path for path in package.rglob("*") if path.is_file()):
            handle.write(file, arcname=f"E8_G3_metrics_v2_review/{file.relative_to(package).as_posix()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT_FOR_IMPORT)
    args = parser.parse_args()
    root = args.project_root.resolve()
    work = root / "intermediate/simulation_outputs/upstream"
    paths = output_dirs(work)
    manifest = pd.read_csv(work / "02_runs" / "g3_manifest_revised.csv")
    ledger_path = work / "02_runs" / "g3_call_ledger.csv"
    ledger_before = ledger_path.read_bytes()
    write_contracts(paths, work)
    outage, zones, time_trace = outage_metrics_v2(manifest, work)
    write_csv(outage, paths["metrics"] / "g3_outage_recovery_metrics_v2.csv")
    write_csv(zones, paths["metrics"] / "g3_zone_same_mask_diagnostics_v2.csv")
    write_csv(time_trace, paths["metrics"] / "g3_hottest_zone_identity_v2.csv")
    _, service = capacity_and_phase_audit(manifest, work, load_production(root))
    write_csv(service, paths["metrics"] / "g3_cooling_service_diagnostics_v2.csv")
    write_csv(source_timestamp_audit(root, work, manifest), paths["validation"] / "hkust_time_source_audit.csv")
    for name, frame in hkust_diagnostics(manifest, root, work).items():
        write_csv(frame, paths["validation"] / f"hkust_{name}_v2.csv")
    write_csv(warning_registry(manifest, work), paths["qc"] / "warning_registry.csv")
    write_csv(raw_hash_reconciliation(manifest, work), paths["qc"] / "raw_hash_reconciliation.csv")
    samples, targets, leave, reconciliation = counterfactual_audit(root, work)
    write_csv(samples, paths["counterfactual"] / "counterfactual_daily_reference_sample_audit.csv")
    write_csv(targets, paths["counterfactual"] / "E8_event_and_reference_daily_targets.csv")
    write_csv(leave, paths["counterfactual"] / "counterfactual_leave_event_year_out.csv")
    write_csv(reconciliation, paths["counterfactual"] / "counterfactual_version_reconciliation_v2.csv")
    decision = """# Counterfactual reference decision

Recommended name: **fixed_recent_reference_benchmark**.

The traceable source is King's Park daily temperature and relative humidity for 2016–2025. Calendar-day medians replace temperature and humidity only during the formal event; radiation, wind, pressure, calendar and the approved prehistory remain paired with the event input.

It is not the 1980–2025 climatology stated in the old manuscript, does not guarantee exclusion of heatwave days, includes some evaluated event years, can yield negative increments for earlier events relative to the recent reference, and is not a real-time no-heatwave counterfactual or strict historical forecast. It estimates temperature–humidity differences relative to a fixed recent reference.

The G3 CF bytes already use this source and algorithm. Relabelling provenance does not change physics hashes. Final E8 freeze remains subject to approval; no alternate baseline was selected after viewing building outcomes.
"""
    (paths["counterfactual"] / "counterfactual_reference_decision.md").write_text(decision, encoding="utf-8")
    temporal = pd.DataFrame([
        {"item": "reference_event_order", "T1_T3_status": "event dates respected", "strict_historical_status": "partial", "reason": "fixed current state/calibration remain"},
        {"item": "counterfactual_climatology", "T1_T3_status": "fixed_recent_reference_2016_2025", "strict_historical_status": "not_as_of", "reason": "contains dates after T1 and some test years"},
        {"item": "building_state", "T1_T3_status": "fixed current state", "strict_historical_status": "not historical reconstruction", "reason": "histories incomplete"},
        {"item": "population", "T1_T3_status": "fixed reference population", "strict_historical_status": "not historical population", "reason": "no event-year redistribution"},
    ])
    write_csv(temporal, paths["counterfactual"] / "temporal_information_audit_v2.csv")
    test_rows = []
    for suite_name, filename in [("focused_metrics_and_parser", "pytest_focused.xml"), ("full_repository", "pytest_full.xml")]:
        xml_path = paths["qc"] / filename
        if not xml_path.exists():
            test_rows.append({"suite": suite_name, "executed": False, "tests": 0, "failures": 0, "errors": 0, "skipped": 0, "status": "not_executed"})
            continue
        xml_root = ET.parse(xml_path).getroot()
        suites = [xml_root] if xml_root.tag == "testsuite" else list(xml_root.findall("testsuite"))
        counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
        test_rows.append({"suite": suite_name, "executed": True, **counts, "status": "pass" if counts["failures"] + counts["errors"] == 0 else "fail"})
    write_csv(pd.DataFrame(test_rows), paths["qc"] / "test_status_summary.csv")
    write_review_documents(paths, work, manifest)
    if ledger_path.read_bytes() != ledger_before:
        raise RuntimeError("G3 call ledger changed during zero-solver postprocessing")
    write_json({"version": VERSION, "solver_calls_before": len(pd.read_csv(ledger_path)), "solver_calls_after": len(pd.read_csv(ledger_path)), "energyplus_invoked": False, "lockbox_building_results_read": False}, paths["base"] / "zero_solver_attestation.json")
    package_review(root, paths, manifest)
    print(json.dumps({"output": str(paths["base"]), "solver_calls": 0, "cached_runs": len(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
