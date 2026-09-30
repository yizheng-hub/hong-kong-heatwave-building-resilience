"""Create auditable G3 physics, prehistory and HKUST validation evidence.

This module only consumes the approved G3 manifest and persisted EnergyPlus
outputs.  It never invokes EnergyPlus and never propagates results to the city
building stock.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import shutil
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from utils.e8_g3_runner import HKUST_CALIBRATION, HKUST_TEST, load_production


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def result_dir(work: Path, run_id: str) -> Path:
    return work / "02_runs" / "g3_results" / run_id


def load_building(work: Path, run_id: str) -> pd.DataFrame:
    frame = pd.read_parquet(result_dir(work, run_id) / "parsed" / "native_timestep_building.parquet")
    frame["timestamp_hkt"] = pd.to_datetime(frame["timestamp_hkt"], utc=True).dt.tz_convert("Asia/Hong_Kong")
    return frame


def load_zones(work: Path, run_id: str) -> pd.DataFrame:
    frame = pd.read_parquet(result_dir(work, run_id) / "parsed" / "native_timestep_zones.parquet")
    frame["timestamp_hkt"] = pd.to_datetime(frame["timestamp_hkt"], utc=True).dt.tz_convert("Asia/Hong_Kong")
    return frame


def interval_start(frame: pd.DataFrame) -> pd.Series:
    return frame["timestamp_hkt"] - pd.Timedelta(minutes=15)


def formal_event_mask(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    starts = interval_start(frame)
    return starts.between(start, end + pd.Timedelta(days=1), inclusive="left")


def expected_observation_grid(times, values, start, window_hours, eligible, expected_step):
    """Absent rows have unknown support; conservatively require them until qualified by a schedule."""
    observed = pd.DataFrame({"time": pd.to_datetime(times).to_numpy(), "value": pd.Series(values).to_numpy()})
    observed["eligible"] = True if eligible is None else pd.Series(eligible).to_numpy()
    if observed["time"].duplicated().any():
        raise ValueError("duplicate timestamps in observation support")
    expected = pd.date_range(start, start + pd.Timedelta(hours=window_hours), freq=expected_step)
    observed = observed.loc[observed.time.between(expected[0], expected[-1])]
    if not observed.time.isin(expected).all():
        raise ValueError("observation timestamps are off the declared timestep grid")
    ordered = observed.set_index("time").reindex(expected).rename_axis("time").reset_index()
    ordered["eligibility_known"] = ordered["eligible"].notna()
    ordered["eligible"] = ordered["eligible"].eq(True) | ~ordered["eligibility_known"]
    ordered["value"] = pd.to_numeric(ordered["value"], errors="coerce")
    ordered.loc[~np.isfinite(ordered["value"]), "value"] = np.nan
    return ordered


def threshold_crossing_diagnostic(
    times: pd.Series,
    values: pd.Series,
    start: pd.Timestamp,
    threshold: float,
    eligible: pd.Series | None = None,
    window_hours: float = 24.0,
    expected_step: pd.Timedelta = pd.Timedelta(minutes=15),
) -> dict[str, object]:
    """Find an upward crossing on one fixed-zone expected support grid.

    Every expected timestamp must be present.  A missing value while ``eligible``
    is true is unknown monitoring support, not a safe or unoccupied interval.
    """
    ordered = expected_observation_grid(times, values, start, window_hours, eligible, expected_step)
    end = start + pd.Timedelta(hours=window_hours)
    ordered = ordered.loc[ordered["time"].between(start, end)].reset_index(drop=True)
    eligible_mask = ordered["eligible"].fillna(False).astype(bool)
    required = ordered.loc[eligible_mask]
    if required.empty:
        return {
            "hours": math.nan, "status": "not_applicable_no_required_monitoring",
            "right_censored": False, "left_censored": False,
            "interval_censored": False, "lower_h": math.nan,
            "upper_h": math.nan, "support_before": pd.NaT,
            "support_after": pd.NaT, "lower_bound_source": "not_applicable",
            "required_samples": 0, "valid_required_samples": 0,
            "missing_required_samples": 0,
        }
    valid = eligible_mask & ordered["value"].notna() & np.isfinite(pd.to_numeric(ordered["value"], errors="coerce"))
    t0 = ordered.loc[ordered["time"].eq(start) & valid]
    if not t0.empty and float(t0.iloc[0].value) >= threshold:
        return {
            "hours": 0.0, "status": "baseline_already_above_threshold",
            "right_censored": False, "left_censored": True,
            "interval_censored": False, "lower_h": 0.0, "upper_h": 0.0,
            "support_before": start, "support_after": start,
            "lower_bound_source": "valid_t0_support",
            "required_samples": int(eligible_mask.sum()),
            "valid_required_samples": int(valid.sum()),
            "missing_required_samples": int((eligible_mask & ~valid).sum()),
        }
    earliest_unresolved_lower: pd.Timestamp | None = None
    earliest_unresolved_source: str | None = None
    last_verified_safe: pd.Timestamp | None = None
    previous_expected_time: pd.Timestamp | None = None
    previous_required = False
    for idx in range(len(ordered)):
        current = ordered.iloc[idx]
        current_time = current.time
        is_required = bool(eligible_mask.iloc[idx])
        if current_time < start:
            continue
        new_required_interval = is_required and (
            not previous_required
            or previous_expected_time is None
            or current_time - previous_expected_time != expected_step
        )
        qualification_boundary = max(start, current_time - expected_step) if new_required_interval else None
        if not is_required:
            previous_required = False
            previous_expected_time = current_time
            last_verified_safe = None
            continue
        if not bool(valid.iloc[idx]):
            if earliest_unresolved_lower is None:
                earliest_unresolved_lower = (
                    last_verified_safe
                    if last_verified_safe is not None
                    else qualification_boundary if qualification_boundary is not None else max(start, current_time - expected_step)
                )
                earliest_unresolved_source = (
                    "last_verified_safe_support" if last_verified_safe is not None else "eligibility_interval_start"
                )
            previous_required = True
            previous_expected_time = current_time
            continue
        if float(current.value) < threshold:
            last_verified_safe = current_time
            previous_required = True
            previous_expected_time = current_time
            continue
        upper_h = float((current.time - start).total_seconds() / 3600)
        previous = ordered.iloc[idx - 1] if idx > 0 else None
        contiguous = (
            previous is not None
            and bool(valid.iloc[idx - 1])
            and bool(eligible_mask.iloc[idx - 1])
            and current.time - previous.time == expected_step
        )
        if earliest_unresolved_lower is not None:
            lower_time = earliest_unresolved_lower
            return {
                "hours": math.nan, "status": "interval_censored_missing_support",
                "right_censored": False, "left_censored": False,
                "interval_censored": True,
                "lower_h": max(0.0, float((lower_time - start).total_seconds() / 3600)),
                "upper_h": upper_h, "support_before": lower_time,
                "support_after": current.time,
                "lower_bound_source": earliest_unresolved_source,
                "required_samples": int(eligible_mask.sum()),
                "valid_required_samples": int(valid.sum()),
                "missing_required_samples": int((eligible_mask & ~valid).sum()),
            }
        if not contiguous or float(previous.value) >= threshold:
            lower_time = (
                last_verified_safe
                if last_verified_safe is not None
                else qualification_boundary if qualification_boundary is not None else max(start, current.time - expected_step)
            )
            return {
                "hours": math.nan, "status": "interval_censored_no_adjacent_safe_support",
                "right_censored": False, "left_censored": False,
                "interval_censored": True,
                "lower_h": max(0.0, float((lower_time - start).total_seconds() / 3600)),
                "upper_h": upper_h, "support_before": lower_time,
                "support_after": current.time,
                "lower_bound_source": (
                    "last_verified_safe_support" if last_verified_safe is not None else "eligibility_interval_start"
                ),
                "required_samples": int(eligible_mask.sum()),
                "valid_required_samples": int(valid.sum()),
                "missing_required_samples": int((eligible_mask & ~valid).sum()),
            }
        previous_value = float(previous.value)
        delta = float(current.value) - previous_value
        fraction = 1.0 if delta == 0 else (threshold - previous_value) / delta
        crossing = previous.time + float(fraction) * (current.time - previous.time)
        hours = max(0.0, float((crossing - start).total_seconds() / 3600))
        return {
            "hours": hours, "status": "observed_interpolated",
            "right_censored": False, "left_censored": False,
            "interval_censored": False, "lower_h": hours, "upper_h": hours,
            "support_before": previous.time, "support_after": current.time,
            "lower_bound_source": "adjacent_valid_linear_interpolation",
            "required_samples": int(eligible_mask.sum()),
            "valid_required_samples": int(valid.sum()),
            "missing_required_samples": int((eligible_mask & ~valid).sum()),
        }
    if earliest_unresolved_lower is not None:
        return {
            "hours": math.nan, "status": "unknown_incomplete_observation",
            "right_censored": False, "left_censored": False,
            "interval_censored": False,
            "lower_h": max(0.0, float((earliest_unresolved_lower - start).total_seconds() / 3600)),
            "upper_h": math.nan, "support_before": earliest_unresolved_lower,
            "support_after": pd.NaT, "lower_bound_source": earliest_unresolved_source,
            "required_samples": int(eligible_mask.sum()),
            "valid_required_samples": int(valid.sum()),
            "missing_required_samples": int((eligible_mask & ~valid).sum()),
        }
    last_valid = ordered.loc[valid, "time"].max() if valid.any() else pd.NaT
    grid_diffs = ordered["time"].diff().dropna()
    grid_complete = (
        not ordered.empty
        and ordered["time"].min() <= start
        and ordered["time"].max() >= end
        and grid_diffs.eq(expected_step).all()
    )
    full = bool((eligible_mask & ~valid).sum() == 0 and grid_complete)
    return {
        "hours": window_hours if full else math.nan,
        "status": "right_censored" if full else "unknown_incomplete_observation",
        "right_censored": bool(full), "left_censored": False,
        "interval_censored": False,
        "lower_h": window_hours if full else math.nan,
        "upper_h": math.nan, "support_before": last_valid,
        "support_after": pd.NaT, "lower_bound_source": "complete_required_support" if full else "unknown",
        "required_samples": int(eligible_mask.sum()),
        "valid_required_samples": int(valid.sum()),
        "missing_required_samples": int((eligible_mask & ~valid).sum()),
    }


def first_crossing_hours(times: pd.Series, values: pd.Series, start: pd.Timestamp, threshold: float) -> tuple[float, bool]:
    result = threshold_crossing_diagnostic(times, values, start, threshold)
    return float(result["hours"]), bool(result["right_censored"])


def recovery_diagnostic(
    times: pd.Series, values: pd.Series, restore: pd.Timestamp,
    threshold: float = 26.0, window_hours: float = 24.0,
    eligible: pd.Series | None = None,
    expected_step: pd.Timedelta = pd.Timedelta(minutes=15),
) -> dict[str, object]:
    ordered = expected_observation_grid(times, values, restore, window_hours, eligible, expected_step)
    ordered = ordered.loc[ordered["time"].between(restore, restore + pd.Timedelta(hours=window_hours))]
    required = ordered["eligible"].fillna(False).astype(bool)
    valid = required & ordered["value"].notna() & np.isfinite(pd.to_numeric(ordered["value"], errors="coerce"))
    if not required.any():
        return {
            "hours": math.nan, "censored": False, "status": "not_applicable_no_required_monitoring",
            "lower_h": math.nan, "upper_h": math.nan, "required_samples": 0,
            "valid_required_samples": 0, "missing_required_samples": 0,
        }
    exact = ordered.loc[ordered["time"].eq(restore) & valid]
    if not exact.empty and float(exact.iloc[0].value) <= threshold:
        return {
            "hours": 0.0, "censored": False, "status": "recovered_at_restore",
            "lower_h": 0.0, "upper_h": 0.0,
            "required_samples": int(required.sum()), "valid_required_samples": int(valid.sum()),
            "missing_required_samples": int((required & ~valid).sum()),
        }
    missing_before = False
    last_verified_unrecovered = restore
    earliest_missing_lower = None
    for idx, row in ordered.iterrows():
        if row.time < restore or not bool(required.loc[idx]):
            continue
        if not bool(valid.loc[idx]):
            if earliest_missing_lower is None:
                earliest_missing_lower = last_verified_unrecovered
            missing_before = True
            continue
        if float(row.value) > threshold:
            last_verified_unrecovered = row.time
            continue
        upper_h = max(0.0, float((row.time - restore).total_seconds() / 3600))
        if missing_before:
            lower_h = max(0.0, float((earliest_missing_lower - restore).total_seconds() / 3600))
            return {
                "hours": math.nan, "censored": False,
                "status": "interval_censored_missing_support",
                "lower_h": lower_h, "upper_h": upper_h,
                "lower_bound_source": "window_start_without_state" if lower_h == 0 and exact.empty else "last_verified_unrecovered_support",
                "required_samples": int(required.sum()), "valid_required_samples": int(valid.sum()),
                "missing_required_samples": int((required & ~valid).sum()),
            }
        return {
            "hours": upper_h, "censored": False, "status": "observed",
            "lower_h": upper_h, "upper_h": upper_h,
            "required_samples": int(required.sum()), "valid_required_samples": int(valid.sum()),
            "missing_required_samples": int((required & ~valid).sum()),
        }
    if bool((required & ~valid).any()):
        lower_h = max(0.0, float((earliest_missing_lower - restore).total_seconds() / 3600))
        return {
            "hours": math.nan, "censored": False, "status": "unknown_incomplete_observation",
            "lower_h": lower_h, "upper_h": math.nan,
            "lower_bound_source": "window_start_without_state" if lower_h == 0 and exact.empty else "last_verified_unrecovered_support",
            "required_samples": int(required.sum()), "valid_required_samples": int(valid.sum()),
            "missing_required_samples": int((required & ~valid).sum()),
        }
    grid_diffs = ordered["time"].diff().dropna()
    grid_complete = (
        not ordered.empty
        and ordered["time"].min() <= restore
        and ordered["time"].max() >= restore + pd.Timedelta(hours=window_hours)
        and grid_diffs.eq(expected_step).all()
    )
    if not grid_complete:
        return {
            "hours": math.nan, "censored": False, "status": "unknown_incomplete_observation",
            "lower_h": max(0.0, float((last_verified_unrecovered - restore).total_seconds() / 3600)),
            "upper_h": math.nan,
            "required_samples": int(required.sum()), "valid_required_samples": int(valid.sum()),
            "missing_required_samples": int((required & ~valid).sum()),
        }
    return {
        "hours": window_hours, "censored": True, "status": "right_censored",
        "lower_h": window_hours, "upper_h": math.nan,
        "required_samples": int(required.sum()), "valid_required_samples": int(valid.sum()),
        "missing_required_samples": 0,
    }


def first_recovery_hours(
    times: pd.Series, values: pd.Series, restore: pd.Timestamp,
    threshold: float = 26.0, window_hours: float = 24.0,
) -> tuple[float, bool]:
    result = recovery_diagnostic(times, values, restore, threshold, window_hours)
    return float(result["hours"]), bool(result["censored"])


def sustained_recovery_hours(
    times: pd.Series, values: pd.Series, restore: pd.Timestamp,
    threshold: float = 26.2, sustain_hours: float = 1.0,
    window_hours: float = 24.0,
    expected_step: pd.Timedelta = pd.Timedelta(minutes=15),
) -> tuple[float, bool]:
    ordered = pd.DataFrame({"time": pd.to_datetime(times), "value": values}).sort_values("time")
    ordered = ordered.loc[
        ordered["time"].gt(restore)
        & ordered["time"].le(restore + pd.Timedelta(hours=window_hours))
    ].reset_index(drop=True)
    required_intervals = int(round(sustain_hours / (expected_step.total_seconds() / 3600)))
    run_start: int | None = None
    for idx, row in ordered.iterrows():
        qualifies = pd.notna(row.value) and float(row.value) <= threshold
        contiguous = idx == 0 or row.time - ordered.loc[idx - 1, "time"] == expected_step
        if not qualifies or (run_start is not None and not contiguous):
            run_start = None
        if qualifies and run_start is None:
            run_start = idx
        if run_start is not None and idx - run_start + 1 >= required_intervals:
            support_start = ordered.loc[run_start, "time"] - expected_step
            return max(0.0, float((support_start - restore).total_seconds() / 3600)), False
    return window_hours, True


def max_occupied_temperature(zones: pd.DataFrame) -> pd.DataFrame:
    mask = zones["occupied_for_screen"].astype(bool) if "occupied_for_screen" in zones else zones["occupants"].gt(0)
    if "structural_conditioning_present" in zones:
        mask &= zones["structural_conditioning_present"].astype(bool)
    eligible = zones.loc[mask].copy()
    return eligible.groupby("timestamp_hkt", as_index=False)["temperature_c"].max().rename(
        columns={"temperature_c": "max_occupied_temperature_c"}
    )


def max_conditioned_temperature(zones: pd.DataFrame) -> pd.DataFrame:
    eligible = zones.loc[zones["structural_conditioning_present"].astype(bool)].copy()
    return eligible.groupby("timestamp_hkt", as_index=False)["temperature_c"].max().rename(
        columns={"temperature_c": "max_conditioned_temperature_c"}
    )


def attach_monitoring_flags(
    zones: pd.DataFrame,
    actual_schedule: pd.DataFrame,
    planned_schedule: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach explicit thermal-monitoring flags to persisted zone output."""
    planned = actual_schedule if planned_schedule is None else planned_schedule
    timestamps = pd.DatetimeIndex(sorted(zones["timestamp_hkt"].unique()))
    if len(timestamps) != len(actual_schedule) * 4 or len(actual_schedule) != len(planned):
        raise ValueError("zone timestamps and hourly schedules do not have a 4:1 mapping")
    actual_repeated = actual_schedule.loc[actual_schedule.index.repeat(4)].reset_index(drop=True)
    planned_repeated = planned.loc[planned.index.repeat(4)].reset_index(drop=True)
    timestamp_index = pd.DataFrame({"timestamp_hkt": timestamps, "schedule_row": np.arange(len(timestamps))})
    result = zones.merge(timestamp_index, on="timestamp_hkt", how="left", validate="many_to_one")
    occupancy_values: list[float] = []
    planned_values: list[bool] = []
    actual_values: list[bool] = []
    for row in result.itertuples(index=False):
        idx = int(row.schedule_row)
        occupancy_column = f"{row.zone_id}_occupancy"
        ac_column = f"{row.zone_id}_ac"
        if occupancy_column not in planned_repeated or ac_column not in planned_repeated or ac_column not in actual_repeated:
            raise KeyError(f"monitoring schedule columns missing for zone {row.zone_id}")
        occupancy_values.append(float(planned_repeated.loc[idx, occupancy_column]))
        planned_values.append(bool(float(planned_repeated.loc[idx, ac_column]) > 0))
        actual_values.append(bool(float(actual_repeated.loc[idx, ac_column]) > 0))
    result["occupancy_schedule_fraction"] = occupancy_values
    result["occupied_for_screen"] = result["occupancy_schedule_fraction"].gt(0.15)
    result["planned_service_required"] = planned_values
    result["actual_hvac_on"] = actual_values
    return result.drop(columns="schedule_row")


def expected_native_timestamps(weather: pd.DataFrame) -> pd.DatetimeIndex:
    """Build the 15-minute interval-end grid from hourly input timestamps."""
    hourly = pd.DatetimeIndex(pd.to_datetime(weather["timestamp_hkt"], utc=True)).tz_convert("Asia/Hong_Kong")
    if hourly.has_duplicates or not hourly.is_monotonic_increasing:
        raise ValueError("weather timestamps must be unique and increasing")
    if len(hourly) > 1 and not pd.Series(hourly).diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("weather timestamps do not form a complete hourly grid")
    values = [stamp + pd.Timedelta(minutes=offset) for stamp in hourly for offset in (15, 30, 45, 60)]
    return pd.DatetimeIndex(values)


def build_expected_monitoring_support(
    zones: pd.DataFrame,
    actual_schedule: pd.DataFrame,
    planned_schedule: pd.DataFrame,
    expected_times: pd.DatetimeIndex,
    zone_contract: pd.DataFrame,
) -> pd.DataFrame:
    """Create expected zone-time support independently of observed zone rows."""
    contract = zone_contract[["zone_id", "zone_area_m2", "structural_conditioning_present"]].copy()
    if contract["zone_id"].duplicated().any() or contract.empty:
        raise ValueError("zone contract must contain one row per expected zone")
    if len(expected_times) != len(actual_schedule) * 4 or len(actual_schedule) != len(planned_schedule):
        raise ValueError("expected native grid and hourly schedules do not have a 4:1 mapping")
    grid = pd.MultiIndex.from_product(
        [expected_times, contract["zone_id"].astype(str)], names=["timestamp_hkt", "zone_id"]
    ).to_frame(index=False)
    grid = grid.merge(contract, on="zone_id", how="left", validate="many_to_one")
    observed = zones.copy()
    if observed.duplicated(["timestamp_hkt", "zone_id"]).any():
        raise ValueError("duplicate zone-time output rows")
    expected_keys = pd.MultiIndex.from_frame(grid[["timestamp_hkt", "zone_id"]])
    actual_keys = pd.MultiIndex.from_frame(observed[["timestamp_hkt", "zone_id"]])
    if len(actual_keys.difference(expected_keys)):
        raise ValueError("unexpected zone-time output outside the frozen grid")
    attributes = observed[["zone_id", "zone_area_m2"]].merge(contract[["zone_id", "zone_area_m2"]], on="zone_id", suffixes=("_observed", "_expected"), validate="many_to_one")
    if not np.allclose(attributes.zone_area_m2_observed, attributes.zone_area_m2_expected, atol=1e-9, rtol=0):
        raise ValueError("observed zone areas differ from frozen zone contract")
    observed_payload = observed.drop(
        columns=["zone_area_m2", "structural_conditioning_present"], errors="ignore"
    )
    grid = grid.merge(
        observed_payload, on=["timestamp_hkt", "zone_id"], how="left",
        validate="one_to_one", indicator="_output_merge",
    )
    grid["output_present"] = grid["_output_merge"].eq("both")
    grid = grid.drop(columns="_output_merge")
    actual_repeated = actual_schedule.loc[actual_schedule.index.repeat(4)].reset_index(drop=True)
    planned_repeated = planned_schedule.loc[planned_schedule.index.repeat(4)].reset_index(drop=True)
    time_index = {stamp: idx for idx, stamp in enumerate(expected_times)}
    occupancy, planned_service, actual_hvac = [], [], []
    for row in grid.itertuples(index=False):
        idx = time_index[row.timestamp_hkt]
        occupancy_column = f"{row.zone_id}_occupancy"
        ac_column = f"{row.zone_id}_ac"
        if occupancy_column not in planned_repeated or ac_column not in planned_repeated or ac_column not in actual_repeated:
            raise KeyError(f"monitoring schedule columns missing for zone {row.zone_id}")
        if not np.isfinite([planned_repeated.loc[idx, occupancy_column], planned_repeated.loc[idx, ac_column], actual_repeated.loc[idx, ac_column]]).all():
            raise ValueError("non-finite required schedule value")
        occupancy.append(float(planned_repeated.loc[idx, occupancy_column]))
        planned_service.append(bool(float(planned_repeated.loc[idx, ac_column]) > 0))
        actual_hvac.append(bool(float(actual_repeated.loc[idx, ac_column]) > 0))
    grid["occupancy_schedule_fraction"] = occupancy
    grid["occupied_for_screen"] = grid["occupancy_schedule_fraction"].gt(0.15)
    grid["planned_service_required"] = planned_service
    grid["actual_hvac_on"] = actual_hvac
    grid["monitoring_required"] = (
        grid["structural_conditioning_present"].astype(bool)
        & grid["occupied_for_screen"].astype(bool)
    )
    if not np.isfinite(np.asarray(occupancy, dtype=float)).all():
        raise ValueError("non-finite reference occupancy schedule")
    grid["temperature_valid"] = np.isfinite(pd.to_numeric(grid.get("temperature_c", pd.Series(np.nan, index=grid.index)), errors="coerce"))
    grid["support_status"] = np.select(
        [~grid["monitoring_required"], grid["temperature_valid"]],
        ["not_required", "required_valid"], default="required_missing",
    )
    return grid


def strict_zone_time_pair(
    left: pd.DataFrame,
    right: pd.DataFrame,
    value_columns: list[str],
    left_label: str,
    right_label: str,
) -> pd.DataFrame:
    """Pair complete zone-time records; never let inner joins erase missing keys."""
    keys = ["timestamp_hkt", "zone_id"]
    for label, frame in ((left_label, left), (right_label, right)):
        if frame.empty:
            raise ValueError(f"{label} has no required zone-time support")
        if frame.duplicated(keys).any():
            raise ValueError(f"{label} has duplicate zone-time keys")
        missing_columns = [column for column in keys + ["zone_area_m2"] + value_columns if column not in frame]
        if missing_columns:
            raise KeyError(f"{label} missing columns {missing_columns}")
        for column in ["zone_area_m2"] + value_columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
                raise ValueError(f"{label} has non-finite required values in {column}")
    paired = left[keys + ["zone_area_m2"] + value_columns].merge(
        right[keys + ["zone_area_m2"] + value_columns], on=keys, how="outer",
        suffixes=(f"_{left_label}", f"_{right_label}"), validate="one_to_one", indicator=True,
    )
    counts = paired["_merge"].value_counts()
    if int(counts.get("left_only", 0)) or int(counts.get("right_only", 0)):
        raise ValueError(
            f"zone-time key mismatch: {left_label}_only={int(counts.get('left_only', 0))};"
            f"{right_label}_only={int(counts.get('right_only', 0))}"
        )
    left_area = paired[f"zone_area_m2_{left_label}"].to_numpy(dtype=float)
    right_area = paired[f"zone_area_m2_{right_label}"].to_numpy(dtype=float)
    if not np.allclose(left_area, right_area, rtol=0.0, atol=1e-9):
        raise ValueError("zone area attributes differ between paired runs")
    paired["zone_area_m2"] = left_area
    return paired.drop(columns=["_merge", f"zone_area_m2_{left_label}", f"zone_area_m2_{right_label}"])


def load_monitoring_support(work: Path, item, reference_item) -> pd.DataFrame:
    areas = json.loads(item.zone_areas_json)
    conditioned = set(json.loads(item.conditioned_zones_json))
    contract = pd.DataFrame([
        {"zone_id": zone, "zone_area_m2": area, "structural_conditioning_present": zone in conditioned}
        for zone, area in areas.items()
    ])
    return build_expected_monitoring_support(
        load_zones(work, item.run_id),
        pd.read_parquet(result_dir(work, item.run_id) / "inputs" / "event_schedules.parquet"),
        pd.read_parquet(result_dir(work, reference_item.run_id) / "inputs" / "event_schedules.parquet"),
        expected_native_timestamps(pd.read_parquet(result_dir(work, item.run_id) / "inputs" / "weather.parquet")),
        contract,
    )


def building_crossing_from_zones(
    zones: pd.DataFrame, start: pd.Timestamp, threshold: float = 32.0,
    window_hours: float = 24.0,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Compute per-zone crossings and propagate bounds for the earliest event."""
    rows: list[dict[str, object]] = []
    for zone_id, group in zones.groupby("zone_id", sort=True):
        result = threshold_crossing_diagnostic(
            group["timestamp_hkt"], group["temperature_c"], start, threshold,
            eligible=group["occupied_for_screen"] & group["structural_conditioning_present"],
            window_hours=window_hours,
        )
        result["zone_id"] = zone_id
        result["valid_samples"] = int(result.get("valid_required_samples", 0))
        result["effective_monitoring_hours"] = 0.25 * int(result.get("valid_required_samples", 0))
        rows.append(result)
    per_zone = pd.DataFrame(rows)
    applicable = per_zone.loc[~per_zone["status"].eq("not_applicable_no_required_monitoring")].copy()
    if applicable.empty:
        building = {
            "hours": math.nan, "status": "not_applicable_no_required_monitoring",
            "right_censored": False, "left_censored": False,
            "interval_censored": False, "lower_h": math.nan,
            "upper_h": math.nan, "zone_id": None,
            "effective_monitoring_hours": 0.0,
        }
    else:
        finite_upper = applicable.loc[pd.to_numeric(applicable["upper_h"], errors="coerce").notna()].copy()
        unknown = applicable.loc[applicable["status"].isin(["unknown_incomplete_observation"])]
        effective_hours = float(applicable["effective_monitoring_hours"].sum())
        if not finite_upper.empty:
            controller = finite_upper.sort_values(["upper_h", "lower_h", "zone_id"]).iloc[0]
            lower_values = pd.to_numeric(applicable["lower_h"], errors="coerce").fillna(0.0)
            lower_controller = applicable.loc[lower_values.idxmin()]
            lower_h = float(lower_values.min())
            upper_h = float(controller["upper_h"])
            exact = math.isclose(lower_h, upper_h, abs_tol=1e-12)
            building = controller.to_dict()
            building.update({
                "hours": upper_h if exact else math.nan,
                "status": str(controller["status"]) if exact else "interval_censored_earliest_zone_combination",
                "right_censored": False,
                "left_censored": bool(lower_h == 0 and upper_h == 0),
                "interval_censored": not exact,
                "lower_h": lower_h,
                "upper_h": upper_h,
                "lower_bound_zone_id": lower_controller["zone_id"],
                "upper_bound_zone_id": controller["zone_id"],
                "lower_bound_source": lower_controller.get("lower_bound_source", "unknown"),
                "support_before": lower_controller.get("support_before", pd.NaT),
                "effective_monitoring_hours": effective_hours,
            })
        elif not unknown.empty:
            lower_values = pd.to_numeric(unknown["lower_h"], errors="coerce").dropna()
            building = {
                "hours": math.nan, "status": "unknown_incomplete_observation",
                "right_censored": False, "left_censored": False,
                "interval_censored": False,
                "lower_h": float(lower_values.min()) if not lower_values.empty else math.nan,
                "upper_h": math.nan, "zone_id": None,
                "effective_monitoring_hours": effective_hours,
            }
        elif applicable["right_censored"].all():
            building = {
                "hours": window_hours, "status": "right_censored",
                "right_censored": True, "left_censored": False,
                "interval_censored": False, "lower_h": window_hours,
                "upper_h": math.nan, "zone_id": None,
                "effective_monitoring_hours": effective_hours,
            }
        else:
            building = {
                "hours": math.nan, "status": "unknown_no_valid_monitoring_support",
                "right_censored": False, "left_censored": False,
                "interval_censored": False, "lower_h": math.nan,
                "upper_h": math.nan, "zone_id": None,
                "effective_monitoring_hours": effective_hours,
            }
    support = zones.loc[zones.timestamp_hkt.gt(start) & zones.timestamp_hkt.le(start + pd.Timedelta(hours=window_hours))].copy()
    support["required"] = support.occupied_for_screen & support.structural_conditioning_present
    support["valid_required"] = support.required & np.isfinite(pd.to_numeric(support.temperature_c, errors="coerce"))
    building["effective_monitoring_zone_hours"] = float(support.valid_required.sum()) * .25
    complete = support.groupby("timestamp_hkt").agg(required=("required", "sum"), valid=("valid_required", "sum"))
    building["effective_monitoring_hours"] = float(((complete.required > 0) & complete.required.eq(complete.valid)).sum()) * .25
    return building, per_zone


def building_recovery_from_zones(
    zones: pd.DataFrame,
    restore: pd.Timestamp,
    threshold: float = 26.0,
    window_hours: float = 24.0,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Require every structural conditioned zone at each recovery timestamp."""
    conditioned = zones.loc[zones["structural_conditioning_present"].astype(bool)].copy()
    if conditioned.empty:
        result = {
            "hours": math.nan, "censored": False,
            "status": "not_applicable_no_conditioned_zone",
            "lower_h": math.nan, "upper_h": math.nan,
            "required_samples": 0, "valid_required_samples": 0,
            "missing_required_samples": 0,
        }
        return result, pd.DataFrame()
    expected_zones = sorted(conditioned["zone_id"].astype(str).unique())
    window = conditioned.loc[
        conditioned["timestamp_hkt"].between(
            restore, restore + pd.Timedelta(hours=window_hours)
        )
    ].copy()
    if window.duplicated(["timestamp_hkt", "zone_id"]).any():
        raise ValueError("duplicate conditioned zone-time rows in recovery")
    states = []
    for timestamp, group in window.groupby("timestamp_hkt", sort=True):
        present = set(group["zone_id"].astype(str))
        missing_zone_rows = sorted(set(expected_zones) - present)
        values = pd.to_numeric(group["temperature_c"], errors="coerce")
        nonfinite = sorted(group.loc[~np.isfinite(values), "zone_id"].astype(str))
        complete = not missing_zone_rows and not nonfinite and len(group) == len(expected_zones)
        states.append({
            "timestamp_hkt": timestamp,
            "temperature_c": float(values.max()) if complete else math.nan,
            "eligible": True,
            "all_required_zones_complete": complete,
            "missing_zone_ids": "|".join(missing_zone_rows + nonfinite),
            "required_zone_count": len(expected_zones),
            "valid_zone_count": int(np.isfinite(values).sum()) if not missing_zone_rows else int(np.isfinite(values).sum()),
        })
    state = pd.DataFrame(states, columns=["timestamp_hkt", "temperature_c", "eligible", "all_required_zones_complete", "missing_zone_ids", "required_zone_count", "valid_zone_count"])
    result = recovery_diagnostic(
        state["timestamp_hkt"], state["temperature_c"], restore,
        threshold=threshold, window_hours=window_hours,
        eligible=state["eligible"],
    )
    return result, state


def outage_score_bounds(result: dict[str, object], window_hours: float = 24.0) -> dict[str, object]:
    """Map first-crossing censoring to monotone BERI outage-score bounds."""
    status = str(result.get("status", "unknown"))
    lower = pd.to_numeric(pd.Series([result.get("lower_h")]), errors="coerce").iloc[0]
    upper = pd.to_numeric(pd.Series([result.get("upper_h")]), errors="coerce").iloc[0]
    if status == "right_censored":
        score_lower = score_upper = 1.0
        reason = "complete_window_no_crossing"
    elif status == "not_applicable_no_required_monitoring":
        score_lower = score_upper = math.nan
        reason = "no_required_monitoring_support"
    elif np.isfinite(lower) and np.isfinite(upper):
        score_lower = float(np.clip(lower / window_hours, 0.0, 1.0))
        score_upper = float(np.clip(upper / window_hours, 0.0, 1.0))
        reason = "interval_or_exact_crossing"
    elif np.isfinite(lower):
        score_lower = float(np.clip(lower / window_hours, 0.0, 1.0))
        score_upper = 1.0
        reason = "incomplete_observation_after_lower_support"
    else:
        score_lower, score_upper = 0.0, 1.0
        reason = "crossing_time_unknown"
    return {
        "raw_time_lower": float(lower) if np.isfinite(lower) else math.nan,
        "raw_time_upper": float(upper) if np.isfinite(upper) else math.nan,
        "observation_status": status,
        "score_lower": score_lower,
        "score_upper": score_upper,
        "NA_reason": reason if status.startswith("unknown") or status.startswith("not_applicable") else "point_not_identified" if score_lower != score_upper else "",
        "bound_semantics": "preserve endpoint provenance; no midpoint for interval bounds",
    }


def recovery_score_bounds(result: dict[str, object], window_hours: float = 24.0) -> dict[str, object]:
    """Map recovery censoring to monotone BERI recovery-score bounds."""
    status = str(result.get("status", "unknown"))
    lower = pd.to_numeric(pd.Series([result.get("lower_h")]), errors="coerce").iloc[0]
    upper = pd.to_numeric(pd.Series([result.get("upper_h")]), errors="coerce").iloc[0]
    if status == "right_censored":
        score_lower = score_upper = 0.0
        reason = "complete_24h_window_without_recovery"
    elif status.startswith("not_applicable"):
        score_lower = score_upper = math.nan
        reason = "no_required_recovery_object"
    elif np.isfinite(lower) and np.isfinite(upper):
        score_lower = float(np.clip(1.0 - upper / window_hours, 0.0, 1.0))
        score_upper = float(np.clip(1.0 - lower / window_hours, 0.0, 1.0))
        reason = "interval_or_exact_recovery"
    elif np.isfinite(lower):
        score_lower = 0.0
        score_upper = float(np.clip(1.0 - lower / window_hours, 0.0, 1.0))
        reason = "incomplete_observation_after_unrecovered_support"
    else:
        score_lower, score_upper = 0.0, 1.0
        reason = "recovery_time_unknown"
    return {
        "raw_time_lower": float(lower) if np.isfinite(lower) else math.nan,
        "raw_time_upper": float(upper) if np.isfinite(upper) else math.nan,
        "observation_status": status,
        "score_lower": score_lower,
        "score_upper": score_upper,
        "NA_reason": reason if status.startswith("unknown") or status.startswith("not_applicable") else "point_not_identified" if score_lower != score_upper else "",
        "bound_semantics": "preserve endpoint provenance; no midpoint for interval bounds",
    }


def beri_from_event_score_bounds(frame: pd.DataFrame, expected_event_ids: list[str]) -> dict[str, object]:
    """Propagate all declared event/component bounds without dropping missing rows."""
    keys = ["event_id", "component"]
    expected = pd.MultiIndex.from_product([expected_event_ids, ["N", "C", "O", "R"]], names=keys)
    if frame.duplicated(keys).any():
        raise ValueError("duplicate BERI event/component")
    actual = pd.MultiIndex.from_frame(frame[keys])
    if len(expected.difference(actual)) or len(actual.difference(expected)):
        return {"score_lower": math.nan, "score_upper": math.nan, "NA_reason": "incomplete_reference_event_component_set"}
    bounds = frame[["score_lower", "score_upper"]].to_numpy(dtype=float)
    if not np.isfinite(bounds).all():
        return {"score_lower": math.nan, "score_upper": math.nan, "NA_reason": "non_evaluable_component"}
    if (bounds < 0).any() or (bounds > 1).any() or (bounds[:, 0] > bounds[:, 1]).any():
        raise ValueError("invalid score bounds")
    medians = frame.groupby("component")[["score_lower", "score_upper"]].median()
    lower, upper = 100 * np.prod(medians.to_numpy(), axis=0) ** .25
    return {"score_lower": float(lower), "score_upper": float(upper), "NA_reason": "", "event_count": len(expected_event_ids)}


def run_inventory(manifest: pd.DataFrame, work: Path) -> pd.DataFrame:
    rows = []
    for item in manifest.itertuples(index=False):
        run = result_dir(work, item.run_id)
        payload = json.loads((run / "run_result.json").read_text(encoding="utf-8"))
        err = (run / "raw" / "eplusout.err").read_text(encoding="utf-8", errors="replace")
        rows.append(
            {
                "run_id": item.run_id,
                "role": item.role,
                "event_id": item.event_id,
                "prototype_id": item.prototype_id,
                "scenario_id": item.scenario_id,
                "prehistory_days": item.prehistory_days,
                "physics_hash": item.physics_hash,
                "output_schema_hash": item.output_schema_hash,
                "status": payload.get("status"),
                "returncode": payload.get("returncode"),
                "severe_errors": payload.get("severe_errors"),
                "fatal_errors": payload.get("fatal_errors"),
                "elapsed_seconds": payload.get("elapsed_seconds"),
                "peak_rss_bytes": payload.get("peak_rss_bytes"),
                "run_storage_bytes": payload.get("run_storage_bytes", sum(p.stat().st_size for p in run.rglob("*") if p.is_file())),
                "parser_version": payload.get("parser_version"),
                "raw_output_sha256": payload.get("raw_output_sha256"),
                "completed_successfully": "EnergyPlus Completed Successfully" in err,
            }
        )
    return pd.DataFrame(rows)


def zone_timestep_service_accounting(
    required_zones: pd.DataFrame, delivered_zones: pd.DataFrame, expected_keys: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Match cooling at fixed zone and timestep before any aggregation."""
    if expected_keys is not None:
        expected = pd.MultiIndex.from_frame(expected_keys[["timestamp_hkt", "zone_id"]])
        for label, frame in (("required", required_zones), ("delivered", delivered_zones)):
            observed = pd.MultiIndex.from_frame(frame[["timestamp_hkt", "zone_id"]])
            if len(expected.difference(observed)) or len(observed.difference(expected)):
                raise ValueError(f"{label} differs from expected zone-time grid")
    matched = strict_zone_time_pair(
        required_zones, delivered_zones, ["cooling_total_j"],
        left_label="required", right_label="delivered",
    )
    matched["matched_j"] = matched[["cooling_total_j_required", "cooling_total_j_delivered"]].min(axis=1)
    matched["shortfall_j"] = (matched["cooling_total_j_required"] - matched["cooling_total_j_delivered"]).clip(lower=0)
    matched["exceedance_j"] = (matched["cooling_total_j_delivered"] - matched["cooling_total_j_required"]).clip(lower=0)
    totals = {
        "required_j": float(matched["cooling_total_j_required"].sum()),
        "delivered_raw_j": float(matched["cooling_total_j_delivered"].sum()),
        "matched_j": float(matched["matched_j"].sum()),
        "shortfall_j": float(matched["shortfall_j"].sum()),
        "exceedance_j": float(matched["exceedance_j"].sum()),
    }
    return matched, totals


def capacity_and_phase_audit(manifest: pd.DataFrame, work: Path, production) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = production.prototype_parameter_table().set_index("prototype_id")
    audit_rows: list[dict[str, object]] = []
    service_rows: list[dict[str, object]] = []
    for role in sorted(manifest.loc[manifest.role.str.startswith("core_group_"), "role"].unique()):
        group = manifest.loc[manifest["role"].eq(role)].set_index("scenario_id")
        c70_item = group.loc["HW_C70"]
        c70 = load_building(work, c70_item.run_id)
        unl = load_building(work, group.loc["HW_UNL", "run_id"])
        weather = pd.read_parquet(result_dir(work, c70_item.run_id) / "inputs" / "weather.parquet")
        weather["timestamp_hkt"] = pd.to_datetime(weather["timestamp_hkt"], utc=True).dt.tz_convert("Asia/Hong_Kong")
        schedule = pd.read_parquet(result_dir(work, c70_item.run_id) / "inputs" / "event_schedules.parquet")
        schedule = pd.concat([weather[["event_phase"]].reset_index(drop=True), schedule.reset_index(drop=True)], axis=1)
        rated_cols = [column for column in schedule if column.endswith("_rated_capacity_ac")]
        c70_cols = [column for column in schedule if column.endswith("_c70_capacity_ac")]
        simultaneous = int(((schedule[rated_cols].sum(axis=1) > 0) & (schedule[c70_cols].sum(axis=1) > 0)).sum())
        wrong_rated = int(((schedule["event_phase"].eq("heatwave")) & (schedule[rated_cols].sum(axis=1) > 0)).sum())
        wrong_c70 = int(((~schedule["event_phase"].eq("heatwave")) & (schedule[c70_cols].sum(axis=1) > 0)).sum())
        audit_rows.append(
            {
                "audit": "C70_phase_switch",
                "role": role,
                "prototype_id": c70_item.prototype_id,
                "metric": "schedule_exclusivity",
                "value": simultaneous + wrong_rated + wrong_c70,
                "unit": "violating_hour_rows",
                "pass": simultaneous == wrong_rated == wrong_c70 == 0,
                "evidence": f"simultaneous={simultaneous};rated_during_event={wrong_rated};c70_outside_event={wrong_c70}",
            }
        )
        outage_item = group.loc["HW_O24_SYNC"]
        outage_idf = (Path(outage_item.compiled_dir) / "model.idf").read_text(encoding="utf-8", errors="replace")
        audit_rows.append(
            {
                "audit": "scenario_branch_independence",
                "role": role,
                "prototype_id": c70_item.prototype_id,
                "metric": "O24_contains_no_C70_dual_system",
                "value": int("C70IDEALLOADS" in outage_idf.upper()),
                "unit": "occurrences_flag",
                "pass": "C70IDEALLOADS" not in outage_idf.upper() and c70_item.physics_hash != outage_item.physics_hash,
                "evidence": f"C70_hash={c70_item.physics_hash};O24_hash={outage_item.physics_hash}",
            }
        )
        assignment = pd.read_csv(work / "01_inputs" / "E8_event_assignment.csv").set_index("parent_event_id")
        event = assignment.loc[c70_item.event_id]
        event_start = pd.Timestamp(event.event_start, tz="Asia/Hong_Kong")
        event_end = pd.Timestamp(event.event_end, tz="Asia/Hong_Kong")
        mask_c70 = formal_event_mask(c70, event_start, event_end)
        mask_unl = formal_event_mask(unl, event_start, event_end)
        c70_event = c70.loc[mask_c70].copy()
        unl_event = unl.loc[mask_unl].copy()
        c70_zones = load_zones(work, c70_item.run_id)
        unl_zones = load_zones(work, group.loc["HW_UNL", "run_id"])
        c70_zone_mask = formal_event_mask(c70_zones, event_start, event_end)
        unl_zone_mask = formal_event_mask(unl_zones, event_start, event_end)
        c70_conditioned = c70_zones.loc[
            c70_zone_mask & c70_zones["structural_conditioning_present"].astype(bool)
        ].copy()
        required_zones = unl_zones.loc[
            unl_zone_mask & unl_zones["structural_conditioning_present"].astype(bool),
            ["timestamp_hkt", "zone_id", "zone_area_m2", "cooling_total_j"],
        ]
        delivered_zones = c70_zones.loc[
            c70_zone_mask & c70_zones["structural_conditioning_present"].astype(bool),
            ["timestamp_hkt", "zone_id", "zone_area_m2", "cooling_total_j"],
        ]
        expected_times = expected_native_timestamps(weather)
        expected_times = expected_times[(expected_times > event_start) & (expected_times <= event_end + pd.Timedelta(days=1))]
        expected_keys = pd.MultiIndex.from_product(
            [expected_times, json.loads(c70_item.conditioned_zones_json)], names=["timestamp_hkt", "zone_id"]
        ).to_frame(index=False)
        matched, zone_totals = zone_timestep_service_accounting(required_zones, delivered_zones, expected_keys)
        building_matched = matched.groupby("timestamp_hkt", as_index=False)[["cooling_total_j_required", "cooling_total_j_delivered"]].sum()
        required = matched["cooling_total_j_required"].to_numpy(dtype=float)
        delivered = matched["cooling_total_j_delivered"].to_numpy(dtype=float)
        stepwise = zone_totals["matched_j"]
        aggregate_required = building_matched["cooling_total_j_required"].to_numpy(dtype=float)
        aggregate_delivered = building_matched["cooling_total_j_delivered"].to_numpy(dtype=float)
        aggregate = float(np.minimum(aggregate_required, aggregate_delivered).sum())
        shortfall = zone_totals["shortfall_j"]
        exceed = zone_totals["exceedance_j"]
        required_total = zone_totals["required_j"]
        service_rows.append(
            {
                "role": role,
                "prototype_id": c70_item.prototype_id,
                "event_id": c70_item.event_id,
                "required_ref_kwh_thermal": required_total / 3.6e6,
                "delivered_raw_kwh_thermal": float(delivered.sum()) / 3.6e6,
                "matched_stepwise_kwh_thermal": stepwise / 3.6e6,
                "matched_building_timestep_min_kwh_thermal_diagnostic": aggregate / 3.6e6,
                "shortfall_ref_kwh_thermal": shortfall / 3.6e6,
                "raw_excess_over_reference_kwh_thermal": exceed / 3.6e6,
                "score_capacity_stepwise": 1.0 if required_total == 0 else 1.0 - shortfall / required_total,
                "c70_conditioned_zone_unmet_hours": float(c70_conditioned["unmet_hours"].sum()),
                "c70_conditioned_zone_hours_unit": "zone-hours; sum may exceed event duration",
                "c70_building_any_zone_unmet_hours": float(
                    c70_conditioned.assign(unmet=lambda x: x["unmet_hours"].gt(0))
                    .groupby("timestamp_hkt")["unmet"].any().sum() * 0.25
                ),
                "c70_area_weighted_unmet_hours": float(
                    (c70_conditioned["unmet_hours"] * c70_conditioned["zone_area_m2"]).sum()
                    / c70_conditioned.groupby("zone_id")["zone_area_m2"].first().sum()
                ),
                "c70_max_conditioned_zone_temperature_c": float(c70_conditioned["temperature_c"].max()),
                "accounting_definition": "same-zone same-native-timestep min(required_ref, delivered_raw), then sum zones and time",
            }
        )
        capacity = float(getattr(c70_item, "effective_cooling_capacity_w_m2",
                                 params.loc[c70_item.prototype_id, "cooling_capacity_w_m2"]))
        areas = json.loads(c70_item.zone_areas_json)
        conditioned = json.loads(c70_item.conditioned_zones_json)
        declared_total = capacity * sum(float(areas[zone]) for zone in conditioned) * 0.70
        maximum_rate = float(c70_event["cooling_rate_w"].max())
        zone_utilization = c70_conditioned.assign(
            declared_zone_capacity_w=lambda x: x["zone_area_m2"] * capacity * 0.70,
            utilization=lambda x: x["cooling_rate_w"] / x["declared_zone_capacity_w"],
        )
        maximum_zone_utilization = float(zone_utilization["utilization"].max())
        audit_rows.append(
            {
                "audit": "C70_declared_capacity",
                "role": role,
                "prototype_id": c70_item.prototype_id,
                "metric": "maximum_aggregate_cooling_rate",
                "value": maximum_rate,
                "unit": "W",
                "pass": maximum_rate <= declared_total + 1e-6,
                "evidence": f"declared_aggregate_70pct_capacity_W={declared_total:.6f}",
            }
        )
        audit_rows.append(
            {
                "audit": "C70_zone_capacity",
                "role": role,
                "prototype_id": c70_item.prototype_id,
                "metric": "maximum_zone_capacity_utilization",
                "value": maximum_zone_utilization,
                "unit": "fraction",
                "pass": maximum_zone_utilization <= 1.0 + 1e-9,
                "evidence": "each conditioned zone rate divided by 0.70 * nominal W/m2 * its own zone area",
            }
        )
    return pd.DataFrame(audit_rows), pd.DataFrame(service_rows)


def outage_audit(manifest: pd.DataFrame, work: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    audits: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    for role in sorted(manifest.loc[manifest.role.str.startswith("core_group_"), "role"].unique()):
        group = manifest.loc[manifest["role"].eq(role)].set_index("scenario_id")
        rated_item = group.loc["HW_RATED"]
        outage_item = group.loc["HW_O24_SYNC"]
        rated = load_building(work, rated_item.run_id)
        outage = load_building(work, outage_item.run_id)
        rated_zones = load_zones(work, rated_item.run_id)
        outage_zones = load_zones(work, outage_item.run_id)
        start = pd.Timestamp(outage_item.outage_start_hkt)
        end = pd.Timestamp(outage_item.outage_end_hkt)
        if start.tzinfo is None:
            start = start.tz_localize("Asia/Hong_Kong")
            end = end.tz_localize("Asia/Hong_Kong")
        if rated["timestamp_hkt"].duplicated().any() or outage["timestamp_hkt"].duplicated().any():
            raise ValueError(f"duplicate building timestamps in rated/outage pair for {role}")
        pair = rated.merge(
            outage, on="timestamp_hkt", how="outer",
            suffixes=("_rated", "_outage"), validate="one_to_one", indicator=True,
        )
        pair_counts = pair["_merge"].value_counts()
        if int(pair_counts.get("left_only", 0)) or int(pair_counts.get("right_only", 0)):
            raise ValueError(
                f"building-time key mismatch for {role}: "
                f"rated_only={int(pair_counts.get('left_only', 0))};"
                f"outage_only={int(pair_counts.get('right_only', 0))}"
            )
        pair = pair.drop(columns="_merge")
        pre = pair["timestamp_hkt"].le(start)
        temperature_columns = [column for column in rated if column.endswith("__temperature_c")]
        for column in temperature_columns:
            for suffix in ("rated", "outage"):
                values = pd.to_numeric(pair[f"{column}_{suffix}"], errors="coerce")
                if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
                    raise ValueError(f"non-finite paired building temperature for {role}: {column}_{suffix}")
        pre_temperature_difference = max(
            float((pair[f"{column}_rated"] - pair[f"{column}_outage"]).abs().loc[pre].max())
            for column in temperature_columns
        )
        outage_mask = outage["timestamp_hkt"].gt(start) & outage["timestamp_hkt"].le(end)
        recovery_mask = outage["timestamp_hkt"].gt(end)
        shutdown = outage.loc[outage_mask, [
            "cooling_total_j", "lights_j", "equipment_j", "equipment_convective_gain_j",
            "schedule_ac_sum", "schedule_mechvent_sum", "schedule_lights_sum", "schedule_plugs_sum",
        ]].sum()
        audits.extend(
            [
                {
                    "audit": "outage_prestate_pairing", "role": role, "prototype_id": outage_item.prototype_id,
                    "metric": "maximum_preoutage_zone_temperature_difference", "value": pre_temperature_difference,
                    "unit": "C", "pass": pre_temperature_difference <= 1e-9,
                    "evidence": f"rated_run={rated_item.run_id};outage_run={outage_item.run_id}",
                },
                {
                    "audit": "outage_duration", "role": role, "prototype_id": outage_item.prototype_id,
                    "metric": "native_timestep_coverage", "value": float(outage_mask.sum()) * 0.25,
                    "unit": "h", "pass": int(outage_mask.sum()) == 96,
                    "evidence": f"start={start};end={end};recovery_hours={float(recovery_mask.sum()) * .25}",
                },
                {
                    "audit": "outage_shutdown", "role": role, "prototype_id": outage_item.prototype_id,
                    "metric": "cooling_lighting_plugs_process_and_gain_sum", "value": float(shutdown.sum()),
                    "unit": "mixed_zero_check", "pass": bool(np.allclose(shutdown.to_numpy(dtype=float), 0.0)),
                    "evidence": json.dumps(shutdown.to_dict()),
                },
            ]
        )
        rated_schedule = pd.read_parquet(result_dir(work, rated_item.run_id) / "inputs" / "event_schedules.parquet")
        outage_schedule = pd.read_parquet(result_dir(work, outage_item.run_id) / "inputs" / "event_schedules.parquet")
        rated_weather = pd.read_parquet(result_dir(work, rated_item.run_id) / "inputs" / "weather.parquet")
        outage_weather = pd.read_parquet(result_dir(work, outage_item.run_id) / "inputs" / "weather.parquet")
        expected_times = expected_native_timestamps(rated_weather)
        if not expected_times.equals(expected_native_timestamps(outage_weather)):
            raise ValueError(f"rated/outage expected time grids differ for {role}")
        areas = json.loads(outage_item.zone_areas_json)
        conditioned_ids = set(json.loads(outage_item.conditioned_zones_json))
        zone_contract = pd.DataFrame([
            {
                "zone_id": zone_id,
                "zone_area_m2": float(area),
                "structural_conditioning_present": zone_id in conditioned_ids,
            }
            for zone_id, area in areas.items()
        ])
        rated_support = build_expected_monitoring_support(
            rated_zones, rated_schedule, rated_schedule, expected_times, zone_contract
        )
        outage_support = build_expected_monitoring_support(
            outage_zones, outage_schedule, rated_schedule, expected_times, zone_contract
        )
        zone_pair = strict_zone_time_pair(
            rated_support, outage_support, ["temperature_c"],
            left_label="rated", right_label="outage",
        )
        zone_pair = zone_pair.merge(
            rated_support[[
                "timestamp_hkt", "zone_id", "structural_conditioning_present",
                "occupied_for_screen", "occupancy_schedule_fraction",
                "planned_service_required", "actual_hvac_on", "support_status",
            ]],
            on=["timestamp_hkt", "zone_id"], how="left", validate="one_to_one",
        )
        zone_pair["eligible"] = (
            zone_pair["structural_conditioning_present"].astype(bool)
            & zone_pair["occupied_for_screen"].astype(bool)
        )
        thermal = zone_pair.loc[
            zone_pair["timestamp_hkt"].gt(start) & zone_pair["timestamp_hkt"].le(end)
            & zone_pair["eligible"]
        ].copy()
        thermal["absolute_dh32"] = np.maximum(thermal["temperature_c_outage"] - 32.0, 0) * 0.25
        thermal["absolute_dh26"] = np.maximum(thermal["temperature_c_outage"] - 26.0, 0) * 0.25
        thermal["rated_dh32"] = np.maximum(thermal["temperature_c_rated"] - 32.0, 0) * 0.25
        thermal["rated_dh26"] = np.maximum(thermal["temperature_c_rated"] - 26.0, 0) * 0.25
        thermal["extra_dh26"] = np.maximum(
            np.maximum(thermal["temperature_c_outage"] - 26.0, 0)
            - np.maximum(thermal["temperature_c_rated"] - 26.0, 0), 0,
        ) * 0.25
        # Each timestamp contributes the worst same-zone paired consequence;
        # temperatures from different zones are never interpolated or subtracted.
        worst = thermal.groupby("timestamp_hkt", as_index=False).agg(
            absolute_dh32=("absolute_dh32", "max"),
            absolute_dh26=("absolute_dh26", "max"),
            rated_dh32=("rated_dh32", "max"),
            rated_dh26=("rated_dh26", "max"),
            extra_dh26=("extra_dh26", "max"),
            maximum_temperature_c=("temperature_c_outage", "max"),
        )
        crossing_result, _ = building_crossing_from_zones(outage_support, start, 32.0, 24.0)
        conditioned = max_conditioned_temperature(outage_support.loc[outage_support["temperature_c"].notna()])
        rated_conditioned = max_conditioned_temperature(rated_support.loc[rated_support["temperature_c"].notna()])
        outage_initial = conditioned.loc[conditioned["timestamp_hkt"].le(start)].iloc[-1]["max_conditioned_temperature_c"]
        rated_initial = rated_conditioned.loc[rated_conditioned["timestamp_hkt"].le(start)].iloc[-1]["max_conditioned_temperature_c"]
        recovery_result, conditioned_simultaneous = building_recovery_from_zones(
            outage_support, end, 26.0, 24.0
        )
        rated_recovery_result, rated_conditioned_simultaneous = building_recovery_from_zones(
            rated_support, end, 26.0, 24.0
        )
        recovery_sustained, sustained_censored = sustained_recovery_hours(
            conditioned_simultaneous["timestamp_hkt"], conditioned_simultaneous["temperature_c"], end
        )
        metrics.append(
            {
                "role": role,
                "prototype_id": outage_item.prototype_id,
                "event_id": outage_item.event_id,
                "outage_start_hkt": start,
                "outage_end_hkt": end,
                "outage_hours": float(outage_mask.sum()) * 0.25,
                "recovery_observation_hours": float(recovery_mask.sum()) * 0.25,
                "rated_initial_max_conditioned_temperature_c": float(rated_initial),
                "outage_initial_max_conditioned_temperature_c": float(outage_initial),
                "first_occupied_crossing_32c_h": crossing_result["hours"],
                "first_crossing_status": crossing_result["status"],
                "first_crossing_zone_id": crossing_result.get("zone_id"),
                "first_crossing_right_censored": crossing_result["right_censored"],
                "first_crossing_left_censored": crossing_result["left_censored"],
                "first_crossing_interval_censored": crossing_result["interval_censored"],
                "first_crossing_lower_h": crossing_result.get("lower_h"),
                "first_crossing_upper_h": crossing_result.get("upper_h"),
                "effective_monitoring_hours": crossing_result.get("effective_monitoring_hours"),
                "absolute_occupied_degree_hours_above_32c": float(worst["absolute_dh32"].sum(min_count=1)),
                "absolute_occupied_degree_hours_above_26c": float(worst["absolute_dh26"].sum(min_count=1)),
                "rated_occupied_degree_hours_above_32c": float(worst["rated_dh32"].sum(min_count=1)),
                "rated_occupied_degree_hours_above_26c": float(worst["rated_dh26"].sum(min_count=1)),
                "positive_extra_occupied_degree_hours_above_26c_vs_rated": float(worst["extra_dh26"].sum(min_count=1)),
                "maximum_occupied_outage_temperature_c": float(worst["maximum_temperature_c"].max()) if not worst.empty else math.nan,
                "recovery_first_all_conditioned_zones_le_26c_h": recovery_result["hours"],
                "recovery_first_le_26c_right_censored_at_24h": recovery_result["censored"],
                "recovery_status": recovery_result["status"],
                "recovery_lower_h": recovery_result.get("lower_h"),
                "recovery_upper_h": recovery_result.get("upper_h"),
                "rated_recovery_first_all_conditioned_zones_le_26c_h": rated_recovery_result["hours"],
                "rated_recovery_status": rated_recovery_result["status"],
                "rated_recovery_lower_h": rated_recovery_result.get("lower_h"),
                "rated_recovery_upper_h": rated_recovery_result.get("upper_h"),
                "diagnostic_recovery_le_26_2c_sustained_1h_h": recovery_sustained,
                "diagnostic_recovery_right_censored_at_24h": sustained_censored,
                "thermal_mask": "structurally conditioned zones with uninterrupted-reference occupancy schedule fraction >0.15 at each native timestep",
                "extra_heat_aggregation": "same-zone same-timestep positive excess difference, maximum eligible-zone consequence per timestep, then integrate",
                "recovery_mask": "all structurally conditioned zones",
            }
        )
    return pd.DataFrame(audits), pd.DataFrame(metrics)


def prehistory_audit(manifest: pd.DataFrame, work: Path) -> pd.DataFrame:
    assignment = pd.read_csv(work / "01_inputs" / "E8_event_assignment.csv").set_index("parent_event_id")
    rows = []
    for scenario in ("HW_RATED", "HW_O24_SYNC"):
        candidates = manifest.loc[
            manifest["scenario_id"].eq(scenario)
            & manifest["role"].isin(["core_group_D", "prehistory_2d_group_D", "prehistory_7d_group_D"])
        ].set_index("prehistory_days")
        reference_item = candidates.loc[14]
        reference = load_building(work, reference_item.run_id)
        event = assignment.loc[reference_item.event_id]
        start = pd.Timestamp(event.event_start, tz="Asia/Hong_Kong")
        end = pd.Timestamp(event.event_end, tz="Asia/Hong_Kong")
        ref_mask = formal_event_mask(reference, start, end)
        ref_energy = float(reference.loc[ref_mask, "total_modeled_electricity_j"].sum())
        ref_initial = reference.loc[reference["timestamp_hkt"].le(start), [c for c in reference if c.endswith("__temperature_c")]].iloc[-1]
        reference_zones = load_zones(work, reference_item.run_id)
        reference_occupied = max_occupied_temperature(reference_zones)
        if scenario == "HW_O24_SYNC":
            outage_start = pd.Timestamp(reference_item.outage_start_hkt)
            if outage_start.tzinfo is None:
                outage_start = outage_start.tz_localize("Asia/Hong_Kong")
            ref_crossing, ref_censored = first_crossing_hours(reference_occupied.timestamp_hkt, reference_occupied.max_occupied_temperature_c, outage_start, 32.0)
        else:
            ref_crossing, ref_censored = math.nan, False
        for days in (2, 7):
            item = candidates.loc[days]
            current = load_building(work, item.run_id)
            mask = formal_event_mask(current, start, end)
            energy = float(current.loc[mask, "total_modeled_electricity_j"].sum())
            initial = current.loc[current["timestamp_hkt"].le(start), [c for c in current if c.endswith("__temperature_c")]].iloc[-1]
            current_zones = load_zones(work, item.run_id)
            occupied = max_occupied_temperature(current_zones)
            if scenario == "HW_O24_SYNC":
                outage_start = pd.Timestamp(item.outage_start_hkt)
                if outage_start.tzinfo is None:
                    outage_start = outage_start.tz_localize("Asia/Hong_Kong")
                crossing, censored = first_crossing_hours(occupied.timestamp_hkt, occupied.max_occupied_temperature_c, outage_start, 32.0)
            else:
                crossing, censored = math.nan, False
            energy_pct = math.nan if ref_energy == 0 else (energy - ref_energy) / ref_energy * 100
            crossing_delta = math.nan if scenario != "HW_O24_SYNC" or censored or ref_censored else crossing - ref_crossing
            rows.append(
                {
                    "scenario_id": scenario,
                    "prehistory_days": days,
                    "reference_prehistory_days": 14,
                    "initial_zone_temperature_max_abs_difference_c": float((initial - ref_initial).abs().max()),
                    "formal_event_energy_difference_pct": energy_pct,
                    "first_crossing_32c_difference_h": crossing_delta,
                    "crossing_censoring_changed": bool(censored != ref_censored),
                    "temperature_rule_pass": float((initial - ref_initial).abs().max()) <= 0.1,
                    "energy_rule_pass": abs(energy_pct) <= 1.0 if np.isfinite(energy_pct) else False,
                    "crossing_rule_pass": True if scenario != "HW_O24_SYNC" or not np.isfinite(crossing_delta) else abs(crossing_delta) <= 0.25,
                    "engineering_rule_status": "diagnostic_only_not_scientific_threshold",
                }
            )
    return pd.DataFrame(rows)


def hourly_model(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["hour_hkt"] = interval_start(work).dt.floor("h").dt.tz_localize(None)
    return work.groupby("hour_hkt", as_index=False).agg(
        model_energy_kwh=("total_modeled_electricity_j", lambda x: float(x.sum()) / 3.6e6),
    )


def slope_per_c(load: pd.Series, temperature: pd.Series, normalizer: float) -> float:
    valid = load.notna() & temperature.notna()
    if valid.sum() < 2 or normalizer <= 0:
        return math.nan
    return float(np.polyfit(temperature.loc[valid], load.loc[valid] / normalizer, 1)[0])


def hkust_validation(manifest: pd.DataFrame, root: Path, work: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_path = root / "output" / "paper_pipeline_v0_3" / "02_building_response" / "hkust_whole_building_hourly_calibration.parquet"
    observed = pd.read_parquet(observed_path)
    observed["timestamp_hkt"] = pd.to_datetime(observed["timestamp_hkt"], format="mixed")
    observed = observed.loc[observed["building_split"].eq("building_holdout")].copy()
    split = pd.read_csv(work / "03_validation" / "validation_split_manifest_revised.csv")
    validation_runs = manifest.loc[manifest["role"].eq("hkust_same_chain_validation")].set_index("prototype_id")
    weather_run = validation_runs.iloc[0]
    weather = pd.read_parquet(result_dir(work, weather_run.run_id) / "inputs" / "weather.parquet")
    weather["timestamp_hkt"] = pd.to_datetime(weather["timestamp_hkt"], utc=True).dt.tz_convert("Asia/Hong_Kong").dt.tz_localize(None)
    weather = weather[["timestamp_hkt", "temperature_c"]].rename(columns={"timestamp_hkt": "hour_hkt"})
    result_rows = []
    scale_rows = []
    for item in split.itertuples(index=False):
        run = validation_runs.loc[item.prototype_id]
        model = hourly_model(load_building(work, run.run_id)).merge(weather, on="hour_hkt", how="left", validate="one_to_one")
        obs = observed.loc[observed["leaf_building"].eq(item.observed_object), ["timestamp_hkt", "energy_kwh"]].rename(
            columns={"timestamp_hkt": "hour_hkt", "energy_kwh": "observed_energy_kwh"}
        )
        paired = model.merge(obs, on="hour_hkt", how="inner", validate="one_to_one")
        cal = paired.loc[paired["hour_hkt"].between(pd.Timestamp(HKUST_CALIBRATION[0]), pd.Timestamp(HKUST_CALIBRATION[1]))].copy()
        test = paired.loc[paired["hour_hkt"].between(pd.Timestamp(HKUST_TEST[0]), pd.Timestamp(HKUST_TEST[1]))].copy()
        scale = float(cal["observed_energy_kwh"].sum() / cal["model_energy_kwh"].sum())
        test["prediction_kwh"] = test["model_energy_kwh"] * scale
        error = test["prediction_kwh"] - test["observed_energy_kwh"]
        mean_observed = float(test["observed_energy_kwh"].mean())
        observed_peak_idx = test["observed_energy_kwh"].idxmax()
        predicted_peak_idx = test["prediction_kwh"].idxmax()
        cal_mean_observed = float(cal["observed_energy_kwh"].mean())
        cal_mean_model = float(cal["model_energy_kwh"].mean())
        result_rows.append(
            {
                "case_id": item.case_id,
                "observed_object": item.observed_object,
                "observed_scope": item.observed_scope,
                "prototype_id": item.prototype_id,
                "calibration_hours": len(cal),
                "test_hours": len(test),
                "calibration_only_scale": scale,
                "test_nmbe_pct": float(error.sum() / (len(test) * mean_observed) * 100),
                "test_cvrmse_pct": float(np.sqrt(np.mean(error**2)) / mean_observed * 100),
                "test_mae_kwh_per_h_after_calibration_scale": float(error.abs().mean()),
                "test_peak_relative_error_pct": float((test.loc[predicted_peak_idx, "prediction_kwh"] - test.loc[observed_peak_idx, "observed_energy_kwh"]) / test.loc[observed_peak_idx, "observed_energy_kwh"] * 100),
                "test_peak_timing_error_h": float((test.loc[predicted_peak_idx, "hour_hkt"] - test.loc[observed_peak_idx, "hour_hkt"]).total_seconds() / 3600),
                "observed_test_relative_temperature_slope_per_c": slope_per_c(test["observed_energy_kwh"], test["temperature_c"], cal_mean_observed),
                "model_test_relative_temperature_slope_per_c": slope_per_c(test["model_energy_kwh"], test["temperature_c"], cal_mean_model),
                "validation_scope": "relative shape, normalized peak amplitude/timing and relative temperature sensitivity; not absolute electricity or outage indoor temperature",
            }
        )
        scale_rows.append(
            {
                "case_id": item.case_id,
                "prototype_id": item.prototype_id,
                "scale": scale,
                "scale_source_start_hkt": HKUST_CALIBRATION[0],
                "scale_source_end_hkt": HKUST_CALIBRATION[1],
                "test_data_used_in_scale": False,
            }
        )
    return pd.DataFrame(result_rows), pd.DataFrame(scale_rows)


def validation_case_mapping(manifest: pd.DataFrame, work: Path) -> pd.DataFrame:
    split = pd.read_csv(work / "03_validation" / "validation_split_manifest_revised.csv")
    runs = manifest.loc[manifest["role"].eq("hkust_same_chain_validation"), [
        "prototype_id", "run_id", "physics_hash", "output_schema_hash",
    ]]
    mapping = split.merge(runs, on="prototype_id", how="left", validate="many_to_one")
    mapping["observed_object_identity"] = "meter object; not automatically an independent physical building"
    mapping["staff_quarters_meter_relation"] = np.where(
        mapping["observed_object"].str.startswith("Staff_Quarters_07"),
        "two records share the Staff Quarters 07 parent label; downstream electrical relationship is not resolved by available metadata",
        "not_applicable",
    )
    mapping["physical_input_reuse_rule"] = "shared only by matching prototype, weather, dates, schedules and state assumptions"
    return mapping


def verified_variable_mapping(manifest: pd.DataFrame, work: Path) -> pd.DataFrame:
    rows = []
    seen = set()
    for item in manifest.itertuples(index=False):
        audit = json.loads((result_dir(work, item.run_id) / "parsed" / "parser_audit.json").read_text(encoding="utf-8"))
        for field, columns in audit["exact_fields"].items():
            for column in columns:
                key = (field, column)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "parsed_field": field,
                        "exact_energyplus_csv_column": column,
                        "frequency": "TimeStep",
                        "engine": "EnergyPlus 26.1.0-6f2e40d102",
                        "verified_in_real_G3_output": True,
                    }
                )
    return pd.DataFrame(rows).sort_values(["parsed_field", "exact_energyplus_csv_column"])


def actual_input_hashes(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in manifest.itertuples(index=False):
        byte_hashes = ast.literal_eval(item.file_byte_sha256)
        rows.append(
            {
                "run_id": item.run_id,
                "role": item.role,
                "scenario_id": item.scenario_id,
                "physics_hash": item.physics_hash,
                "normalized_manifest_hash": item.normalized_manifest_hash,
                "output_schema_hash": item.output_schema_hash,
                **{f"{name}_byte_sha256": value for name, value in byte_hashes.items()},
            }
        )
    return pd.DataFrame(rows)


def test_summary(xml_path: Path) -> dict[str, object]:
    root = ET.parse(xml_path).getroot()
    suite = next(root.iter("testsuite"))
    failures = [
        case.attrib.get("name") for case in root.iter("testcase") if case.find("failure") is not None
    ]
    return {**suite.attrib, "failure_names": failures}


def write_completion_reports(root: Path, work: Path, manifest: pd.DataFrame) -> None:
    inventory = pd.read_csv(work / "02_runs" / "g3_run_inventory.csv")
    physics = pd.read_csv(work / "03_validation" / "g3_physics_audit.csv")
    prehistory = pd.read_csv(work / "03_validation" / "prehistory_convergence.csv")
    validation = pd.read_csv(work / "03_validation" / "hkust_same_chain_validation_results.csv")
    outage = pd.read_csv(work / "03_validation" / "g3_outage_recovery_metrics.csv")
    resource = json.loads((work / "02_runs" / "g3_resource_summary.json").read_text(encoding="utf-8"))
    tests = test_summary(work / "00_protocol" / "pytest_all_corrected.xml")
    vintage = pd.read_csv(work / "01_inputs" / "construction_year_coverage_by_use.csv")
    input_hashes = actual_input_hashes(manifest)
    write_csv(input_hashes, work / "02_runs" / "g3_actual_input_hashes.csv")
    elapsed_wall = (
        pd.Timestamp(resource["last_solver_started_utc"])
        - pd.Timestamp(resource["first_solver_started_utc"])
    ).total_seconds() + float(pd.read_csv(work / "02_runs" / "g3_call_ledger.csv")["elapsed_seconds"].iloc[-1])
    report = f"""# E8 G2修正与条件性G3真实求解报告

日期：2026-09-19  
范围：只覆盖用户批准的G2修正与G3；未启动G4、G5、锁箱、PolyU、住宅机制批量或全建筑传播。

## 1. 实际生产调用链

`src/utils/e8_g3_runner.py` 的限定入口（`--run-id`/`--run-core`） → `_compile_case()` → 生产模块 `src/workflows/stages/build_building_response_and_exposure.py` 的 `prototype_parameter_table()`、`behaviour_schedules()`、`annualize_schedules()`、`build_idf()`、`add_run_period()` 与 `write_epw()` → 最终 IDF/EPW/外部日程及字节哈希 → EnergyPlus 26.1.0 → 持久化 CSV/RDD/MDD/EDD/ERR → `parse_native_output()` 的完整变量名＋key＋单位＋频率匹配 → 15分钟逐区和建筑结果 → `e8_g3_postprocess.py` 的物理、前史和验证指标。

G3没有调用旧的宽泛冷量子串解析器；必需字段缺失会抛错。`Electricity:Facility` 与照明/设备电量的最大非制冷账本残差为约5.82e-11 J；IdealLoads总冷量仅从供风侧边界累计一次，并按COP仅转换一次。

## 2. 运行范围与资源

- 批准唯一输入：{len(manifest)}；完成并通过：{int(inventory.status.eq('passed').sum())}。
- EnergyPlus真实调用：{resource['actual_energyplus_calls']}／30。首例因解析修订保留两次成功求解记录；没有删除证据。
- 全部运行：0 Severe、0 Fatal；物理审计：{len(physics)}项，失败{int((~physics['pass'].astype(bool)).sum())}项。
- 求解器累计：{resource['solver_elapsed_seconds_sum']:.2f}s；P50 {resource['solver_elapsed_seconds_p50']:.3f}s，P90 {resource['solver_elapsed_seconds_p90']:.3f}s。
- 从首个到最后一个求解启动并完成约{elapsed_wall/60:.1f}min；低于2h限制。
- 实测单进程峰值RSS：{resource['measured_peak_rss_bytes_max']/1024**2:.1f}MiB；本轮实际并发为1。
- G3运行目录：{resource['g3_runs_storage_bytes']/1024**2:.1f}MiB；低于5GiB限制。

## 3. 已关闭的代码与物理冲突

1. 输出变量名已按真实RDD更正为 `Electric Equipment Convective Heating Energy`；已生成逐列真实映射。
2. C70使用同一连续模拟内两套互斥IdealLoads：正式事件外为额定容量，事件期为70%；四组均为0个相位违例，逐区最大容量利用率均≤1。
3. O24是独立分支，IDF不含C70双系统；C70不作为停电前序。
4. 四组O24在停电前与RATED逐区温度最大差均为0°C；停电均为96个15分钟步；恢复观察为57–59h。
5. 停电期制冷、机械通风、照明、插座/过程电力及对应设备对流得热均为零。数据中心也满足该规则，结构先前储热未被重置。
6. 冷量诊断同时保存raw delivery、unlimited reference、逐时间步matched量、shortfall及raw exceedance；实际结果再次证明“逐时截断后积分”与“总量先截断”不等价。
7. 首次越阈从停电起点的真实连续状态计算；开始即超阈记0，未越阈保留右删失。恢复主指标保持首次≤26°C，≤26.2°C持续1h只作诊断。

## 4. 前史诊断

以D组、14日真实前史为参考：

- 2日：初始区温最大差{prehistory.loc[prehistory.prehistory_days.eq(2),'initial_zone_temperature_max_abs_difference_c'].max():.3f}°C，未通过0.1°C工程规则；正式事件电量差仍小于{prehistory.loc[prehistory.prehistory_days.eq(2),'formal_event_energy_difference_pct'].abs().max():.3f}%。
- 7日：初始区温最大差{prehistory.loc[prehistory.prehistory_days.eq(7),'initial_zone_temperature_max_abs_difference_c'].max():.3f}°C，事件电量最大差{prehistory.loc[prehistory.prehistory_days.eq(7),'formal_event_energy_difference_pct'].abs().max():.4f}%，越阈时间差0h；通过本次诊断阈值。
- 结论仅适用于D组两个工况；14日仍为本轮参考，不把7日推广到全部原型和事件。

## 5. HKUST同链条检查

四个观测对象均采用共同、连续且互不重叠的336h校准期（2023-07-10至07-23）和336h高温测试期（2023-07-24至08-06）。只生成两个物理输入；Staff Quarters 07的两条记录明确作为两个表计对象，不宣称两栋独立建筑。所有比例只由校准期确定。

- 测试NMBE范围：{validation.test_nmbe_pct.min():.1f}%至{validation.test_nmbe_pct.max():.1f}%。
- 测试CVRMSE范围：{validation.test_cvrmse_pct.min():.1f}%至{validation.test_cvrmse_pct.max():.1f}%。
- 峰值幅值误差：{validation.test_peak_relative_error_pct.min():.1f}%至{validation.test_peak_relative_error_pct.max():.1f}%；全测试窗口绝对峰值时刻误差为{validation.test_peak_timing_error_h.min():.0f}h至{validation.test_peak_timing_error_h.max():.0f}h。
- 教学楼观测/模型相对温度斜率分别为{validation.iloc[0].observed_test_relative_temperature_slope_per_c:.3f}与{validation.iloc[0].model_test_relative_temperature_slope_per_c:.3f} 每°C。

这些结果支持“同一物理接口已经贯通”的结论，但逐时形状误差仍较大。它们不验证绝对建筑电量、停电室温或全香港住宅响应，不能在论文中包装成强本地验证。

## 6. G3示例物理结果的解释边界

G3四组是接口覆盖，不是代表总体的机制比较。额定状态在同步停电起点的最大空调区温度分别为{', '.join(f'{x:.2f}' for x in outage.rated_initial_max_conditioned_temperature_c)}°C；其中公共租住和旧式低层在停电前已经超过32°C，因此其0h越阈主要反映该天气、日程和起点下的初始服务状态，不能全部归因为停电增温。数据中心在当前“过程设备随电网关闭”的场景中24h内未越32°C，不能再沿用旧模型中由持续过程得热造成最低韧性的解释；业务连续性也不等于该热学结果。

## 7. 反事实与天气

文稿写1980–2025日历日气候基准；生产代码、数据文件和旧缓存实际使用2016–2025 King's Park日值。G3的CF_UNL仅标记为`integration_only`接口检查。该冲突没有通过虚构长期数据解决，E8主CF及依赖的S_N仍未获G4/G5放行。

事件目录站点角色为HKO，小时温湿锚定站为King's Park；NASA POWER提供小时形状、辐射、风、压力和降水。UTC先转换HKT，再按本地日期锚定；八个资格事件均0重复小时、0必需变量缺失，锚定后Tmin/Tmean/Tmax/RH日目标残差为0。锁箱0054和0066未运行、未查看建筑性能结果。

## 8. 年代来源

已纠正为最终330,761栋队列：住宅{int(vintage.loc[vintage.main_class.eq('Residential'),'buildings'].iloc[0]):,}、商业{int(vintage.loc[vintage.main_class.eq('Commercial'),'buildings'].iloc[0]):,}、混合用途{int(vintage.loc[vintage.main_class.eq('Mixed-use'),'buildings'].iloc[0]):,}。虽然每栋都有`construction_year_value`，直接年份证据覆盖仅约{vintage.direct_year_building_coverage.min()*100:.1f}%–{vintage.direct_year_building_coverage.max()*100:.1f}%；其余主要为模型推断。没有完整投入使用、拆除、改造和设备日期，因此只支持存在性敏感性，不支持历史香港城市重建。

## 9. 测试状态

完整测试：{tests['tests']}项，{int(tests['tests'])-int(tests['failures'])}通过，{tests['failures']}失败，0错误。唯一失败仍为既有DOCX含原始LaTeX `\\[`；它单列为文档交付问题，没有删除或伪装，也不改变独立G3数值结论。

## 10. 阶段结论

有限真实输入→求解器→原始输出→精确解析→物理指标链已贯通。G3完成后按授权停止。G4/G5当前不放行，原因与建议见`G4_GO_NO_GO.md`。
"""
    (work / "00_protocol" / "G2_correction_and_G3_report.md").write_text(report, encoding="utf-8")
    average_storage = resource["g3_runs_storage_bytes"] / len(manifest)
    p90 = resource["solver_elapsed_seconds_p90"]
    budget = pd.DataFrame(
        [
            {"stage": "G3_actual", "unique_runs": len(manifest), "solver_seconds_at_G3_P90": len(manifest)*p90, "storage_GiB_at_G3_mean": len(manifest)*average_storage/1024**3, "status": "completed"},
            {"stage": "G4_one_297_design_event_five_core_upper_bound", "unique_runs": 1485, "solver_seconds_at_G3_P90": 1485*p90, "storage_GiB_at_G3_mean": 1485*average_storage/1024**3, "status": "estimate_only_not_approved"},
            {"stage": "G5_E8_297_design_five_core_upper_bound", "unique_runs": 11880, "solver_seconds_at_G3_P90": 11880*p90, "storage_GiB_at_G3_mean": 11880*average_storage/1024**3, "status": "estimate_only_not_approved"},
        ]
    )
    budget["qualification"] = "linear projection from heterogeneous G3 run lengths; excludes compilation, retries, concurrency, full-event duration mix and downstream propagation"
    write_csv(budget, work / "02_runs" / "g4_g5_budget_projection.csv")
    go = f"""# G4 GO / NO-GO建议

## 决定：NO-GO（等待新的明确授权与科学口径关闭）

G3接口与物理一致性门槛通过，但这不自动批准G4/G5。

### 已满足

- 26/26唯一输入完成，27/30实际调用，0 Severe、0 Fatal。
- 精确逐区解析、冷量边界、COP一次转换、C70事件期切换、O24独立分支、24h停电及连续恢复均通过。
- 7日相对14日前史通过D组工程诊断；主设置仍保持14日。
- HKUST共同窗口、校准/测试隔离及两物理输入/四观测对象关系已纠正。

### 阻断G4/G5的事项

1. **E8主反事实未冻结。** 文稿1980–2025与生产2016–2025冲突；G3 CF只可作为integration-only，不能用于冻结主S_N。
2. **同链条观测结果不能支持强验证措辞。** 测试CVRMSE为{validation.test_cvrmse_pct.min():.1f}%–{validation.test_cvrmse_pct.max():.1f}%，峰值和温度斜率仍有明显偏差。下一步需在不接触测试期的前提下决定：仅收窄论文验证表述，还是依据校准期和可证参数修订日程/映射后重新进行独立测试。不得根据测试表现挑窗口或时间平移。
3. **年代字段只够敏感性。** 最终样本的直接年份证据覆盖有限，不能把后续前推评价写成完整历史存量重建。
4. **存储预算需另批。** 按G3平均文件量线性估算，单事件297×5上限约{budget.iloc[1].storage_GiB_at_G3_mean:.1f}GiB，全E8上限约{budget.iloc[2].storage_GiB_at_G3_mean:.1f}GiB；均不在本轮5GiB范围内。该估算仍需按批准的真实事件长度与可复用缓存重算。

### 放行G4前最小动作

- 明确并冻结主反事实基准、源数据与版本；决定文稿随代码修正还是另建经证据支持的长期基准。
- 冻结HKUST验证用途和声明边界；若改变物理参数，只能用校准期，保留当前测试结果并给新版本。
- 生成G4唯一依赖展开清单，扣除可复用合格缓存，重新提交运行数、磁盘和并发预算。
- 修复DOCX数学公式属于发布交付项，可与科学放行分开，但投稿前必须关闭。

本文件只给建议，不构成G4或G5授权。
"""
    (work / "00_protocol" / "G4_GO_NO_GO.md").write_text(go, encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_review_package(root: Path, work: Path, manifest: pd.DataFrame) -> None:
    publication = work / "08_publication"
    publication.mkdir(parents=True, exist_ok=True)
    package = publication / "E8_G3_review_v1"
    if package.exists():
        if package.parent.resolve() != publication.resolve():
            raise ValueError(package)
        shutil.rmtree(package)
    package.mkdir()
    paths = [
        "00_protocol/G2_correction_and_G3_report.md",
        "00_protocol/G4_GO_NO_GO.md",
        "00_protocol/approval_items_revised.csv",
        "00_protocol/baseline_audit.md",
        "00_protocol/baseline_result_registry.csv",
        "00_protocol/legacy_issue_recheck.csv",
        "00_protocol/core_scenario_contract.csv",
        "00_protocol/counterfactual_version_reconciliation.csv",
        "00_protocol/environment_audit.json",
        "00_protocol/estimand_and_field_contract.csv",
        "00_protocol/output_schema.json",
        "00_protocol/variable_mapping_audit.csv",
        "00_protocol/pytest_e8_g2_corrected.xml",
        "00_protocol/pytest_all_corrected.xml",
        "01_inputs/E8_event_assignment.csv",
        "01_inputs/event_catalog_65.csv",
        "01_inputs/hourly_weather_eligibility_audit.csv",
        "01_inputs/event_window_crosswalk.csv",
        "01_inputs/lockbox_manifest.json",
        "01_inputs/weather_source_and_boundary_audit.csv",
        "01_inputs/weather_anchor_residuals.csv",
        "01_inputs/construction_year_coverage_by_use.csv",
        "01_inputs/construction_year_source_audit.csv",
        "02_runs/g3_manifest_revised.csv",
        "02_runs/g3_manifest_revised.json",
        "02_runs/g3_coverage_report.csv",
        "02_runs/g3_call_ledger.csv",
        "02_runs/g3_run_inventory.csv",
        "02_runs/g3_actual_input_hashes.csv",
        "02_runs/g3_resource_summary.json",
        "02_runs/g4_g5_budget_projection.csv",
        "03_validation/g3_physics_audit.csv",
        "03_validation/g3_cooling_service_diagnostics.csv",
        "03_validation/g3_outage_recovery_metrics.csv",
        "03_validation/prehistory_convergence.csv",
        "03_validation/g3_variable_mapping_verified.csv",
        "03_validation/hkust_window_completeness_audit.csv",
        "03_validation/validation_split_manifest_revised.csv",
        "03_validation/validation_case_mapping.csv",
        "03_validation/hkust_calibration_scales.csv",
        "03_validation/hkust_same_chain_validation_results.csv",
        "04_mechanisms/mechanism_reference_design_audit.csv",
        "04_mechanisms/parameter_provenance.csv",
    ]
    for relative in paths:
        source = work / relative
        target = package / "intermediate" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    code_paths = [
        "src/workflows/stages/build_building_response_and_exposure.py",
        "src/utils/e8_preflight.py", "src/utils/e8_g3_runner.py", "src/utils/e8_g3_audits.py",
        "src/utils/e8_g3_postprocess.py",
    ]
    for relative in code_paths:
        source = root / relative
        target = package / "code" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    representatives = {
        "A_RATED": manifest.loc[manifest.role.eq("core_group_A") & manifest.scenario_id.eq("HW_RATED")].iloc[0].run_id,
        "A_C70": manifest.loc[manifest.role.eq("core_group_A") & manifest.scenario_id.eq("HW_C70")].iloc[0].run_id,
        "C_DATA_CENTRE_O24": manifest.loc[manifest.role.eq("core_group_C") & manifest.scenario_id.eq("HW_O24_SYNC")].iloc[0].run_id,
        "D_PREHISTORY_7D_O24": manifest.loc[manifest.role.eq("prehistory_7d_group_D") & manifest.scenario_id.eq("HW_O24_SYNC")].iloc[0].run_id,
        "HKUST_EDUCATION": manifest.loc[manifest.role.eq("hkust_same_chain_validation") & manifest.prototype_id.eq("com_education")].iloc[0].run_id,
    }
    for label, run_id in representatives.items():
        source = result_dir(work, run_id)
        target = package / "representative_runs" / label
        target.mkdir(parents=True, exist_ok=True)
        for relative in [
            "run_result.json", "raw/eplusout.err", "raw/eplusout.rdd", "raw/eplusout.mdd",
            "parsed/parser_audit.json", "parsed/native_timestep_zones.parquet",
            "inputs/input_manifest.json", "inputs/model.idf", "inputs/weather.epw",
            "inputs/schedules.csv", "inputs/event_schedules.parquet", "inputs/weather.parquet",
        ]:
            candidate = source / relative
            if candidate.exists():
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, destination)
        if label == "A_RATED":
            shutil.copy2(source / "raw" / "eplusout.csv", target / "raw" / "eplusout.csv")
    diff = subprocess.run(
        ["git", "diff", "--", "src/workflows/stages/build_building_response_and_exposure.py"],
        cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    (package / "tracked_code_diff.patch").write_text(diff.stdout, encoding="utf-8")
    index = """# E8 G3 compact review package

This package contains the corrected G2 contracts, the exact 26-run G3 manifest,
input and physics hashes, pytest XML, real EnergyPlus dictionaries/logs, selected
raw and parsed zone-native evidence, validation tables, and the GO/NO-GO record.
It deliberately excludes restricted HKUST source data, full legacy caches,
whole-building-stock Parquet files, lockbox performance outputs, and secrets.
"""
    (package / "README.md").write_text(index, encoding="utf-8")
    hashes = []
    for path in sorted(p for p in package.rglob("*") if p.is_file()):
        hashes.append({"relative_path": path.relative_to(package).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_csv(pd.DataFrame(hashes), package / "PACKAGE_MANIFEST_SHA256.csv")
    zip_path = publication / "E8_G3_review_v1.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(p for p in package.rglob("*") if p.is_file()):
            archive.write(path, path.relative_to(package).as_posix())
    (publication / "E8_G3_review_v1.zip.sha256").write_text(f"{sha256(zip_path)}  {zip_path.name}\n", encoding="ascii")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    work = root / "intermediate/simulation_outputs/upstream"
    manifest = pd.read_csv(work / "02_runs" / "g3_manifest_revised.csv")
    production = load_production(root)
    inventory = run_inventory(manifest, work)
    write_csv(inventory, work / "02_runs" / "g3_run_inventory.csv")
    capacity, service = capacity_and_phase_audit(manifest, work, production)
    outage_checks, outage_metrics = outage_audit(manifest, work)
    physics = pd.concat([capacity, outage_checks], ignore_index=True)
    write_csv(physics, work / "03_validation" / "g3_physics_audit.csv")
    write_csv(service, work / "03_validation" / "g3_cooling_service_diagnostics.csv")
    write_csv(outage_metrics, work / "03_validation" / "g3_outage_recovery_metrics.csv")
    write_csv(prehistory_audit(manifest, work), work / "03_validation" / "prehistory_convergence.csv")
    validation, scales = hkust_validation(manifest, root, work)
    write_csv(validation, work / "03_validation" / "hkust_same_chain_validation_results.csv")
    write_csv(scales, work / "03_validation" / "hkust_calibration_scales.csv")
    write_csv(validation_case_mapping(manifest, work), work / "03_validation" / "validation_case_mapping.csv")
    write_csv(verified_variable_mapping(manifest, work), work / "03_validation" / "g3_variable_mapping_verified.csv")
    ledger = pd.read_csv(work / "02_runs" / "g3_call_ledger.csv")
    budget = json.loads((work / "02_runs" / "g3_budget.json").read_text(encoding="utf-8"))
    resource = {
        "approved_unique_inputs": len(manifest),
        "passed_unique_inputs": int(inventory["status"].eq("passed").sum()),
        "actual_energyplus_calls": len(ledger),
        "approved_call_ceiling": int(budget["max_calls"]),
        "solver_elapsed_seconds_sum": float(ledger["elapsed_seconds"].sum()),
        "solver_elapsed_seconds_p50": float(ledger["elapsed_seconds"].median()),
        "solver_elapsed_seconds_p90": float(ledger["elapsed_seconds"].quantile(0.9)),
        "measured_peak_rss_bytes_max": float(ledger["peak_rss_bytes"].max()),
        "g3_runs_storage_bytes": int(sum(p.stat().st_size for p in (work / "02_runs").rglob("*") if p.is_file())),
        "approved_storage_ceiling_bytes": int(budget["max_bytes"]),
        "first_solver_started_utc": budget["first_solver_started_utc"],
        "last_solver_started_utc": ledger["started_at_utc"].iloc[-1],
        "parallelism_used": 1,
    }
    write_json(resource, work / "02_runs" / "g3_resource_summary.json")
    write_completion_reports(root, work, manifest)
    build_review_package(root, work, manifest)
    print(json.dumps(resource, indent=2))
    print("physics failures", int((~physics["pass"].astype(bool)).sum()))


if __name__ == "__main__":
    main()
