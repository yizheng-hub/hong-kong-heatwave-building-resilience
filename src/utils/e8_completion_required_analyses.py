"""Approved, solver-free analyses required by the E8 completion contract.

The functions in this module are production functions. They accept ordinary
tables, validate the full decision schema, and retain unknown/censored states.
Fixtures exercise these same functions; no fixture-only numerical shortcut is
used here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.e8_completion_common import Blocked, canonical
from utils.e8_g3_postprocess import build_expected_monitoring_support, expected_native_timestamps


REQUIRED_DECISIONS = (
    "absolute_performance_thresholds",
    "multi_stage_threshold_set",
    "day_night_windows",
    "region_snapshot_times_hkt",
    "construction_year_provenance",
)


def _approved(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Blocked(f"analysis decision {name} must be an object")
    if value.get("approval_status") != "approved":
        raise Blocked(f"analysis decision {name} is not explicitly approved")
    return value


def validate_analysis_decisions(contract: dict[str, Any]) -> dict[str, Any]:
    """Validate the formal decision schema before any solver can be reached."""
    values = contract.get("decision_parameters")
    if not isinstance(values, dict):
        raise Blocked("analysis contract has no decision_parameters object")
    missing = [name for name in REQUIRED_DECISIONS if values.get(name) in (None, "", [], {})]
    if missing:
        raise Blocked("required analysis decisions are not frozen: " + ", ".join(missing))

    absolute = _approved(values["absolute_performance_thresholds"], "absolute_performance_thresholds")
    rules = absolute.get("rules")
    if not isinstance(rules, list) or not rules:
        raise Blocked("absolute_performance_thresholds.rules must be non-empty")
    allowed_ops = {">", ">=", "<", "<=", "=="}
    rule_ids = []
    for rule in rules:
        required = {"rule_id", "stage", "metric", "operator", "threshold", "unit", "eligibility"}
        if not isinstance(rule, dict) or required - set(rule):
            raise Blocked(f"absolute threshold rule has invalid schema: {rule}")
        if rule["operator"] not in allowed_ops or not np.isfinite(float(rule["threshold"])):
            raise Blocked(f"absolute threshold rule has invalid operator/value: {rule['rule_id']}")
        rule_ids.append(str(rule["rule_id"]))
    if len(rule_ids) != len(set(rule_ids)):
        raise Blocked("absolute threshold rule_id values are not unique")

    multi = _approved(values["multi_stage_threshold_set"], "multi_stage_threshold_set")
    sets = multi.get("sets")
    if not isinstance(sets, list) or not sets:
        raise Blocked("multi_stage_threshold_set.sets must be non-empty")
    for item in sets:
        if not isinstance(item, dict) or not {"set_id", "rule_ids", "minimum_failed"} <= set(item):
            raise Blocked(f"multi-stage set has invalid schema: {item}")
        selected = [str(x) for x in item["rule_ids"]]
        if not selected or not set(selected) <= set(rule_ids):
            raise Blocked(f"multi-stage set references unknown rules: {item.get('set_id')}")
        if not 1 <= int(item["minimum_failed"]) <= len(selected):
            raise Blocked(f"multi-stage minimum_failed is invalid: {item.get('set_id')}")

    windows = _approved(values["day_night_windows"], "day_night_windows")
    periods = windows.get("periods")
    if not isinstance(periods, list) or {x.get("period_id") for x in periods if isinstance(x, dict)} != {"day", "night"}:
        raise Blocked("day_night_windows requires exactly day and night periods")
    for period in periods:
        for name in ("start_hour_hkt", "end_hour_hkt"):
            if name not in period or not 0 <= int(period[name]) <= 23:
                raise Blocked(f"invalid {name} in day/night contract")
    humidity = windows.get("humidity_metric", {})
    if humidity.get("name") not in {"same_zone_heat_index_c", "same_zone_temperature_rh"}:
        raise Blocked("humidity_metric must retain same-zone temperature/RH identity")
    recovery = windows.get("sustained_recovery", {})
    if float(recovery.get("threshold_c", np.nan)) != 26.2 or float(recovery.get("support_hours", np.nan)) != 1.0:
        raise Blocked("sustained recovery diagnostic must remain 26.2 C for one supported hour")

    snapshots = _approved(values["region_snapshot_times_hkt"], "region_snapshot_times_hkt")
    roles = snapshots.get("snapshot_roles")
    if not isinstance(roles, list) or not roles:
        raise Blocked("region snapshot roles must be non-empty")
    allowed_roles = {"event_start", "hw_peak", "outage_start", "restore", "restore_plus_6h", "restore_plus_24h"}
    if not set(roles) <= allowed_roles:
        raise Blocked("region snapshot role is not supported")
    regions = snapshots.get("regions")
    if not isinstance(regions, list) or "Hong Kong total" not in regions:
        raise Blocked("region snapshots must include Hong Kong total")

    provenance = _approved(values["construction_year_provenance"], "construction_year_provenance")
    if provenance.get("unknown_source_policy") != "retain_unknown":
        raise Blocked("construction-year source must retain unknown records")
    if provenance.get("same_year_policy") != "unknown_with_start_end_sensitivity":
        raise Blocked("construction-year same-year policy is not supported")
    if set(provenance.get("allowed_source_classes", [])) != {"direct", "inferred", "unknown"}:
        raise Blocked("construction-year provenance classes must be direct/inferred/unknown")
    return values


def analysis_decision_hash(decisions: dict[str, Any]) -> str:
    return canonical({name: decisions[name] for name in REQUIRED_DECISIONS})


def _compare(value: pd.Series, operator: str, threshold: float) -> pd.Series:
    if operator == ">": return value > threshold
    if operator == ">=": return value >= threshold
    if operator == "<": return value < threshold
    if operator == "<=": return value <= threshold
    if operator == "==": return value == threshold
    raise Blocked(f"unsupported comparison operator: {operator}")


def compute_absolute_performance_thresholds(building_events: pd.DataFrame, decision: dict[str, Any]) -> pd.DataFrame:
    """Evaluate approved physical thresholds without filling missing outcomes."""
    _approved(decision, "absolute_performance_thresholds")
    keys = ["BUILDINGSTRUCTUREID", "event_id"]
    if any(c not in building_events for c in keys) or building_events.duplicated(keys).any():
        raise Blocked("building-event input requires unique BUILDINGSTRUCTUREID x event_id")
    source_frame = building_events.copy()
    if "relative_delta" not in source_frame and {"net_delta_kwh", "E_CF_kwh"} <= set(source_frame):
        denominator = pd.to_numeric(source_frame.E_CF_kwh, errors="coerce")
        source_frame["relative_delta"] = (pd.to_numeric(source_frame.net_delta_kwh, errors="coerce") /
                                           denominator.where(denominator.gt(0)))
    if "cooling_shortfall_fraction" not in source_frame:
        shortfall = next((x for x in ("cooling_shortfall_kwh", "shortfall_ref_kwh_thermal") if x in source_frame), None)
        required = next((x for x in ("required_ref_kwh_thermal", "required_reference_kwh_thermal") if x in source_frame), None)
        if shortfall and required:
            denominator = pd.to_numeric(source_frame[required], errors="coerce")
            source_frame["cooling_shortfall_fraction"] = (pd.to_numeric(source_frame[shortfall], errors="coerce") /
                                                            denominator.where(denominator.gt(0)))
    if "recovery_failed_within_24h" not in source_frame and "recovery_status" in source_frame:
        status = source_frame.recovery_status.fillna("").astype(str)
        source_frame["recovery_failed_within_24h"] = np.where(
            status.eq("right_censored"), 1.0,
            np.where(status.isin(["exact", "interval_censored", "left_censored"]), 0.0, np.nan))
    rows = []
    for rule in decision["rules"]:
        metric = str(rule["metric"])
        if metric not in source_frame:
            raise Blocked(f"absolute threshold metric is absent: {metric}")
        values = pd.to_numeric(source_frame[metric], errors="coerce")
        eligible = values.notna()
        if rule["eligibility"] == "positive_reference_demand":
            field = str(rule.get("eligibility_field", "required_ref_kwh_thermal"))
            if field not in source_frame:
                raise Blocked(f"threshold eligibility field is absent: {field}")
            eligible &= pd.to_numeric(source_frame[field], errors="coerce").gt(0)
        elif rule["eligibility"] == "valid_monitoring_support":
            field = str(rule.get("eligibility_field", "effective_monitoring_hours"))
            if field not in source_frame:
                raise Blocked(f"threshold monitoring eligibility field is absent: {field}")
            eligible &= pd.to_numeric(source_frame[field], errors="coerce").gt(0)
        elif rule["eligibility"] != "nonmissing_metric":
            raise Blocked(f"unsupported eligibility rule: {rule['eligibility']}")
        failed = pd.Series(pd.NA, index=building_events.index, dtype="boolean")
        failed.loc[eligible] = _compare(values.loc[eligible], rule["operator"], float(rule["threshold"]))
        reason = np.where(eligible, "", np.where(values.isna(), "metric_missing_or_censored", "not_eligible"))
        for i in building_events.index:
            rows.append({"BUILDINGSTRUCTUREID": source_frame.at[i, "BUILDINGSTRUCTUREID"],
                         "event_id": str(source_frame.at[i, "event_id"]),
                         "rule_id": str(rule["rule_id"]), "stage": str(rule["stage"]),
                         "metric": metric, "value": values.at[i], "unit": str(rule["unit"]),
                         "operator": str(rule["operator"]), "threshold": float(rule["threshold"]),
                         "eligible": bool(eligible.at[i]), "failed": failed.at[i], "NA_reason": reason[i],
                         "eligibility_denominator": str(rule["eligibility"]),
                         "time_window": str(rule.get("time_window", "complete_mother_event_or_declared_24h_stage")),
                         "source": str(rule.get("source", "E8 building-event physical outcomes"))})
    out = pd.DataFrame(rows)
    out["decision_version"] = str(decision.get("version", "unspecified"))
    return out


def compute_multi_stage_threshold_sets(absolute: pd.DataFrame, decision: dict[str, Any]) -> pd.DataFrame:
    """Combine threshold states with interval-safe unknown propagation."""
    _approved(decision, "multi_stage_threshold_set")
    key = ["BUILDINGSTRUCTUREID", "event_id", "rule_id"]
    if any(c not in absolute for c in key) or absolute.duplicated(key).any():
        raise Blocked("absolute threshold table has invalid primary key")
    outputs = []
    for item in decision["sets"]:
        ids = [str(x) for x in item["rule_ids"]]
        subset = absolute.loc[
            absolute.rule_id.astype(str).isin(ids),
            ["BUILDINGSTRUCTUREID", "event_id", "rule_id", "failed"],
        ].copy()
        subset["known"] = subset.failed.notna().astype("int8")
        subset["failed_known"] = subset.failed.fillna(False).astype(bool).astype("int8")
        grouped = subset.groupby(["BUILDINGSTRUCTUREID", "event_id"], sort=False, as_index=False).agg(
            present_rules=("rule_id", "nunique"),
            rows=("rule_id", "size"),
            failed_known=("failed_known", "sum"),
            evaluable_rules=("known", "sum"),
        )
        invalid = grouped.present_rules.ne(len(ids)) | grouped.rows.ne(len(ids))
        if invalid.any():
            example = grouped.loc[invalid, ["BUILDINGSTRUCTUREID", "event_id"]].iloc[0].tolist()
            raise Blocked(f"multi-stage support differs for {tuple(example)}")
        minimum = int(item["minimum_failed"])
        grouped["required_rules"] = len(ids)
        grouped["unknown_rules"] = grouped.required_rules - grouped.evaluable_rules
        state = pd.Series(pd.NA, index=grouped.index, dtype="boolean")
        state.loc[grouped.failed_known.ge(minimum)] = True
        state.loc[(grouped.failed_known + grouped.unknown_rules).lt(minimum)] = False
        grouped["set_failed"] = state
        grouped["set_id"] = str(item["set_id"])
        grouped["rule_ids"] = "|".join(ids)
        grouped["minimum_failed"] = minimum
        grouped["unit"] = "binary screening state"
        grouped["eligibility_denominator"] = "all listed rules"
        grouped["time_window"] = str(item.get("time_window", "stage-specific windows"))
        grouped["NA_reason"] = np.where(
            grouped.set_failed.isna(), "one_or_more_required_thresholds_unknown", ""
        )
        grouped["source"] = "absolute_performance_thresholds.csv"
        outputs.append(grouped.drop(columns=["present_rules", "rows"]))
    result = pd.concat(outputs, ignore_index=True)
    return result[[
        "BUILDINGSTRUCTUREID", "event_id", "set_id", "rule_ids", "minimum_failed",
        "failed_known", "evaluable_rules", "required_rules", "unknown_rules", "set_failed",
        "unit", "eligibility_denominator", "time_window", "NA_reason", "source",
    ]]


def _period(hour: int, periods: list[dict[str, Any]]) -> str:
    for item in periods:
        start, end = int(item["start_hour_hkt"]), int(item["end_hour_hkt"])
        inside = start <= hour <= end if start <= end else hour >= start or hour <= end
        if inside: return str(item["period_id"])
    raise Blocked(f"hour {hour} is not covered by day/night periods")


def _heat_index_c(temperature_c: pd.Series, rh_pct: pd.Series) -> tuple[pd.Series, pd.Series]:
    t_c = pd.to_numeric(temperature_c, errors="coerce"); rh = pd.to_numeric(rh_pct, errors="coerce")
    t_f = t_c * 9 / 5 + 32
    hi_f = (-42.379 + 2.04901523*t_f + 10.14333127*rh - .22475541*t_f*rh
            - .00683783*t_f*t_f - .05481717*rh*rh + .00122874*t_f*t_f*rh
            + .00085282*t_f*rh*rh - .00000199*t_f*t_f*rh*rh)
    valid = t_c.ge(26.7) & rh.ge(40) & t_c.notna() & rh.notna()
    hi = ((hi_f - 32) * 5 / 9).where(valid)
    reason = pd.Series(np.where(valid, "", "outside_declared_heat_index_range_or_missing"), index=t_c.index)
    return hi, reason


def compute_day_night_humid_recovery(zone_intervals: pd.DataFrame, decision: dict[str, Any]) -> pd.DataFrame:
    """Summarise same-zone interval T/RH and sustained recovery by day/night."""
    _approved(decision, "day_night_windows")
    required = ["event_id", "base_case_id", "prototype_id", "floor_position", "zone_id", "timestamp_hkt",
                "temperature_c", "relative_humidity_pct", "eligible_for_screen", "phase", "interval_hours"]
    missing = [c for c in required if c not in zone_intervals]
    if missing: raise Blocked(f"day/night input missing fields: {missing}")
    key = ["event_id", "base_case_id", "zone_id", "timestamp_hkt", "phase"]
    if zone_intervals.duplicated(key).any(): raise Blocked("day/night input has duplicate zone intervals")
    frame = zone_intervals.copy(); frame["timestamp_hkt"] = pd.to_datetime(frame.timestamp_hkt, errors="coerce")
    frame["interval_hours"] = pd.to_numeric(frame.interval_hours, errors="coerce")
    valid_time = frame.timestamp_hkt.notna() & frame.interval_hours.gt(0)
    frame["monitoring_valid"] = frame.eligible_for_screen.fillna(False).astype(bool) & valid_time
    frame["period_id"] = frame.timestamp_hkt.dt.hour.map(lambda h: _period(int(h), decision["periods"]) if pd.notna(h) else "unknown")
    hi, reason = _heat_index_c(frame.temperature_c, frame.relative_humidity_pct)
    frame["heat_index_c"] = hi; frame["heat_index_NA_reason"] = reason
    frame["dh32"] = np.where(frame.monitoring_valid,
                              np.maximum(pd.to_numeric(frame.temperature_c, errors="coerce") - 32.0, 0) * frame.interval_hours,
                              np.nan)
    frame["supported_recovery_interval"] = (frame.monitoring_valid & frame.phase.eq("recovery") &
                                              pd.to_numeric(frame.temperature_c, errors="coerce").le(
                                                  float(decision["sustained_recovery"]["threshold_c"])))
    base_keys = ["event_id", "base_case_id", "prototype_id", "floor_position"]
    expected_zones = {
        keys: set(group.loc[group.eligible_for_screen.fillna(False).astype(bool), "zone_id"].astype(str))
        for keys, group in frame.groupby(base_keys, dropna=False, sort=False)
    }
    group_keys = ["event_id", "base_case_id", "prototype_id", "floor_position", "period_id"]
    rows = []
    for keys, group in frame.groupby(group_keys, dropna=False, sort=False):
        valid = group.monitoring_valid & pd.to_numeric(group.temperature_c, errors="coerce").notna()
        hi_valid = valid & group.heat_index_c.notna(); supported_h = group.loc[valid, "interval_hours"].sum()
        humid_h = group.loc[hi_valid, "interval_hours"].sum(); recovery = group[group.phase.eq("recovery")].sort_values(["zone_id", "timestamp_hkt"])
        sustained = False
        recovery_support_status = "not_observed"
        if not recovery.empty:
            required_zones = expected_zones.get(tuple(keys[:4]), set())
            by_time = []
            for timestamp, interval in recovery.groupby("timestamp_hkt", sort=True):
                observed = set(interval.zone_id.astype(str))
                complete = bool(required_zones) and observed == required_zones and len(interval) == len(required_zones)
                finite = (pd.to_numeric(interval.temperature_c, errors="coerce").notna().all() and
                          pd.to_numeric(interval.interval_hours, errors="coerce").notna().all())
                widths = pd.to_numeric(interval.interval_hours, errors="coerce")
                same_width = finite and widths.nunique(dropna=True) == 1
                all_below = complete and finite and interval.supported_recovery_interval.fillna(False).astype(bool).all()
                by_time.append((pd.Timestamp(timestamp), float(widths.iloc[0]) if same_width else np.nan,
                                bool(all_below and same_width), complete and finite and same_width))
            run = 0.0; previous = None
            all_complete = bool(by_time) and all(record[3] for record in by_time)
            for timestamp, width, supported, complete in by_time:
                contiguous = (previous is None or (pd.notna(width) and
                              timestamp - previous <= pd.Timedelta(hours=float(width) + 1e-9)))
                run = run + width if supported and contiguous else (width if supported else 0.0)
                previous = timestamp
                if run >= float(decision["sustained_recovery"]["support_hours"]):
                    sustained = True; break
            recovery_support_status = ("complete_supported_recovery" if sustained else
                                       "complete_observation_no_sustained_recovery" if all_complete else
                                       "incomplete_required_zone_support")
        rows.append({"event_id": keys[0], "base_case_id": keys[1], "prototype_id": keys[2], "floor_position": keys[3],
                     "period_id": keys[4], "eligible_zone_intervals": int(valid.sum()),
                     "effective_monitoring_hours": float(supported_h),
                     "temperature_mean_c": pd.to_numeric(group.loc[valid, "temperature_c"], errors="coerce").mean(),
                     "temperature_max_c": pd.to_numeric(group.loc[valid, "temperature_c"], errors="coerce").max(),
                     "relative_humidity_mean_pct": pd.to_numeric(group.loc[valid, "relative_humidity_pct"], errors="coerce").mean(),
                     "same_zone_heat_index_mean_c": group.loc[hi_valid, "heat_index_c"].mean(),
                     "heat_index_supported_hours": float(humid_h), "absolute_dh32_k_h": group.dh32.sum(min_count=1),
                     "sustained_recovery_26_2c_1h": (sustained if recovery_support_status != "incomplete_required_zone_support"
                                                       else pd.NA) if not recovery.empty else pd.NA,
                     "recovery_support_status": recovery_support_status,
                     "unit": "C|%|K h|h", "time_window": "approved HKT day/night over outage and recovery intervals",
                     "eligibility_denominator": "eligible zone-interval hours",
                     "NA_reason": "" if valid.any() else "no_valid_monitoring_support",
                     "source": "same-zone native 15-minute temperature/RH/control trajectories"})
    return pd.DataFrame(rows)


def compute_region_hourly_snapshots(snapshot_input: pd.DataFrame, decision: dict[str, Any]) -> pd.DataFrame:
    """Aggregate preselected building snapshots without rendering frames."""
    _approved(decision, "region_snapshot_times_hkt")
    required = ["BUILDINGSTRUCTUREID", "event_id", "study_region", "snapshot_role", "timestamp_hkt",
                "electric_power_kw", "temperature_c", "thermal_evaluable", "population_reference_2021"]
    missing = [c for c in required if c not in snapshot_input]
    if missing: raise Blocked(f"regional snapshot input missing fields: {missing}")
    key = ["BUILDINGSTRUCTUREID", "event_id", "study_region", "snapshot_role"]
    if snapshot_input.duplicated(key).any(): raise Blocked("regional snapshot input has duplicate building-role records")
    expected_roles = set(decision["snapshot_roles"]); expected_regions = set(decision["regions"])
    extra = set(snapshot_input.snapshot_role.astype(str)) - expected_roles
    if extra: raise Blocked(f"regional snapshot input contains unapproved roles: {sorted(extra)}")
    frame = snapshot_input[snapshot_input.study_region.astype(str).isin(expected_regions)].copy(); rows = []
    for keys, group in frame.groupby(["event_id", "study_region", "snapshot_role", "timestamp_hkt"], dropna=False):
        thermal = group.thermal_evaluable.fillna(False).astype(bool) & pd.to_numeric(group.temperature_c, errors="coerce").notna()
        population = pd.to_numeric(group.population_reference_2021, errors="coerce"); temperature = pd.to_numeric(group.temperature_c, errors="coerce")
        valid_population = thermal & population.gt(0); pop_denom = population[valid_population].sum()
        rows.append({"event_id": keys[0], "study_region": keys[1], "snapshot_role": keys[2], "timestamp_hkt": keys[3],
                     "buildings": int(group.BUILDINGSTRUCTUREID.nunique()),
                     "energy_evaluable_buildings": int(pd.to_numeric(group.electric_power_kw, errors="coerce").notna().sum()),
                     "thermal_evaluable_buildings": int(thermal.sum()),
                     "electric_power_kw": pd.to_numeric(group.electric_power_kw, errors="coerce").sum(min_count=1),
                     "temperature_mean_c": temperature[thermal].mean(), "temperature_max_c": temperature[thermal].max(),
                     "population_weighted_temperature_c": ((temperature[valid_population] * population[valid_population]).sum() / pop_denom if pop_denom > 0 else np.nan),
                     "population_denominator_people": pop_denom if pop_denom > 0 else np.nan,
                     "unit": "kW|C|people", "time_window": "single approved 15-minute snapshot",
                     "eligibility_denominator": "region buildings with metric support",
                     "NA_reason": "" if thermal.any() else "no_thermally_evaluable_buildings",
                     "source": "building snapshot values joined to frozen spatial links"})
    return pd.DataFrame(rows)


def compute_construction_year_tables(buildings: pd.DataFrame, event_dates: pd.DataFrame,
                                     decision: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retain explicit source classes and produce start/end same-year sensitivity."""
    _approved(decision, "construction_year_provenance")
    if "BUILDINGSTRUCTUREID" not in buildings or buildings.BUILDINGSTRUCTUREID.duplicated().any():
        raise Blocked("construction-year input requires unique BUILDINGSTRUCTUREID")
    year_field = str(decision.get("year_field", "construction_year_value")); source_field = str(decision.get("source_class_field", "construction_year_source_class"))
    if year_field not in buildings: raise Blocked(f"construction-year value field is absent: {year_field}")
    keep = ["BUILDINGSTRUCTUREID", year_field] + ([source_field] if source_field in buildings else [])
    provenance = buildings[keep].copy().rename(columns={year_field: "construction_year_value", source_field: "construction_year_source_class"})
    if "construction_year_source_class" not in provenance: provenance["construction_year_source_class"] = "unknown"
    provenance["construction_year_source_class"] = provenance.construction_year_source_class.fillna("unknown").astype(str)
    allowed = set(decision["allowed_source_classes"])
    provenance.loc[~provenance.construction_year_source_class.isin(allowed), "construction_year_source_class"] = "unknown"
    provenance["construction_year_value"] = pd.to_numeric(provenance.construction_year_value, errors="coerce")
    provenance["construction_year_known"] = provenance.construction_year_value.notna()
    provenance["source_specific_to_building"] = provenance.construction_year_source_class.ne("unknown")
    provenance["NA_reason"] = np.where(provenance.construction_year_value.isna(), "construction_year_missing",
                                        np.where(provenance.construction_year_source_class.eq("unknown"),
                                                 "year_value_present_but_per_building_source_unknown", ""))
    provenance["source"] = str(decision.get("source_description", "frozen stock year field; provenance retained per building"))
    provenance["unit"] = "calendar year"
    date_field = "mother_start_hkt" if "mother_start_hkt" in event_dates else "event_start"
    if "parent_event_id" not in event_dates or date_field not in event_dates: raise Blocked("event dates lack parent_event_id and mother start")
    events = event_dates[["parent_event_id", date_field]].drop_duplicates("parent_event_id").copy()
    events["event_year"] = pd.to_datetime(events[date_field], errors="coerce", format="mixed").dt.year
    if events.event_year.isna().any(): raise Blocked("event year cannot be derived")
    cross = provenance.assign(_k=1).merge(events.assign(_k=1), on="_k").drop(columns="_k")
    y = cross.construction_year_value; e = cross.event_year
    cross["exists_year_start_assumption"] = pd.array(np.where(y.isna(), pd.NA, y.le(e)), dtype="boolean")
    cross["exists_year_end_assumption"] = pd.array(np.where(y.isna(), pd.NA, y.lt(e)), dtype="boolean")
    cross["existence_primary_status"] = np.select([y.isna(), y.lt(e), y.gt(e), y.eq(e)],
                                                   ["unknown_year", "yes", "no", "unknown_same_year"], default="unknown")
    cross["historical_scope"] = "current surviving stock existence sensitivity; not historical city reconstruction"
    cross["population_reference_year"] = 2021; cross["unit"] = "binary/unknown"
    cross["NA_reason"] = np.where(y.isna(), "construction_year_missing", np.where(y.eq(e), "completion_month_or_in_use_date_unknown", ""))
    cross = cross.rename(columns={"parent_event_id": "event_id", date_field: "event_start_hkt"})
    return provenance, cross


def iter_zone_intervals_from_run_index(run_index: pd.DataFrame):
    """Yield one frozen O24 group's native intervals at a time."""
    required = {"event_id", "base_case_id", "prototype_id", "floor_position", "scenario_id", "source_result_dir",
                "outage_start_hkt", "outage_end_hkt"}
    if missing := sorted(required - set(run_index)):
        raise Blocked(f"run index cannot support day/night analysis: {missing}")
    identity = ["event_id", "base_case_id", "prototype_id", "floor_position"]
    rated = run_index[run_index.scenario_id.eq("HW_RATED")].copy()
    if rated.duplicated(identity).any():
        raise Blocked("run index has duplicate HW_RATED reference schedules")
    rated_lookup = rated.set_index(identity, drop=False)
    yielded = False
    for row in run_index[run_index.scenario_id.eq("HW_O24_SYNC")].itertuples(index=False):
        key = (row.event_id, row.base_case_id, row.prototype_id, row.floor_position)
        if key not in rated_lookup.index:
            raise Blocked(f"HW_RATED reference schedule is missing for {key}")
        reference = rated_lookup.loc[key]
        if isinstance(reference, pd.DataFrame):
            raise Blocked(f"HW_RATED reference schedule is not unique for {key}")
        path = Path(row.source_result_dir) / "parsed/native_timestep_zones.parquet"
        if not path.is_file(): raise Blocked(f"native zone trajectory is missing: {path}")
        frame = pd.read_parquet(path)
        aliases = {
            "zone_relative_humidity_pct": "relative_humidity_pct",
            "relative_humidity": "relative_humidity_pct",
            "occupied_for_screen": "eligible_for_screen",
        }
        for old, new in aliases.items():
            if new not in frame and old in frame: frame[new] = frame[old]
        try:
            areas = json.loads(row.zone_areas_json)
            conditioned = set(json.loads(row.conditioned_zones_json))
        except (TypeError, ValueError) as exc:
            raise Blocked(f"frozen zone contract is invalid for {row.run_id}: {exc}") from exc
        contract = pd.DataFrame([
            {"zone_id": str(zone), "zone_area_m2": float(area),
             "structural_conditioning_present": str(zone) in conditioned}
            for zone, area in areas.items()
        ])
        try:
            frame = build_expected_monitoring_support(
                frame,
                pd.read_parquet(Path(row.source_result_dir) / "inputs/event_schedules.parquet"),
                pd.read_parquet(Path(reference.source_result_dir) / "inputs/event_schedules.parquet"),
                expected_native_timestamps(
                    pd.read_parquet(Path(row.source_result_dir) / "inputs/weather.parquet")
                ),
                contract,
            )
        except (KeyError, ValueError) as exc:
            raise Blocked(f"monitoring support reconstruction failed for {row.run_id}: {exc}") from exc
        frame["eligible_for_screen"] = frame["monitoring_required"]
        needed = {"timestamp_hkt", "zone_id", "temperature_c", "relative_humidity_pct", "eligible_for_screen"}
        if missing := sorted(needed - set(frame)): raise Blocked(f"native zone fields missing in {path}: {missing}")
        timestamp = pd.to_datetime(frame.timestamp_hkt, errors="coerce")
        outage = pd.Timestamp(row.outage_start_hkt); restore = pd.Timestamp(row.outage_end_hkt)
        if timestamp.dt.tz is None:
            timestamp = timestamp.dt.tz_localize("Asia/Hong_Kong")
        if outage.tzinfo is None: outage = outage.tz_localize("Asia/Hong_Kong")
        if restore.tzinfo is None: restore = restore.tz_localize("Asia/Hong_Kong")
        frame = frame.assign(timestamp_hkt=timestamp)
        frame = frame[(frame.timestamp_hkt > outage) & (frame.timestamp_hkt <= restore + pd.Timedelta(hours=24))].copy()
        frame["phase"] = np.where(frame.timestamp_hkt <= restore, "outage", "recovery")
        frame["interval_hours"] = .25
        frame["event_id"] = row.event_id; frame["base_case_id"] = row.base_case_id
        frame["prototype_id"] = row.prototype_id; frame["floor_position"] = row.floor_position
        yielded = True
        yield frame[["event_id", "base_case_id", "prototype_id", "floor_position", "zone_id", "timestamp_hkt",
                     "temperature_c", "relative_humidity_pct", "eligible_for_screen", "phase", "interval_hours"]]
    if not yielded:
        raise Blocked("no O24 trajectories were available for day/night analysis")


def load_zone_intervals_from_run_index(run_index: pd.DataFrame) -> pd.DataFrame:
    """Load all frozen O24 intervals; tests and small callers only."""
    return pd.concat(iter_zone_intervals_from_run_index(run_index), ignore_index=True)
