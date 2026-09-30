"""E8 scoring, tie-aware screening, event holdout and mechanism effects.

This module is solver-free and accepts ordinary tables.  It deliberately keeps
censoring intervals and reports endpoint scenarios as sensitivity analyses,
never as mathematical confidence bounds for rank statistics.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

from utils.e8_completion_common import ALL_EVENTS, Blocked, atomic_csv
from utils import e8_g3_postprocess as science


COMPONENTS = ("N", "C", "O", "R")
FORWARD = {
    "T1": (["HKO_LI_0049", "HKO_LI_0052", "HKO_LI_0054", "HKO_LI_0058", "HKO_LI_0066"], "HKO_LI_0068"),
    "T2": (["HKO_LI_0049", "HKO_LI_0052", "HKO_LI_0054", "HKO_LI_0058", "HKO_LI_0066", "HKO_LI_0068"], "HKO_LI_0085"),
    "T3": (["HKO_LI_0049", "HKO_LI_0052", "HKO_LI_0054", "HKO_LI_0058", "HKO_LI_0066", "HKO_LI_0068", "HKO_LI_0085"], "HKO_LI_0090"),
}


def geometric_score(frame: pd.DataFrame, suffix: str) -> pd.Series:
    cols = [f"score_{c}_{suffix}" for c in COMPONENTS]
    values = frame[cols].apply(pd.to_numeric, errors="coerce")
    valid = values.notna().all(axis=1)
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    result.loc[valid] = 100.0 * np.power(values.loc[valid].clip(0, 1).prod(axis=1), 0.25)
    return result


def build_rating(event_scores: pd.DataFrame, train_events: Iterable[str], key="BUILDINGSTRUCTUREID") -> pd.DataFrame:
    required = tuple(dict.fromkeys(str(x) for x in train_events))
    required_columns = [key, "event_id"] + [f"score_{c}_{b}" for c in COMPONENTS for b in ("lower", "upper")]
    missing_columns = [column for column in required_columns if column not in event_scores]
    if missing_columns:
        raise Blocked(f"BERI input missing required columns: {missing_columns}")
    if event_scores.duplicated([key, "event_id"]).any():
        raise Blocked("BERI input has duplicate building-event records")
    train = event_scores[event_scores.event_id.astype(str).isin(required)].copy()
    if train.empty:
        raise Blocked("BERI training event set is empty")
    score_columns = required_columns[2:]
    numeric = train[score_columns].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float)) | numeric.isna().to_numpy()
    if not finite.all():
        raise Blocked("BERI score bounds contain non-finite values")
    for component in COMPONENTS:
        lo, hi = numeric[f"score_{component}_lower"], numeric[f"score_{component}_upper"]
        if (lo.isna() ^ hi.isna()).any():
            raise Blocked(f"BERI {component} has only one interval endpoint")
        known = lo.notna()
        if ((lo[known] < 0) | (hi[known] > 1) | (lo[known] > hi[known])).any():
            raise Blocked(f"BERI {component} bounds are outside [0,1] or reversed")
    rows = []
    for building_id, group in train.groupby(key, sort=False):
        present = set(group.event_id.astype(str))
        missing_events = sorted(set(required) - present)
        unknown = []
        row = {key: building_id, "training_events": "|".join(required),
               "effective_events": int(len(present)), "required_events": len(required)}
        for component in COMPONENTS:
            for bound in ("lower", "upper"):
                values = pd.to_numeric(group[f"score_{component}_{bound}"], errors="coerce")
                row[f"score_{component}_{bound}"] = values.median() if not missing_events and values.notna().all() else np.nan
            if group[f"score_{component}_lower"].isna().any() or group[f"score_{component}_upper"].isna().any():
                unknown.append(component)
        if missing_events:
            row["NA_reason"] = "missing_required_events:" + "|".join(missing_events)
        elif unknown:
            reasons = sorted(set(str(x) for x in group.get("NA_reason", pd.Series(dtype=str)).dropna() if str(x)))
            row["NA_reason"] = "unknown_event_components:" + "|".join(sorted(set(unknown)))
            if reasons:
                row["NA_reason"] += ";source=" + "|".join(reasons)
        else:
            row["NA_reason"] = ""
        rows.append(row)
    result = pd.DataFrame(rows)
    result["BERI_lower"] = geometric_score(result, "lower")
    result["BERI_upper"] = geometric_score(result, "upper")
    result["BERI_point_identified"] = result.BERI_lower.eq(result.BERI_upper)
    invalid = result[["BERI_lower", "BERI_upper"]].isna().any(axis=1)
    result.loc[invalid & result.NA_reason.eq(""), "NA_reason"] = "one_or_more_components_not_evaluable"
    return result


def exchangeable_selection_probability(score: pd.Series, fraction: float, lower_is_risk=True) -> pd.Series:
    score = pd.to_numeric(score, errors="coerce")
    out = pd.Series(np.nan, index=score.index, dtype=float)
    valid = score.dropna()
    if valid.empty:
        return out
    m = int(np.ceil(fraction * len(valid)))
    ordered = valid.sort_values(ascending=lower_is_risk, kind="mergesort")
    cutoff = ordered.iloc[m - 1]
    worse = valid.lt(cutoff) if lower_is_risk else valid.gt(cutoff)
    tied = valid.eq(cutoff)
    p = (m - int(worse.sum())) / int(tied.sum())
    out.loc[valid.index] = 0.0
    out.loc[worse.index[worse]] = 1.0
    out.loc[tied.index[tied]] = p
    return out


def burden_capture(probability: pd.Series, burden: pd.Series) -> tuple[float, str]:
    p = pd.to_numeric(probability, errors="coerce")
    b = pd.to_numeric(burden, errors="coerce")
    valid = p.notna() & b.notna() & b.ge(0)
    if not valid.any():
        return np.nan, "no_jointly_evaluable_objects"
    denominator = b.loc[valid].sum()
    if denominator <= 0:
        return np.nan, "zero_total_burden"
    return float((p.loc[valid] * b.loc[valid]).sum() / denominator), ""


def _rank_stats(x: pd.Series, y: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce"); y = pd.to_numeric(y, errors="coerce")
    valid = x.notna() & y.notna()
    if valid.sum() < 3 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
        return {"n": int(valid.sum()), "spearman": np.nan, "kendall_tau_b": np.nan,
                "NA_reason": "insufficient_or_constant_data"}
    return {"n": int(valid.sum()), "spearman": float(spearmanr(x[valid], y[valid]).statistic),
            "kendall_tau_b": float(kendalltau(x[valid], y[valid], variant="b").statistic), "NA_reason": ""}


def evaluate_fold(event_scores: pd.DataFrame, target: pd.DataFrame, train_events: list[str], test_event: str,
                  fold_id: str, key="BUILDINGSTRUCTUREID",
                  frozen_rating: pd.DataFrame | None = None,
                  frozen_cohorts: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rating = build_rating(event_scores, train_events, key) if frozen_rating is None else frozen_rating.copy()
    if rating.duplicated(key).any():
        raise Blocked("rating has duplicate building identities")
    rating = rating[[column for column in rating.columns if column == key or column.startswith("score_")
                     or column.startswith("prediction__")
                     or column.startswith("BERI_") or column in {"training_events", "effective_events", "required_events", "NA_reason"}]]
    raw_test = target[target.event_id.eq(test_event)].drop(columns=[c for c in target if c.startswith("score_")])
    test = raw_test.merge(rating, on=key, how="inner", validate="one_to_one")
    methods = {"BERI": "BERI_lower", "stage_N": "score_N_lower", "stage_C": "score_C_lower",
               "stage_O": "score_O_lower", "stage_R": "score_R_lower"}
    target_columns = [c for c in ("cooling_shortfall_kwh", "absolute_dh32", "extra_dh26",
                                  "recovery_hours", "population_degree_hours") if c in test]
    summary = []
    selection = []

    def frozen_probability(method, outcome, fraction):
        if frozen_cohorts is None:
            return None
        selected = frozen_cohorts[
            frozen_cohorts.method.eq(method) & frozen_cohorts.outcome.eq(outcome)
            & np.isclose(pd.to_numeric(frozen_cohorts.fraction), fraction)
        ]
        if selected.duplicated(key).any() or set(test[key]) - set(selected[key]):
            raise Blocked(f"frozen cohort is incomplete for {method}/{outcome}/{fraction}")
        return test[key].map(selected.set_index(key).selection_probability)
    for method, score_col in methods.items():
        for outcome in target_columns:
            stat = _rank_stats(test[score_col], -pd.to_numeric(test[outcome], errors="coerce"))
            summary.append({"fold_id": fold_id, "train_events": "|".join(train_events), "test_event": test_event,
                            "method": method, "outcome": outcome, **stat})
            for q in (0.05, 0.10, 0.20):
                p = frozen_probability(method, outcome, q)
                if p is None:
                    p = exchangeable_selection_probability(test[score_col], q, lower_is_risk=True)
                capture, reason = burden_capture(p, test[outcome])
                selection.append({"fold_id": fold_id, "test_event": test_event, "method": method,
                                  "outcome": outcome, "fraction": q, "evaluable": int(p.notna().sum()),
                                  "expected_burden_capture": capture, "NA_reason": reason})
    for method, fields in (("building_type", ["main_class"]),
                           ("detailed_use_floor", ["sub_class", "prototype_floor_signature"]),
                           ("construction_era", ["construction_era"])):
        for outcome in target_columns:
            if frozen_rating is not None:
                prediction_column = f"prediction__{method}__{outcome}"
                if prediction_column not in test:
                    raise Blocked(f"frozen first-look prediction missing: {prediction_column}")
                scored = test
                score = pd.to_numeric(scored[prediction_column], errors="coerce")
            else:
                train_target = target[target.event_id.isin(train_events)].copy()
                train_target["construction_era"] = pd.cut(pd.to_numeric(train_target.construction_year_value, errors="coerce"),
                                                           [-np.inf, 1979, 1999, 2010, np.inf], labels=["pre1980", "1980s90s", "2000s", "2011plus"])
                test["construction_era"] = pd.cut(pd.to_numeric(test.construction_year_value, errors="coerce"),
                                                  [-np.inf, 1979, 1999, 2010, np.inf], labels=["pre1980", "1980s90s", "2000s", "2011plus"])
                lookup = train_target.groupby(fields, dropna=False, observed=False)[outcome].median().rename("training_group_burden").reset_index()
                scored = test.merge(lookup, on=fields, how="left", validate="many_to_one")
                score = -pd.to_numeric(scored.training_group_burden, errors="coerce")
            stat = _rank_stats(score, -pd.to_numeric(scored[outcome], errors="coerce"))
            summary.append({"fold_id": fold_id, "train_events": "|".join(train_events), "test_event": test_event,
                            "method": method, "outcome": outcome, **stat})
            for q in (.05, .10, .20):
                p = frozen_probability(method, outcome, q)
                if p is None:
                    p = exchangeable_selection_probability(score, q, lower_is_risk=True)
                capture, reason = burden_capture(p, scored[outcome])
                selection.append({"fold_id": fold_id, "test_event": test_event, "method": method,
                                  "outcome": outcome, "fraction": q, "evaluable": int(p.notna().sum()),
                                  "expected_burden_capture": capture, "NA_reason": reason})
    return pd.DataFrame(summary), pd.DataFrame(selection)


def run_event_evaluation(event_scores: pd.DataFrame, target: pd.DataFrame, output_dir: Path) -> dict:
    summaries, selections = [], []
    for endpoint in ("lower", "upper"):
        endpoint_scores = event_scores.copy()
        if endpoint == "upper":
            for component in COMPONENTS:
                endpoint_scores[f"score_{component}_lower"] = endpoint_scores[f"score_{component}_upper"]
        for event in ALL_EVENTS:
            train = [e for e in ALL_EVENTS if e != event]
            a, b = evaluate_fold(endpoint_scores, target, train, event, f"LOEO_{event}")
            a["score_endpoint_scenario"] = endpoint; b["score_endpoint_scenario"] = endpoint
            summaries.append(a); selections.append(b)
        for fold, (train, test) in FORWARD.items():
            a, b = evaluate_fold(endpoint_scores, target, train, test, fold)
            a["score_endpoint_scenario"] = endpoint; b["score_endpoint_scenario"] = endpoint
            summaries.append(a); selections.append(b)
    summary = pd.concat(summaries, ignore_index=True)
    selection = pd.concat(selections, ignore_index=True)
    atomic_csv(output_dir / "event_holdout_rank_metrics.csv", summary)
    atomic_csv(output_dir / "event_holdout_screening_capture.csv", selection)
    full = build_rating(event_scores, list(ALL_EVENTS))
    atomic_csv(output_dir / "full_e8_reference_rating.csv", full)
    methods = {"BERI": "BERI_lower", **{f"stage_{c}": f"score_{c}_lower" for c in COMPONENTS}}
    probabilities = {name: {q: exchangeable_selection_probability(full[column], q) for q in (.05, .10, .20)}
                     for name, column in methods.items()}
    overlap_rows = []
    for q in (.05, .10, .20):
        for left, right in combinations(methods, 2):
            p_left, p_right = probabilities[left][q], probabilities[right][q]
            valid = p_left.notna() & p_right.notna()
            denominator = float(p_left[valid].sum())
            independent_joint = float((p_left[valid] * p_right[valid]).sum())
            lower_joint = float(np.maximum(0.0, p_left[valid] + p_right[valid] - 1.0).sum())
            upper_joint = float(np.minimum(p_left[valid], p_right[valid]).sum())
            overlap_rows.append({"fraction": q, "method_left": left, "method_right": right,
                                 "joint_expected_selected": independent_joint,
                                 "joint_expected_selected_independent": independent_joint,
                                 "joint_selected_lower_bound": lower_joint,
                                 "joint_selected_upper_bound": upper_joint,
                                 "expected_selected": denominator,
                                 "overlap_fraction": (independent_joint / denominator
                                                      if denominator else np.nan),
                                 "joint_rule": "independent_exchangeable_tie_draws",
                                 "tie_policy": "exchangeable_expected_selection"})
    atomic_csv(output_dir / "screening_queue_overlap.csv", pd.DataFrame(overlap_rows))
    weight_rows = []
    presets = {"equal_NCOR": {c: .25 for c in COMPONENTS}}
    presets.update({f"leave_out_{omitted}": {c: (0 if c == omitted else 1 / 3) for c in COMPONENTS}
                    for omitted in COMPONENTS})
    for name, weights in presets.items():
        values = pd.Series(0.0, index=full.index)
        valid = pd.Series(True, index=full.index)
        has_positive_weight_zero = pd.Series(False, index=full.index)
        for component, weight in weights.items():
            if weight == 0:
                continue
            component_values = pd.to_numeric(full[f"score_{component}_lower"], errors="coerce")
            valid &= component_values.notna()
            has_positive_weight_zero |= component_values.eq(0)
            values += weight * np.log(component_values.where(component_values.gt(0)))
        beri = 100 * np.exp(values.where(valid & ~has_positive_weight_zero))
        beri = beri.mask(valid & has_positive_weight_zero, 0.0)
        weight_rows.append({"weight_scenario": name, "weights_json": json.dumps(weights, sort_keys=True),
                            "buildings": len(full), "evaluable": int(beri.notna().sum()),
                            "median_BERI_lower": beri.median() if beri.notna().any() else np.nan})
    atomic_csv(output_dir / "sensitivity_beri_weights.csv", pd.DataFrame(weight_rows))
    failure_rows = []
    for building, group in target.groupby("BUILDINGSTRUCTUREID", sort=False):
        for outcome in ("cooling_shortfall_kwh", "absolute_dh32", "extra_dh26", "recovery_hours"):
            if outcome not in group:
                continue
            values = pd.to_numeric(group[outcome], errors="coerce")
            if outcome == "recovery_hours" and "recovery_status" in group:
                status = group.recovery_status.astype(str)
                censored = status.eq("right_censored")
                lower = pd.to_numeric(group.get("recovery_lower_h", np.nan), errors="coerce")
                failure = values.gt(0) | censored
                severity = values.where(~censored, lower)
                evaluable = values.notna() | censored
            else:
                censored = pd.Series(False, index=group.index)
                failure = values.gt(0)
                severity = values
                evaluable = values.notna()
            failure_rows.append({"BUILDINGSTRUCTUREID": building, "outcome": outcome,
                                 "evaluable_events": int(evaluable.sum()),
                                 "failure_events_positive": int(failure.sum()),
                                 "right_censored_events": int(censored.sum()),
                                 "median_severity": severity.median() if severity.notna().any() else np.nan,
                                 "maximum_severity": severity.max() if severity.notna().any() else np.nan,
                                 "severity_semantics": ("observed_or_censoring_lower_bound_hours"
                                                        if outcome == "recovery_hours" else "observed_point_value")})
    atomic_csv(output_dir / "cross_event_failure_frequency_severity.csv", pd.DataFrame(failure_rows))
    return {"folds": 11, "endpoint_scenarios": 2, "full_ratings": len(full),
            "rank_rows": len(summary), "capture_rows": len(selection)}


def mechanism_effects(group_metrics: pd.DataFrame, logical_map: pd.DataFrame) -> pd.DataFrame:
    keys = ["mechanism_id", "prototype_id", "factor", "level", "event_id"]
    required_map = set(keys + ["reference"])
    required_metrics = {"base_case_id", "prototype_id", "event_id"}
    if missing := sorted(required_map - set(logical_map.columns)):
        raise Blocked(f"mechanism logical map missing columns: {missing}")
    if missing := sorted(required_metrics - set(group_metrics.columns)):
        raise Blocked(f"mechanism metrics missing columns: {missing}")
    identities = logical_map[keys + ["reference"]].drop_duplicates()
    if identities.duplicated(["mechanism_id", "event_id"]).any():
        raise Blocked("mechanism identity is inconsistent within event")
    if group_metrics.duplicated(["base_case_id", "prototype_id", "event_id"]).any():
        raise Blocked("mechanism metrics contain duplicate complete keys")
    merged = identities.merge(
        group_metrics, left_on=["mechanism_id", "prototype_id", "event_id"],
        right_on=["base_case_id", "prototype_id", "event_id"], how="left", validate="one_to_one")
    if merged["base_case_id"].isna().any():
        raise Blocked("mechanism metrics are missing one or more logical groups")
    numeric = [c for c in ("net_delta_kwh", "relative_delta", "prototype_peak_difference_kw",
                           "shortfall_ref_kwh_thermal", "absolute_occupied_degree_hours_above_32c",
                           "positive_extra_occupied_degree_hours_above_26c_vs_rated",
                           "first_crossing_lower_h", "first_crossing_upper_h", "recovery_lower_h", "recovery_upper_h")
               if c in merged]
    def metric_unit(name):
        if name.endswith("_kwh"): return "kWh"
        if name.endswith("_kw"): return "kW"
        if name.endswith("_h"): return "h"
        if name.endswith("_c"): return "degC"
        if "relative" in name or name.endswith("_fraction"): return "fraction"
        return "dimensionless"
    reference_rows = merged[merged.reference]
    if reference_rows.duplicated(["prototype_id", "factor", "event_id"]).any():
        raise Blocked("mechanism reference is not unique")
    refs = reference_rows.set_index(["prototype_id", "factor", "event_id"])
    rows = []
    for row in merged[~merged.reference].itertuples(index=False):
        identity = (row.prototype_id, row.factor, row.event_id)
        if identity not in refs.index:
            raise Blocked(f"mechanism reference missing: {identity}")
        ref = refs.loc[identity]
        out = {k: getattr(row, k) for k in keys}
        for col in numeric:
            out[f"reference_{col}"] = ref[col]
            out[f"treatment_{col}"] = getattr(row, col)
            if pd.notna(ref[col]) and pd.notna(getattr(row, col)):
                out[f"effect_{col}"] = getattr(row, col) - ref[col]
            else:
                out[f"effect_{col}"] = np.nan
            if not col.endswith(("_lower_h", "_upper_h")):
                out[f"effect_{col}_lower"] = out[f"effect_{col}"]
                out[f"effect_{col}_upper"] = out[f"effect_{col}"]
                out[f"effect_{col}_unit"] = metric_unit(col)
        for stem in ("first_crossing", "recovery"):
            lo, hi = f"{stem}_lower_h", f"{stem}_upper_h"
            if lo in numeric and hi in numeric and all(pd.notna(x) for x in (ref[lo], ref[hi], getattr(row, lo), getattr(row, hi))):
                out[f"effect_{stem}_lower_h"] = getattr(row, lo) - ref[hi]
                out[f"effect_{stem}_upper_h"] = getattr(row, hi) - ref[lo]
                out[f"effect_{stem}_bound_semantics"] = "interval_difference_Tlower-Rupper_to_Tupper-Rlower"
        rows.append(out)
    return pd.DataFrame(rows)


def threshold_sensitivity_from_runs(run_index: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Recompute 30/32/34 C outage screening from persisted zone trajectories."""
    required = {"run_id", "event_id", "base_case_id", "prototype_id", "floor_position", "scenario_id",
                "source_result_dir", "outage_start_hkt", "zone_areas_json", "conditioned_zones_json"}
    if missing := sorted(required - set(run_index.columns)):
        raise Blocked(f"threshold source index missing columns: {missing}")
    identities = ["event_id", "base_case_id", "prototype_id", "floor_position"]
    if run_index.duplicated(identities + ["scenario_id"]).any():
        raise Blocked("threshold source index contains duplicate scenario identities")

    def hkt(value):
        stamp = pd.Timestamp(value)
        return stamp.tz_localize("Asia/Hong_Kong") if stamp.tzinfo is None else stamp.tz_convert("Asia/Hong_Kong")

    rows = []
    subset = run_index[run_index.scenario_id.eq("HW_O24_SYNC")]
    for item in subset.itertuples(index=False):
        base = Path(item.source_result_dir)
        zones = pd.read_parquet(base / "parsed/native_timestep_zones.parquet")
        zones["timestamp_hkt"] = zones.timestamp_hkt.map(hkt)
        weather = pd.read_parquet(base / "inputs/weather.parquet")
        weather["timestamp_hkt"] = weather.timestamp_hkt.map(hkt)
        actual_schedule = pd.read_parquet(base / "inputs/event_schedules.parquet")
        reference = run_index[
            run_index.scenario_id.eq("HW_RATED")
            & run_index.event_id.eq(item.event_id)
            & run_index.base_case_id.eq(item.base_case_id)
            & run_index.prototype_id.eq(item.prototype_id)
            & run_index.floor_position.eq(item.floor_position)
        ]
        if len(reference) != 1:
            raise Blocked(f"exact rated reference missing for outage run: {item.run_id}")
        planned_schedule = pd.read_parquet(Path(reference.iloc[0].source_result_dir) / "inputs/event_schedules.parquet")
        areas = json.loads(item.zone_areas_json)
        conditioned = set(json.loads(item.conditioned_zones_json))
        contract = pd.DataFrame([{"zone_id": str(zone), "zone_area_m2": float(area),
                                  "structural_conditioning_present": str(zone) in conditioned}
                                 for zone, area in areas.items()])
        try:
            support = science.build_expected_monitoring_support(
                zones, actual_schedule, planned_schedule,
                science.expected_native_timestamps(weather), contract)
        except (KeyError, ValueError) as exc:
            raise Blocked(f"invalid monitoring support for {item.run_id}: {exc}") from exc
        start = hkt(item.outage_start_hkt)
        window = support[support.timestamp_hkt.gt(start) & support.timestamp_hkt.le(start + pd.Timedelta(hours=24))].copy()
        for threshold in (30.0, 32.0, 34.0):
            crossing, _ = science.building_crossing_from_zones(support, start, threshold=threshold)
            required_support = window[window.monitoring_required]
            has_monitored_objects = not required_support.empty
            complete = (len(window.timestamp_hkt.unique()) == 96 and has_monitored_objects
                        and not required_support.support_status.eq("required_missing").any())
            maxima = required_support.groupby("timestamp_hkt").temperature_c.max()
            dh = float((maxima - threshold).clip(lower=0).sum() * .25) if complete else np.nan
            rows.append({"event_id": item.event_id, "base_case_id": item.base_case_id,
                         "prototype_id": item.prototype_id, "floor_position": item.floor_position,
                         "threshold_c": threshold, "degree_hours": dh,
                         "crossing_lower_h": crossing.get("lower_h"), "crossing_upper_h": crossing.get("upper_h"),
                         "observation_status": crossing.get("status"), "monitoring_complete": bool(complete),
                         "monitoring_object_count": int(required_support.zone_id.nunique()) if "zone_id" in required_support else 0,
                         "NA_reason": ("" if complete else
                                       "no_eligible_monitoring_objects" if not has_monitored_objects else
                                       "incomplete_required_zone_time_support")})
    result = pd.DataFrame(rows)
    atomic_csv(output_path, result)
    return result
