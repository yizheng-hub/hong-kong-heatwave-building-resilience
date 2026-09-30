"""Pure consumers for Open6/E8 score tables; never dispatch simulations."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from utils.e8_open6_prepare import OPEN_EVENTS, LOCKBOX_EVENTS
from utils.e8_preflight import tie_aware_selection_probabilities

ALL_E8 = (
    "HKO_LI_0049", "HKO_LI_0052", "HKO_LI_0054", "HKO_LI_0058",
    "HKO_LI_0066", "HKO_LI_0068", "HKO_LI_0085", "HKO_LI_0090",
)
COMPONENTS = ("N", "C", "O", "R")
FORWARD_FOLDS = (
    ("T1", ("HKO_LI_0049", "HKO_LI_0052", "HKO_LI_0054", "HKO_LI_0058", "HKO_LI_0066"), "HKO_LI_0068"),
    ("T2", ("HKO_LI_0049", "HKO_LI_0052", "HKO_LI_0054", "HKO_LI_0058", "HKO_LI_0066", "HKO_LI_0068"), "HKO_LI_0085"),
    ("T3", ("HKO_LI_0049", "HKO_LI_0052", "HKO_LI_0054", "HKO_LI_0058", "HKO_LI_0066", "HKO_LI_0068", "HKO_LI_0085"), "HKO_LI_0090"),
)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".writing")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, path)


def beri_bounds(scores: pd.DataFrame, events: tuple[str, ...]) -> pd.DataFrame:
    """Median each component across the declared events, then geometric aggregation."""
    keys = ["base_case_id", "prototype_id", "floor_position"]
    required = set(events)
    seen = set(scores.event_id.astype(str))
    missing = sorted(required - seen)
    if missing:
        raise ValueError(f"Required events missing: {missing}")
    use = scores.loc[scores.event_id.isin(events)].copy()
    counts = use.groupby(keys + ["component"]).event_id.nunique()
    if not (counts == len(events)).all():
        raise ValueError("At least one design-component lacks a required event; no dropna aggregation")
    med = (use.groupby(keys + ["component"], as_index=False)
           .agg(component_score_lower=("score_lower", "median"),
                component_score_upper=("score_upper", "median"),
                event_count=("event_id", "nunique")))
    wide_l = med.pivot(index=keys, columns="component", values="component_score_lower")
    wide_u = med.pivot(index=keys, columns="component", values="component_score_upper")
    if set(wide_l.columns) != set(COMPONENTS) or wide_l.isna().any().any() or wide_u.isna().any().any():
        raise ValueError("Four component bounds are not complete")
    lower = 100.0 * np.prod(np.clip(wide_l[list(COMPONENTS)].to_numpy(float), 0, 1), axis=1) ** 0.25
    upper = 100.0 * np.prod(np.clip(wide_u[list(COMPONENTS)].to_numpy(float), 0, 1), axis=1) ** 0.25
    result = wide_l.reset_index()[keys].copy()
    result["beri_lower"] = lower
    result["beri_upper"] = upper
    result["reference_events"] = ";".join(events)
    result["event_count"] = len(events)
    result["scope"] = "open6_descriptive_reference" if tuple(events) == tuple(OPEN_EVENTS) else "complete_E8_reference"
    return result


def dependency_status(scores: pd.DataFrame | None) -> dict:
    available = sorted(set(scores.event_id.astype(str))) if scores is not None and not scores.empty else []
    missing = sorted(set(ALL_E8) - set(available))
    return {
        "available_events": available,
        "required_complete_E8_events": list(ALL_E8),
        "missing_required_events": missing,
        "lockbox_events": sorted(LOCKBOX_EVENTS),
        "open6_descriptive_reference": "READY" if set(OPEN_EVENTS) <= set(available) else "WAITING_FOR_OPEN6_RESULTS",
        "eight_fold_event_holdout": "READY" if not missing else "WAITING_FOR_REQUIRED_EVENTS",
        "three_retrospective_forward_tests": "READY" if not missing else "WAITING_FOR_REQUIRED_EVENTS",
        "final_complete_E8_BERI": "READY" if not missing else "WAITING_FOR_REQUIRED_EVENTS",
        "note": "No missing event is dropped, imputed by an interval midpoint, or replaced by an unlocked event.",
    }


def rank_and_decile(train: pd.DataFrame, test: pd.DataFrame, score_col: str, target_col: str,
                    fraction: float = 0.10) -> dict:
    """A deterministic continuous diagnostic; tie-aware production selection stays in the frozen evaluator."""
    keys = ["base_case_id", "prototype_id", "floor_position"]
    merged = train[keys + [score_col]].merge(test[keys + [target_col]], on=keys, validate="one_to_one")
    valid = merged[[score_col, target_col]].notna().all(axis=1)
    m = merged.loc[valid]
    if len(m) < 3:
        return {"n": len(m), "spearman": math.nan, "lowest_fraction": fraction, "status": "insufficient"}
    return {"n": len(m), "spearman": float(m[score_col].corr(m[target_col], method="spearman")),
            "lowest_fraction": fraction, "status": "continuous_rank_only"}


def _folds(kind: str) -> list[tuple[str, tuple[str, ...], str]]:
    if kind == "whole_event_leave_one_out":
        return [(f"LOEO_{test}", tuple(e for e in ALL_E8 if e != test), test) for test in ALL_E8]
    if kind == "retrospective_forward":
        return list(FORWARD_FOLDS)
    raise ValueError(f"Unknown fold kind: {kind}")


def fold_manifest(kind: str) -> pd.DataFrame:
    return pd.DataFrame([
        {"evaluation": kind, "fold_id": fold, "training_event_ids": ";".join(train),
         "training_event_count": len(train), "test_event_id": test,
         "whole_parent_event_grouping": True, "status": "WAITING_FOR_REQUIRED_EVENTS"}
        for fold, train, test in _folds(kind)
    ])


def _event_composite(scores: pd.DataFrame, event_id: str) -> pd.DataFrame:
    out = beri_bounds(scores, (event_id,)).rename(columns={
        "beri_lower": "test_composite_lower", "beri_upper": "test_composite_upper"})
    return out.drop(columns=["reference_events", "event_count", "scope"])


def _predictors(scores: pd.DataFrame, training_events: tuple[str, ...]) -> pd.DataFrame:
    keys = ["base_case_id", "prototype_id", "floor_position"]
    rating = beri_bounds(scores, training_events).rename(columns={
        "beri_lower": "BERI_lower", "beri_upper": "BERI_upper"})
    by_type = rating.groupby("prototype_id")[["BERI_lower", "BERI_upper"]].median().rename(columns={
        "BERI_lower": "prototype_baseline_lower", "BERI_upper": "prototype_baseline_upper"})
    by_floor = rating.groupby(["prototype_id", "floor_position"])[["BERI_lower", "BERI_upper"]].median().rename(columns={
        "BERI_lower": "prototype_floor_baseline_lower", "BERI_upper": "prototype_floor_baseline_upper"})
    result = rating[keys + ["BERI_lower", "BERI_upper"]].merge(by_type, on="prototype_id", validate="many_to_one")
    return result.merge(by_floor, on=["prototype_id", "floor_position"], validate="many_to_one")


def _selection_overlap(left, right, fraction: float) -> float:
    lp = tie_aware_selection_probabilities(np.asarray(left, dtype=float), fraction, lower_is_worse=True)
    rp = tie_aware_selection_probabilities(np.asarray(right, dtype=float), fraction, lower_is_worse=True)
    selected = int(np.ceil(fraction * len(lp)))
    return float(np.dot(lp, rp) / selected) if selected else math.nan


def evaluate_event_folds(scores: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate frozen ratings without midpoint imputation or row-level random splits.

    The test composite is an internal transfer diagnostic, not an independent
    physical validation target. Physical burden capture is reported separately.
    """
    missing = sorted(set(ALL_E8) - set(scores.event_id.astype(str)))
    if missing:
        raise ValueError(f"Complete E8 fold evaluation requires events: {missing}")
    keys = ["base_case_id", "prototype_id", "floor_position"]
    rows, selections = [], []
    for fold_id, train_events, test_event in _folds(kind):
        predictor = _predictors(scores, train_events)
        test = _event_composite(scores, test_event)
        merged = predictor.merge(test[keys + ["test_composite_lower", "test_composite_upper"]],
                                 on=keys, validate="one_to_one")
        for method, lower, upper in (
            ("BERI", "BERI_lower", "BERI_upper"),
            ("prototype", "prototype_baseline_lower", "prototype_baseline_upper"),
            ("prototype_floor", "prototype_floor_baseline_lower", "prototype_floor_baseline_upper"),
        ):
            rows.append({
                "evaluation": kind, "fold_id": fold_id, "test_event_id": test_event,
                "training_event_ids": ";".join(train_events), "method": method, "n": len(merged),
                "spearman_lower_bound": merged[lower].corr(merged.test_composite_lower, method="spearman"),
                "spearman_upper_bound": merged[upper].corr(merged.test_composite_upper, method="spearman"),
                "target_identity": "same-scale held-event four-component composite; internal transfer diagnostic",
            })
            for fraction in (0.05, 0.10, 0.20):
                selections.append({
                    "evaluation": kind, "fold_id": fold_id, "test_event_id": test_event,
                    "method": method, "selected_fraction": fraction,
                    "tie_aware_expected_overlap_lower": _selection_overlap(merged[lower], merged.test_composite_lower, fraction),
                    "tie_aware_expected_overlap_upper": _selection_overlap(merged[upper], merged.test_composite_upper, fraction),
                    "interval_policy": "lower and upper ranks reported separately; no midpoint",
                })
    return pd.DataFrame(rows), pd.DataFrame(selections)


def _burden_capture(scores: pd.DataFrame, energy: pd.DataFrame, thermal: pd.DataFrame,
                    kind: str) -> pd.DataFrame:
    if energy.empty or thermal.empty:
        return pd.DataFrame()
    keys = ["base_case_id", "prototype_id", "floor_position"]
    rows = []
    for fold_id, train_events, test_event in _folds(kind):
        predictor = _predictors(scores, train_events)
        outcome = energy.loc[energy.event_id.eq(test_event)].merge(
            thermal.loc[thermal.event_id.eq(test_event)], on=keys + ["event_id"], validate="one_to_one",
            suffixes=("_energy", "_thermal"))
        merged = predictor.merge(outcome, on=keys, validate="one_to_one")
        metrics = [name for name in (
            "positive_event_net_delta_kwh", "shortfall_ref_kwh_thermal",
            "absolute_occupied_degree_hours_above_32c",
            "positive_extra_occupied_degree_hours_above_26c_vs_rated") if name in merged]
        for method, field in (("BERI", "BERI_lower"), ("prototype", "prototype_baseline_lower"),
                              ("prototype_floor", "prototype_floor_baseline_lower")):
            for fraction in (0.05, 0.10, 0.20):
                probability = tie_aware_selection_probabilities(merged[field].to_numpy(float), fraction, lower_is_worse=True)
                for metric in metrics:
                    burden = pd.to_numeric(merged[metric], errors="coerce")
                    valid = burden.notna() & burden.ge(0)
                    total = float(burden.loc[valid].sum())
                    rows.append({
                        "evaluation": kind, "fold_id": fold_id, "test_event_id": test_event,
                        "method": method, "selected_fraction": fraction, "captured_metric": metric,
                        "valid_designs": int(valid.sum()),
                        "tie_aware_expected_captured_share": float(np.dot(probability[valid], burden.loc[valid]) / total) if total > 0 else math.nan,
                        "target_total": total, "zero_total_is_NA": total <= 0,
                    })
    return pd.DataFrame(rows)


def write_analysis_dependencies(scores_path: Path, output_dir: Path,
                                energy_path: Path | None = None,
                                thermal_path: Path | None = None) -> dict:
    scores = pd.read_csv(scores_path) if scores_path.exists() else None
    status = dependency_status(scores)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(fold_manifest("whole_event_leave_one_out"), output_dir / "E8_whole_event_holdout_fold_manifest.csv")
    _write_csv(fold_manifest("retrospective_forward"), output_dir / "E8_retrospective_forward_fold_manifest.csv")
    (output_dir / "E8_analysis_dependency_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    if scores is not None and set(OPEN_EVENTS) <= set(scores.event_id.astype(str)):
        try:
            _write_csv(beri_bounds(scores, OPEN_EVENTS), output_dir / "open6_descriptive_beri_bounds.csv")
        except ValueError as exc:
            status["open6_descriptive_reference"] = "WAITING_FOR_COMPLETE_DESIGN_EVENT_COVERAGE"
            status["open6_coverage_issue"] = str(exc)
            (output_dir / "E8_analysis_dependency_status.json").write_text(
                json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    if scores is not None and set(ALL_E8) <= set(scores.event_id.astype(str)):
        energy = pd.read_csv(energy_path) if energy_path and energy_path.exists() else pd.DataFrame()
        thermal = pd.read_csv(thermal_path) if thermal_path and thermal_path.exists() else pd.DataFrame()
        for kind, stem in (("whole_event_leave_one_out", "E8_whole_event_holdout"),
                           ("retrospective_forward", "E8_retrospective_forward")):
            performance, overlap = evaluate_event_folds(scores, kind)
            capture = _burden_capture(scores, energy, thermal, kind)
            _write_csv(performance, output_dir / f"{stem}_performance.csv")
            _write_csv(overlap, output_dir / f"{stem}_tie_aware_overlap.csv")
            _write_csv(capture, output_dir / f"{stem}_physical_burden_capture.csv")
        status["eight_fold_event_holdout"] = "COMPLETE"
        status["three_retrospective_forward_tests"] = "COMPLETE"
        (output_dir / "E8_analysis_dependency_status.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return status
