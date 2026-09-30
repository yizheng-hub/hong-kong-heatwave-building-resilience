"""Manual, resumable E8 completion orchestrator.

No solver is reachable from doctor/dry-run/status/summarize/package-review.
Only ``execute --resume`` with a bound approval may invoke the proven local
scheduler, first for mechanisms and then for the two released lockboxes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import pandas as pd

from utils.e8_completion_common import (
    ALL_EVENTS, SCENARIOS, Blocked, Waiting, approval_contract, atomic_csv, atomic_json, atomic_parquet,
    imported_module_audit, load_config, load_task_state, make_upstream_receipt,
    save_task_state, sha256, validate_approval, verify_upstream,
)
from utils.e8_completion_manifests import (SOURCE_INPUTS, SOURCE_PARSED, audit_targeted_c70_migration,
                                           compile_lockboxes, compile_mechanisms,
                                           effective_open6_source_index, hydrate_source_index)
from utils.e8_completion_analysis import (build_rating, evaluate_fold, exchangeable_selection_probability,
                                           mechanism_effects, run_event_evaluation,
                                           threshold_sensitivity_from_runs)
from utils.e8_completion_required_analyses import (
    analysis_decision_hash, compute_absolute_performance_thresholds,
    compute_construction_year_tables, compute_day_night_humid_recovery,
    compute_multi_stage_threshold_sets, compute_region_hourly_snapshots,
    iter_zone_intervals_from_run_index, load_zone_intervals_from_run_index, validate_analysis_decisions,
)
from utils.e8_completion_release import (finalize_source_contract, freeze_release, package_review,
                                         verify_final, write_dependency_matrices)
from utils.e8_completion_evidence import audit_external_evidence
from utils.e8_completion_stock import (aggregate_design_metrics, city_event_energy_trajectories, propagate_event,
                                       propagate_stock, stock_components, apply_simultaneous_recovery,
                                       region_snapshot_input_from_runs)
from utils import g4a_local_execution as runner


def _task(c, task_id, status, outputs=None, error=None):
    state = load_task_state(c); item = state["tasks"][task_id]
    now = time.time()
    if status == "running" and item.get("status") != "running":
        item["attempts"] = int(item.get("attempts", 0)) + 1
        item["active_started_at"] = now
    if status in {"complete", "blocked", "failed"} and item.get("active_started_at"):
        item["spent_seconds"] = float(item.get("spent_seconds", 0.0)) + max(0.0, now - float(item.pop("active_started_at")))
    item.update(status=status, outputs=outputs or item.get("outputs", {}), error=error, updated_at=now)
    save_task_state(c, state)


def _artifact_hashes(paths):
    return {str(Path(path).resolve()): sha256(path) for path in paths
            if Path(path).is_file() and Path(path).stat().st_size > 0}


def _task_current(c, task_id, paths, dependency_hash=None):
    item = load_task_state(c)["tasks"][task_id]
    expected = item.get("outputs", {}).get("artifact_hashes", {})
    paths = [Path(path) for path in paths]
    dependency_current = dependency_hash is None or item.get("outputs", {}).get("dependency_hash") == dependency_hash
    return (item.get("status") == "complete" and dependency_current and len(expected) == len(paths)
            and all(path.is_file() and expected.get(str(path.resolve())) == sha256(path) for path in paths))


def _completion_resource_usage(c):
    root = Path(c["results_root"])
    state = load_task_state(c)
    now = time.time()
    task_seconds = 0.0
    for item in state.get("tasks", {}).values():
        task_seconds += float(item.get("spent_seconds", 0.0) or 0.0)
        if item.get("active_started_at"):
            task_seconds += max(0.0, now - float(item["active_started_at"]))
    child_calls = child_retries = 0
    for label in ("lockbox", "mechanisms"):
        path = root / "runtime" / label / "state.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        attempts = data.get("attempts", [])
        child_calls += sum(bool(item.get("launched")) for item in attempts)
        child_retries += sum(bool(item.get("launched")) and bool(item.get("is_retry")) for item in attempts)
    return {"task_seconds": task_seconds, "bytes_on_disk": runner.unique_tree_bytes(root),
            "solver_calls": child_calls, "retry_calls": child_retries}


def _check_completion_budget(c, approval, phase):
    usage = _completion_resource_usage(c)
    resources = approval["resources"]
    if usage["bytes_on_disk"] > int(resources["max_bytes"]):
        raise Blocked(f"completion disk budget exhausted before {phase}")
    if usage["task_seconds"] > int(resources["wall_seconds"]):
        raise Blocked(f"completion wall-time budget exhausted before {phase}")
    if usage["solver_calls"] > int(resources["max_calls"]) or usage["retry_calls"] > int(resources["max_retries"]):
        raise Blocked(f"completion call budget exhausted before {phase}")
    remaining_bytes = max(0, int(resources["max_bytes"]) - int(usage["bytes_on_disk"]))
    free = shutil.disk_usage(c["results_root"]).free
    if free < remaining_bytes + int(resources["min_free_bytes"]):
        raise Blocked(f"insufficient free space for remaining completion budget before {phase}")
    state = load_task_state(c)
    state["resource_usage"] = {**usage, "remaining_approved_bytes": remaining_bytes,
                               "checked_phase": phase, "checked_at": time.time()}
    save_task_state(c, state)
    return usage


def _reconcile_delegated_task_clock(c, task_id, label):
    """Replace a stale parent timer with the child ledger's active time.

    A child scheduler may be resumed directly after a resource stop. In that
    case the parent process is no longer alive, but its task row can still say
    running and would otherwise count offline waiting as active wall time.
    The child ledger is authoritative for solver-stage active time. This
    reconciliation never marks the task complete; the normal child verifier
    must still run before the parent can advance.
    """
    child_path = Path(c["results_root"]) / "runtime" / label / "state.json"
    if not child_path.is_file():
        return
    child = json.loads(child_path.read_text(encoding="utf-8"))
    if child.get("active_session") is not None:
        return
    state = load_task_state(c)
    item = state["tasks"][task_id]
    if item.get("status") != "running" or not item.get("active_started_at"):
        return
    item["spent_seconds"] = max(float(item.get("spent_seconds", 0.0) or 0.0),
                                  float(child.get("spent_seconds", 0.0) or 0.0))
    item.pop("active_started_at", None)
    item["status"] = "blocked"
    item["error"] = f"delegated child stopped: {child.get('stop_reason', 'unknown')}"
    item["updated_at"] = time.time()
    save_task_state(c, state)


def _reconcile_interrupted_local_tasks(c):
    """Close stale local-stage timers left by a prior foreground exit."""
    state = load_task_state(c)
    stop_at = float(state.get("updated_at", time.time()) or time.time())
    changed = False
    for task_id, item in state.get("tasks", {}).items():
        if task_id in {"T05_lockbox_run", "T08_mechanism_run"}:
            continue
        started = item.get("active_started_at")
        if item.get("status") != "running" or not started:
            continue
        stopped = max(float(started), stop_at, float(item.get("updated_at", started) or started))
        item["spent_seconds"] = float(item.get("spent_seconds", 0.0) or 0.0) + max(0.0, stopped - float(started))
        item.pop("active_started_at", None)
        item["status"] = "blocked"
        item["error"] = "prior local stage exited before final state commit; safe to resume"
        item["updated_at"] = stopped
        changed = True
    if changed:
        save_task_state(c, state)


def _execute_or_accept_verified_child(child_cfg, retry_failed=False):
    """Return a current, complete child without re-entering its resource gate."""
    state = runner.load_state(child_cfg)
    recorded_complete = bool(state and state.get("rows") and
                             all(item.get("status") == runner.COMPLETE for item in state["rows"].values()))
    if recorded_complete and state.get("active_session") is None:
        with runner.exclusive_lock(Path(child_cfg["output_dir"]) / "execution.lock"):
            version = runner.verify_versions(child_cfg)
            frame = pd.read_csv(child_cfg["manifest"]).fillna("")
            runner.reconcile_execution_identity(child_cfg, state, version)
            runner.validate_state(state, frame)
            runner.restore_state(child_cfg, state)
            complete = runner.reports(child_cfg, state)
            runner.atomic_json(Path(child_cfg["output_dir"]) / "state.json", state)
        if complete:
            return 0
        raise Blocked("Recorded-complete child failed current evidence verification; refusing automatic rerun")
    return runner.execute(child_cfg, retry_failed)


def _validate_required_analysis_decisions(c):
    path = Path(c["workspace_root"]) / "config/completion_analysis_contract.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    return validate_analysis_decisions(contract)


def _measured_upstream_resource_basis(c):
    """Read only completed Open6 event summaries; never scan growing run trees."""
    base = Path(c["upstream_project_root"]) / "intermediate/simulation_outputs/upstream/E8_open6/events"
    rows = []
    for event in ALL_EVENTS:
        path = base / event / "resource_usage.json"
        if not path.is_file():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("complete") is not True or int(value.get("actual_calls", 0)) <= 0:
            continue
        rows.append({"event_id": event, "actual_calls": int(value["actual_calls"]),
                     "active_seconds": float(value.get("active_seconds", 0.0)),
                     "bytes_on_disk": int(value.get("bytes_on_disk", 0)),
                     "source_path": str(path), "source_sha256": sha256(path)})
    calls = sum(x["actual_calls"] for x in rows)
    return {"completed_event_summaries": rows,
            "measured_calls": calls,
            "weighted_active_seconds_per_call": (sum(x["active_seconds"] for x in rows) / calls) if calls else None,
            "maximum_event_bytes_per_call": max((x["bytes_on_disk"] / x["actual_calls"] for x in rows), default=None),
            "identity": "provisional planning basis from completed Open6 summaries; final free space is rechecked at execution"}


def _protocol(c):
    root = Path(c["results_root"]); root.mkdir(parents=True, exist_ok=True)
    for folder in ("00_protocol", "01_core_e8", "02_mechanisms", "03_stock", "04_validation",
                   "05_evidence", "06_source_data", "07_release", "runtime"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    write_dependency_matrices(root, Path(c["upstream_project_root"]), Path(c["paper_panel_registry"]))
    atomic_csv(root / "00_protocol/imported_module_audit.csv", pd.DataFrame(imported_module_audit(Path(c["workspace_root"]))))


def doctor(c, deep=False):
    _protocol(c)
    candidate_version = c["upstream"].get("candidate_version_lock_sha256")
    verification = verify_upstream(c, deep=deep, accepted_version_lock_sha256=candidate_version)
    disk = shutil.disk_usage(c["results_root"])
    migration = audit_targeted_c70_migration(
        Path(c["upstream_project_root"]),
        Path(c["results_root"]) / "00_protocol/targeted_c70_migration",
        candidate_version,
        c["upstream"].get("expected_c70_replacements"),
    )
    contract = approval_contract(c)
    proposal = json.loads(Path(c["approval_template"]).read_text(encoding="utf-8"))
    proposal.update(contract, approval_status="pending_user_confirmation", generated_at=time.time())
    proposal["mechanism_levels_sha256"] = contract["mechanism_levels_sha256"]
    proposal["resources"] = dict(c["resources"])
    atomic_json(Path(c["results_root"]) / "00_protocol/completion_approval.proposed.json", proposal)
    try:
        decisions = _validate_required_analysis_decisions(c)
        decision_view = {"status": "FORMAL_SCHEMA_APPROVED", "decision_hash": analysis_decision_hash(decisions), "issues": []}
    except Blocked as exc:
        decision_view = {"status": "PENDING_USER_CONFIRMATION", "decision_hash": None, "issues": [str(exc)],
                         "proposal": c["analysis_decision_proposal"], "schema": c["analysis_decision_schema"]}
    panels = pd.read_csv(c["paper_panel_registry"]).fillna("")
    panel_missing = []
    for table in panels.source_table.unique():
        path = Path(c["results_root"]) / str(table)
        if not path.is_file(): panel_missing.append(str(table))
    levels = json.loads((Path(c["workspace_root"]) / "config/mechanism_levels_recommended.json").read_text(encoding="utf-8"))
    pair_comparison_rows = len(pd.read_csv(
        Path(c["workspace_root"]) / "frozen_upstream/mechanisms/residential_pair_design.csv"))
    variant_count_per_prototype = 1 + 1 + 2 + 2 + 2
    mechanism_unique_upper = 5 * variant_count_per_prototype * 3 * len(SCENARIOS)
    mechanism_logical = 5 * (2 + 3 + 3 + 3) * 3 * len(SCENARIOS)
    lockbox_logical = 2 * 297 * 5
    conservative_new_upper = lockbox_logical + mechanism_unique_upper
    required_with_retries = conservative_new_upper + int(c["resources"]["max_retries"])
    measured_basis = _measured_upstream_resource_basis(c)
    per_call_bytes = measured_basis["maximum_event_bytes_per_call"]
    estimated_solver_bytes = int(per_call_bytes * conservative_new_upper) if per_call_bytes else None
    analysis_margin = 4 * 1024**3
    resource_view = {"lockbox_logical_runs": lockbox_logical,
                     "mechanism_logical_runs_before_physical_dedup": mechanism_logical,
                     "mechanism_pair_comparison_rows": pair_comparison_rows,
                     "mechanism_unique_physics_upper_before_core_cache_reuse": mechanism_unique_upper,
                     "conservative_unique_new_upper_bound_before_physical_dedup": conservative_new_upper,
                     "approved_retry_allowance": int(c["resources"]["max_retries"]),
                     "conservative_call_budget_needed": required_with_retries,
                     "configured_max_calls": int(c["resources"]["max_calls"]),
                     "budget_shortfall_before_actual_dedup": max(0, required_with_retries-int(c["resources"]["max_calls"])),
                     "final_unique_manifest_status": "WAITING_FOR_APPROVAL_AND_COMPILE",
                     "measured_open6_basis": measured_basis,
                     "estimated_solver_bytes_before_dedup": estimated_solver_bytes,
                     "analysis_and_source_data_margin_bytes": analysis_margin,
                     "estimated_total_bytes_before_dedup": (estimated_solver_bytes + analysis_margin) if estimated_solver_bytes is not None else None,
                     "configured_max_bytes": int(c["resources"]["max_bytes"]),
                     "execution_free_space_gate_bytes": int(c["resources"]["max_bytes"]) + int(c["resources"]["min_free_bytes"])}
    estimated_total = resource_view["estimated_total_bytes_before_dedup"]
    resource_view["storage_budget_shortfall_before_actual_dedup"] = (
        max(0, int(estimated_total) - int(c["resources"]["max_bytes"]))
        if estimated_total is not None else None
    )
    upstream_view = {"status": verification["status"], "numerical_complete": verification["numerical_complete"],
                     "issues": verification["issues"],
                     "actual_version_lock_sha256": verification["actual_version_lock_sha256"],
                     "candidate_version_lock_sha256": candidate_version,
                     "workorder_lock_held": verification["workorder"]["lock_held"],
                     "workorder_lock_path": verification["workorder"]["path"],
                     "active_events": [Path(x["path"]).parent.name for x in verification["activity"] if x["process_alive"] is True],
                     "source_tables": {k: {q: v.get(q) for q in ("exists", "rows", "expected_rows", "valid")}
                                       for k, v in verification["sources"].items()}}
    blockers = list(verification["issues"]) + decision_view["issues"]
    if not migration["evidence_complete"]: blockers.extend(migration["issues"])
    blockers.extend(["targeted C70 scientific migration acceptance pending", "lockbox release pending",
                     "residential mechanism levels pending"])
    if resource_view["budget_shortfall_before_actual_dedup"]:
        blockers.append("configured solver-call budget may be insufficient before actual physical dedup")
    if resource_view["storage_budget_shortfall_before_actual_dedup"]:
        blockers.append("configured storage budget is below the conservative measured estimate before actual physical dedup")
    result = {"status": "CODE_COMPLETE_WAITING_FOR_USER_APPROVAL" if blockers else "READY_FOR_APPROVED_EXECUTION",
              "energyplus_calls": 0, "upstream": upstream_view,
              "accepted_version_and_supersession": migration,
              "code_functionality": {"five_required_analysis_functions": "IMPLEMENTED",
                                     "paper_only": True, "animation": False,
                                     "production_test_record": str(Path(c["workspace_root"]) / "test-results")},
              "analysis_contracts_and_sources": decision_view,
              "paper_panels": {"registered_rows": len(panels), "unique_source_tables": panels.source_table.nunique(),
                               "missing_until_production": panel_missing,
                               "registry_sha256": sha256(c["paper_panel_registry"])},
              "approvals": {"mechanism_levels": levels.get("status"), "lockbox_release": proposal.get("lockbox_release"),
                            "targeted_c70_migration": "pending_user_confirmation"},
              "physical_manifest_and_resources": resource_view,
              "unresolved_items": blockers,
              "workspace": c["workspace_root"], "results_root": c["results_root"],
              "free_disk_bytes": disk.free, "estimated_required_bytes": c["resources"]["max_bytes"],
              "module_imports_isolated": all(x["inside_isolated_workspace"] for x in imported_module_audit(Path(c["workspace_root"]))),
              "approval_proposal": str(Path(c["results_root"]) / "00_protocol/completion_approval.proposed.json")}
    atomic_json(Path(c["results_root"]) / "00_protocol/doctor.json", result)
    return result


def _copy_open6(c, receipt):
    root = Path(c["results_root"]); combined = Path(c["upstream_project_root"]) / "intermediate/simulation_outputs/upstream/E8_open6/combined"
    mapping = {
        "open6_design_event_energy_service.csv": "open6_energy.csv",
        "open6_design_event_thermal_recovery.csv": "open6_thermal.csv",
        "open6_stage_score_bounds.csv": "open6_scores.csv",
    }
    for source, dest in mapping.items():
        shutil.copy2(combined / source, root / "01_core_e8" / dest)
    effective = effective_open6_source_index(Path(c["upstream_project_root"]))
    atomic_csv(root / "01_core_e8/open6_source_index.csv", effective)
    atomic_json(root / "00_protocol/upstream_receipt.json", receipt)
    _task(c, "T00_upstream_guard", "complete", {"receipt": sha256(root / "00_protocol/upstream_receipt.json")})
    imported = {k: sha256(root / "01_core_e8" / v) for k, v in mapping.items()}
    imported["effective_open6_source_index"] = sha256(root / "01_core_e8/open6_source_index.csv")
    _task(c, "T01_open6_import", "complete", imported)


def _freeze_open6_stock(c):
    root = Path(c["results_root"]); upstream = Path(c["upstream_project_root"])
    production = runner.load_production(upstream, code_root=Path(c["workspace_root"]))
    response = aggregate_design_metrics(pd.read_csv(root / "01_core_e8/open6_energy.csv"),
                                        pd.read_csv(root / "01_core_e8/open6_thermal.csv"),
                                        pd.read_csv(root / "01_core_e8/open6_scores.csv"),
                                        production.prototype_parameter_table())
    source = upstream / "output/paper_pipeline_v0_3/02_building_response"
    buildings, components = stock_components(pd.read_parquet(source / "building_hourly_component_map.parquet"),
                                              pd.read_parquet(source / "all_sector_building_model_inputs.parquet"))
    folder = root / "03_stock/open6_prediction_events"; folder.mkdir(parents=True, exist_ok=True)
    open_events = ["HKO_LI_0049", "HKO_LI_0052", "HKO_LI_0058", "HKO_LI_0068", "HKO_LI_0085", "HKO_LI_0090"]
    for event in open_events:
        frame = propagate_event(buildings, components, response, event)
        frame.to_parquet(folder / f"{event}.parquet", index=False)
    open_index = hydrate_source_index(pd.read_csv(root / "01_core_e8/open6_source_index.csv").fillna(""))
    apply_simultaneous_recovery(open_index, components, folder,
                                root / "03_stock/open6_recovery_trajectories", event_ids=open_events)
    event_frames = [pd.read_parquet(folder / f"{event}.parquet") for event in open_events]
    open_targets = pd.concat(event_frames, ignore_index=True)
    open_targets.to_parquet(root / "03_stock/open6_prediction_targets.parquet", index=False)
    parts = [frame[["BUILDINGSTRUCTUREID", "event_id"] + [x for x in frame if x.startswith("score_")]]
             for frame in event_frames]
    attributes = event_frames[0][[c for c in ("BUILDINGSTRUCTUREID", "main_class", "sub_class",
                                               "prototype_floor_signature", "construction_year_value")
                                  if c in event_frames[0]]]
    rating = build_rating(pd.concat(parts, ignore_index=True), open_events)
    if attributes is not None:
        rating = rating.merge(attributes, on="BUILDINGSTRUCTUREID", how="left", validate="one_to_one")
    for q in (.05, .10, .20):
        rating[f"low_score_probability_q{int(q*100):02d}"] = exchangeable_selection_probability(rating.BERI_lower, q)
    rating.to_parquet(root / "03_stock/open6_prediction_stock.parquet", index=False)
    _task(c, "T02_open6_stock_freeze", "complete", {"buildings": len(rating), "events": len(open_events)})
    return rating


def verify_lockbox_prediction_receipt(c):
    root = Path(c["results_root"])
    receipt_path = root / "00_protocol/lockbox_prediction_receipt.json"
    if not receipt_path.is_file():
        raise Blocked("lockbox prediction receipt is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "FROZEN_BEFORE_LOCKBOX" or set(receipt.get("event_ids", [])) != {"HKO_LI_0054", "HKO_LI_0066"}:
        raise Blocked("lockbox prediction receipt identity is invalid")
    for label in ("prediction", "cohorts", "baselines", "rules", "upstream_receipt"):
        path = Path(receipt.get(label + "_path", ""))
        if not path.is_file() or sha256(path) != receipt.get(label + "_sha256"):
            raise Blocked(f"frozen lockbox {label} is missing or changed")
    if receipt.get("analysis_contract_sha256") != sha256(Path(c["workspace_root"]) / "config/completion_analysis_contract.json"):
        raise Blocked("analysis contract changed after prediction freeze")
    return receipt


def freeze_lockbox_predictions(c, rating, approval, training_target=None):
    root = Path(c["results_root"])
    receipt_path = root / "00_protocol/lockbox_prediction_receipt.json"
    if receipt_path.exists():
        verify_lockbox_prediction_receipt(c)
        return receipt_path
    protected = [root / "04_validation/lockbox_predictions_frozen.csv",
                 root / "04_validation/lockbox_cohorts_frozen.csv",
                 root / "04_validation/lockbox_baselines_frozen.csv",
                 root / "00_protocol/lockbox_prediction_rules.json"]
    if any(path.exists() for path in protected):
        raise Blocked("partial prior lockbox freeze exists without a valid receipt")
    if training_target is None:
        raise Blocked("open-event training targets are required before lockbox prediction freeze")
    required_attributes = {"BUILDINGSTRUCTUREID", "event_id", "main_class", "sub_class",
                           "prototype_floor_signature", "construction_year_value"}
    if missing := sorted(required_attributes - set(training_target.columns)):
        raise Blocked(f"lockbox training target missing fields: {missing}")
    if training_target.duplicated(["BUILDINGSTRUCTUREID", "event_id"]).any():
        raise Blocked("lockbox training target contains duplicate building-event rows")
    prediction = rating.copy()
    attributes = training_target.sort_values("event_id").drop_duplicates("BUILDINGSTRUCTUREID")
    attribute_columns = ["BUILDINGSTRUCTUREID", "main_class", "sub_class",
                         "prototype_floor_signature", "construction_year_value"]
    for column in attribute_columns[1:]:
        if column in prediction:
            prediction = prediction.drop(columns=column)
    prediction = prediction.merge(attributes[attribute_columns], on="BUILDINGSTRUCTUREID",
                                  how="left", validate="one_to_one")
    target = training_target.copy()
    target["construction_era"] = pd.cut(pd.to_numeric(target.construction_year_value, errors="coerce"),
                                         [-np.inf, 1979, 1999, 2010, np.inf],
                                         labels=["pre1980", "1980s90s", "2000s", "2011plus"]).astype("string")
    building_attributes = target.sort_values("event_id").drop_duplicates("BUILDINGSTRUCTUREID")
    outcomes = [column for column in ("cooling_shortfall_kwh", "absolute_dh32", "extra_dh26",
                                       "recovery_hours", "population_degree_hours") if column in target]
    baseline_rows = []
    for method, fields in (("building_type", ["main_class"]),
                           ("detailed_use_floor", ["sub_class", "prototype_floor_signature"]),
                           ("construction_era", ["construction_era"])):
        for outcome in outcomes:
            lookup = target.groupby(fields, dropna=False, observed=False)[outcome].agg(
                training_group_burden="median", training_observations="count").reset_index()
            scored = building_attributes[["BUILDINGSTRUCTUREID"] + fields].merge(
                lookup, on=fields, how="left", validate="many_to_one")
            prediction = prediction.merge(
                scored[["BUILDINGSTRUCTUREID", "training_group_burden"]].rename(
                    columns={"training_group_burden": f"prediction__{method}__{outcome}"}),
                on="BUILDINGSTRUCTUREID", how="left", validate="one_to_one")
            prediction[f"prediction__{method}__{outcome}"] *= -1.0
            lookup["method"] = method; lookup["outcome"] = outcome
            lookup["group_fields"] = "|".join(fields)
            baseline_rows.append(lookup)
    prediction["prediction_role"] = "six_open_events_frozen_before_lockbox_first_look"
    path = root / "04_validation/lockbox_predictions_frozen.csv"
    atomic_csv(path, prediction)
    cohort_rows = []
    score_methods = {"BERI": "BERI_lower", "stage_N": "score_N_lower", "stage_C": "score_C_lower",
                     "stage_O": "score_O_lower", "stage_R": "score_R_lower"}
    for outcome in outcomes:
        methods = dict(score_methods)
        methods.update({method: f"prediction__{method}__{outcome}"
                        for method in ("building_type", "detailed_use_floor", "construction_era")})
        for method, column in methods.items():
            for q in (.05, .10, .20):
                probabilities = exchangeable_selection_probability(prediction[column], q)
                cohort_rows.extend({"BUILDINGSTRUCTUREID": building_id, "method": method,
                                    "outcome": outcome, "fraction": q, "selection_probability": probability}
                                   for building_id, probability in zip(prediction.BUILDINGSTRUCTUREID, probabilities))
    cohorts_path = root / "04_validation/lockbox_cohorts_frozen.csv"
    atomic_csv(cohorts_path, pd.DataFrame(cohort_rows))
    baselines_path = root / "04_validation/lockbox_baselines_frozen.csv"
    atomic_csv(baselines_path, pd.concat(baseline_rows, ignore_index=True, sort=False))
    rules_path = root / "00_protocol/lockbox_prediction_rules.json"
    rules = {"rating_events": list(c.get("upstream_open_events", ["HKO_LI_0049", "HKO_LI_0052", "HKO_LI_0058", "HKO_LI_0068", "HKO_LI_0085", "HKO_LI_0090"])),
             "selection_fractions": [0.05, 0.10, 0.20], "tie_policy": "exchangeable_expected_selection",
             "component_order": ["N", "C", "O", "R"], "recovery_composition": "simultaneous_all_components",
             "baseline_methods": ["building_type", "detailed_use_floor", "construction_era"],
             "outcomes": outcomes, "first_look_refit_forbidden": True}
    atomic_json(rules_path, rules)
    upstream_receipt = root / "00_protocol/upstream_receipt.json"
    receipt = {"status": "FROZEN_BEFORE_LOCKBOX", "event_ids": ["HKO_LI_0054", "HKO_LI_0066"],
               "created_at": time.time(), "prediction_path": str(path), "prediction_sha256": sha256(path),
               "cohorts_path": str(cohorts_path), "cohorts_sha256": sha256(cohorts_path),
               "baselines_path": str(baselines_path), "baselines_sha256": sha256(baselines_path),
               "rules_path": str(rules_path), "rules_sha256": sha256(rules_path),
               "upstream_receipt_path": str(upstream_receipt), "upstream_receipt_sha256": sha256(upstream_receipt),
               "analysis_contract_sha256": sha256(Path(c["workspace_root"]) / "config/completion_analysis_contract.json"),
               "approval_contract_sha256": approval["completion_code_lock_sha256"],
               "first_event_cannot_update_second": True, "not_historical_forward_prediction": True}
    atomic_json(receipt_path, receipt)
    _task(c, "T03_lockbox_predictions", "complete", {"receipt": sha256(receipt_path)})
    return receipt_path


def _write_version_lock(c, name):
    workspace = Path(c["workspace_root"]); upstream = Path(c["upstream_project_root"])
    files = ["src/utils/g4a_local_execution.py", "src/utils/e8_g3_runner.py", "src/utils/e8_g3_postprocess.py",
             "src/utils/simulation_metrics.py", "config/contracts/estimand_and_field_contract.csv",
             "config/contracts/thermal_monitoring_contract.csv"]
    rels = {}
    for rel in files:
        path = workspace / rel
        rels[os.path.relpath(path, upstream).replace("\\", "/")] = sha256(path)
    production = runner.load_production(upstream, code_root=workspace)
    import hashlib
    prototype = production.prototype_parameter_table().to_csv(index=False)
    lock = {"name": name, "files": rels,
            "prototype_parameter_table_sha256": hashlib.sha256(prototype.encode()).hexdigest()}
    path = workspace / "config" / f"{name}_version_lock.json"; atomic_json(path, lock)
    return path


def child_config(c, approval, label, manifest, output_dir, solver_only=False):
    frame = pd.read_csv(manifest).fillna("")
    workspace = Path(c["workspace_root"]); upstream = Path(c["upstream_project_root"])
    lock = _write_version_lock(c, label)
    resources = approval["resources"]
    # The two child ledgers have disjoint ceilings whose sums cannot exceed the
    # one approved completion budget; a new batch ID cannot manufacture quota.
    new_physics = int(frame.loc[frame.disposition.eq("new"), "physics_hash"].nunique())
    amendment = approval.get("resource_only_amendment", {})
    lockbox_wall_seconds = int(amendment.get("new_lockbox_wall_seconds", 36000))
    lockbox_max_bytes = int(amendment.get("new_lockbox_max_bytes", 10 * 1024**3))
    if label == "lockbox":
        retry_allowance = 150
        allocation = {"max_calls": new_physics + retry_allowance, "max_retries": retry_allowance,
                      "wall_seconds": lockbox_wall_seconds, "max_bytes": lockbox_max_bytes}
    else:
        retry_allowance = 50
        allocation = {"max_calls": new_physics + retry_allowance, "max_retries": retry_allowance,
                      "wall_seconds": int(resources["wall_seconds"]) - lockbox_wall_seconds,
                      "max_bytes": int(resources["max_bytes"]) - lockbox_max_bytes}
        if allocation["wall_seconds"] <= 0 or allocation["max_bytes"] <= 0:
            raise Blocked("approved completion resources leave no positive mechanism allocation")
    for key in allocation:
        if allocation[key] > int(resources[key]):
            raise Blocked(f"child resource allocation exceeds approved total: {key}")
    dispositions = frame.disposition.value_counts().to_dict()
    cfg = {
        "project_root": str(upstream), "input_root": str(upstream), "code_root": str(workspace),
        "batch_id": c["batch_id"] + "_" + label,
        "output_dir": str(Path(output_dir)), "development_dir": str(Path(c["results_root"]) / "00_protocol" / "dry_runs" / label),
        "approved_output_roots": [str(Path(c["results_root"]).resolve())],
        "manifest": str(Path(manifest)), "manifest_sha256": sha256(manifest),
        "cache_root": str(upstream / "intermediate/simulation_outputs/upstream"),
        "energyplus": str(Path(c["energyplus"])),
        "version_lock": str(lock), "expected_environment": "hk-heatwave-repro", "expected_python": "3.11",
        "paper_only": True, "concurrency": min(4, int(resources["concurrency"])),
        "max_calls": allocation["max_calls"], "max_retries": allocation["max_retries"],
        "wall_seconds": allocation["wall_seconds"], "max_bytes": allocation["max_bytes"],
        "min_free_bytes": int(resources["min_free_bytes"]), "grace_seconds": 30,
        "run_timeout_seconds": int(resources.get("run_timeout_seconds", 1800)), "progress_seconds": 15,
        "max_worker_rss_bytes": int(resources.get("max_worker_rss_bytes", 2 * 1024**3)),
        "count_hardlinks_once": True, "delegated_workorder_start": True,
        "scope": {"expected_runs": len(frame), "expected_unique_physics": int(frame.physics_hash.nunique()),
                  "expected_groups": int(frame.groupby(["event_id", "base_case_id", "prototype_id", "floor_position"]).ngroups),
                  "events": sorted(frame.event_id.unique()), "prehistory_days": 14,
                  "expected_dispositions": {str(k): int(v) for k, v in dispositions.items()},
                  "label": label, "report_prefix": label.lower(), "solver_only": solver_only,
                  "event_isolated_groups": True,
                  "review_archive": label + "_review.zip", "confirmation_scope": "approved E8 completion only",
                  "owner_registry": str(Path(c["results_root"]) / "runtime" / "owners" /
                                        f"{c['batch_id']}_{label}_owner.json"),
                  "initial_ceiling": {"max_calls": allocation["max_calls"], "max_retries": allocation["max_retries"],
                                      "wall_seconds": allocation["wall_seconds"], "max_bytes": allocation["max_bytes"],
                                      "min_free_bytes": int(resources["min_free_bytes"])}},
    }
    cfg["analysis_version"] = "E8-completion-closure-v1"
    path = workspace / "config" / f"{label}_runtime.json"
    state_path = Path(output_dir) / "state.json"
    if path.is_file() and state_path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        # The frozen initial ceiling belongs to the existing work order and is
        # not rewritten by a later, explicitly approved resource amendment.
        cfg["scope"]["initial_ceiling"] = existing.get("scope", {}).get(
            "initial_ceiling", cfg["scope"]["initial_ceiling"])
        changed = {key for key in set(existing) | set(cfg) if existing.get(key) != cfg.get(key)}
        if changed - runner.RESOURCE_KEYS:
            raise Blocked(f"existing {label} child configuration changed outside resource keys: "
                          f"{sorted(changed - runner.RESOURCE_KEYS)}")
        for key in runner.RESOURCE_KEYS:
            if key in cfg:
                existing[key] = cfg[key]
        cfg = existing
    atomic_json(path, cfg)
    return runner.load_config(path)


def _collect_group_tables(base_dirs, destination):
    frames = {"energy": [], "thermal": [], "scores": [], "qc": []}
    for base in base_dirs:
        for name in frames:
            frames[name].extend(pd.read_csv(p) for p in Path(base).glob(f"**/{name}.csv"))
    destination.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, parts in frames.items():
        if not parts: raise Blocked(f"No {name} group tables found under {base_dirs}")
        frame = pd.concat(parts, ignore_index=True)
        atomic_csv(destination / f"e8_design_event_{name}.csv", frame)
        out[name] = frame
    return out


def _materialize_mechanism_groups(c, mech_cfg):
    root = Path(c["results_root"]); work = Path(mech_cfg["output_dir"])
    logical = pd.read_csv(root / "02_mechanisms/manifests/mechanism_logical_map.csv").fillna("")
    state = runner.load_state(mech_cfg)
    if state is None or any(x.get("status") != runner.COMPLETE for x in state["rows"].values()):
        raise Blocked("mechanism physical runs are not completely verified")
    # Project complete inputs and parsed evidence into the isolated work tree;
    # downstream science must never receive a parsed-only alias.
    for row in logical.to_dict("records"):
        target = work / "02_runs/g3_results" / str(row["run_id"])
        source = (Path(row["source_result_dir"]) if row["disposition"] == "reuse_complete"
                  else work / "02_runs/g3_results" / str(row["resolved_run_id"]))
        if target.resolve() == source.resolve():
            continue
        for folder, names in (("inputs", SOURCE_INPUTS), ("parsed", SOURCE_PARSED)):
            (target / folder).mkdir(parents=True, exist_ok=True)
            for name in names:
                source_file = source / folder / name
                if not source_file.is_file():
                    raise Blocked(f"mechanism source evidence missing: {source_file}")
                destination = target / folder / name
                if not destination.exists():
                    shutil.copy2(source_file, destination)
    production = runner.load_production(Path(c["upstream_project_root"]), code_root=Path(c["workspace_root"]))
    for _, group in logical.groupby(["mechanism_id", "event_id"], sort=False):
        if len(group) != 5 or set(group.scenario_id) != set(runner.CORE_SCENARIO_IDS):
            raise Blocked("mechanism logical group is not a complete five-scenario pair")
        runner.group_metrics(mech_cfg, group.to_dict("records"), production)
    tables = _collect_group_tables([work / "groups"], root / "02_mechanisms/group_tables")
    merged = tables["energy"].merge(tables["thermal"], on=["event_id", "base_case_id", "prototype_id", "floor_position"],
                                     suffixes=("", "_thermal"), validate="one_to_one")
    effects = mechanism_effects(merged, logical)
    atomic_csv(root / "02_mechanisms/controlled_pair_results.csv", effects)
    summary = effects.groupby(["factor", "prototype_id", "level"], as_index=False).median(numeric_only=True)
    atomic_csv(root / "02_mechanisms/controlled_effect_summary.csv", summary)
    bounds = [c for c in effects if c.startswith("effect_") and (c.endswith("_lower_h") or c.endswith("_upper_h"))]
    paired = effects.groupby(["event_id", "prototype_id", "factor", "level"], as_index=False).agg(
        actual_pair_count=("mechanism_id", "size"))
    for column in bounds:
        values = effects.groupby(["event_id", "prototype_id", "factor", "level"])[column].median().reset_index(name=column)
        paired = paired.merge(values, on=["event_id", "prototype_id", "factor", "level"], validate="one_to_one")
    atomic_csv(root / "02_mechanisms/residential_paired_event_prototype_effects.csv", paired)
    return {"groups": len(merged), "effects": len(effects)}


def _complete_analysis(c, approval):
    root = Path(c["results_root"]); upstream = Path(c["upstream_project_root"])
    prediction_receipt = verify_lockbox_prediction_receipt(c)
    energy = pd.read_csv(root / "01_core_e8/e8_design_event_energy_service.csv")
    thermal = pd.read_csv(root / "01_core_e8/e8_design_event_thermal_recovery.csv")
    scores = pd.read_csv(root / "01_core_e8/e8_stage_score_bounds.csv")
    stock_artifacts = ([root / "03_stock/events" / f"{event}.parquet" for event in ALL_EVENTS]
                       + [root / "03_stock/stock_event_coverage.csv", root / "03_stock/city_event_energy_peak.csv",
                          root / "03_stock/simultaneous_recovery_coverage.csv",
                          root / "04_validation/sensitivity_temperature_thresholds.csv"])
    if not _task_current(c, "T10_stock_e8", stock_artifacts):
        production = runner.load_production(upstream, code_root=Path(c["workspace_root"]))
        response = aggregate_design_metrics(energy, thermal, scores, production.prototype_parameter_table())
        atomic_csv(root / "01_core_e8/prototype_floor_event_response.csv", response)
        _task(c, "T10_stock_e8", "running")
        stock = propagate_stock(upstream, response, root / "03_stock")
        open_index = pd.read_csv(root / "01_core_e8/open6_source_index.csv").fillna("")
        lock_manifest = pd.read_csv(root / "00_protocol/lockbox_manifests/lockbox_run_manifest.csv").fillna("")
        lock_manifest["source_result_dir"] = lock_manifest.run_id.map(
            lambda run_id: str(root / "runtime/lockbox/02_runs/g3_results" / str(run_id)))
        open_index = hydrate_source_index(open_index)
        lock_index = hydrate_source_index(lock_manifest, result_root=root / "runtime/lockbox")
        run_index = pd.concat([open_index, lock_index], ignore_index=True, sort=False)
        event_dates = pd.read_csv(Path(c["workspace_root"]) / "frozen_upstream/inputs/E8_event_assignment.csv")
        components = pd.read_parquet(root / "03_stock/building_component_map.parquet")
        city_event_energy_trajectories(run_index, components, production.prototype_parameter_table(), event_dates,
                                       root / "03_stock/city_trajectories")
        apply_simultaneous_recovery(run_index, components, root / "03_stock/events", root / "03_stock/recovery_trajectories")
        threshold = threshold_sensitivity_from_runs(run_index, root / "04_validation/sensitivity_temperature_thresholds.csv")
        _task(c, "T10_stock_e8", "complete", {**stock, "artifact_hashes": _artifact_hashes(stock_artifacts)})
    else:
        threshold = pd.read_csv(root / "04_validation/sensitivity_temperature_thresholds.csv")
    event_frames = [pd.read_parquet(root / "03_stock/events" / f"{event}.parquet") for event in ALL_EVENTS]
    events = pd.concat(event_frames, ignore_index=True)
    score_cols = [c for c in events if c.startswith("score_")]
    event_scores = events[["BUILDINGSTRUCTUREID", "event_id"] + score_cols
                          + (["NA_reason"] if "NA_reason" in events else [])]
    first = []
    open_events = list(c["upstream_open_events"] if "upstream_open_events" in c else ["HKO_LI_0049", "HKO_LI_0052", "HKO_LI_0058", "HKO_LI_0068", "HKO_LI_0085", "HKO_LI_0090"])
    frozen = pd.read_csv(prediction_receipt["prediction_path"])
    frozen_cohorts = pd.read_csv(prediction_receipt["cohorts_path"])
    first_receipt_path = root / "00_protocol/lockbox_first_look_receipt.json"
    if first_receipt_path.exists():
        existing = json.loads(first_receipt_path.read_text(encoding="utf-8"))
        if existing.get("prediction_receipt_sha256") != sha256(root / "00_protocol/lockbox_prediction_receipt.json"):
            raise Blocked("first-look receipt is bound to a different prediction freeze")
        for label in ("rank", "capture"):
            p = Path(existing[label + "_path"])
            if not p.is_file() or sha256(p) != existing[label + "_sha256"]:
                raise Blocked("immutable lockbox first-look output changed")
        first = None
    elif any((root / "04_validation" / name).exists() for name in
             ("lockbox_first_look_rank.csv", "lockbox_first_look_capture.csv")):
        raise Blocked("first-look output exists without its immutable receipt")
    for lock in (() if first is None else ("HKO_LI_0054", "HKO_LI_0066")):
        a, b = evaluate_fold(event_scores, events, open_events, lock, f"LOCKBOX_FIRST_LOOK_{lock}",
                             frozen_rating=frozen, frozen_cohorts=frozen_cohorts)
        a["first_look"] = True; b["first_look"] = True; first.append((a, b))
    if first is not None:
        rank_path = root / "04_validation/lockbox_first_look_rank.csv"
        capture_path = root / "04_validation/lockbox_first_look_capture.csv"
        atomic_csv(rank_path, pd.concat([x[0] for x in first], ignore_index=True))
        atomic_csv(capture_path, pd.concat([x[1] for x in first], ignore_index=True))
        atomic_json(first_receipt_path, {"status": "IMMUTABLE_FIRST_LOOK", "created_at": time.time(),
                    "prediction_receipt_sha256": sha256(root / "00_protocol/lockbox_prediction_receipt.json"),
                    "rank_path": str(rank_path), "rank_sha256": sha256(rank_path),
                    "capture_path": str(capture_path), "capture_sha256": sha256(capture_path)})
    _task(c, "T06_lockbox_first_look", "complete")
    evaluation_artifacts = [root / "04_validation/event_holdout_rank_metrics.csv",
                            root / "04_validation/event_holdout_screening_capture.csv",
                            root / "04_validation/full_e8_reference_rating.csv",
                            root / "04_validation/screening_queue_overlap.csv",
                            root / "04_validation/sensitivity_beri_weights.csv",
                            root / "04_validation/cross_event_failure_frequency_severity.csv",
                            root / "04_validation/sensitivity_design_event_range.csv",
                            root / "03_stock/building_reference_beri.parquet"]
    if _task_current(c, "T11_evaluation", evaluation_artifacts):
        evaluation = load_task_state(c)["tasks"]["T11_evaluation"].get("outputs", {})
    else:
        _task(c, "T11_evaluation", "running")
        evaluation = run_event_evaluation(event_scores, events, root / "04_validation")
        rating = pd.read_csv(root / "04_validation/full_e8_reference_rating.csv")
        attributes = events.sort_values("event_id").drop_duplicates("BUILDINGSTRUCTUREID")
        attribute_columns = [c for c in ("BUILDINGSTRUCTUREID", "main_class", "sub_class",
                                          "prototype_floor_signature", "construction_year_value") if c in attributes]
        rating = rating.merge(attributes[attribute_columns], on="BUILDINGSTRUCTUREID", how="left", validate="one_to_one")
        rating.to_parquet(root / "03_stock/building_reference_beri.parquet", index=False)
        uncertainty_metrics = [x for x in ("net_delta_kwh", "relative_delta", "shortfall_ref_kwh_thermal",
                                           "absolute_occupied_degree_hours_above_32c",
                                           "positive_extra_occupied_degree_hours_above_26c_vs_rated") if x in energy or x in thermal]
        design = energy.merge(thermal, on=["event_id", "base_case_id", "prototype_id", "floor_position"], suffixes=("", "_thermal"))
        rows = []
        for keys, group in design.groupby(["event_id", "prototype_id", "floor_position"]):
            for metric in uncertainty_metrics:
                values = pd.to_numeric(group[metric], errors="coerce").dropna()
                rows.append({"event_id": keys[0], "prototype_id": keys[1], "floor_position": keys[2],
                             "metric": metric, "design_cases": len(values), "p05": values.quantile(.05) if len(values) else np.nan,
                             "median": values.median() if len(values) else np.nan, "p95": values.quantile(.95) if len(values) else np.nan,
                             "uncertainty_identity": "discrete_design_case_range_not_iid_building_CI"})
        atomic_csv(root / "04_validation/sensitivity_design_event_range.csv", pd.DataFrame(rows))
        _task(c, "T11_evaluation", "complete", {**evaluation, "artifact_hashes": _artifact_hashes(evaluation_artifacts)})
    analysis_decisions = _validate_required_analysis_decisions(c)
    decision_hash = analysis_decision_hash(analysis_decisions)
    required_analysis_artifacts = [
        root / "04_validation/absolute_performance_thresholds.csv",
        root / "04_validation/absolute_performance_thresholds_building_event.parquet",
        root / "04_validation/multi_stage_threshold_sets.csv",
        root / "04_validation/multi_stage_threshold_sets_building_event.parquet",
        root / "04_validation/day_night_humid_recovery.csv",
        root / "03_stock/region_hourly_snapshots.csv",
        root / "03_stock/construction_year_provenance.csv",
        root / "03_stock/historical_existence_sensitivity.parquet",
        root / "03_stock/historical_existence_sensitivity_summary.csv",
    ]
    if not _task_current(c, "T12_sensitivity", required_analysis_artifacts, dependency_hash=decision_hash):
        _task(c, "T12_sensitivity", "running")
        absolute_detail = compute_absolute_performance_thresholds(events, analysis_decisions["absolute_performance_thresholds"])
        absolute_summary = absolute_detail.groupby(["event_id", "rule_id", "stage", "metric", "unit", "operator", "threshold",
                                                    "eligibility_denominator", "time_window", "source", "decision_version"],
                                                   dropna=False, as_index=False).agg(
            buildings=("BUILDINGSTRUCTUREID", "nunique"), eligible_buildings=("eligible", "sum"),
            failed_buildings=("failed", lambda x: int(x.dropna().astype(bool).sum())),
            unknown_buildings=("failed", lambda x: int(x.isna().sum())))
        atomic_parquet(required_analysis_artifacts[1], absolute_detail)
        atomic_csv(required_analysis_artifacts[0], absolute_summary)
        multi_detail = compute_multi_stage_threshold_sets(absolute_detail, analysis_decisions["multi_stage_threshold_set"])
        multi_summary = multi_detail.groupby(["event_id", "set_id", "rule_ids", "minimum_failed", "unit", "time_window", "source"],
                                             dropna=False, as_index=False).agg(
            buildings=("BUILDINGSTRUCTUREID", "nunique"),
            failed_buildings=("set_failed", lambda x: int(x.dropna().astype(bool).sum())),
            evaluable_buildings=("set_failed", lambda x: int(x.notna().sum())),
            unknown_buildings=("set_failed", lambda x: int(x.isna().sum())))
        atomic_parquet(required_analysis_artifacts[3], multi_detail); atomic_csv(required_analysis_artifacts[2], multi_summary)

        if c.get("execution_profile") == "acceptance_fixture":
            zone_intervals = pd.read_parquet(root / "00_protocol/fixture_zone_intervals.parquet")
            snapshot_input = pd.read_parquet(root / "00_protocol/fixture_region_snapshot_input.parquet")
        else:
            open_index = hydrate_source_index(pd.read_csv(root / "01_core_e8/open6_source_index.csv").fillna(""))
            lock_manifest = pd.read_csv(root / "00_protocol/lockbox_manifests/lockbox_run_manifest.csv").fillna("")
            lock_manifest["source_result_dir"] = lock_manifest.run_id.map(
                lambda run_id: str(root / "runtime/lockbox/02_runs/g3_results" / str(run_id)))
            lock_index = hydrate_source_index(lock_manifest, result_root=root / "runtime/lockbox")
            run_index = pd.concat([open_index, lock_index], ignore_index=True, sort=False)
            zone_intervals = None
            production = runner.load_production(upstream, code_root=Path(c["workspace_root"]))
            snapshot_input = region_snapshot_input_from_runs(
                run_index, pd.read_parquet(root / "03_stock/building_component_map.parquet"),
                pd.read_parquet(root / "03_stock/building_spatial_links.parquet"),
                production.prototype_parameter_table(),
                pd.read_csv(Path(c["workspace_root"]) / "frozen_upstream/inputs/E8_event_assignment.csv"),
                pd.read_csv(root / "03_stock/city_event_energy_peak.csv"),
                analysis_decisions["region_snapshot_times_hkt"])
        if zone_intervals is None:
            day_night_parts = [
                compute_day_night_humid_recovery(part, analysis_decisions["day_night_windows"])
                for part in iter_zone_intervals_from_run_index(run_index)
            ]
            if not day_night_parts:
                raise Blocked("no O24 trajectories were available for day/night analysis")
            day_night = pd.concat(day_night_parts, ignore_index=True)
        else:
            day_night = compute_day_night_humid_recovery(zone_intervals, analysis_decisions["day_night_windows"])
        atomic_csv(required_analysis_artifacts[4], day_night)
        regional = compute_region_hourly_snapshots(snapshot_input, analysis_decisions["region_snapshot_times_hkt"])
        atomic_csv(required_analysis_artifacts[5], regional)
        provenance, existence = compute_construction_year_tables(
            pd.read_parquet(root / "03_stock/fixed_stock_buildings.parquet") if c.get("execution_profile") != "acceptance_fixture"
            else events.sort_values("event_id").drop_duplicates("BUILDINGSTRUCTUREID"),
            pd.read_csv(Path(c["workspace_root"]) / "frozen_upstream/inputs/E8_event_assignment.csv"),
            analysis_decisions["construction_year_provenance"])
        atomic_csv(required_analysis_artifacts[6], provenance); atomic_parquet(required_analysis_artifacts[7], existence)
        existence_summary = existence.groupby(["event_id", "existence_primary_status", "construction_year_source_class"],
                                               dropna=False, as_index=False).agg(
            buildings=("BUILDINGSTRUCTUREID", "nunique"),
            year_start_exists=("exists_year_start_assumption", lambda x: int(x.fillna(False).sum())),
            year_end_exists=("exists_year_end_assumption", lambda x: int(x.fillna(False).sum())))
        existence_summary["unit"] = "buildings"; existence_summary["scope"] = existence.historical_scope.iloc[0]
        atomic_csv(required_analysis_artifacts[8], existence_summary)
        decisions = pd.DataFrame([
            {"item": name, "required": True, "status": "approved_and_computed",
             "contract_field": f"decision_parameters.{name}", "decision_hash": decision_hash,
             "reason": "formal schema validated; production numerical function completed"}
            for name in analysis_decisions
        ] + [{"item": "future_building_reuse", "required": True,
              "status": "excluded_by_scope_approval" if approval["future_building_results_decision"] == "exclude_incompatible_old_results" else "pending_hash_audit",
              "contract_field": "future_building_results_decision", "decision_hash": decision_hash,
              "reason": "no silent compatibility assumption"}])
        atomic_csv(root / "00_protocol/decisions_required.csv", decisions)
        unresolved = decisions[decisions.required & ~decisions.status.isin({"approved_and_computed", "excluded_by_scope_approval"})]
        if not unresolved.empty: raise Blocked("required analysis decisions remain unresolved: " + ", ".join(unresolved.item))
        _task(c, "T12_sensitivity", "complete", {"dependency_hash": decision_hash,
              "artifact_hashes": _artifact_hashes(required_analysis_artifacts),
              "screening_fractions": "0.05|0.10|0.20", "threshold_rows": len(threshold),
              "thresholds_c": "30|32|34", "endpoint_identity": "sensitivity_not_mathematical_bounds"})
    assignment = pd.read_csv(Path(c["workspace_root"]) / "frozen_upstream/inputs/E8_event_assignment.csv")
    catalog = pd.read_csv(Path(c["workspace_root"]) / "frozen_upstream/inputs/event_catalog_65.csv")
    climate = catalog.merge(assignment[["parent_event_id", "E8_role"]], left_on="event_id", right_on="parent_event_id", how="left")
    atomic_csv(root / "05_evidence/climate_catalog_and_event_roles.csv", climate)
    future_path = root / "05_evidence/future_compatibility.csv"
    future = pd.read_csv(future_path)
    if approval["future_building_results_decision"] == "exclude_incompatible_old_results":
        future["compatibility"] = "excluded_by_explicit_scope_decision"
        future["final_decision"] = "old future-building energy/thermal outputs excluded; unchanged climate hazard background retained separately"
    else:
        raise Blocked("reuse_only_if_hash_compatible requires a separate verified compatibility audit; none is available in this delivery")
    atomic_csv(future_path, future)
    evidence = audit_external_evidence(Path(c["workspace_root"]), upstream, root / "05_evidence",
                                       approval["future_building_results_decision"],
                                       Path(c["external_evidence_registry"]))
    if evidence["missing_required_available_sources"]:
        raise Blocked("required external sources marked reusable are missing: " +
                      ", ".join(evidence["missing_required_available_sources"]))
    evidence_artifacts = [root / "05_evidence/external_source_compatibility.csv",
                          root / "05_evidence/validation_evidence_compatibility.csv", future_path]
    _task(c, "T13_evidence_compatibility", "complete",
          {**evidence, "artifact_hashes": _artifact_hashes(evidence_artifacts)})
    finalize_source_contract(root)
    _task(c, "T14_source_data", "complete")


def _execute_locked(c, approval_path, retry_failed=False):
    if c.get("execution_profile") == "acceptance_fixture":
        approval = json.loads(Path(approval_path).read_text(encoding="utf-8"))
        if approval.get("approval_status") != "approved_test_only" or approval.get("execution_profile") != "acceptance_fixture":
            raise Blocked("acceptance profile requires an explicit test-only approval")
        from utils.e8_completion_fixture_acceptance import execute_acceptance_profile
        return execute_acceptance_profile(c, approval, retry_failed=retry_failed)
    approval = validate_approval(c, approval_path)
    _reconcile_interrupted_local_tasks(c)
    _reconcile_delegated_task_clock(c, "T05_lockbox_run", "lockbox")
    _reconcile_delegated_task_clock(c, "T08_mechanism_run", "mechanisms")
    verification = verify_upstream(c, deep=True,
                                   accepted_version_lock_sha256=approval["accepted_upstream_version_lock_sha256"])
    if verification["status"] != "QUALIFIED": raise Waiting(str(verification["issues"]))
    _protocol(c)
    _validate_required_analysis_decisions(c)
    _check_completion_budget(c, approval, "open6_import")
    open_paths = [Path(c["results_root"]) / "01_core_e8" / name for name in
                  ("open6_energy.csv", "open6_thermal.csv", "open6_scores.csv", "open6_source_index.csv")]
    if not _task_current(c, "T01_open6_import", open_paths):
        receipt = make_upstream_receipt(c, verification); _copy_open6(c, receipt)
        state = load_task_state(c); state["tasks"]["T01_open6_import"]["outputs"] = {"artifact_hashes": _artifact_hashes(open_paths)}; save_task_state(c, state)
    rating_path = Path(c["results_root"]) / "03_stock/open6_prediction_stock.parquet"
    target_path = Path(c["results_root"]) / "03_stock/open6_prediction_targets.parquet"
    if _task_current(c, "T02_open6_stock_freeze", [rating_path, target_path]):
        open6_rating = pd.read_parquet(rating_path)
    else:
        open6_rating = _freeze_open6_stock(c)
        state = load_task_state(c); state["tasks"]["T02_open6_stock_freeze"]["outputs"] = {"artifact_hashes": _artifact_hashes([rating_path, target_path])}; save_task_state(c, state)
    prediction_receipt = freeze_lockbox_predictions(c, open6_rating, approval,
                                                    training_target=pd.read_parquet(target_path))
    _check_completion_budget(c, approval, "lockbox_compile")
    _task(c, "T04_lockbox_compile", "running")
    lockbox = compile_lockboxes(c, approval, prediction_receipt)
    _task(c, "T04_lockbox_compile", "complete", {"manifest": sha256(Path(c["results_root"]) / "00_protocol/lockbox_manifests/lockbox_run_manifest.csv")})
    lock_cfg = child_config(c, approval, "lockbox", Path(c["results_root"]) / "00_protocol/lockbox_manifests/lockbox_run_manifest.csv",
                            Path(c["results_root"]) / "runtime/lockbox")
    _task(c, "T05_lockbox_run", "running")
    code = _execute_or_accept_verified_child(lock_cfg, retry_failed)
    if code: raise Blocked(f"Lockbox scheduler stopped with code {code}; resume the same completion command")
    _task(c, "T05_lockbox_run", "complete")
    _task(c, "T07_mechanism_compile", "running")
    _check_completion_budget(c, approval, "mechanism_compile")
    logical, physical = compile_mechanisms(c, approval)
    _task(c, "T07_mechanism_compile", "complete", {"physical_manifest": sha256(Path(c["results_root"]) / "02_mechanisms/manifests/mechanism_physical_run_manifest.csv")})
    mech_cfg = child_config(c, approval, "mechanisms", Path(c["results_root"]) / "02_mechanisms/manifests/mechanism_physical_run_manifest.csv",
                            Path(c["results_root"]) / "runtime/mechanisms", solver_only=True)
    _task(c, "T08_mechanism_run", "running")
    code = _execute_or_accept_verified_child(mech_cfg, retry_failed)
    if code: raise Blocked(f"Mechanism scheduler stopped with code {code}; resume the same completion command")
    _task(c, "T08_mechanism_run", "complete")
    _materialize_mechanism_groups(c, mech_cfg)
    summarize(c)
    _check_completion_budget(c, approval, "analysis")
    _complete_analysis(c, approval)
    _task(c, "T15_release", "running")
    _check_completion_budget(c, approval, "release")
    verification = freeze_release(Path(c["results_root"]), Path(c["task_matrix"]))
    if not verification["complete"]:
        _task(c, "T15_release", "blocked", error=str(verification["issues"]))
        raise Blocked(f"final evidence gate failed: {verification['issues']}")
    _task(c, "T15_release", "complete", {"final_verification": sha256(Path(c["results_root"]) / "07_release/final_verification.json")})
    return 0


def execute(c, approval_path, retry_failed=False):
    """Hold one completion-wide lock across freeze, solve, analysis and release."""
    lock = Path(c["results_root"]) / "runtime/completion_workorder.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with runner.exclusive_lock(lock):
        return _execute_locked(c, approval_path, retry_failed)


def summarize(c):
    root = Path(c["results_root"]); _protocol(c)
    # Core E8 group outputs: imported Open6 + newly completed lockboxes.
    open6 = root / "01_core_e8"
    lock_groups = root / "runtime/lockbox/groups"
    if lock_groups.exists():
        imported = {n: pd.read_csv(open6 / f"open6_{n}.csv") for n in ("energy", "thermal", "scores")}
        new = _collect_group_tables([lock_groups], root / "01_core_e8/lockbox_group_tables")
        mapping = {"energy": "e8_design_event_energy_service.csv", "thermal": "e8_design_event_thermal_recovery.csv", "scores": "e8_stage_score_bounds.csv"}
        combined = {}
        for name, file in mapping.items():
            combined[name] = pd.concat([imported[name], new[name]], ignore_index=True)
            atomic_csv(root / "01_core_e8" / file, combined[name])
        if len(combined["energy"]) != 2376 or len(combined["thermal"]) != 2376 or len(combined["scores"]) != 9504:
            raise Blocked("E8 core integration is not 2,376 groups / 9,504 component scores")
        open_index = pd.read_csv(open6 / "open6_source_index.csv")
        lock_index = pd.read_csv(root / "00_protocol/lockbox_manifests/lockbox_run_manifest.csv")
        lock_index["source_batch"] = c["batch_id"] + "_lockbox"
        lock_index["source_run_id"] = lock_index.run_id
        lock_index["source_result_dir"] = lock_index.run_id.map(lambda x: str(root / "runtime/lockbox/02_runs/g3_results" / str(x)))
        index_columns = list(open_index.columns)
        source_index = pd.concat([open_index, lock_index[index_columns]], ignore_index=True)
        if len(source_index) != 11880: raise Blocked("E8 logical source index is not 11,880 rows")
        atomic_csv(root / "01_core_e8/e8_logical_physical_source_index.csv", source_index)
        _task(c, "T09_core_e8_integrate", "complete")
    state = load_task_state(c)
    atomic_json(root / "00_protocol/status.json", state)
    return verify_final(root, Path(c["task_matrix"]))


def status(c):
    verification = verify_upstream(c, deep=False,
                                   accepted_version_lock_sha256=c["upstream"].get("candidate_version_lock_sha256"))
    state = load_task_state(c)
    child = {}
    for label in ("lockbox", "mechanisms"):
        path = Path(c["results_root"]) / "runtime" / label / "state.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        attempts = data.get("attempts", [])
        child[label] = {"calls": sum(bool(item.get("launched")) for item in attempts),
                        "retry_calls": sum(bool(item.get("launched")) and bool(item.get("is_retry")) for item in attempts),
                        "spent_seconds": data.get("spent_seconds", 0),
                        "bytes_on_disk": runner.unique_tree_bytes(Path(c["results_root"]) / "runtime" / label)}
    total = _completion_resource_usage(c)
    return {"upstream": verification["status"], "upstream_issues": verification["issues"],
            "completion_status": state.get("completion_label") or state.get("status"),
            "tasks": {k: v.get("status") for k, v in state["tasks"].items()},
            "resource_usage_across_resume": child,
            "solver_calls": sum(int(x.get("calls", 0) or 0) for x in child.values()),
            "retry_calls": sum(int(x.get("retry_calls", 0) or 0) for x in child.values()),
            "completion_total_usage": total}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Complete approved E8 numerical work after Open6; no argument only shows this help")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "config/e8_completion.json"))
    actions = parser.add_mutually_exclusive_group()
    for name in ("doctor", "dry-run", "execute", "status", "summarize", "package-review"):
        actions.add_argument("--" + name, action="store_true")
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--approval")
    args = parser.parse_args(argv)
    if not any(getattr(args, x) for x in ("doctor", "dry_run", "execute", "status", "summarize", "package_review")):
        parser.print_help(); return 0
    if (args.resume or args.retry_failed) and not args.execute: parser.error("--resume/--retry-failed require --execute")
    if args.execute and (not args.resume or not args.approval): parser.error("execution requires --execute --resume --approval FILE")
    try:
        c = load_config(args.config)
        if args.doctor or args.dry_run:
            print(json.dumps(doctor(c, deep=args.dry_run), ensure_ascii=False, indent=2)); return 0
        if args.status:
            print(json.dumps(status(c), ensure_ascii=False, indent=2)); return 0
        if args.summarize:
            lock = Path(c["results_root"]) / "runtime/completion_workorder.lock"
            lock.parent.mkdir(parents=True, exist_ok=True)
            with runner.exclusive_lock(lock):
                print(json.dumps(summarize(c), ensure_ascii=False, indent=2)); return 0
        if args.package_review:
            lock = Path(c["results_root"]) / "runtime/completion_workorder.lock"
            lock.parent.mkdir(parents=True, exist_ok=True)
            with runner.exclusive_lock(lock):
                print(package_review(Path(c["results_root"]), Path(c["task_matrix"]))); return 0
        return execute(c, args.approval, args.retry_failed)
    except Waiting as exc:
        print(f"WAITING_FOR_UPSTREAM: {exc}", file=sys.stderr); return 2
    except (Blocked, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr); return 3
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr); return 4
