"""Small no-solver acceptance flow through the real completion interfaces."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from utils.e8_completion_analysis import (build_rating, evaluate_fold, exchangeable_selection_probability, mechanism_effects,
                                           run_event_evaluation, threshold_sensitivity_from_runs)
from utils.e8_completion_common import ALL_EVENTS, OPEN_EVENTS, atomic_csv, atomic_json, sha256
from utils.e8_completion_execution import (_artifact_hashes, _complete_analysis, _protocol, _task,
                                           _task_current, execute, freeze_lockbox_predictions,
                                           verify_lockbox_prediction_receipt)
from utils.e8_completion_release import freeze_release, package_review
from utils.e8_completion_stock import aggregate_design_metrics, propagate_event


class FixtureInterrupted(RuntimeError):
    pass


def _scores_and_targets():
    rows = []
    kinds = [(1, "Residential", "Private", "p_r@middle"),
             (2, "Commercial", "Office", "p_c@middle"),
             (3, "Mixed", "Mixed", "p_c@middle|p_r@middle")]
    for event_index, event in enumerate(ALL_EVENTS):
        for building, main, sub, signature in kinds:
            value = .2 + building * .15 + event_index * .005
            row = {"BUILDINGSTRUCTUREID": building, "event_id": event, "main_class": main,
                   "sub_class": sub, "prototype_floor_signature": signature,
                   "construction_year_value": 1970 + building * 20,
                   "E_HW_kwh": 110.0 + event_index, "E_CF_kwh": 100.0, "net_delta_kwh": 10.0 + event_index,
                   "required_ref_kwh_thermal": 20.0, "effective_monitoring_hours": 24.0,
                   "cooling_shortfall_kwh": 12 - building + event_index / 10,
                   "absolute_dh32": 8 - building + event_index / 10,
                   "crossing_lower_h": 2.0 + building / 10,
                   "crossing_upper_h": 2.25 + building / 10,
                   "extra_dh26": 6 - building + event_index / 10,
                   "recovery_hours": 5 - building / 2 + event_index / 10,
                   "recovery_status": "exact", "recovery_lower_h": 5 - building / 2 + event_index / 10,
                   "recovery_upper_h": 5 - building / 2 + event_index / 10,
                   "population_degree_hours": (20 - building) if main != "Commercial" else 0,
                   "NA_reason": ""}
            for component in "NCOR":
                row[f"score_{component}_lower"] = value
                row[f"score_{component}_upper"] = min(1.0, value + .05)
            if building == 2 and event == "HKO_LI_0066":
                row["recovery_hours"] = np.nan; row["recovery_status"] = "right_censored"
                row["recovery_lower_h"] = 24.0; row["recovery_upper_h"] = np.nan
            if building == 3 and event == "HKO_LI_0090":
                row["score_R_lower"] = row["score_R_upper"] = np.nan
                row["NA_reason"] = "required_recovery_support_unknown"
            rows.append(row)
    return pd.DataFrame(rows)


def _design_tables():
    energy, thermal, scores = [], [], []
    for event_index, event in enumerate(ALL_EVENTS):
        for prototype in ("p_r", "p_c"):
            for design in ("d1", "d2"):
                key = {"event_id": event, "base_case_id": f"{prototype}_{design}",
                       "prototype_id": prototype, "floor_position": "middle"}
                energy.append({**key, "E_HW_kwh": 10 + event_index, "E_CF_kwh": 8,
                               "net_delta_kwh": 2 + event_index, "sum_positive_timestep_delta_kwh": 3,
                               "required_ref_kwh_thermal": 5, "delivered_raw_kwh_thermal": 4,
                               "shortfall_ref_kwh_thermal": 1})
                thermal.append({**key, "absolute_occupied_degree_hours_above_32c": 2 + event_index,
                                "absolute_occupied_degree_hours_above_26c": 4 + event_index,
                                "positive_extra_occupied_degree_hours_above_26c_vs_rated": 1 + event_index,
                                "first_crossing_lower_h": 2, "first_crossing_upper_h": 2.25,
                                "recovery_lower_h": 5, "recovery_upper_h": 5.25,
                                "maximum_occupied_outage_temperature_c": 34})
                for component in "NCOR":
                    scores.append({**key, "component": component, "score_lower": .5,
                                   "score_upper": .6, "NA_reason": ""})
    prototypes = pd.DataFrame({"prototype_id": ["p_r", "p_c"], "floor_area_m2": [100.0, 200.0]})
    return pd.DataFrame(energy), pd.DataFrame(thermal), pd.DataFrame(scores), prototypes


def _threshold_fixture(root: Path):
    start = pd.Timestamp("2020-07-01 00:00", tz="Asia/Hong_Kong")
    weather = pd.DataFrame({"timestamp_hkt": pd.date_range(start, periods=24, freq="h")})
    schedule = pd.DataFrame({"Z_occupancy": np.ones(24), "Z_ac": np.ones(24)})
    times = [stamp + pd.Timedelta(minutes=offset) for stamp in weather.timestamp_hkt for offset in (15, 30, 45, 60)]
    zones = pd.DataFrame({"timestamp_hkt": times, "zone_id": "Z", "zone_area_m2": 10.0,
                          "structural_conditioning_present": True, "temperature_c": 34.0})
    records = []
    for scenario, name in (("HW_O24_SYNC", "outage"), ("HW_RATED", "rated")):
        base = root / "source" / name
        (base / "inputs").mkdir(parents=True, exist_ok=True); (base / "parsed").mkdir(exist_ok=True)
        weather.to_parquet(base / "inputs/weather.parquet", index=False)
        schedule.to_parquet(base / "inputs/event_schedules.parquet", index=False)
        if scenario == "HW_O24_SYNC":
            zones.to_parquet(base / "parsed/native_timestep_zones.parquet", index=False)
        records.append({"run_id": name, "event_id": ALL_EVENTS[0], "base_case_id": "b",
                        "prototype_id": "p_r", "floor_position": "middle", "scenario_id": scenario,
                        "source_result_dir": str(base), "outage_start_hkt": start.isoformat(),
                        "zone_areas_json": '{"Z":10}', "conditioned_zones_json": '["Z"]'})
    return pd.DataFrame(records)


def _fixture_context(root: Path, interrupt_after: str | None):
    workspace = root / "fixture_workspace"
    (workspace / "config").mkdir(parents=True, exist_ok=True)
    (workspace / "frozen_upstream/inputs").mkdir(parents=True, exist_ok=True)
    (workspace / "frozen_upstream/docs").mkdir(parents=True, exist_ok=True)
    contract_path = workspace / "config/completion_analysis_contract.json"
    if not contract_path.exists():
        proposal = json.loads((Path(__file__).resolve().parents[2] / "config/completion_analysis_decisions.proposed.json").read_text(encoding="utf-8"))
        decisions = proposal["decision_parameters"]
        for value in decisions.values(): value["approval_status"] = "approved"
        atomic_json(contract_path, {"profile": "acceptance_fixture", "contract_version": "formal-fixture-v2",
                                    "decision_parameters": decisions})
    matrix = workspace / "config/matrix.csv"
    if not matrix.exists():
        task_rows = [{"task_id": task, "required": True} for task in
            ("T03_lockbox_predictions", "T06_lockbox_first_look", "T08_mechanism_run", "T09_core_e8_integrate",
             "T10_stock_e8", "T11_evaluation", "T12_sensitivity", "T13_evidence_compatibility", "T14_source_data")]
        task_rows.append({"task_id": "T15_release", "required": False})
        atomic_csv(matrix, pd.DataFrame(task_rows))
    assignment = pd.DataFrame({"parent_event_id": list(ALL_EVENTS), "E8_role": ["core"] * 8,
                               "mother_start_hkt": ["2020-07-01"] * 8, "mother_end_hkt": ["2020-07-02"] * 8})
    atomic_csv(workspace / "frozen_upstream/inputs/E8_event_assignment.csv", assignment)
    catalog_ids = list(ALL_EVENTS) + [f"HKO_LI_X{i:03d}" for i in range(57)]
    atomic_csv(workspace / "frozen_upstream/inputs/event_catalog_65.csv",
               pd.DataFrame({"event_id": catalog_ids, "start_year": range(1950, 2015),
                             "duration_days": [3] * 65, "mean_intensity_c": [1.0] * 65}))
    atomic_csv(workspace / "frozen_upstream/docs/validation_evidence_matrix.csv", pd.DataFrame([
        {"validation_layer": layer, "dataset": f"fixture_{layer}", "primary_metric": "fixture_metric",
         "value": 1.0, "claim_boundary": "test_profile_only", "status": "partial"}
        for layer in ("climate", "population", "building", "future")]))
    source_root = Path(__file__).resolve().parents[2]
    panel_registry = workspace / "config/paper_panel_registry.csv"
    shutil.copy2(source_root / "config/paper_panel_registry.csv", panel_registry)
    evidence_rows = []
    for layer in ("climate", "population", "building", "future"):
        relative = f"frozen_upstream/docs/fixture_source_{layer}.csv"
        source = workspace / relative
        atomic_csv(source, pd.DataFrame([{"fixture": layer, "value": 1.0}]))
        evidence_rows.append({"source_id": f"fixture_{layer}", "dataset_match": f"fixture_{layer}",
                              "validation_layer": layer, "source_path": relative,
                              "method_identity": "acceptance fixture source-specific method",
                              "claim_boundary": "test profile only", "disposition": "verified_reuse",
                              "required_for_final": True})
    evidence_registry = workspace / "config/external_evidence_registry.csv"
    atomic_csv(evidence_registry, pd.DataFrame(evidence_rows))
    approval_path = workspace / "config/acceptance_approval.json"
    if not approval_path.exists():
        atomic_json(approval_path, {"approval_status": "approved_test_only", "execution_profile": "acceptance_fixture",
                                    "completion_code_lock_sha256": "f" * 64,
                                    "future_building_results_decision": "exclude_incompatible_old_results"})
    return {"workspace_root": str(workspace), "upstream_project_root": str(workspace), "results_root": str(root),
            "task_matrix": str(matrix), "batch_id": "E8_acceptance_fixture", "execution_profile": "acceptance_fixture",
            "upstream_open_events": list(OPEN_EVENTS), "interrupt_after": interrupt_after,
            "paper_panel_registry": str(panel_registry), "external_evidence_registry": str(evidence_registry)}, approval_path


def execute_acceptance_profile(c, approval, retry_failed=False):
    root = Path(c["results_root"]); _protocol(c)
    profile_path = root / "00_protocol/execution_profile.json"
    if not profile_path.exists():
        atomic_json(profile_path, {"profile": "acceptance_fixture", "energyplus_calls": 0,
                                   "fake_solver": "persisted same-schema synthetic outputs",
                                   "code_root": str(Path(__file__).resolve().parents[2]),
                                   "expected_counts": {"core_groups": 32, "stage_scores": 128,
                                                       "logical_runs": 160, "buildings": 3,
                                                       "source_contract_rows": len(pd.read_csv(c["paper_panel_registry"]))}})
    else:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        if profile.get("code_root") != str(Path(__file__).resolve().parents[2]):
            profile["code_root"] = str(Path(__file__).resolve().parents[2])
            atomic_json(profile_path, profile)
    future_path = root / "05_evidence/future_compatibility.csv"
    if not future_path.exists():
        atomic_csv(future_path, pd.DataFrame([{"result": "fixture_future", "compatibility": "pending_hash_audit",
                                               "default_decision": "exclude", "alternative": "none",
                                               "requires_new_solver": False}]))
    atomic_json(root / "00_protocol/upstream_receipt.json", {"fixture": True, "read_only": True, "energyplus_calls": 0})
    data = _scores_and_targets()
    score_columns = [column for column in data if column.startswith("score_")]
    event_scores = data[["BUILDINGSTRUCTUREID", "event_id", "NA_reason"] + score_columns]
    new_freeze = not (root / "00_protocol/lockbox_prediction_receipt.json").exists()
    if new_freeze:
        rating = build_rating(event_scores, OPEN_EVENTS)
        for q in (.05, .10, .20):
            rating[f"low_score_probability_q{int(q*100):02d}"] = exchangeable_selection_probability(rating.BERI_lower, q)
        freeze_lockbox_predictions(c, rating, approval, training_target=data[data.event_id.isin(OPEN_EVENTS)])
    verify_lockbox_prediction_receipt(c)
    if new_freeze and c.get("interrupt_after") == "freeze":
        raise FixtureInterrupted("freeze")

    mechanism_path = root / "02_mechanisms/controlled_pair_results.csv"
    if not _task_current(c, "T08_mechanism_run", [mechanism_path]):
        mechanism_events = ("HKO_LI_0068", "HKO_LI_0085", "HKO_LI_0090")
        logical = pd.DataFrame([
            {"mechanism_id": identity, "prototype_id": "p_r", "factor": "capacity", "level": level,
             "event_id": event, "reference": reference}
            for event in mechanism_events for identity, level, reference in (("ref", "x1", True), ("treat", "x.9", False))])
        metrics = pd.DataFrame([
            {"base_case_id": identity, "prototype_id": "p_r", "event_id": event,
             "net_delta_kwh": 10.0 if reference else 9.0}
            for event in mechanism_events for identity, reference in (("ref", True), ("treat", False))])
        atomic_csv(mechanism_path, mechanism_effects(metrics, logical))
        _task(c, "T08_mechanism_run", "complete", {"artifact_hashes": _artifact_hashes([mechanism_path])})
        if c.get("interrupt_after") == "mechanism":
            raise FixtureInterrupted("mechanism")

    energy, thermal, scores, _ = _design_tables()
    core_paths = [root / "01_core_e8/e8_design_event_energy_service.csv",
                  root / "01_core_e8/e8_design_event_thermal_recovery.csv",
                  root / "01_core_e8/e8_stage_score_bounds.csv",
                  root / "01_core_e8/e8_logical_physical_source_index.csv"]
    if not _task_current(c, "T09_core_e8_integrate", core_paths):
        atomic_csv(core_paths[0], energy); atomic_csv(core_paths[1], thermal); atomic_csv(core_paths[2], scores)
        logical_rows = []
        for row in energy.itertuples(index=False):
            for scenario in ("CF_UNL", "HW_UNL", "HW_RATED", "HW_C70", "HW_O24_SYNC"):
                logical_rows.append({"event_id": row.event_id, "base_case_id": row.base_case_id,
                                     "prototype_id": row.prototype_id, "floor_position": row.floor_position,
                                     "scenario_id": scenario, "run_id": f"{row.base_case_id}_{row.event_id}_{scenario}",
                                     "physics_hash": f"alias_{row.event_id}_{row.prototype_id}_{row.floor_position}_{scenario}"})
        atomic_csv(core_paths[3], pd.DataFrame(logical_rows))
        _task(c, "T09_core_e8_integrate", "complete", {"artifact_hashes": _artifact_hashes(core_paths)})

    stock_artifacts = ([root / "03_stock/events" / f"{event}.parquet" for event in ALL_EVENTS]
                       + [root / "03_stock/stock_event_coverage.csv", root / "03_stock/city_event_energy_peak.csv",
                          root / "03_stock/simultaneous_recovery_coverage.csv",
                          root / "04_validation/sensitivity_temperature_thresholds.csv"])
    if not _task_current(c, "T10_stock_e8", stock_artifacts):
        (root / "03_stock/events").mkdir(parents=True, exist_ok=True)
        for event in ALL_EVENTS:
            data[data.event_id.eq(event)].to_parquet(root / "03_stock/events" / f"{event}.parquet", index=False)
        atomic_csv(root / "03_stock/stock_event_coverage.csv",
                   pd.DataFrame({"event_id": list(ALL_EVENTS), "rows": [3] * 8,
                                 "evaluable": [3] * 8, "not_evaluable": [0] * 8}))
        atomic_csv(root / "03_stock/city_event_energy_peak.csv", pd.DataFrame([
            {"event_id": event, "E_HW_kwh": 100 + index, "E_CF_kwh": 90, "net_delta_kwh": 10 + index,
             "simultaneous_HW_peak_kw": 20 + index, "simultaneous_CF_peak_kw": 18,
             "positive_timestep_delta_kwh": 12 + index, "negative_timestep_delta_kwh": -2,
             "paired_difference_at_HW_peak_kw": 2 + index,
             "HW_peak_time_hkt": "2020-07-01T12:00:00+08:00"} for index, event in enumerate(ALL_EVENTS)]))
        response = aggregate_design_metrics(energy, thermal, scores,
                                            pd.DataFrame({"prototype_id": ["p_r", "p_c"],
                                                          "floor_area_m2": [100.0, 200.0]}))
        atomic_csv(root / "01_core_e8/prototype_floor_event_response.csv", response)
        atomic_csv(root / "03_stock/simultaneous_recovery_coverage.csv",
                   pd.DataFrame({"event_id": list(ALL_EVENTS), "buildings": [3] * 8,
                                 "evaluable_recovery": [3] * 8, "right_censored": [0] * 8}))
        threshold_sensitivity_from_runs(_threshold_fixture(root), stock_artifacts[-1])
        _task(c, "T10_stock_e8", "complete", {"artifact_hashes": _artifact_hashes(stock_artifacts)})
        if c.get("interrupt_after") == "stock":
            raise FixtureInterrupted("stock")

    zone_rows = []
    for event in ALL_EVENTS:
        for prototype in ("p_r", "p_c"):
            for design in ("d1", "d2"):
                for hour, phase, temperature in ((9.25, "outage", 33.0), (20.25, "outage", 34.0),
                                                  (9.5, "recovery", 26.0), (9.75, "recovery", 26.0),
                                                  (10.0, "recovery", 26.0), (10.25, "recovery", 26.0)):
                    zone_rows.append({"event_id": event, "base_case_id": f"{prototype}_{design}",
                                      "prototype_id": prototype, "floor_position": "middle", "zone_id": "Z",
                                      "timestamp_hkt": pd.Timestamp("2020-07-01", tz="Asia/Hong_Kong") + pd.Timedelta(hours=hour),
                                      "temperature_c": temperature, "relative_humidity_pct": 65.0,
                                      "eligible_for_screen": True, "phase": phase, "interval_hours": .25})
    pd.DataFrame(zone_rows).to_parquet(root / "00_protocol/fixture_zone_intervals.parquet", index=False)
    snapshot_rows = []
    roles = ["event_start", "hw_peak", "outage_start", "restore", "restore_plus_6h", "restore_plus_24h"]
    regions = ["Hong Kong total", "Ho Man Tin", "Mong Kok", "Tsim Sha Tsui"]
    for event_index, event in enumerate(ALL_EVENTS):
        for building in (1, 2, 3):
            for region in regions:
                for role_index, role in enumerate(roles):
                    snapshot_rows.append({"BUILDINGSTRUCTUREID": building, "event_id": event, "study_region": region,
                                          "snapshot_role": role,
                                          "timestamp_hkt": (pd.Timestamp("2020-07-01", tz="Asia/Hong_Kong") + pd.Timedelta(hours=role_index)).isoformat(),
                                          "electric_power_kw": 2.0 + building + event_index / 10,
                                          "temperature_c": 30.0 + role_index / 2, "thermal_evaluable": True,
                                          "population_reference_2021": 10.0 if building != 2 else 0.0})
    pd.DataFrame(snapshot_rows).to_parquet(root / "00_protocol/fixture_region_snapshot_input.parquet", index=False)

    _complete_analysis(c, approval)
    if c.get("interrupt_after") == "analysis":
        raise FixtureInterrupted("analysis")
    verification = freeze_release(root, Path(c["task_matrix"]))
    if not verification["complete"]:
        raise RuntimeError(f"acceptance final gate failed: {verification['issues']}")
    if c.get("interrupt_after") == "publish":
        raise FixtureInterrupted("publish")
    atomic_json(root / "00_protocol/acceptance_execution_summary.json",
                {"status": verification["status"], "profile": verification["profile"],
                 "production_ready": verification["production_ready"], "energyplus_calls": 0,
                 "formal_route": ["execute", "_execute_locked", "execute_acceptance_profile",
                                  "_complete_analysis", "freeze_release", "package_review"]})
    archive = package_review(root, Path(c["task_matrix"]))
    _task(c, "T15_release", "complete", {"archive_sha256": sha256(archive)})
    return 0


def run_fixture_acceptance(root: Path, interrupt_after: str | None = None) -> dict:
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=True)
    c, approval_path = _fixture_context(root, interrupt_after)
    execute(c, approval_path)
    final = json.loads((root / "07_release/final_verification.json").read_text(encoding="utf-8"))
    archive = root / "07_release/E8_completion_review.zip"
    final.update({"energyplus_calls": 0, "events": list(ALL_EVENTS), "buildings": 3,
                  "archive": str(archive), "archive_sha256": sha256(archive)})
    return final
