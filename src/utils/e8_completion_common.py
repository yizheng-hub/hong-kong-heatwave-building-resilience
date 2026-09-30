"""Shared contracts for the isolated E8 completion work-order.

This module is deliberately solver-free.  It verifies the live Open6 batch by
read-only evidence, validates the one-time approval receipt, and provides
transactional state helpers for every later analysis node.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time
from typing import Any

import pandas as pd

try:
    import psutil
except ImportError:  # doctor reports this as an environment failure
    psutil = None


OPEN_EVENTS = (
    "HKO_LI_0049", "HKO_LI_0052", "HKO_LI_0058",
    "HKO_LI_0068", "HKO_LI_0085", "HKO_LI_0090",
)
LOCKBOX_EVENTS = ("HKO_LI_0054", "HKO_LI_0066")
ALL_EVENTS = (
    "HKO_LI_0049", "HKO_LI_0052", "HKO_LI_0054", "HKO_LI_0058",
    "HKO_LI_0066", "HKO_LI_0068", "HKO_LI_0085", "HKO_LI_0090",
)
SCENARIOS = ("CF_UNL", "HW_UNL", "HW_RATED", "HW_C70", "HW_O24_SYNC")
FINAL_LABEL = "READY_FOR_FIGURES_AND_MANUSCRIPT"
CODE_LABEL = "CODE_COMPLETE_WAITING_FOR_USER_APPROVAL"


class Blocked(RuntimeError):
    pass


class Waiting(RuntimeError):
    pass


def sha256(path: str | Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".writing.{os.getpid()}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temp, path)


def atomic_csv(path: str | Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".writing.{os.getpid()}")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, path)


def atomic_parquet(path: str | Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".writing.{os.getpid()}")
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    c = read_json(path)
    if not isinstance(c, dict):
        raise Blocked(f"Invalid configuration: {path}")
    workspace = path.parents[1]
    c["workspace_root"] = str(workspace)
    c["config_path"] = str(path)
    for key in ("upstream_project_root", "results_root", "approval_template", "task_matrix",
                "analysis_decision_schema", "analysis_decision_proposal", "paper_panel_registry",
                "external_evidence_registry", "energyplus"):
        value = Path(c[key])
        c[key] = str(value if value.is_absolute() else (workspace / value).resolve())
    if tuple(c["all_events"]) != ALL_EVENTS or tuple(c["lockbox_events"]) != LOCKBOX_EVENTS:
        raise Blocked("E8 event identity differs from the frozen protocol")
    if tuple(c["scenarios"]) != SCENARIOS:
        raise Blocked("Five-scenario identity differs")
    if int(c["resources"]["concurrency"]) not in range(1, 5):
        raise Blocked("concurrency must be 1..4")
    if not c.get("paper_only") or c.get("allow_animation"):
        raise Blocked("Completion work-order is paper-only and forbids animation")
    return c


def process_alive(pid: int | None, created: float | None, host: str | None) -> bool | None:
    if not pid or not created or host != socket.gethostname() or psutil is None:
        return None
    try:
        process = psutil.Process(int(pid))
        return math.isclose(process.create_time(), float(created), abs_tol=2.0) and process.is_running()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def file_lock_held(path: str | Path) -> bool:
    """Probe an existing one-byte work-order lock without changing its contents."""
    path = Path(path)
    if not path.exists():
        return False
    # The completion workspace must not request write access to any live Open6
    # evidence. Both supported lock APIs can probe the byte read-only.
    stream = path.open("rb")
    try:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                return True
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            return False
        import fcntl
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return False
    finally:
        stream.close()


def imported_module_audit(workspace: Path) -> list[dict[str, Any]]:
    import sys
    rows = []
    for name, module in sorted(sys.modules.items()):
        if not name.startswith("utils."):
            continue
        file = getattr(module, "__file__", None)
        if not file:
            continue
        resolved = Path(file).resolve()
        rows.append({"module": name, "file": str(resolved),
                     "inside_isolated_workspace": workspace in resolved.parents,
                     "sha256": sha256(resolved) if resolved.exists() else ""})
    return rows


def _state_activity(state_path: Path) -> tuple[bool, dict[str, Any]]:
    state = read_json(state_path, {})
    session = state.get("active_session") or {}
    alive = process_alive(session.get("pid"), session.get("created"), session.get("host"))
    counts = pd.Series([x.get("status", "unknown") for x in state.get("rows", {}).values()]).value_counts()
    return alive is True, {"path": str(state_path), "active_session": session, "process_alive": alive,
                           "row_states": {str(k): int(v) for k, v in counts.items()}}


def verify_upstream(c: dict[str, Any], *, deep: bool = False,
                    accepted_version_lock_sha256: str | None = None) -> dict[str, Any]:
    """Verify Open6 without invoking its summarize/package writers."""
    root = Path(c["upstream_project_root"])
    issues: list[str] = []
    version_path = root / "config/open_event_version_lock.json"
    manifest_path = root / "intermediate/simulation_outputs/upstream/open6_delivery/open6_target_manifest.csv"
    expected_version = accepted_version_lock_sha256 or c["upstream"]["version_lock_sha256"]
    actual_version = sha256(version_path) if version_path.exists() else None
    if actual_version != expected_version:
        issues.append("upstream version-lock hash differs or is missing")
    if not manifest_path.exists() or sha256(manifest_path) != c["upstream"]["manifest_sha256"]:
        issues.append("upstream target-manifest hash differs or is missing")
    activity = []
    workorder_lock = root / "intermediate/simulation_outputs/upstream/E8_open6/workorder.lock"
    workorder_lock_held = file_lock_held(workorder_lock)
    confirmations = read_json(root / "intermediate/simulation_outputs/upstream/E8_open6/manual_start_confirmations.json", []) or []
    workorder = {"path": str(workorder_lock), "lock_held": workorder_lock_held,
                 "latest_confirmation": confirmations[-1] if confirmations else None}
    if workorder_lock_held:
        issues.append("active upstream workorder lock")
    for event in OPEN_EVENTS:
        state_path = root / "intermediate/simulation_outputs/upstream/E8_open6/events" / event / "state.json"
        if state_path.exists():
            active, detail = _state_activity(state_path)
            activity.append(detail)
            if active:
                issues.append(f"active upstream writer: {event}")
    combined = root / "intermediate/simulation_outputs/upstream/E8_open6/combined"
    aggregate = read_json(combined / "open6_aggregate_status.json", {})
    numerical_complete = aggregate.get("status") == "NUMERICAL_COMPLETE_OPEN6"
    source_specs = {
        "energy": (combined / "open6_design_event_energy_service.csv", 1782),
        "thermal": (combined / "open6_design_event_thermal_recovery.csv", 1782),
        "scores": (combined / "open6_stage_score_bounds.csv", 7128),
        "source_index": (combined / "open6_logical_physical_source_index.csv", 8910),
    }
    sources: dict[str, Any] = {}
    for label, (path, expected) in source_specs.items():
        item = {"path": str(path), "exists": path.exists(), "expected_rows": expected}
        if path.exists() and numerical_complete:
            frame = pd.read_csv(path)
            item.update(rows=len(frame), sha256=sha256(path), valid=len(frame) == expected)
            if len(frame) != expected:
                issues.append(f"{label} rows={len(frame)} expected={expected}")
        else:
            item.update(rows=None, sha256=None, valid=False)
            if numerical_complete:
                issues.append(f"missing completed source table: {label}")
        sources[label] = item
    if numerical_complete and deep:
        for event in OPEN_EVENTS:
            state_path = root / "intermediate/simulation_outputs/upstream/E8_open6/events" / event / "state.json"
            state = read_json(state_path, {})
            rows = state.get("rows", {})
            if len(rows) != 1485 or any(x.get("status") != "metrics_complete" for x in rows.values()):
                issues.append(f"{event} state is not 1485 metrics_complete rows")
            if not (state.get("last_verification") or {}).get("complete"):
                issues.append(f"{event} last evidence verification is incomplete")
    waiting = (not numerical_complete or workorder_lock_held or
               any("active upstream writer" in x for x in issues))
    status = "WAITING_FOR_UPSTREAM" if waiting else ("BLOCKED" if issues else "QUALIFIED")
    return {"status": status, "numerical_complete": numerical_complete, "issues": issues,
            "actual_version_lock_sha256": actual_version, "expected_version_lock_sha256": expected_version,
            "candidate_version_lock_sha256": c["upstream"].get("candidate_version_lock_sha256"),
            "aggregate": aggregate, "activity": activity, "workorder": workorder, "sources": sources,
            "checked_at": time.time(), "read_only": True}


def make_upstream_receipt(c: dict[str, Any], verification: dict[str, Any]) -> dict[str, Any]:
    if verification["status"] != "QUALIFIED":
        raise Waiting(f"Upstream is not qualified: {verification['status']} {verification['issues']}")
    receipt = {"receipt_version": "open6-readonly-receipt-v1", "created_at": time.time(),
               "source_project_root": c["upstream_project_root"],
               "target_manifest_sha256": c["upstream"]["manifest_sha256"],
               "version_lock_sha256": verification["actual_version_lock_sha256"],
               "superseded_version_lock_sha256": c["upstream"]["version_lock_sha256"],
               "source_tables": verification["sources"], "source_batch": "E8_open6_core_nonlockbox_v1",
               "source_scope": list(OPEN_EVENTS), "no_upstream_write": True}
    receipt["receipt_sha256"] = canonical(receipt)
    return receipt


def approval_contract(c: dict[str, Any]) -> dict[str, Any]:
    workspace = Path(c["workspace_root"])
    plan = workspace / "TASK_PLAN.md"
    matrix = Path(c["task_matrix"])
    readiness_matrix = workspace / "FORMAL_PRODUCTION_READINESS_MATRIX.csv"
    analysis = workspace / "config/completion_analysis_contract.json"
    mechanisms = workspace / "config/mechanism_levels_recommended.json"
    code_files = [workspace / x for x in ("src/workflows/run_e8_completion.py",
                  "src/utils/e8_completion_common.py",
                  "src/utils/e8_completion_manifests.py", "src/utils/e8_completion_execution.py",
                  "src/utils/e8_completion_analysis.py", "src/utils/e8_completion_required_analyses.py",
                  "src/utils/e8_completion_evidence.py", "src/utils/e8_completion_stock.py", "src/utils/e8_completion_release.py",
                  "src/utils/g4a_local_execution.py", "src/utils/e8_g3_runner.py", "src/utils/e8_g3_postprocess.py")]
    code_lock = canonical({str(x.relative_to(workspace)): sha256(x) for x in code_files})
    return {"plan_sha256": sha256(plan), "task_matrix_sha256": sha256(matrix),
            "formal_readiness_matrix_sha256": sha256(readiness_matrix),
            "analysis_contract_sha256": sha256(analysis),
            "mechanism_levels_sha256": sha256(mechanisms),
            "completion_code_lock_sha256": code_lock,
            "upstream_manifest_sha256": c["upstream"]["manifest_sha256"],
            "upstream_version_lock_sha256": c["upstream"]["version_lock_sha256"],
            "accepted_upstream_version_lock_sha256": c["upstream"].get("candidate_version_lock_sha256", c["upstream"]["version_lock_sha256"]),
            "lockbox_events": list(LOCKBOX_EVENTS), "all_events": list(ALL_EVENTS),
            "results_root": str(Path(c["results_root"]).resolve()), "resources": c["resources"]}


def validate_approval(c: dict[str, Any], path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    approval = read_json(path)
    if not isinstance(approval, dict) or approval.get("approval_status") != "approved":
        raise Blocked("The completion approval is absent or not explicitly approved")
    expected = approval_contract(c)
    mismatches = {key: {"expected": value, "actual": approval.get(key)}
                  for key, value in expected.items() if approval.get(key) != value}
    if mismatches:
        raise Blocked(f"Approval does not bind the current plan: {mismatches}")
    if approval.get("lockbox_release") != "released_after_prediction_freeze":
        raise Blocked("Lockbox release must require prediction freeze")
    if approval.get("mechanism_levels_status") != "approved":
        raise Blocked("Residential mechanism levels remain unapproved")
    if approval.get("results_root_confirmed") is not True:
        raise Blocked("The completion results location has not been explicitly confirmed")
    if approval.get("future_building_results_decision") not in {"exclude_incompatible_old_results", "reuse_only_if_hash_compatible"}:
        raise Blocked("Future-building compatibility decision is unresolved")
    resources = approval.get("resources", {})
    for name in ("max_calls", "max_retries", "wall_seconds", "max_bytes", "min_free_bytes"):
        if int(resources.get(name, 0)) <= 0:
            raise Blocked(f"Approved resource is missing: {name}")
    return approval


def initial_task_state(c: dict[str, Any]) -> dict[str, Any]:
    matrix = pd.read_csv(c["task_matrix"])
    return {"batch_id": c["batch_id"], "created_at": time.time(), "updated_at": time.time(),
            "status": CODE_LABEL, "tasks": {row.task_id: {"status": "pending", "attempts": 0,
                                                             "dependency_hash": None, "outputs": {}}
                                                 for row in matrix.itertuples(index=False)},
            "solver_calls": 0, "retry_calls": 0, "active_seconds": 0.0,
            "completion_label": None, "last_error": None}


def load_task_state(c: dict[str, Any]) -> dict[str, Any]:
    path = Path(c["results_root"]) / "00_protocol/task_state.json"
    state = read_json(path)
    if state is None:
        state = initial_task_state(c)
    if state.get("batch_id") != c["batch_id"]:
        raise Blocked("Results directory belongs to a different completion batch")
    return state


def save_task_state(c: dict[str, Any], state: dict[str, Any]) -> None:
    state["updated_at"] = time.time()
    atomic_json(Path(c["results_root"]) / "00_protocol/task_state.json", state)


def verified_outputs(paths: list[str | Path]) -> dict[str, str]:
    result = {}
    for value in paths:
        path = Path(value)
        if not path.exists() or not path.is_file() or path.stat().st_size == 0:
            raise Blocked(f"Required output missing or empty: {path}")
        result[str(path)] = sha256(path)
    return result
