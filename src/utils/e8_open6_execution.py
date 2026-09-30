"""Continuous local work-order for the six approved, non-lockbox E8 events."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import socket
import time
import zipfile

import numpy as np
import pandas as pd

from utils import g4a_local_execution as core
from utils.e8_open6_analysis import write_analysis_dependencies
from utils.e8_open6_prepare import OPEN_EVENTS, LOCKBOX_EVENTS, sha256

VERSION = "e8-open6-local-v1"
RAW_RETAIN = ["eplusout.csv", "eplusout.err", "eplusout.end", "eplusout.rdd", "eplusout.mdd"]
ALL_EVENTS = ("HKO_LI_0049", "HKO_LI_0052", "HKO_LI_0054", "HKO_LI_0058",
              "HKO_LI_0066", "HKO_LI_0068", "HKO_LI_0085", "HKO_LI_0090")


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic(path: Path, value) -> None:
    core.atomic_json(path, value)


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    c = _json(path)
    root = (path.parent / c["project_root"]).resolve()
    c["project_root"] = str(root)
    c["config_path"] = str(path)
    for key in ("delivery_dir", "output_dir", "energyplus", "version_lock"):
        c[key] = str((root / c[key]).resolve())
    enabled = tuple(c["enabled_events"])
    if enabled != tuple(OPEN_EVENTS):
        if enabled != ALL_EVENTS:
            raise core.Blocked("Work order must be the approved Open6 order or the full frozen E8 order")
        release_path = c.get("lockbox_release_record")
        if not release_path:
            raise core.Blocked("Full E8 requires an explicit lockbox_release_record")
        release_path = (root / release_path).resolve()
        release = _json(release_path)
        if release.get("status") != "released" or set(release.get("event_ids", [])) != LOCKBOX_EVENTS:
            raise core.Blocked("Lockbox release record is not valid for 0054/0066")
        c["lockbox_release_record"] = str(release_path)
    if c["concurrency"] < 1 or c["concurrency"] > 4:
        raise core.Blocked("concurrency must be 1..4")
    if not c.get("paper_only", False) or c.get("allow_animation", True):
        raise core.Blocked("Open6 is paper-only and forbids animation/frame production")
    return c


def delivery_summary(c: dict) -> dict:
    path = Path(c["delivery_dir"]) / "open6_dry_run_summary.json"
    if not path.exists():
        raise core.Blocked("Open6 manifests have not been compiled")
    summary = _json(path)
    events = tuple(c["enabled_events"])
    expected_runs, expected_groups = len(events) * 1485, len(events) * 297
    if summary.get("logical_runs") != expected_runs or summary.get("logical_groups") != expected_groups:
        raise core.Blocked("Open6 frozen target cardinality differs")
    if tuple(summary.get("events", [])) != events:
        raise core.Blocked("Open6 event/lockbox identity differs")
    if events == tuple(OPEN_EVENTS) and set(summary.get("lockboxes_excluded", [])) != LOCKBOX_EVENTS:
        raise core.Blocked("Open6 lockbox exclusion identity differs")
    target = Path(c["delivery_dir"]) / "open6_target_manifest.csv"
    if sha256(target) != summary["manifest_sha256"]:
        raise core.Blocked("Open6 total manifest hash differs")
    return summary


def _event_runtime_config(c: dict, event_id: str, summary: dict, remaining: dict) -> dict:
    meta = summary["event_manifests"][event_id]
    dispositions = {str(k): int(v) for k, v in meta["dispositions"].items()}
    out = Path(c["output_dir"]) / "events" / event_id
    existing = core.load_state({"output_dir": str(out)}) if (out / "state.json").exists() else None
    attempts = len(existing.get("attempts", [])) if existing else 0
    retries = sum(bool(a.get("is_retry")) for a in existing.get("attempts", [])) if existing else 0
    spent = float(existing.get("spent_seconds", 0)) if existing else 0.0
    event_existing_bytes = core.tree_bytes(out)
    config = {
        "project_root": c["project_root"], "batch_id": f"E8_open6_{event_id}_v1",
        "output_dir": str(out), "development_dir": str(Path(c["delivery_dir"]) / "dry_run" / event_id),
        "manifest": meta["path"], "manifest_sha256": meta["sha256"],
        "cache_root": str(Path(c["project_root"]) / "intermediate/simulation_outputs/upstream"), "energyplus": c["energyplus"],
        "expected_environment": c["expected_environment"], "expected_python": c["expected_python"],
        "analysis_version": c["analysis_version"], "version_lock": c["version_lock"],
        "paper_only": True, "concurrency": c["concurrency"],
        "max_calls": attempts + max(0, remaining["calls"]),
        "max_retries": retries + max(0, remaining["retries"]),
        "wall_seconds": int(spent + max(1, remaining["seconds"])),
        "max_bytes": int(event_existing_bytes + max(1, remaining["bytes"])),
        "min_free_bytes": c["min_free_bytes"], "grace_seconds": c["grace_seconds"],
        "run_timeout_seconds": c["run_timeout_seconds"], "progress_seconds": c["progress_seconds"],
        "max_worker_rss_bytes": c["max_worker_rss_bytes"], "hardlink_inputs": True,
        "approved_output_roots": ["intermediate/simulation_outputs/upstream"],
        "retained_raw_files": RAW_RETAIN, "compact_attempt_after_qc": True,
        "count_hardlinks_once": True,
        "delegated_workorder_start": True,
        "storage_scan_seconds": 60,
        "human_progress": True,
        "scope": {
            "label": f"E8 Open6 {event_id}", "report_prefix": event_id.lower(),
            "event_isolated_groups": True,
            "events": [event_id], "prehistory_days": 14, "expected_runs": 1485,
            "expected_unique_physics": 1485, "expected_groups": 297,
            "expected_dispositions": dispositions,
            "review_archive": f"{event_id}_local_review.zip",
            "execution_report": f"{event_id}_execution_report.md",
            "confirmation_scope": f"Open6 work order event {event_id}",
            "owner_registry": f"intermediate/simulation_outputs/upstream/E8_open6/owners/{event_id}.json",
            "initial_ceiling": {"max_calls": c["max_calls"], "max_retries": c["max_retries"],
                                "wall_seconds": c["wall_seconds"], "max_bytes": c["max_bytes"],
                                "min_free_bytes": c["min_free_bytes"]},
            "review_code_files": ["src/workflows/run_e8_local.py", "src/utils/e8_open6_execution.py",
                                  "src/utils/g4a_local_execution.py", "src/utils/e8_open6_analysis.py"],
        },
    }
    if event_id == "HKO_LI_0049":
        config["reuse_complete_state"] = str(Path(c["project_root"]) / "intermediate/simulation_outputs/upstream/G4a_local/state.json")
    return config


def _event_config_file(c: dict, event_id: str, summary: dict, remaining: dict) -> dict:
    value = _event_runtime_config(c, event_id, summary, remaining)
    path = Path(c["output_dir"]) / "event_configs" / f"{event_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic(path, value)
    return core.load_config(path)


def _states(c: dict) -> dict[str, dict | None]:
    result = {}
    for event_id in c["enabled_events"]:
        path = Path(c["output_dir"]) / "events" / event_id / "state.json"
        result[event_id] = _json(path) if path.exists() else None
    return result


def unique_tree_bytes(paths) -> int:
    """Count hard-linked bytes once; fall back conservatively when file identity is unavailable."""
    seen = set()
    total = 0
    for root in paths:
        root = Path(root)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino) if stat.st_ino else (str(path.resolve()),)
            if identity in seen:
                continue
            seen.add(identity)
            total += stat.st_size
    return total


def resource_usage(c: dict) -> dict:
    states = _states(c)
    attempts = [a for s in states.values() if s for a in s.get("attempts", [])]
    unique_bytes = unique_tree_bytes([c["delivery_dir"], c["output_dir"]])
    return {
        "attempt_reservations": len(attempts),
        "actual_calls": sum(bool(a.get("launched")) for a in attempts),
        "retries": sum(bool(a.get("is_retry")) for a in attempts),
        "active_seconds": sum(float(s.get("spent_seconds", 0)) for s in states.values() if s),
        "new_bytes_unique_file_identity": unique_bytes,
        "new_bytes_tree_accounting": unique_bytes,
        "free_disk_bytes": shutil.disk_usage(c["output_dir"]).free if Path(c["output_dir"]).exists() else shutil.disk_usage(c["project_root"]).free,
    }


def remaining(c: dict) -> dict:
    used = resource_usage(c)
    return {
        "calls": c["max_calls"] - used["attempt_reservations"],
        "retries": c["max_retries"] - used["retries"],
        "seconds": c["wall_seconds"] - used["active_seconds"],
        "bytes": c["max_bytes"] - used["new_bytes_tree_accounting"],
    }


def status(c: dict) -> dict:
    summary = delivery_summary(c)
    events = {}
    total_complete = 0
    for event_id, state in _states(c).items():
        if state is None:
            view = {"status": "not_started", "completed": 0, "total": 1485, "actual_calls": 0}
        else:
            view = core.status_view(state)
        total_complete += int(view.get("completed", 0))
        events[event_id] = view
    used = resource_usage(c)
    expected_total = len(c["enabled_events"]) * 1485
    complete = total_complete == expected_total and all(v.get("recorded_complete") for v in events.values())
    return {"batch_id": c["batch_id"], "scope": "CORE_NONLOCKBOX", "events": events,
            "logical_total": expected_total, "logical_complete_recorded": total_complete,
            "recorded_complete": complete, "identity": "stored event states; use --summarize for current evidence verification",
            "resources": used, "remaining": remaining(c), "target_manifest_sha256": summary["manifest_sha256"]}


def preflight_all(c: dict, *, write: bool = True) -> dict:
    summary = delivery_summary(c)
    results = {}
    rem = remaining(c)
    for event_id in c["enabled_events"]:
        event_c = _event_config_file(c, event_id, summary, rem)
        frame, env, version, actions = core.preflight(event_c)
        results[event_id] = {"runs": len(frame), "groups": frame.groupby(["event_id", "base_case_id"]).ngroups,
                             "unique_physics": frame.physics_hash.nunique(),
                             "dispositions": frame.disposition.value_counts().to_dict(),
                             "actions": actions.effective_action.value_counts().to_dict(),
                             "manifest_sha256": event_c["manifest_sha256"], "version_lock_sha256": version}
    result = {"status": "READY_TO_RUN", "energyplus_calls": 0, "manual_start_confirmed": False,
              "events": results, "target": {"runs": len(c['enabled_events']) * 1485, "groups": len(c['enabled_events']) * 297},
              "resources": resource_usage(c), "remaining": remaining(c),
              "environment": env, "lockboxes_excluded": sorted(LOCKBOX_EVENTS)}
    if write:
        _atomic(Path(c["delivery_dir"]) / "open6_runtime_dry_run.json", result)
    return result


def _concat(paths: list[Path], required_columns: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=required_columns)


def aggregate(c: dict, *, redraw: bool = False) -> dict:
    out = Path(c["output_dir"])
    combined = out / "combined"
    combined.mkdir(parents=True, exist_ok=True)
    summary = delivery_summary(c)
    verification = {}
    recorded_completed = {}
    rem = remaining(c)
    for event_id in c["enabled_events"]:
        ec = _event_config_file(c, event_id, summary, rem)
        state = core.load_state(ec)
        if state:
            core.reports(ec, state)
            _atomic(Path(ec["output_dir"]) / "state.json", state)
            verification[event_id] = state.get("last_verification", {"complete": False})
            recorded_completed[event_id] = int(core.status_view(state).get("completed", 0))
        else:
            verification[event_id] = {"complete": False, "status": "NOT STARTED", "issues": ["No state"]}
            recorded_completed[event_id] = 0
    active = tuple(c["enabled_events"])
    identity_columns = ["pair_id", "event_id", "base_case_id", "prototype_id", "floor_position"]
    energy = _concat([out / "events" / e / f"{e.lower()}_energy_service_metrics.csv" for e in active],
                     identity_columns + ["relative_delta", "net_delta_kwh"])
    thermal = _concat([out / "events" / e / f"{e.lower()}_thermal_recovery_metrics.csv" for e in active],
                      identity_columns + ["absolute_occupied_degree_hours_above_32c"])
    scores = _concat([out / "events" / e / f"{e.lower()}_stage_score_bounds.csv" for e in active],
                     identity_columns + ["component", "score_lower", "score_upper"])
    qc = _concat([out / "events" / e / "physics_qc.csv" for e in active], ["pair_id"])
    warnings = _concat([out / "events" / e / "warning_summary.csv" for e in active], ["run_id"])
    coverage = _concat([out / "events" / e / "coverage_and_pairing.csv" for e in active], ["pair_id"])
    core.csv_write(combined / "open6_design_event_energy_service.csv", energy)
    core.csv_write(combined / "open6_design_event_thermal_recovery.csv", thermal)
    core.csv_write(combined / "open6_stage_score_bounds.csv", scores)
    core.csv_write(combined / "open6_physics_qc.csv", qc)
    core.csv_write(combined / "open6_warning_summary.csv", warnings)
    core.csv_write(combined / "open6_coverage_and_pairing.csv", coverage)
    source_data = pd.DataFrame([
        {"file": "open6_design_event_energy_service.csv", "unit_of_observation": "design x parent event",
         "scope": "six non-lockbox complete-event prototype simulations", "role": "energy and cooling-service estimands"},
        {"file": "open6_design_event_thermal_recovery.csv", "unit_of_observation": "design x parent event",
         "scope": "synchronous 24 h outage and continuous recovery", "role": "thermal, censoring, and recovery estimands"},
        {"file": "open6_stage_score_bounds.csv", "unit_of_observation": "design x parent event x component",
         "scope": "N/C/O/R score bounds", "role": "BERI component source data; no interval midpoint"},
        {"file": "open6_physics_qc.csv", "unit_of_observation": "design-event QC assertion",
         "scope": "frozen production contracts", "role": "physical and pairing QC"},
        {"file": "open6_logical_physical_source_index.csv", "unit_of_observation": "logical solver input",
         "scope": "8,910 approved open-event items", "role": "cache, input, and evidence provenance"},
    ])
    core.csv_write(combined / "open6_source_data_index.csv", source_data)
    target = pd.read_csv(Path(c["delivery_dir"]) / "open6_target_manifest.csv")
    source_cols = ["event_id", "base_case_id", "prototype_id", "floor_position", "scenario_id", "run_id",
                   "physics_hash", "raw_output_schema_hash", "analysis_hash", "disposition", "source_batch",
                   "source_run_id", "source_result_dir"]
    core.csv_write(combined / "open6_logical_physical_source_index.csv", target[source_cols])
    event_summary = []
    for event_id in active:
        event_summary.append({"event_id": event_id, "runs_complete": recorded_completed[event_id],
                              "groups_in_energy": int(energy.event_id.eq(event_id).sum()) if "event_id" in energy else 0,
                              "groups_in_thermal": int(thermal.event_id.eq(event_id).sum()) if "event_id" in thermal else 0,
                              "scores": int(scores.event_id.eq(event_id).sum()) if "event_id" in scores else 0,
                              "current_evidence_complete": bool(verification[event_id].get("complete")),
                              "verification_status": verification[event_id].get("status", "")})
    event_summary = pd.DataFrame(event_summary)
    core.csv_write(combined / "open6_event_coverage_summary.csv", event_summary)
    expected_groups = len(active) * 297
    all_complete = (len(energy) == expected_groups and len(thermal) == expected_groups and len(scores) == expected_groups * 4
                    and event_summary.current_evidence_complete.all())
    dependencies = write_analysis_dependencies(
        combined / "open6_stage_score_bounds.csv", combined,
        combined / "open6_design_event_energy_service.csv",
        combined / "open6_design_event_thermal_recovery.csv")
    if all_complete:
        descriptive = (energy.groupby(["event_id", "prototype_id"], as_index=False)
                       .agg(designs=("base_case_id", "size"), median_relative_delta=("relative_delta", "median"),
                            median_net_delta_kwh=("net_delta_kwh", "median")))
        core.csv_write(combined / "open6_event_type_descriptive_summary.csv", descriptive)
    if redraw or all_complete:
        draw_static(c, energy, thermal, event_summary)
    result = {"status": "NUMERICAL_COMPLETE_OPEN6" if all_complete else "PARTIAL/BLOCKED",
              "scope_status": "CORE_NONLOCKBOX_COMPLETE" if all_complete else "CORE_NONLOCKBOX_INCOMPLETE",
              "energy_rows": len(energy), "thermal_rows": len(thermal), "score_rows": len(scores),
              "expected": {"energy": expected_groups, "thermal": expected_groups, "scores": expected_groups * 4},
              "event_verification": verification, "analysis_dependencies": dependencies,
              "resources": resource_usage(c), "generated_at": time.time()}
    _atomic(combined / "open6_aggregate_status.json", result)
    report = "# E8 Open6 numerical summary\n\n" + json.dumps(result, ensure_ascii=False, indent=2)
    (combined / "OPEN6_EXECUTION_REPORT.md").write_text(report, encoding="utf-8")
    return result


def draw_static(c: dict, energy: pd.DataFrame, thermal: pd.DataFrame, coverage: pd.DataFrame) -> dict:
    out = Path(c["output_dir"]) / "combined/static_diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    result = {"status": "NO_DATA", "figures": [], "solver_calls": 0, "frames": 0, "animations": 0}
    if energy.empty:
        _atomic(out / "static_figure_status.json", result)
        return result
    try:
        import matplotlib.pyplot as plt
        panels = 2 if (not thermal.empty and "absolute_occupied_degree_hours_above_32c" in thermal) else 1
        fig, axes = plt.subplots(1, panels, figsize=(11 if panels == 2 else 8, 4.2))
        axes = np.atleast_1d(axes)
        ax = axes[0]
        table = energy.groupby("event_id").relative_delta.median().reindex(c["enabled_events"])
        ax.bar(range(len(table)), 100 * table.to_numpy(float), color="#3B7EA1")
        ax.set_xticks(range(len(table)), [x.replace("HKO_LI_", "") for x in table.index])
        ax.set_ylabel("Median relative event electricity change (%)")
        ax.set_xlabel("Parent heatwave event")
        ax.axhline(0, color="black", lw=.8)
        if panels == 2:
            ax = axes[1]
            heat = thermal.groupby("event_id").absolute_occupied_degree_hours_above_32c.median().reindex(c["enabled_events"])
            ax.bar(range(len(heat)), heat.to_numpy(float), color="#C65D3A")
            ax.set_xticks(range(len(heat)), [x.replace("HKO_LI_", "") for x in heat.index])
            ax.set_ylabel("Median absolute outage heat (°C h above 32°C)")
            ax.set_xlabel("Parent heatwave event")
        fig.tight_layout()
        path = out / "open6_event_energy_diagnostic.png"
        fig.savefig(path, dpi=220); plt.close(fig)
        result = {"status": "COMPLETE", "figures": [str(path)], "solver_calls": 0, "frames": 0, "animations": 0,
                  "source": str(Path(c["output_dir"]) / "combined/open6_design_event_energy_service.csv")}
    except Exception as exc:
        result = {"status": "STATIC_FIGURE_FAILED_NUMERICS_RETAINED", "error": f"{type(exc).__name__}: {exc}",
                  "figures": [], "solver_calls": 0, "frames": 0, "animations": 0}
    _atomic(out / "static_figure_status.json", result)
    return result


def package_review(c: dict) -> tuple[Path, dict]:
    result = aggregate(c)
    out = Path(c["output_dir"])
    target = out / "E8_open6_local_review.zip"
    temp = target.with_name(target.name + ".writing")
    entries = {}
    for root in (out / "combined", Path(c["delivery_dir"])):
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".json", ".csv", ".md", ".png"} and p.stat().st_size < 25 * 1024**2:
                entries[str(p.relative_to(Path(c["project_root"])))] = p
    for rel in ("src/workflows/run_e8_local.py", "src/utils/e8_open6_execution.py", "src/utils/e8_open6_prepare.py",
                "src/utils/e8_open6_analysis.py", "src/utils/g4a_local_execution.py", "config/e8_core_nonlockbox.json"):
        p = Path(c["project_root"]) / rel
        if p.exists(): entries["code/" + rel] = p
    hashes = [{"path": name, "bytes": p.stat().st_size, "sha256": core.digest(p)} for name, p in sorted(entries.items())]
    core.csv_write(out / "combined/review_file_hashes.csv", pd.DataFrame(hashes))
    entries["review_file_hashes.csv"] = out / "combined/review_file_hashes.csv"
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, p in entries.items(): archive.write(p, name)
    with zipfile.ZipFile(temp) as archive:
        if archive.testzip() is not None: raise core.Blocked("Review archive CRC failure")
    os.replace(temp, target)
    target.with_name(target.name + ".sha256").write_text(core.digest(target) + "  " + target.name + "\n", encoding="utf-8")
    return target, result


def execute(c: dict, retry_failed: bool = False) -> int:
    summary = delivery_summary(c)
    out = Path(c["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    owner = out / "workorder_owner.json"
    with core.exclusive_lock(out / "workorder.lock"):
        identity = {"batch_id": c["batch_id"], "target_manifest_sha256": summary["manifest_sha256"],
                    "events": list(c["enabled_events"]), "config_path": c["config_path"]}
        if owner.exists() and _json(owner) != identity:
            raise core.Blocked("Open6 work order is already owned by a different identity")
        if not owner.exists(): _atomic(owner, identity)
        preflight_all(c)
        confirmations = _json(out / "manual_start_confirmations.json") if (out / "manual_start_confirmations.json").exists() else []
        confirmations.append({"at": time.time(), "manual_cli_start": True, "pid": os.getpid(),
                              "host": socket.gethostname(), "target_manifest_sha256": summary["manifest_sha256"],
                              "config_sha256": core.canonical(c), "events": list(c["enabled_events"]),
                              "retry_failed": retry_failed})
        _atomic(out / "manual_start_confirmations.json", confirmations)
        for event_id in c["enabled_events"]:
            rem = remaining(c)
            if rem["calls"] <= 0 or rem["seconds"] <= 0 or rem["bytes"] <= 0:
                raise core.ResourceStop(f"Open6 global resource exhausted before {event_id}: {rem}")
            event_c = _event_config_file(c, event_id, summary, rem)
            state = core.load_state(event_c)
            if state:
                current = core.verify_completion(event_c, state)
                if current["complete"]:
                    print(f"[{event_id}] verified complete; continuing", flush=True)
                    continue
            print(f"[{event_id}] starting/resuming; global remaining={rem}", flush=True)
            code = core.execute(event_c, retry_failed=retry_failed)
            state = core.load_state(event_c)
            verified = core.verify_completion(event_c, state)
            if not verified["complete"] and code == 4 and not retry_failed:
                print(f"[{event_id}] applying bounded technical retry pass", flush=True)
                rem = remaining(c)
                event_c = _event_config_file(c, event_id, summary, rem)
                code = core.execute(event_c, retry_failed=True)
                state = core.load_state(event_c); verified = core.verify_completion(event_c, state)
            if not verified["complete"]:
                aggregate(c)
                return code or 4
            _atomic(out / "workorder_progress.json", status(c))
        final = aggregate(c, redraw=True)
        package_review(c)
        return 0 if final["status"] == "NUMERICAL_COMPLETE_OPEN6" else 4


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="E8 Open6: no arguments starts/resumes the frozen six-event work order")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "config/e8_core_nonlockbox.json"))
    actions = parser.add_mutually_exclusive_group()
    for name in ("doctor", "dry-run", "execute", "status", "summarize", "package-review", "redraw"):
        actions.add_argument("--" + name, action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    explicit = any(getattr(args, x) for x in ("doctor", "dry_run", "execute", "status", "summarize", "package_review", "redraw"))
    if not explicit and not args.resume and not args.retry_failed:
        args.execute = args.resume = True
    if (args.resume or args.retry_failed) and not args.execute:
        parser.error("--resume/--retry-failed require --execute")
    if args.execute and not args.resume:
        parser.error("Use --execute --resume, or run the script with no arguments")
    try:
        c = load_config(args.config)
        if args.status:
            print(json.dumps(status(c), ensure_ascii=False, indent=2)); return 0
        if args.doctor or args.dry_run:
            result = preflight_all(c)
            print(json.dumps(result, ensure_ascii=False, indent=2)); return 0
        if args.execute:
            return execute(c, retry_failed=args.retry_failed)
        if args.redraw:
            result = aggregate(c, redraw=True); print(json.dumps(result, ensure_ascii=False, indent=2)); return 0
        if args.package_review:
            path, result = package_review(c); print(path); return 0 if result["status"] == "NUMERICAL_COMPLETE_OPEN6" else 4
        result = aggregate(c); print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "NUMERICAL_COMPLETE_OPEN6" else 4
    except core.ResourceStop as exc:
        print(f"RESOURCE STOP: {exc}", file=__import__('sys').stderr); return 2
    except (core.Blocked, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"BLOCKED: {exc}", file=__import__('sys').stderr); return 3
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=__import__('sys').stderr); return 4


if __name__ == "__main__":
    raise SystemExit(main())
