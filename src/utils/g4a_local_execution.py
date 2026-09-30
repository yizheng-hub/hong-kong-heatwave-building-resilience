"""G4-a local, manual-only scheduler. No remote service or model API is used.

state.json is authoritative; CSV/JSON reports are disposable projections.
EnergyPlus is only launched by execute(): direct manual start or explicit execute/resume.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
import zipfile

import numpy as np
import pandas as pd
import psutil

from utils import e8_g3_postprocess as science
from utils.e8_g3_runner import CORE_SCENARIO_IDS, load_production, parse_native_output

MANIFEST_HASH = "7d74083df70130b0f4f86d447259b3b56c5520cfc24f9be2f608f99eba6be1a8"
VERSION = "g4a-local-v4-scoped-workorder"
COMPATIBLE_CODE_ONLY_LOCKS = {
    "7b7806da62a81e6cf787377f31b50aa413b20ff86a8cfdbf3012e9227efbc603",
    "5ff561fe63349d65781cd74c94f82d5ac1633dfac1a2df9cb29f2df6f17158fc",
    "2e5f4cfb57fdbd840d2039603ebabbe6914434e09dbcedd8e8cd75c0c75dc789",
    "ea83768dee40e15031d3e9388cf5135a28e1e28c37254929f9312a2b8b61b793",
    "217aafe25a25cfff0d271465aaa946ed0213991255c1a59629b6bcd28428457d",
    "c973ab004808d3c5359c95e3559b05c38346e53eb6e4c6b18ef182654062a8c5",
    "b6d4178c0140dfa3acd3934d56425fae636e8a1a6862250e1eeaf65fcfcfb56e",
}
RESOURCE_KEYS = {"concurrency", "max_calls", "max_retries", "wall_seconds", "max_bytes",
                 "min_free_bytes", "grace_seconds", "run_timeout_seconds", "progress_seconds", "max_worker_rss_bytes"}
COMPLETE = "metrics_complete"
GOOD = {"qc_passed", COMPLETE}
RAW_FILES = ("eplusout.csv", "eplusout.err", "eplusout.end", "eplusout.rdd", "eplusout.mdd")


class Blocked(RuntimeError):
    pass


class ResourceStop(RuntimeError):
    pass


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A unique name prevents a stopped process, scanner, or concurrent read-only
    # command from sharing the next writer's staging path on Windows.
    temp = path.with_name(f"{path.name}.writing.{os.getpid()}.{uuid.uuid4().hex}")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=str, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    # Windows readers, antivirus and indexers can deny destination replacement for
    # longer than the former two-second window. The staged file is already durable;
    # retry only this commit with bounded backoff, never the scientific work.
    deadline = time.monotonic() + 30.0
    delay = .02
    while True:
        try:
            os.replace(temp, path)
            return
        except PermissionError as exc:
            if time.monotonic() >= deadline:
                raise PermissionError(
                    exc.errno,
                    f"Windows denied atomic checkpoint replacement for 30 s; "
                    f"durable staged state retained at {temp}: {exc}",
                    str(path),
                ) from exc
            time.sleep(delay)
            delay = min(.5, delay * 1.5)


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise Blocked(f"Unreadable state/evidence {path}: {exc}; do not reset/delete it") from exc


def csv_write(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".writing")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, path)


def tree_bytes(path):
    total = 0
    for p in Path(path).rglob("*") if Path(path).exists() else ():
        try:
            if p.is_file():
                total += p.stat().st_size
        except FileNotFoundError:
            # EnergyPlus creates and removes transient files (for example eplusout.dbg)
            # while workers are live. A file that vanished during this scan consumes no
            # current storage and must not abort or terminate the owned solver process.
            continue
    return total


def unique_tree_bytes(path):
    seen, total = set(), 0
    for p in Path(path).rglob("*") if Path(path).exists() else ():
        try:
            if not p.is_file():
                continue
            stat = p.stat()
        except FileNotFoundError:
            # See tree_bytes: tolerate only a verified disappearance, not permission or
            # other I/O errors that could hide evidence or defeat the storage ceiling.
            continue
        identity = (stat.st_dev, stat.st_ino) if stat.st_ino else (str(p.resolve()),)
        if identity in seen:
            continue
        seen.add(identity); total += stat.st_size
    return total


def storage_bytes(c):
    return unique_tree_bytes(c["output_dir"]) if c.get("count_hardlinks_once") else tree_bytes(c["output_dir"])


def manifest_hash(c):
    """Return the frozen manifest identity for this scoped work order."""
    return str(c.get("manifest_sha256", MANIFEST_HASH))


def expected_count(c, key, default):
    return int(c.get("scope", {}).get(key, default))


def load_config(path):
    path = Path(path).resolve()
    c = read_json(path)
    root = (path.parent / c["project_root"]).resolve()
    c["project_root"] = str(root)
    c["config_path"] = str(path)
    for key in ("output_dir", "development_dir", "manifest", "cache_root", "energyplus", "version_lock"):
        c[key] = str((root / c[key]).resolve())
    if c.get("reuse_complete_state"):
        c["reuse_complete_state"] = str((root / c["reuse_complete_state"]).resolve())
    if not c.get("paper_only"):
        raise Blocked("Only paper-only numeric output is implemented")
    positive_resources = RESOURCE_KEYS - {"max_retries"}
    if (not 1 <= c["concurrency"] <= 4 or any(c[k] <= 0 for k in positive_resources)
            or c["max_retries"] < 0
            or any(not isinstance(c[k], int) or isinstance(c[k], bool) for k in ("concurrency", "max_calls", "max_retries", "max_bytes", "min_free_bytes"))):
        raise Blocked("Invalid resource settings; concurrency must be 1..4")
    out = Path(c["output_dir"])
    approved_roots = [
        (root / value).resolve() for value in c.get("approved_output_roots", ["intermediate/simulation_outputs/upstream"])
    ]
    if (out == root or out == Path(c["cache_root"])
            or not any(out != approved and out.is_relative_to(approved) for approved in approved_roots)):
        raise Blocked("Output must be a dedicated directory below an approved output root")
    c["approved_output_roots"] = [str(value) for value in approved_roots]
    scope = c.setdefault("scope", {})
    if not re.fullmatch(r"[0-9a-f]{64}", str(c.get("manifest_sha256", ""))):
        raise Blocked("manifest_sha256 must be a frozen SHA256")
    scope.setdefault("expected_runs", 280)
    scope.setdefault("expected_groups", 56)
    scope.setdefault("events", ["HKO_LI_0049"])
    scope.setdefault("prehistory_days", 14)
    scope.setdefault("expected_dispositions", {"new": 265, "reextract": 15})
    scope.setdefault("label", "G4-a")
    scope.setdefault("report_prefix", "g4a")
    scope.setdefault("review_archive", "G4a_local_review.zip")
    scope.setdefault("confirmation_scope", "G4a only")
    scope.setdefault("owner_registry", "intermediate/simulation_outputs/upstream/G4a_manifest_owner.json")
    scope.setdefault("initial_ceiling", {"max_calls": 280, "max_retries": 15,
                                          "wall_seconds": 3600, "max_bytes": 5 * 1024**3,
                                          "min_free_bytes": 10 * 1024**3})
    return c


def environment(c):
    dependencies = {}
    for name in ("numpy", "pandas", "pyarrow", "psutil", "scipy", "pytest"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise Blocked(f"Missing {name}; install only this package in hk-heatwave-repro")
    if Path(sys.prefix).name != c["expected_environment"] or not sys.version.startswith(c["expected_python"] + "."):
        raise Blocked(f"Wrong interpreter: {sys.executable}; activate {c['expected_environment']}")
    engine = Path(c["energyplus"])
    if not engine.is_file():
        raise Blocked(f"EnergyPlus missing: {engine}")
    return dict(executable=sys.executable, python=sys.version, prefix=sys.prefix,
                dependencies=dependencies, energyplus=str(engine), engine_sha256=digest(engine),
                engine_version_from_metadata=read_json(Path(c["project_root"]) / "software/energyplus/energyplus_release.json"),
                free_disk_bytes=shutil.disk_usage(c["project_root"]).free)


def verify_versions(c):
    lock = read_json(c["version_lock"])
    for rel, sha in lock["files"].items():
        if digest(Path(c["project_root"]) / rel) != sha:
            raise Blocked(f"Code/contract changed: {rel}; requires reviewed version update")
    production = load_production(Path(c["project_root"]), code_root=Path(c.get("code_root", c["project_root"])))
    parameters = production.prototype_parameter_table().to_csv(index=False)
    if hashlib.sha256(parameters.encode()).hexdigest() != lock["prototype_parameter_table_sha256"]:
        raise Blocked("Parameter table used for COP/nominal-capacity QC changed")
    return digest(c["version_lock"])


def raw_complete(raw):
    raw = Path(raw)
    if any(not (raw / name).is_file() for name in RAW_FILES):
        return False
    text = (raw / "eplusout.end").read_text(errors="replace")
    err = (raw / "eplusout.err").read_text(errors="replace")
    return "Completed Successfully" in text and not re.search(r"\*\*\s+(Severe|Fatal)\s+\*\*", err)


def verify_cache(c, row):
    source = science.result_dir(Path(c["cache_root"]), row["cached_g3_run_id"])
    meta = read_json(source / "run_result.json")
    if meta.get("status") != "passed" or not raw_complete(source / "raw"):
        raise Blocked(f"G3 cache lacks successful complete output: {row['run_id']}")
    hashes = ast.literal_eval(row["file_byte_sha256"])
    for key, name in (("idf", "model.idf"), ("epw", "weather.epw"), ("schedule", "schedules.csv"), ("event_schedule", "event_schedules.parquet")):
        if digest(source / "inputs" / name) != hashes[key]:
            raise Blocked(f"G3 cache input hash mismatch: {row['run_id']} {key}")
    evidence = pd.read_csv(Path(c["cache_root"]) / "G3_metrics_v2_closure/raw_hash_reconciliation.csv")
    match = evidence.loc[evidence.run_id.eq(row["cached_g3_run_id"])]
    if len(match) != 1 or digest(source / "raw/eplusout.csv") != match.iloc[0].computed_raw_sha256:
        raise Blocked(f"G3 raw cache hash mismatch: {row['run_id']}")
    return source


def preflight(c):
    env = environment(c)
    version = verify_versions(c)
    frozen = manifest_hash(c)
    if digest(c["manifest"]) != frozen:
        raise Blocked("Manifest SHA256 mismatch")
    frame = pd.read_csv(c["manifest"]).fillna("")
    expected_runs = expected_count(c, "expected_runs", 280)
    if len(frame) != expected_runs or frame.run_id.nunique() != expected_runs:
        raise Blocked(f"Expected {expected_runs} unique logical run IDs")
    expected_physics = expected_count(c, "expected_unique_physics", expected_runs)
    if frame.physics_hash.nunique() != expected_physics:
        raise Blocked(f"Expected {expected_physics} unique physical inputs")
    if set(frame.event_id) != set(c["scope"]["events"]) or set(frame.prehistory_days) != {int(c["scope"]["prehistory_days"])}:
        raise Blocked("Wrong event or prehistory")
    expected_groups = expected_count(c, "expected_groups", 56)
    actual_groups = frame.groupby(["event_id", "base_case_id", "prototype_id", "floor_position"]).ngroups
    if actual_groups != expected_groups:
        raise Blocked(f"Expected {expected_groups} groups, found {actual_groups}")
    if not c.get("scope", {}).get("solver_only"):
        for _, g in frame.groupby(["event_id", "base_case_id", "prototype_id", "floor_position"]):
            if len(g) != 5 or set(g.scenario_id) != set(CORE_SCENARIO_IDS):
                raise Blocked("Incomplete or duplicate five-scenario group")
    expected_dispositions = {str(k): int(v) for k, v in c["scope"]["expected_dispositions"].items()}
    if frame.disposition.value_counts().to_dict() != expected_dispositions:
        raise Blocked("Frozen disposition counts changed")
    checks = []
    for row in frame.to_dict("records"):
        hashes = ast.literal_eval(row["file_byte_sha256"])
        if hashes["engine"] != env["engine_sha256"]:
            raise Blocked("Engine differs from frozen compiled inputs")
        for key in ("idf", "epw", "schedule", "event_schedule"):
            if digest(row[key + "_path"]) != hashes[key]:
                raise Blocked(f"Input byte hash mismatch: {row['run_id']} {key}")
        # Weather parquet is parser metadata and is separately pinned in the delivery lock.
        if not Path(row["weather_path"]).is_file():
            raise Blocked(f"Missing parser weather: {row['run_id']}")
        if row.get("weather_parser_sha256") and digest(row["weather_path"]) != row["weather_parser_sha256"]:
            raise Blocked(f"Parser weather hash mismatch: {row['run_id']}")
        input_manifest = Path(row["compiled_dir"]) / "input_manifest.json"
        if row.get("input_manifest_sha256") and digest(input_manifest) != row["input_manifest_sha256"]:
            raise Blocked(f"Compiled input manifest hash mismatch: {row['run_id']}")
        if row["disposition"] == "reextract":
            verify_cache(c, row)
        elif row["disposition"] == "reuse_complete":
            source = Path(row["source_result_dir"])
            if not source.is_dir() or not raw_complete(source / "raw"):
                raise Blocked(f"Completed external cache is missing: {row['run_id']}")
        checks.append(dict(run_id=row["run_id"], action=row["disposition"], effective_action=row["disposition"],
                           pair_id=pair_key(row), scenario_id=row["scenario_id"], input_qc="passed", physics_hash=row["physics_hash"]))
    return frame, env, version, pd.DataFrame(checks)


@contextmanager
def exclusive_lock(path):
    """OS lock is released by process exit; never delete a live/stale lock file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("a+b")
    f.seek(0, 2)
    if f.tell() == 0:
        f.write(b"0"); f.flush()
    f.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        f.close()
        raise Blocked("Batch is already active; use --status, do not delete locks") from exc
    try:
        yield
    finally:
        f.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def process_alive(pid, created, host):
    if host != socket.gethostname():
        raise Blocked("Process owner is another host; cannot safely determine liveness")
    if not pid or not created:
        raise Blocked("Uncommitted child identity: inspect this attempt before any retry")
    try:
        return abs(psutil.Process(int(pid)).create_time() - float(created)) < .02
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied as exc:
        raise Blocked("Cannot inspect old process identity") from exc


def verify_files(hashes):
    return bool(hashes) and all(Path(p).is_file() and digest(p) == sha for p, sha in hashes.items())


def file_hashes(folder):
    return {str(p): digest(p) for p in sorted(Path(folder).rglob("*")) if p.is_file()}


def load_state(c):
    p = Path(c["output_dir"]) / "state.json"
    if not p.exists():
        return None
    state = read_json(p)
    # Recover launch facts for read-only status too; never disguise a committed receipt as zero calls.
    for journal in (Path(c['output_dir']) / 'attempts').glob('*/*/launch_receipt.json'):
        receipt = read_json(journal)
        matches = [a for a in state['attempts'] if a['attempt_id'] == receipt['attempt_id'] and a['run_id'] == receipt['run_id']]
        if len(matches) != 1:
            raise Blocked(f'Unlinked launch receipt: {journal}')
        if not matches[0].get('launched'):
            matches[0].update(receipt)
    return state


def initial_state(c, frame, version):
    state = dict(schema=VERSION, batch_id=c["batch_id"], manifest_sha256=manifest_hash(c),
                version_lock_sha256=version, config_history=[], confirmations=[], attempts=[],
                spent_seconds=0.0, active_session=None, pilot_passed=False, stop_reason="not_started",
                rows={r["run_id"]: {"status": "pending", "attempts": [], "error": "", "row": r}
                      for r in frame.to_dict("records")}, groups={})
    seed_complete_cache(c, state)
    return state


def seed_complete_cache(c, state):
    """Seed explicitly mapped, fully verified prior-batch evidence without copying it."""
    source_path = c.get("reuse_complete_state")
    if not source_path:
        return
    source = read_json(source_path)
    wanted = {rid for rid, item in state["rows"].items()
              if item["row"].get("disposition") == "reuse_complete"}
    if not wanted:
        return
    if not wanted <= set(source.get("rows", {})):
        raise Blocked("reuse_complete mapping is not covered by its source state")
    for rid in wanted:
        old = source["rows"][rid]
        if old.get("status") != COMPLETE or canonical(old["row"].get("file_byte_sha256")) != canonical(state["rows"][rid]["row"].get("file_byte_sha256")):
            raise Blocked(f"reuse_complete source is not compatible: {rid}")
        state["rows"][rid].update(
            status=COMPLETE, raw_dir=old.get("raw_dir", ""),
            raw_hashes=dict(old.get("raw_hashes", {})),
            parsed_hashes=dict(old.get("parsed_hashes", {})),
            parser_version=old.get("parser_version", ""), qc=old.get("qc", ""),
            result_dir=str(Path(state["rows"][rid]["row"]["source_result_dir"])),
            error="", source_batch=source.get("batch_id", ""))
    grouped = {}
    for item in state["rows"].values():
        grouped.setdefault(pair_key(item["row"]), []).append(item)
    for key, items in grouped.items():
        if not all(i["row"].get("disposition") == "reuse_complete" for i in items):
            continue
        old = source.get("groups", {}).get(key)
        if not old or old.get("status") != COMPLETE or not verify_files(old.get("hashes", {})):
            raise Blocked(f"reuse_complete group evidence is missing: {key}")
        seeded = dict(old)
        seeded["dependencies"] = group_dependencies(c, items)
        state["groups"][key] = seeded


def validate_state(state, frame):
    expected = {r["run_id"]: r for r in frame.to_dict("records")}
    if set(state["rows"]) != set(expected):
        raise Blocked("State run coverage differs from the frozen manifest")
    for run_id, item in state["rows"].items():
        if canonical(item["row"]) != canonical(expected[run_id]):
            raise Blocked(f"State input identity differs: {run_id}")
    ids = [a["attempt_id"] for a in state["attempts"]]
    if len(ids) != len(set(ids)) or not math.isfinite(state["spent_seconds"]) or state["spent_seconds"] < 0:
        raise Blocked("Invalid cumulative state/attempt accounting")
    for run_id, item in state["rows"].items():
        actual = [a["attempt_id"] for a in state["attempts"] if a["run_id"] == run_id]
        if item["attempts"] != actual:
            raise Blocked(f"Attempt linkage differs: {run_id}")


def reconcile_execution_identity(c, state, version):
    """Allow only a reviewed code-only lock transition; never a physics/analysis change."""
    if state["manifest_sha256"] != manifest_hash(c) or state["batch_id"] != c["batch_id"]:
        raise Blocked("Batch/physical identity mismatch; refusing silent mixed results")
    if state["version_lock_sha256"] == version:
        return
    previous = state['version_lock_sha256']
    if previous not in COMPATIBLE_CODE_ONLY_LOCKS:
        raise Blocked("Unreviewed code/analysis version mismatch; refusing silent mixed results")
    state.setdefault('code_version_migrations', []).append(dict(
        at=time.time(), from_sha256=previous, to_sha256=version,
        scope='reviewed execution-control fixes only; physics and analysis contracts unchanged',
        physics_changed=False, analysis_changed=False))
    state['version_lock_sha256'] = version


def reserve_attempt(state, c, run_id):
    calls = len(state["attempts"])
    repeats = sum(a["is_retry"] for a in state["attempts"])
    item = state["rows"][run_id]
    retry = bool(item["attempts"])
    if calls >= c["max_calls"] or (retry and repeats >= c["max_retries"]):
        raise ResourceStop("call_or_retry_limit")
    if item["row"]["disposition"] != "new":
        raise Blocked("A re-extraction cache cannot become a new solver call")
    a = dict(attempt_id=f"attempt_{calls + 1:04d}", run_id=run_id, is_retry=retry,
             reservation="reserved_pending_launch", launched=False, pid=None, created=None, host=socket.gethostname(),
             started=time.time(), finished=None, returncode=None)
    a.update({k: item["row"].get(k, "") for k in ("event_id", "base_case_id", "prototype_id", "floor_position", "scenario_id", "physics_hash", "raw_output_schema_hash")})
    a["analysis_version"] = c["analysis_version"]
    state["attempts"].append(a)
    item["attempts"].append(a["attempt_id"])
    item["status"] = "running"
    return a


class Session:
    def __init__(self, c, state):
        self.c, self.state = c, state
        self.output = Path(c["output_dir"])
        self.last_tick = time.monotonic()
        self.interrupts = 0
        self.interrupted_at = None
        self.live = {}
        self.last_storage_scan = 0.0
        self.cached_storage_bytes = None
        self.last_checkpoint = 0.0

    def save(self, force=True):
        now = time.monotonic()
        self.state["spent_seconds"] += max(0, now - self.last_tick)
        self.last_tick = now
        if self.state["active_session"] is not None:
            self.state["active_session"]["heartbeat"] = time.time()
        if force or now - self.last_checkpoint >= float(self.c.get("progress_seconds", 10)):
            atomic_json(self.output / "state.json", self.state)
            self.last_checkpoint = now

    def check_resources(self, headroom_seconds=0):
        self.save(force=False)
        if self.state["spent_seconds"] + headroom_seconds >= self.c["wall_seconds"]:
            raise ResourceStop("active_wall_limit")
        now = time.monotonic()
        scan_seconds = float(self.c.get("storage_scan_seconds", 0))
        if self.cached_storage_bytes is None or now - self.last_storage_scan >= scan_seconds:
            self.cached_storage_bytes = storage_bytes(self.c)
            self.last_storage_scan = now
        if self.cached_storage_bytes >= self.c["max_bytes"]:
            raise ResourceStop("storage_limit")
        if shutil.disk_usage(self.output).free < self.c["min_free_bytes"]:
            raise ResourceStop("free_disk_limit")

    def signal(self, *_):
        self.interrupts += 1
        if self.interrupted_at is None:
            self.interrupted_at = time.monotonic()
        print("Interrupt: stop dispatch; second Ctrl+C terminates owned children.", flush=True)

    def kill_owned(self):
        for run_id, (proc, attempt, handles) in list(self.live.items()):
            identity_verified = attempt.get('created') is not None and attempt.get('identity_status') != 'unconfirmed'
            if proc.poll() is None and identity_verified and process_alive(attempt["pid"], attempt["created"], attempt["host"]):
                parent = psutil.Process(attempt["pid"])
                try:
                    children = parent.children(recursive=True)
                except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
                    children = []
                    attempt['tree_cleanup_note'] = f'Cannot enumerate descendants: {exc}; owned Popen handle retained'
                for child in reversed(children):
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                try:
                    parent.terminate()
                except psutil.NoSuchProcess:
                    pass
                _, alive = psutil.wait_procs(children + [parent], timeout=3)
                for child in alive:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
            # Windows Popen retains the owned process handle even if PID identity lookup failed.
            # Do not enumerate or signal unknown PID descendants in that case.
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
            for handle in handles:
                handle.close()
            attempt.update(finished=time.time(), returncode=proc.returncode, reservation="interrupted", identity_status='exited_owned_handle')
            self.state["rows"][run_id].update(status="interrupted", error="Owned process terminated; attempt retained")
            del self.live[run_id]
        self.save()


def restore_state(c, state):
    """No solver. Validate evidence and reject live or unidentifiable orphans."""
    for journal in (Path(c['output_dir']) / 'attempts').glob('*/*/launch_receipt.json'):
        receipt = read_json(journal)
        matches = [a for a in state['attempts'] if a['attempt_id'] == receipt['attempt_id'] and a['run_id'] == receipt['run_id']]
        if len(matches) != 1:
            raise Blocked(f'Unlinked actual launch evidence: {journal}')
        attempt = matches[0]
        if not attempt.get('launched'):
            attempt.update(receipt)
        if receipt.get('registration_failed'):
            raw = Path(receipt.get('raw_dir', ''))
            safely_exited = (receipt.get('identity_status') == 'exited_owned_handle'
                             and receipt.get('finished') is not None
                             and receipt.get('returncode') is not None)
            if safely_exited:
                attempt['registration_failure_reconciled'] = True
                attempt['registration_failure_receipt'] = str(journal)
                attempt['reservation'] = 'interrupted_registration_reconciled'
                if raw_complete(raw):
                    state['rows'][attempt['run_id']].update(
                        status='raw_complete', raw_dir=str(raw), raw_hashes=file_hashes(raw),
                        error='Registration commit failed after launch; complete raw output recovered')
                else:
                    state['rows'][attempt['run_id']].update(
                        status='interrupted',
                        error='Registration commit failed after actual launch; exited worker retained as a technical retry')
            else:
                raise Blocked(f'Launch registration failed; actual call retained, inspect {journal}')
    for run_id, item in state["rows"].items():
        if item["status"] in {"running", "interrupted", "retryable_failed"} and item["attempts"]:
            attempt = next(a for a in state["attempts"] if a["attempt_id"] == item["attempts"][-1])
            if attempt["reservation"] == "launch_failed":
                continue
            owned_exit = attempt.get('identity_status') == 'exited_owned_handle' and attempt.get('returncode') is not None
            if not owned_exit and process_alive(attempt["pid"], attempt["created"], attempt["host"]):
                raise Blocked(f"Old worker still alive: {run_id}; wait, do not duplicate")
            raw = Path(attempt["raw_dir"])
            if raw_complete(raw):
                item.update(status="raw_complete", raw_dir=str(raw), raw_hashes=file_hashes(raw))
                attempt["reservation"] = "recovered_complete_after_crash"
            elif item["status"] != "retryable_failed":
                item.update(status="interrupted", error="Prior process ended with incomplete raw output")
                attempt["reservation"] = "interrupted"
        if item["status"] in GOOD | {"parsed", "raw_complete", "parse_failed"}:
            if not verify_files(item.get("raw_hashes", {})):
                item.update(status="blocked", error="Raw evidence changed/missing; not automatically re-solved")
            elif item["status"] in GOOD | {"parsed"} and not verify_files(item.get("parsed_hashes", {})):
                item.update(status="raw_complete", error="Parsed evidence missing: re-extract raw only")
            elif item["status"] == "parsed":
                item.update(status="raw_complete", error="Parsed checkpoint preceded QC commit: re-extract/QC only")
        if item['status'] not in GOOD and pair_key(item['row']) in state['groups']:
            state['groups'][pair_key(item['row'])]['status'] = 'needs_dependency_verification'
    # A crashed active session has an uncertain tail. Charge elapsed wall up to its old cap.
    previous = state.get("active_session")
    if previous:
        gap = max(0, time.time() - previous["heartbeat"])
        cap = previous["wall_limit"]
        charge = min(gap, max(0, cap - state["spent_seconds"]))
        state["spent_seconds"] += charge
        state.setdefault("crash_time_reconciliation", []).append(dict(uncertain_tail_seconds=charge, rule="conservative_cap_no_reset"))
        state["active_session"] = None


def stage_inputs(c, row, target):
    target.mkdir(parents=True, exist_ok=True)
    declared = ast.literal_eval(row["file_byte_sha256"])
    names = {"model.idf": "idf", "weather.epw": "epw", "schedules.csv": "schedule", "event_schedules.parquet": "event_schedule"}
    for name in ("model.idf", "weather.epw", "schedules.csv", "event_schedules.parquet", "weather.parquet", "input_manifest.json"):
        source = Path(row["compiled_dir"]) / name
        dest = target / name
        if name in names and digest(source) != declared[names[name]]:
            raise Blocked(f"Input changed after preflight: {source}")
        if dest.exists() and digest(dest) != digest(source):
            raise Blocked(f"Staged input mismatch: {dest}")
        if not dest.exists():
            if c.get("hardlink_inputs"):
                try:
                    os.link(source, dest)
                except OSError:
                    shutil.copy2(source, dest)
            else:
                shutil.copy2(source, dest)
        if digest(dest) != digest(source):
            raise Blocked(f"Staged input copy failed integrity: {dest}")


def parse_run(c, row, item, production, checkpoint=None):
    work = Path(c["output_dir"])
    dest = science.result_dir(work, row["run_id"])
    stage_inputs(c, row, dest / "inputs")
    raw_source = Path(item["raw_dir"])
    raw = dest / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    retained = set(c.get("retained_raw_files", []))
    for source in raw_source.iterdir():
        if source.is_file() and source.resolve() != (raw / source.name).resolve():
            if retained and source.name not in retained:
                continue
            # Hardlinks avoid duplicating this batch's attempt bytes; copy is a safe fallback.
            target = raw / source.name
            if not target.exists():
                try:
                    if row["disposition"] == "new":
                        os.link(source, target)
                    else:
                        shutil.copy2(source, target)
                except OSError:
                    shutil.copy2(source, target)
            elif digest(source) != digest(target):
                raise Blocked("Existing canonical raw output differs; retain both and inspect")
    if not raw_complete(raw):
        raise Blocked("Raw completion/QC marker missing")
    weather = pd.read_parquet(dest / "inputs/weather.parquet")
    weather["timestamp_hkt"] = pd.to_datetime(weather.timestamp_hkt, utc=True).dt.tz_convert("Asia/Hong_Kong")
    cop = float(production.prototype_parameter_table().set_index("prototype_id").loc[row["prototype_id"], "baseline_ac_cop"])
    building, zones, audit = parse_native_output(raw / "eplusout.csv", weather,
        dest / "inputs/event_schedules.parquet", json.loads(row["zone_areas_json"]),
        set(json.loads(row["conditioned_zones_json"])), cop)
    if not np.isfinite(zones[["temperature_c", "relative_humidity_pct", "cooling_total_j", "occupants"]].to_numpy()).all():
        raise Blocked("Non-finite required zone outputs")
    if not np.isfinite(building.select_dtypes(include="number").to_numpy()).all():
        raise Blocked("Non-finite required building energy/state outputs")
    audit["timestamp_semantics"] = "TimeStep interval means with end labels; adjacent-mean interpolation is a diagnostic model"
    parsed = dest / "parsed"
    parsed.mkdir(parents=True, exist_ok=True)
    for frame, name in ((building, "native_timestep_building.parquet"), (zones, "native_timestep_zones.parquet")):
        temp = parsed / (name + ".writing")
        frame.to_parquet(temp, index=False)
        os.replace(temp, parsed / name)
    atomic_json(parsed / "parser_audit.json", audit)
    raw_evidence = file_hashes(raw) if retained else {**file_hashes(raw_source), **file_hashes(raw)}
    item.update(status="parsed", raw_hashes=raw_evidence, parsed_hashes=file_hashes(parsed))
    if checkpoint:
        checkpoint()
    item.update(status="qc_passed", error="", raw_hashes=raw_evidence, parsed_hashes=file_hashes(parsed),
                parser_version=VERSION, qc="exact_parser_and_finite_fields_passed", raw_dir=str(raw),
                result_dir=str(dest))
    if retained and row.get("disposition") == "new" and c.get("compact_attempt_after_qc"):
        attempt = raw_source.parent
        for child in (attempt / "raw", attempt / "inputs"):
            if child.exists():
                shutil.rmtree(child)


def pair_key(row):
    return "|".join(str(row[k]) for k in ("event_id", "base_case_id", "prototype_id", "floor_position"))


def file_issues(hashes, label):
    if not hashes:
        return [f'{label}: missing evidence manifest']
    return [f'{label}: {p}: missing or SHA256 changed' for p, sha in hashes.items()
            if not Path(p).is_file() or digest(p) != sha]


def run_evidence_issues(c, item):
    rid = item['row']['run_id']
    issues = []
    root = Path(item.get('result_dir') or science.result_dir(Path(c['output_dir']), rid))
    required = {'raw_hashes': [root / 'raw' / n for n in RAW_FILES],
                'parsed_hashes': [root / 'parsed' / n for n in
                    ('native_timestep_building.parquet', 'native_timestep_zones.parquet', 'parser_audit.json')]}
    for field, paths in required.items():
        hashes = item.get(field, {})
        issues.extend(file_issues(hashes, rid + ':' + field))
        for path in paths:
            if str(path) not in hashes:
                issues.append(f'{rid}: required evidence not registered: {path}')
    if 'file_byte_sha256' in item['row']:
        declared = ast.literal_eval(item['row']['file_byte_sha256'])
        for key, name in [('idf', 'model.idf'), ('epw', 'weather.epw'),
                          ('schedule', 'schedules.csv'), ('event_schedule', 'event_schedules.parquet')]:
            path = root / 'inputs' / name
            if not path.is_file() or digest(path) != declared[key]:
                issues.append(f'{rid}: retained physical input missing/changed: {path}')
    if not raw_complete(root / 'raw'):
        issues.append(f'{rid}: canonical raw completion/QC evidence missing')
    return issues


def group_dependencies(c, items):
    return dict(analysis_version=c['analysis_version'], members={i['row']['run_id']: {
        'physics_hash': i['row']['physics_hash'], 'output_schema_hash': i['row']['raw_output_schema_hash'],
        'raw_hashes': dict(i.get('raw_hashes', {})), 'parsed_hashes': dict(i.get('parsed_hashes', {}))}
        for i in sorted(items, key=lambda x: x['row']['run_id'])})


def verify_completion(c, state):
    """Current evidence, not stored success flags. No repair or solver side effects."""
    issues, valid_groups = [], []
    if not state:
        return dict(complete=False, status='NOT STARTED', checked_at=time.time(), issues=['No execution state'], valid_groups=[])
    try:
        version = verify_versions(c)
        if digest(c['manifest']) != manifest_hash(c) or state['version_lock_sha256'] != version:
            raise Blocked('Manifest or execution code version mismatch')
        validate_state(state, pd.read_csv(c['manifest']).fillna(''))
    except Exception as exc:
        issues.append(f'Batch identity: {exc}')
    if c.get('scope', {}).get('solver_only'):
        for rid, item in state['rows'].items():
            errors = run_evidence_issues(c, item)
            if item['status'] != COMPLETE:
                errors.append(f'{rid}: recorded stage {item["status"]}, not solver evidence complete')
            issues.extend(errors)
        expected_runs = expected_count(c, 'expected_runs', len(state['rows']))
        if len(state['rows']) != expected_runs:
            issues.append(f'Required solver-only coverage is {expected_runs} runs')
        complete = not issues
        return dict(complete=complete, status='COMPLETE' if complete else 'PARTIAL / BLOCKED',
                    checked_at=time.time(), issues=issues, valid_groups=[])
    grouped = {}
    run_errors = {}
    for rid, item in state['rows'].items():
        errors = run_evidence_issues(c, item)
        if item['status'] != COMPLETE:
            errors.append(f'{rid}: recorded stage {item["status"]}, not metrics complete')
        run_errors[rid] = errors
        issues.extend(errors)
        grouped.setdefault(pair_key(item['row']), []).append(item)
    for key, items in grouped.items():
        group = state['groups'].get(key, {})
        errors = file_issues(group.get('hashes', {}), key)
        if len(items) != 5 or {i['row']['scenario_id'] for i in items} != set(CORE_SCENARIO_IDS):
            errors.append(f'{key}: incomplete five-scenario identity')
        if group.get('status') != COMPLETE or group.get('dependencies') != group_dependencies(c, items):
            errors.append(f'{key}: group dependency/status mismatch')
        if not {'energy.csv', 'thermal.csv', 'scores.csv', 'qc.csv'} <= {Path(p).name for p in group.get('hashes', {})}:
            errors.append(f'{key}: required group tables missing')
        if not errors and not any(run_errors[i['row']['run_id']] for i in items):
            valid_groups.append(key)
        issues.extend(errors)
    expected_runs = expected_count(c, 'expected_runs', 280)
    expected_groups = expected_count(c, 'expected_groups', 56)
    if len(state['rows']) != expected_runs or len(grouped) != expected_groups or set(state['groups']) != set(grouped):
        issues.append(f'Required coverage is {expected_runs} runs / {expected_groups} groups')
    complete = not issues
    return dict(complete=complete, status='COMPLETE' if complete else 'PARTIAL / BLOCKED',
                checked_at=time.time(), issues=issues, valid_groups=valid_groups)


def group_metrics(c, rows, production):
    """Only adapt manifest identity; all thermal/service definitions use closed production functions."""
    work = Path(c["output_dir"])
    assignment = pd.read_csv(Path(c["cache_root"]) / "01_inputs/E8_event_assignment.csv")
    event_id = str(rows[0]["event_id"])
    csv_write(work / "01_inputs/E8_event_assignment.csv", assignment.loc[assignment.parent_event_id.eq(event_id)])
    frame = pd.DataFrame(rows).copy()
    key = pair_key(rows[0])
    frame["role"] = "core_group_" + rows[0]["base_case_id"]
    ca, service = science.capacity_and_phase_audit(frame, work, production)
    oa, thermal = science.outage_audit(frame, work)
    audit = pd.concat([ca, oa], ignore_index=True)
    if not audit["pass"].all():
        bad = audit.loc[~audit["pass"].astype(bool)].to_dict("records")
        raise Blocked(f"Pair physics QC failed: {bad}")
    indexed = frame.set_index("scenario_id")
    hw = science.load_building(work, indexed.loc["HW_UNL", "run_id"])
    cf = science.load_building(work, indexed.loc["CF_UNL", "run_id"])
    if not hw.timestamp_hkt.equals(cf.timestamp_hkt):
        raise Blocked("HW/CF timestamp mismatch")
    weather = pd.read_parquet(indexed.loc["HW_UNL", "weather_path"])
    event = weather.loc[weather.event_phase.eq("heatwave")]
    start = pd.Timestamp(event.timestamp_hkt.iloc[0])
    end = pd.Timestamp(event.timestamp_hkt.iloc[-1]).normalize()
    mask = science.formal_event_mask(hw, start, end)
    eh = hw.loc[mask, "total_modeled_electricity_j"].to_numpy() / 3.6e6
    ec = cf.loc[mask, "total_modeled_electricity_j"].to_numpy() / 3.6e6
    if not np.isfinite(eh).all() or not np.isfinite(ec).all():
        raise Blocked("Non-finite energy")
    delta = float(eh.sum() - ec.sum())
    relative = delta / ec.sum() if ec.sum() > 1e-9 else math.nan
    energy = service.copy()
    energy["E_HW_kwh"] = eh.sum(); energy["E_CF_kwh"] = ec.sum()
    energy["net_delta_kwh"] = delta
    energy["sum_positive_timestep_delta_kwh"] = np.maximum(eh - ec, 0).sum()
    energy["positive_event_net_delta_kwh"] = max(0, delta)
    energy["relative_delta"] = relative
    energy["prototype_peak_difference_kw"] = (eh.max() - ec.max()) / .25
    c70_zones = science.load_zones(work, indexed.loc["HW_C70", "run_id"])
    c70_mask = science.formal_event_mask(c70_zones, start, end) & c70_zones.structural_conditioning_present.astype(bool)
    energy["c70_max_conditioned_temperature_c"] = c70_zones.loc[c70_mask, "temperature_c"].max()
    energy["c70_max_conditioned_relative_humidity_pct"] = c70_zones.loc[c70_mask, "relative_humidity_pct"].max()
    scores = []
    normal = 1 / (1 + max(0, relative)) if np.isfinite(relative) else math.nan
    for component, value in (("N", normal), ("C", float(service.score_capacity_stepwise.iloc[0]))):
        scores.append(dict(component=component, score_lower=value, score_upper=value,
                           observation_status="defined" if np.isfinite(value) else "zero_baseline_not_evaluable"))
    rated, outage = indexed.loc["HW_RATED"], indexed.loc["HW_O24_SYNC"]
    support = science.load_monitoring_support(work, outage, rated)
    cross, _ = science.building_crossing_from_zones(support, pd.Timestamp(outage.outage_start_hkt))
    rec, _ = science.building_recovery_from_zones(support, pd.Timestamp(outage.outage_end_hkt))
    scores.extend([dict(component="O", **science.outage_score_bounds(cross)),
                   dict(component="R", **science.recovery_score_bounds(rec))])
    tables = {"energy": energy, "thermal": thermal, "qc": audit, "scores": pd.DataFrame(scores)}
    folder = (work / "groups" / event_id / rows[0]["base_case_id"]
              if c.get("scope", {}).get("event_isolated_groups")
              else work / "groups" / rows[0]["base_case_id"])
    for name, table in tables.items():
        table["pair_id"] = key
        for field in ("event_id", "base_case_id", "prototype_id", "floor_position"):
            table[field] = rows[0][field]
        table["analysis_version"] = c["analysis_version"]
        table["physical_input_hashes"] = ";".join(frame.physics_hash)
        csv_write(folder / (name + ".csv"), table)
    return dict(status=COMPLETE, hashes=file_hashes(folder), pair_id=key)


def completed_groups(c, state, production, recompute=False):
    if c.get('scope', {}).get('solver_only'):
        for item in state['rows'].values():
            if item['status'] in GOOD and not run_evidence_issues(c, item):
                item['status'] = COMPLETE
        return
    grouped = {}
    for item in state["rows"].values():
        grouped.setdefault(pair_key(item["row"]), []).append(item)
    for key, items in grouped.items():
        if len(items) != 5 or not all(i["status"] in GOOD for i in items):
            continue
        prior = state["groups"].get(key, {})
        errors = [error for i in items for error in run_evidence_issues(c, i)]
        if errors:
            raise Blocked('; '.join(errors))
        dependencies = group_dependencies(c, items)
        if not recompute and prior.get('dependencies') == dependencies and verify_files(prior.get("hashes", {})):
            # Identical rebuilt evidence can reuse metrics; commit group and all members together.
            prior['status'] = COMPLETE
            for item in items:
                item['status'] = COMPLETE
            continue
        try:
            result = group_metrics(c, [i["row"] for i in items], production)
        except Exception as exc:
            state["groups"][key] = dict(status="blocked", error=f"{type(exc).__name__}: {exc}")
            # A common algorithm/physics problem is not a retryable solver failure.
            raise Blocked(f"Pair {key}: {exc}") from exc
        result['dependencies'] = dependencies
        state["groups"][key] = result
        for item in items:
            item["status"] = COMPLETE


def status_view(state):
    if not state:
        return dict(status="not_started", actual_calls=0, manual_confirmation=False)
    counts = Counter(i["status"] for i in state["rows"].values())
    return dict(batch_id=state["batch_id"], total=len(state["rows"]), states=dict(counts),
                completed=counts.get(COMPLETE, 0), reserved_calls=len(state["attempts"]),
                actual_calls=sum(a.get("launched", False) for a in state["attempts"]),
                uncertain_reservations=sum(a["reservation"] == "reserved_pending_launch" for a in state["attempts"]),
                retry_reservations=sum(a["is_retry"] for a in state["attempts"]),
                active_seconds=state["spent_seconds"], stop_reason=state["stop_reason"],
                active_session=state["active_session"], complete=False,
                identity='stored state only; not current evidence verification',
                recorded_complete=counts.get(COMPLETE, 0) == len(state["rows"]),
                last_verification={k: state.get('last_verification', {}).get(k) for k in ('checked_at', 'status', 'complete')},
                 pending_reextract=sum(i["row"]["disposition"] == "reextract" and i["status"] not in GOOD for i in state["rows"].values()))


def progress_line(c, state):
    view = status_view(state)
    total = max(1, int(view.get("total", 0)))
    done = int(view.get("completed", 0))
    width = 28
    filled = min(width, int(width * done / total))
    bar = "█" * filled + "░" * (width - filled)
    states = view.get("states", {})
    elapsed = float(view.get("active_seconds", 0)) / 60
    return (f"[{c.get('scope', {}).get('label', c['batch_id'])}] {bar} "
            f"{done}/{total} ({100 * done / total:5.1f}%) | running={states.get('running', 0)} "
            f"qc={states.get('qc_passed', 0)} pending={states.get('pending', 0)} "
            f"failed={states.get('retryable_failed', 0) + states.get('blocked', 0) + states.get('parse_failed', 0)} "
            f"calls={view.get('actual_calls', 0)} retries={view.get('retry_reservations', 0)} elapsed={elapsed:.1f} min")


def reports(c, state):
    """Pure result consumer. Never dispatches or silently fills a missing group."""
    out = Path(c["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    view = status_view(state)
    verification = verify_completion(c, state)
    view.update(complete=verification['complete'], identity='current evidence verification', verification=verification)
    if state:
        state['last_verification'] = verification
    atomic_json(out / "progress.json", view)
    resource = dict(view, bytes_on_disk=storage_bytes(c), storage_accounting="unique_file_identity" if c.get("count_hardlinks_once") else "logical_path_bytes", free_disk_bytes=shutil.disk_usage(out).free,
                    configured_limits={k: c[k] for k in sorted(RESOURCE_KEYS)})
    atomic_json(out / "resource_usage.json", resource)
    rows, attempts, coverage, warnings = [], [], [], []
    if state:
        for run_id, item in state["rows"].items():
            row = item["row"]
            rows.append(dict(batch_id=c["batch_id"], run_id=run_id, pair_id=pair_key(row),
                             physics_hash=row["physics_hash"], output_schema_hash=row["raw_output_schema_hash"],
                             analysis_version=c["analysis_version"], event_id=row["event_id"],
                             base_case_id=row["base_case_id"], prototype_id=row["prototype_id"],
                             floor_position=row["floor_position"], scenario_id=row["scenario_id"],
                             disposition=row["disposition"], status=item["status"], error=item.get("error", ""),
                             cached_g3_run_id=row.get("cached_g3_run_id", ""),
                             raw_hashes=json.dumps(item.get("raw_hashes", {})),
                             parsed_hashes=json.dumps(item.get("parsed_hashes", {}))))
            if item.get("raw_dir"):
                err = Path(item["raw_dir"]) / "eplusout.err"
                if err.exists():
                    text = err.read_text(errors="replace")
                    footer = re.search(r"Completed Successfully--\s*(\d+) Warning", text)
                    warnings.append(dict(run_id=run_id, text_warning_entries=len(re.findall(r"\*\* Warning \*\*", text)),
                                         footer_warning_count=int(footer.group(1)) if footer else None,
                                         severe_entries=len(re.findall(r"\*\*\s+Severe\s+\*\*", text)),
                                         err_path=str(err), err_sha256=digest(err)))
        attempts = [dict(batch_id=c["batch_id"], **a) for a in state["attempts"]]
        for key in sorted({pair_key(i["row"]) for i in state["rows"].values()}):
            items = [i for i in state["rows"].values() if pair_key(i["row"]) == key]
            gr = state["groups"].get(key, {})
            verified = key in verification['valid_groups']
            coverage.append(dict(pair_id=key, required=5, qc_passed=sum(i["status"] in GOOD for i in items),
                                 metrics_complete=verified, missing_scenarios=";".join(i["row"]["scenario_id"] for i in items if i["status"] not in GOOD), error=gr.get("error", "")))
    csv_write(out / "run_status.csv", pd.DataFrame(rows, columns=None if rows else ["run_id", "status"]))
    csv_write(out / "execution_ledger.csv", pd.DataFrame(attempts, columns=None if attempts else ["attempt_id", "run_id", "reservation"]))
    csv_write(out / "coverage_and_pairing.csv", pd.DataFrame(coverage, columns=None if coverage else ["pair_id", "required", "qc_passed", "metrics_complete"]))
    csv_write(out / "warning_summary.csv", pd.DataFrame(warnings, columns=None if warnings else ["run_id", "footer_warning_count"]))
    scope = c.get("scope", {})
    prefix = scope.get("report_prefix", "g4a")
    for name, filename in (("energy", f"{prefix}_energy_service_metrics.csv"), ("thermal", f"{prefix}_thermal_recovery_metrics.csv"), ("scores", f"{prefix}_stage_score_bounds.csv"), ("qc", "physics_qc.csv")):
        tables = []
        if state:
            for key, group in state["groups"].items():
                if key not in verification['valid_groups']:
                    continue
                if not all(i["status"] == COMPLETE for i in state["rows"].values() if pair_key(i["row"]) == key):
                    continue
                matches = [p for p in group["hashes"] if Path(p).name == name + ".csv"]
                tables.extend(pd.read_csv(p) for p in matches)
        csv_write(out / filename, pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=["pair_id", "analysis_version"]))
    failures = [r for r in rows if r["status"] in {"blocked", "retryable_failed", "parse_failed", "interrupted"}]
    atomic_json(out / "failure_summary.json", dict(runs=failures, evidence_issues=verification['issues'], groups={k: v for k, v in (state or {}).get("groups", {}).items() if v.get("status") != COMPLETE}))
    complete = verification['complete']
    label = scope.get("label", "G4-a")
    report = (f"# {label} local execution\n\nStatus: {'COMPLETE' if complete else 'PARTIAL / NOT STARTED'}\n\n"
              f"Manifest: {manifest_hash(c)}\n\n{json.dumps(view, ensure_ascii=False, indent=2)}\n\n"
              f"Scope: {', '.join(scope.get('events', ['HKO_LI_0049']))}, {scope.get('prehistory_days', 14)}-day prehistory, "
              f"{expected_count(c, 'expected_groups', 56)} design-event groups x five scenarios. "
              "These are prototype diagnostics, not citywide or complete eight-event BERI results.\n\n"
              "Thermal output uses interval means; crossing interpolation is a stated diagnostic model. "
              "Censoring bounds are preserved. Negative energy deltas and true zero scores are retained. "
              "Reference: King's Park 2016-2025 fixed_recent_reference_benchmark.\n\n"
              "Unchanged evidence limitations: HKUST source interval alignment unresolved and all four "
              "hour-of-week baselines outperformed the generic model; inferred building years do not establish "
              "historical occupancy; the existing DOCX raw-LaTeX test failure remains outside this run.\n")
    (out / scope.get("execution_report", "G4a_execution_report.md")).write_text(report, encoding="utf-8")
    return complete


def package_review(c):
    # Refresh all projections immediately before packing the same evidence snapshot.
    state = load_state(c)
    complete = reports(c, state)
    if state:
        atomic_json(Path(c['output_dir']) / 'state.json', state)
    out = Path(c["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    scope = c.get("scope", {})
    archive_name = scope.get("review_archive", "G4a_local_review.zip")
    marker = out / (Path(archive_name).stem + '_stale.json')
    if marker.exists():
        atomic_json(marker, dict(reason='refreshed current evidence snapshot', usable_as_current=True,
                                 complete=complete, at=time.time()))
    entries = {}
    for p in out.iterdir():
        if p.is_file() and p.suffix in {".json", ".csv", ".md"} and p.name not in {"state.json", "review_file_hashes.csv"}:
            entries[p.name] = p
    entries["inputs/run_manifest.csv"] = Path(c["manifest"])
    entries["inputs/version_lock.json"] = Path(c["version_lock"])
    root = Path(c["project_root"])
    for rel in scope.get("review_code_files", ["src/workflows/run_g4a_local.py", "src/utils/g4a_local_execution.py"]):
        entries["code/" + rel] = root / rel
    # One representative complete raw log/dictionary plus a small CSV trace, not full high-frequency archives.
    dirs = sorted((out / "02_runs/g3_results").glob("*"))
    for run in dirs[:1]:
        for name in ("eplusout.err", "eplusout.end", "eplusout.rdd", "eplusout.mdd"):
            if (run / "raw" / name).exists():
                entries["representative/" + run.name + "/" + name] = run / "raw" / name
        z = run / "parsed/native_timestep_zones.parquet"
        if z.exists():
            csv_write(out / "representative_zone_trace.csv", pd.read_parquet(z).head(96))
            entries["representative_zone_trace.csv"] = out / "representative_zone_trace.csv"
    hashes = [{"path": name, "bytes": p.stat().st_size, "sha256": digest(p)} for name, p in entries.items() if p.exists()]
    csv_write(out / "review_file_hashes.csv", pd.DataFrame(hashes))
    entries["review_file_hashes.csv"] = out / "review_file_hashes.csv"
    target = out / archive_name
    temp = out / (archive_name + ".writing")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, p in entries.items():
            if p.exists():
                z.write(p, name)
    with zipfile.ZipFile(temp) as z:
        if z.testzip() is not None:
            raise Blocked("Review archive CRC failure")
    os.replace(temp, target)
    (out / (archive_name + ".sha256")).write_text(digest(target) + "  " + target.name + "\n", encoding="utf-8")
    return complete


def launch(session, row):
    if session.interrupts:
        return
    session.check_resources(headroom_seconds=30)
    a = reserve_attempt(session.state, session.c, row["run_id"])
    base = session.output / "attempts" / row["run_id"] / a["attempt_id"]
    raw = base / "raw"
    raw.mkdir(parents=True, exist_ok=False)
    stage_inputs(session.c, row, base / "inputs")
    a["raw_dir"] = str(raw)
    session.save()  # Reservation is durable BEFORE Popen.
    handles = [(raw / "stdout.txt").open("wb"), (raw / "stderr.txt").open("wb")]
    command = [session.c["energyplus"], "-w", str(base / "inputs/weather.epw"), "-d", str(raw), str(base / "inputs/model.idf")]
    try:
        proc = subprocess.Popen(command, cwd=str(base), stdout=handles[0], stderr=handles[1],
                                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
    except OSError as exc:
        for f in handles:
            f.close()
        a.update(reservation="launch_failed", finished=time.time(), error=str(exc))
        session.state["rows"][row["run_id"]].update(status="retryable_failed", error=str(exc))
        session.save()
        return
    # Popen success is an actual call, independent of the process table's subsequent lifetime.
    a.update(pid=proc.pid, launched=True, reservation='launched_identity_pending', command=command,
             identity_status='pending')
    session.live[row["run_id"]] = (proc, a, handles)
    receipt = base / 'launch_receipt.json'
    try:
        atomic_json(receipt, dict(a))
        session.save()
        try:
            a['created'] = psutil.Process(proc.pid).create_time()
            a['identity_status'] = 'verified'
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError) as exc:
            code = proc.poll()
            if code is None:
                a.update(identity_status='unconfirmed', identity_error=str(exc))
                raise Blocked('Owned worker live but identity unavailable; stop, retain evidence') from exc
            a.update(identity_status='exited_owned_handle', returncode=code)
        a['reservation'] = 'launched'
        atomic_json(receipt, dict(a))
        session.save()
    except BaseException:
        a['registration_failed'] = True
        try:
            session.kill_owned()
        finally:
            # A second journal survives failure of the authoritative state's atomic commit.
            # If the filesystem itself is unavailable, the durable reservation still blocks retry.
            atomic_json(receipt, dict(a))
        raise


def collect(session):
    for run_id, (proc, a, handles) in list(session.live.items()):
        code = proc.poll()
        if code is None:
            if time.time() - a["started"] >= session.c["run_timeout_seconds"]:
                raise ResourceStop("individual_run_timeout")
            try:
                rss = psutil.Process(proc.pid).memory_info().rss
                a["peak_rss_bytes"] = max(a.get("peak_rss_bytes", 0), rss)
                if rss > session.c["max_worker_rss_bytes"]:
                    raise ResourceStop("worker_memory_limit")
            except psutil.NoSuchProcess:
                pass
            continue
        for handle in handles:
            handle.close()
        a.update(returncode=code, finished=time.time(), elapsed_seconds=time.time() - a["started"], reservation="finished", identity_status='exited_owned_handle')
        item = session.state["rows"][run_id]
        raw = Path(a["raw_dir"])
        if code == 0 and raw_complete(raw):
            item.update(status="raw_complete", raw_dir=str(raw), raw_hashes=file_hashes(raw), error="")
        else:
            err = (raw / "eplusout.err").read_text(errors="replace") if (raw / "eplusout.err").exists() else ""
            physical = bool(re.search(r"\*\*\s+(Severe|Fatal)\s+\*\*", err)) or code == 0
            item.update(status="blocked" if physical else "retryable_failed", error=f"solver_returncode={code}; complete/QC={raw_complete(raw)}")
        del session.live[run_id]
        session.save()


def execute(c, retry_failed=False):
    entered = time.monotonic()
    out = Path(c["output_dir"])
    scope = c.get("scope", {})
    registry = Path(c["project_root"]) / scope.get("owner_registry", "intermediate/simulation_outputs/upstream/G4a_manifest_owner.json")
    # Manifest-level ownership prevents a fresh batch ID from resetting the same input budget.
    with exclusive_lock(registry.with_suffix(".lock")):
        if registry.exists():
            owner = read_json(registry)
            if owner != dict(manifest=manifest_hash(c), batch_id=c["batch_id"], output_dir=str(out)):
                raise Blocked("This manifest already belongs to another batch/output; retain its budget/state")
        with exclusive_lock(out / "execution.lock"):
            frame, env, version, _ = preflight(c)
            state = load_state(c)
            if state is None:
                if registry.exists():
                    raise Blocked("Owner exists but state is missing; do not create a fresh budget")
                ceiling = scope.get("initial_ceiling", {"max_calls": 280, "max_retries": 15, "wall_seconds": 3600,
                                                         "max_bytes": 5 * 1024**3, "min_free_bytes": 10 * 1024**3})
                if (c["max_calls"] > int(ceiling["max_calls"]) or c["max_retries"] > int(ceiling["max_retries"])
                        or c["wall_seconds"] > int(ceiling["wall_seconds"]) or c["max_bytes"] > int(ceiling["max_bytes"])
                        or c["min_free_bytes"] < int(ceiling["min_free_bytes"])):
                    raise Blocked("Initial resource configuration exceeds the frozen work-order ceiling")
                state = initial_state(c, frame, version)
            else:
                reconcile_execution_identity(c, state, version)
            validate_state(state, frame)
            restore_state(c, state)
            runtime_identity = {k: env.get(k) for k in ("python", "dependencies", "engine_sha256")}
            if state.get("runtime_identity") is not None and state["runtime_identity"] != runtime_identity:
                raise Blocked("Runtime versions changed across resume; inspect before mixing outputs")
            state["runtime_identity"] = runtime_identity
            if state["config_history"]:
                old = state["config_history"][-1]["config"]
                changed = {k for k in set(old) | set(c) if old.get(k) != c.get(k)}
                if changed - RESOURCE_KEYS:
                    raise Blocked(f"Non-resource configuration changed: {sorted(changed)}")
            state["spent_seconds"] += time.monotonic() - entered
            state["config_history"].append(dict(at=time.time(), config=c, sha256=canonical(c)))
            state["active_session"] = dict(pid=os.getpid(), created=psutil.Process().create_time(), host=socket.gethostname(),
                                             started=time.time(), heartbeat=time.time(), wall_limit=c["wall_seconds"])
            session = Session(c, state)
            # Save before claiming the manifest; a crash cannot issue a second fresh budget.
            session.save()
            atomic_json(registry, dict(manifest=manifest_hash(c), batch_id=c["batch_id"], output_dir=str(out)))
            try:
                session.check_resources(headroom_seconds=30)
            except ResourceStop as exc:
                state["stop_reason"] = str(exc)
                state["active_session"] = None; session.save()
                reports(c, state)
                raise
            confirmation = dict(at=time.time(), manual_cli_start=not bool(c.get("delegated_workorder_start")),
                                delegated_workorder_start=bool(c.get("delegated_workorder_start")), manifest_sha256=manifest_hash(c),
                                config_sha256=canonical(c), version_lock_sha256=version, pid=os.getpid(),
                                hostname=socket.gethostname(), scope=scope.get("confirmation_scope", "G4a only"), retry_failed=retry_failed)
            state["confirmations"].append(confirmation)
            session.save()
            atomic_json(out / "manual_start_confirmations.json", state["confirmations"])
            atomic_json(out / "effective_config.json", c)
            atomic_json(out / "environment.json", env)
            shutil.copy2(c["manifest"], out / "run_manifest.csv")
            production = load_production(Path(c["project_root"]), code_root=Path(c.get("code_root", c["project_root"])))
            old_handler = signal.signal(signal.SIGINT, session.signal)
            code = 0
            try:
                schedule(session, production, retry_failed)
            except ResourceStop as exc:
                state["stop_reason"] = str(exc); code = 2
                session.kill_owned()
            except Exception as exc:
                state["stop_reason"] = f"{type(exc).__name__}: {exc}"; code = 4
                session.kill_owned()
            finally:
                signal.signal(signal.SIGINT, old_handler)
                state["active_session"] = None
                session.save()
                if state["spent_seconds"] >= c["wall_seconds"] or storage_bytes(c) >= c["max_bytes"]:
                    state["stop_reason"] = "resource_limit_during_finalization"
                    code = 2
                    session.save()
                verified = reports(c, state)
                session.save()
                # Final projections precede packaging; do not rewrite their snapshot afterwards.
                if state["spent_seconds"] + 5 < c["wall_seconds"] and storage_bytes(c) + 8 * 1024**2 < c["max_bytes"] and shutil.disk_usage(out).free >= c["min_free_bytes"]:
                    verified = package_review(c)
                    saved = load_state(c)
                    if saved:
                        state['last_verification'] = saved.get('last_verification')
                session.save()  # Account finalization time; package records its checked-at snapshot.
                if state['spent_seconds'] >= c['wall_seconds'] or storage_bytes(c) >= c['max_bytes']:
                    state['stop_reason'] = 'resource_limit_during_finalization'
                    code = 2
                    session.save()
                    reports(c, state)
                    # Do not deliver a stale COMPLETE archive after a resource stop.
                    marker = out / (Path(scope.get("review_archive", "G4a_local_review.zip")).stem + '_stale.json')
                    atomic_json(marker, dict(reason=state['stop_reason'], usable_as_current=False))
            if verified:
                return code
            return code or 4


def _pilot_complete(c, state, groups, pilot):
    if c.get("scope", {}).get("solver_only"):
        return all(item["status"] == COMPLETE for item in groups[pilot])
    return state["groups"].get(pilot, {}).get("status") == COMPLETE


def schedule(session, production, retry_failed):
    c, state = session.c, session.state
    attempted = set()
    next_print = 0.0
    # First complete wholly-new five-scenario group is the serial pilot, included in the 265.
    groups = {}
    for item in state["rows"].values():
        groups.setdefault(pair_key(item["row"]), []).append(item)
    new_groups = [k for k, items in groups.items() if all(i["row"]["disposition"] == "new" for i in items)]
    pilot = new_groups[0] if new_groups else next(iter(groups))
    if not new_groups:
        state["pilot_passed"] = True
    while True:
        session.check_resources()
        collect(session)
        if session.interrupts:
            state["stop_reason"] = "user_interrupt"
            if session.interrupts > 1 or time.monotonic() - session.interrupted_at >= c["grace_seconds"]:
                raise ResourceStop("user_interrupt_grace_expired")
            if not session.live:
                raise ResourceStop("user_interrupt_saved")
            time.sleep(.2)
            continue
        for run_id, item in state["rows"].items():
            if session.interrupts:
                break
            if item["status"] == "pending" and item["row"]["disposition"] == "reextract":
                source = verify_cache(c, item["row"])
                item.update(status="raw_complete", raw_dir=str(source / "raw"), raw_hashes=file_hashes(source / "raw"))
                session.save()
            if item["status"] in {"raw_complete", "parse_failed"} and (run_id, "parse") not in attempted:
                attempted.add((run_id, "parse"))
                session.check_resources(headroom_seconds=30)
                try:
                    parse_run(c, item["row"], item, production, checkpoint=session.save)
                except Exception as exc:
                    item.update(status="parse_failed", error=f"{type(exc).__name__}: {exc}")
                    session.save()
                    raise Blocked(f"Parser/QC error; raw retained, no re-solve: {run_id}: {exc}") from exc
                session.save()
        if session.interrupts:
            continue
        session.check_resources(headroom_seconds=30)
        completed_groups(c, state, production)
        if _pilot_complete(c, state, groups, pilot):
            state["pilot_passed"] = True
        session.save()
        if all(i["status"] == COMPLETE for i in state["rows"].values()):
            state["stop_reason"] = "complete"; return
        if any(i["status"] == "blocked" for i in state["rows"].values()):
            raise Blocked("Input/physical QC blockage; no further dispatch")
        max_live = c["concurrency"] if state["pilot_passed"] else 1
        # Reduce concurrency when actual available memory cannot retain a conservative worker reserve.
        max_live = min(max_live, max(1, psutil.virtual_memory().available // c["max_worker_rss_bytes"]))
        for run_id, item in state["rows"].items():
            if session.interrupts:
                break
            eligible = item["status"] in {"pending", "interrupted"} or (retry_failed and item["status"] == "retryable_failed")
            if not eligible or run_id in attempted or item["row"]["disposition"] != "new":
                continue
            if not state["pilot_passed"] and pair_key(item["row"]) != pilot:
                continue
            if any(i["status"] in {"retryable_failed", "parse_failed", "blocked"} for i in groups[pair_key(item["row"])]):
                if not (retry_failed and item["status"] == "retryable_failed"):
                    continue
            if len(session.live) >= max_live:
                break
            attempted.add(run_id)
            launch(session, item["row"])
        if time.monotonic() >= next_print:
            print(progress_line(c, state) if c.get("human_progress") else json.dumps(status_view(state), ensure_ascii=False), flush=True)
            # Live progress is explicitly a stored-state projection, not a repeated full-file audit.
            atomic_json(Path(c['output_dir']) / 'progress.json', status_view(state))
            next_print = time.monotonic() + c["progress_seconds"]
        if not session.live:
            state["stop_reason"] = "partial_failures_require_inspection_or_retry"
            return
        time.sleep(.1)


def main(argv=None):
    parser = argparse.ArgumentParser(description="G4-a: direct manual invocation starts/resumes; --doctor/--dry-run/--status/--summarize/--package-review never solve")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "config/g4a_local.json"))
    actions = parser.add_mutually_exclusive_group()
    for flag in ("doctor", "dry-run", "execute", "status", "summarize", "package-review"):
        actions.add_argument("--" + flag, action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    if not any(getattr(args, k) for k in ('doctor', 'dry_run', 'execute', 'status', 'summarize', 'package_review')) and not args.resume and not args.retry_failed:
        args.execute = args.resume = True
    if (args.resume or args.retry_failed) and not args.execute:
        parser.error("--resume/--retry-failed cannot dispatch without --execute")
    if args.execute and not args.resume:
        parser.error("Use --execute --resume for both first start and recovery")
    try:
        c = load_config(args.config)
        if args.execute:
            return execute(c, args.retry_failed)
        if args.status:
            print(json.dumps(status_view(load_state(c)), ensure_ascii=False, indent=2)); return 0
        if args.doctor or args.dry_run:
            frame, env, version, checks = preflight(c)
            saved = load_state(c)
            if saved:
                validate_state(saved, frame)
                for index, check in checks.iterrows():
                    item = saved["rows"][check.run_id]
                    if item["status"] in GOOD and verify_files(item.get("raw_hashes", {})) and verify_files(item.get("parsed_hashes", {})):
                        action = "skip_complete" if item["status"] == COMPLETE else "pair_metrics_only"
                    elif item["status"] in GOOD | {"parsed", "raw_complete", "parse_failed"} and verify_files(item.get("raw_hashes", {})):
                        action = "reextract_existing_raw"
                    elif item["status"] == "running":
                        action = "inspect_old_process_before_resume"
                    elif item["status"] in {"blocked", "retryable_failed"}:
                        action = item["status"]
                    else:
                        action = item["row"]["disposition"]
                    checks.at[index, "effective_action"] = action
            dest = Path(c["development_dir"])
            dest.mkdir(parents=True, exist_ok=True)
            # Filesystem writeability check is a development artifact, not execution approval.
            atomic_json(dest / "doctor.json", dict(environment=env, version_lock_sha256=version, energyplus_calls=0))
            csv_write(dest / "dry_run_actions.csv", checks)
            summary = dict(required=len(frame), unique_physics=frame.physics_hash.nunique(),
                           units=frame.groupby(["prototype_id", "floor_position"]).ngroups,
                           dispositions=frame.disposition.value_counts().to_dict(), energyplus_calls=0,
                           effective_actions=checks.effective_action.value_counts().to_dict(),
                           manual_start_confirmed=False, version_lock_sha256=version)
            atomic_json(dest / "dry_run_summary.json", summary)
            print(json.dumps(dict(summary, environment=env), ensure_ascii=False, indent=2)); return 0
        with exclusive_lock(Path(c["output_dir"]) / "execution.lock"):
            state = load_state(c)
            if args.summarize and state:
                try:
                    verify_versions(c)
                    validate_state(state, pd.read_csv(c["manifest"]).fillna(""))
                    restore_state(c, state)
                    production = load_production(Path(c["project_root"]), code_root=Path(c.get("code_root", c["project_root"])))
                    for item in state["rows"].values():
                        if item["status"] in {"raw_complete", "parse_failed"}:
                            if storage_bytes(c) + 8 * 1024**2 >= c["max_bytes"]:
                                raise ResourceStop("storage_limit_during_reextraction")
                            if shutil.disk_usage(c["output_dir"]).free < c["min_free_bytes"]:
                                raise ResourceStop("free_disk_during_reextraction")
                            parse_run(c, item["row"], item, production)
                            atomic_json(Path(c["output_dir"]) / "state.json", state)
                    completed_groups(c, state, production)
                except Exception as exc:
                    state['stop_reason'] = f'Summary repair stopped: {type(exc).__name__}: {exc}'
                    reports(c, state)
                    atomic_json(Path(c['output_dir']) / 'state.json', state)
                    return 2 if isinstance(exc, ResourceStop) else 4
                atomic_json(Path(c["output_dir"]) / "state.json", state)
            complete = reports(c, state)
            if state:
                atomic_json(Path(c['output_dir']) / 'state.json', state)
            if args.package_review:
                complete = package_review(c)
        return 0 if complete else 4
    except ResourceStop as exc:
        print(f"RESOURCE STOP: {exc}", file=sys.stderr); return 2
    except (Blocked, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr); return 3
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr); return 4


if __name__ == "__main__":
    raise SystemExit(main())
