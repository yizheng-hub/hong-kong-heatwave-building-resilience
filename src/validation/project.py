from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


HISTORICAL_ROOT_MARKERS = (
    "HK_heatwave_public_reproducibility",
    "HK_heatwave_E8_completion_workspace",
    "HK_heatwave_E8_completion_results",
    "D:\\HKBFETD",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def table_rows(path: Path) -> int:
    if path.suffix.lower() == ".parquet":
        return int(pq.ParquetFile(path).metadata.num_rows)
    if path.suffix.lower() == ".csv":
        return int(len(pd.read_csv(path)))
    raise ValueError(f"Unsupported accepted table: {path}")


def tree_stats(path: Path) -> dict[str, int]:
    files = 0
    total_bytes = 0
    stack = [path]
    while stack:
        current = stack.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    files += 1
                    total_bytes += entry.stat(follow_symlinks=False).st_size
    return {"files": files, "bytes": total_bytes}


def _check_current_locks(root: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in ("open_event_version_lock.json", "g4a_version_lock.json"):
        path = root / "config" / name
        lock = json.loads(path.read_text(encoding="utf-8"))
        failures = []
        for relative, expected in lock["files"].items():
            target = root / relative
            if not target.is_file():
                failures.append(f"missing:{relative}")
            elif sha256(target) != expected:
                failures.append(f"hash:{relative}")
        if failures:
            raise ValueError(f"{name} failed: {failures}")
        output[name] = {"sha256": sha256(path), "files": len(lock["files"])}
    return output


def _check_accepted_results(root: Path) -> dict[str, Any]:
    receipt_path = root / "docs" / "audit" / "production_protocol" / "final_verification.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not receipt.get("complete") or not receipt.get("production_ready"):
        raise ValueError("The final production verification receipt is not complete.")
    verified: dict[str, Any] = {}
    for label, expected in receipt["checks"].items():
        old_path = PureWindowsPath(expected["path"])
        target = root / "result" / "accepted" / old_path.parent.name / old_path.name
        if not target.is_file():
            raise FileNotFoundError(target)
        actual_hash = sha256(target)
        actual_rows = table_rows(target)
        if actual_hash != expected["sha256"] or actual_rows != int(expected["rows"]):
            raise ValueError(
                f"Accepted result differs for {label}: rows={actual_rows}, sha256={actual_hash}"
            )
        verified[label] = {
            "path": target.relative_to(root).as_posix(),
            "rows": actual_rows,
            "sha256": actual_hash,
        }
    return verified


def _check_accepted_figures(root: Path) -> dict[str, Any]:
    manifest = root / "docs" / "audit" / "accepted_figure_sha256.csv"
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    if len(rows) != 23:
        raise ValueError(f"Expected 23 accepted figure hashes, found {len(rows)}")
    failures = []
    for row in rows:
        path = root / row["file"]
        if not path.is_file():
            failures.append(f"missing:{row['file']}")
        elif sha256(path) != row["sha256"]:
            failures.append(f"hash:{row['file']}")
    if failures:
        raise ValueError(f"Accepted figures differ: {failures}")
    if list((root / "output" / "figures").glob("*.pdf")):
        raise ValueError("PDF figure output is not part of the current project.")
    return {"figures": len(rows), "manifest": manifest.relative_to(root).as_posix()}


def _check_archive(root: Path) -> dict[str, Any]:
    archive = root / "result" / "archive" / "accepted_eight_event_results.zip"
    checksum = archive.with_name(archive.name + ".sha256")
    expected = checksum.read_text(encoding="utf-8").split()[0].lower()
    actual = sha256(archive)
    if actual != expected:
        raise ValueError(f"Accepted-results archive hash differs: {actual} != {expected}")
    return {"path": archive.relative_to(root).as_posix(), "sha256": actual, "bytes": archive.stat().st_size}


def _check_no_historical_dependencies(root: Path) -> dict[str, Any]:
    files: list[Path] = []
    for base in (root / "src", root / "config"):
        files.extend(path for path in base.rglob("*") if path.is_file())
    for name in ("README.md", "README_CN.md"):
        files.append(root / name)
    for name in ("WORKFLOW.md", "DATA_AVAILABILITY.md", "GETTING_STARTED.md"):
        files.append(root / "docs" / name)
    findings = []
    for path in files:
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.suffix.lower() not in {".py", ".json", ".csv", ".md", ".yml", ".yaml", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for marker in HISTORICAL_ROOT_MARKERS:
            if marker in text:
                findings.append(f"{path.relative_to(root).as_posix()}:{marker}")
    if findings:
        raise ValueError(f"Active files still depend on historical roots: {findings[:20]}")
    return {"files_scanned": len(files), "historical_root_references": 0}


def validate_unified_project(root: Path, *, full_local: bool = False) -> dict[str, Any]:
    root = root.resolve()
    required = [
        root / "data" / "SOURCE_MANIFEST.csv",
        root / "result" / "archive" / "accepted_eight_event_results.zip",
        root / "plot" / "plot.ipynb",
    ]
    if full_local:
        required.extend([
            root
            / "data"
            / "building_stock"
            / "hk_bfetd"
            / "HK_UBEM_Buildings_Public_v1_1.csv",
            root / "software" / "energyplus" / "EnergyPlus-26.1.0-6f2e40d102-Windows-x86_64" / "energyplus.exe",
            root / "intermediate" / "simulation_outputs" / "upstream" / "01_inputs" / "E8_event_assignment.csv",
            root / "intermediate" / "simulation_outputs" / "upstream" / "E8_open6" / "combined" / "open6_aggregate_status.json",
            root / "result" / "accepted" / "03_stock" / "building_reference_beri.parquet",
        ])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Unified project is missing required local files:\n" + "\n".join(missing))

    report: dict[str, Any] = {
        "status": "PASS",
        "project_root": str(root),
        "energyplus_calls": 0,
        "active_dependencies": _check_no_historical_dependencies(root),
        "accepted_figures": _check_accepted_figures(root),
        "accepted_archive": _check_archive(root),
    }
    if full_local:
        report["current_code_locks"] = _check_current_locks(root)
        report["accepted_results"] = _check_accepted_results(root)
        inventories = {
            "source_inputs": tree_stats(root / "data"),
            "compiled_simulation_inputs": tree_stats(root / "intermediate" / "compiled_simulation_inputs"),
            "simulation_outputs": tree_stats(root / "intermediate" / "simulation_outputs"),
            "accepted_results_tree": tree_stats(root / "result" / "accepted"),
            "energyplus_runtime": tree_stats(root / "software" / "energyplus"),
        }
        minimums = {
            "source_inputs": (1000, 10 * 1024**3),
            "compiled_simulation_inputs": (22000, 4 * 1024**3),
            "simulation_outputs": (300000, 80 * 1024**3),
            "accepted_results_tree": (90, 3 * 1024**3),
            "energyplus_runtime": (8000, 800 * 1024**2),
        }
        for label, (minimum_files, minimum_bytes) in minimums.items():
            actual = inventories[label]
            if actual["files"] < minimum_files or actual["bytes"] < minimum_bytes:
                raise ValueError(f"Local inventory is incomplete for {label}: {actual}")
        report["local_inventory"] = inventories
    return report
