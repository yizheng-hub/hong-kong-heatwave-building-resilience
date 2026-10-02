"""Compare regenerated CSV/Parquet tables with the accepted result package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PORTABILITY_METADATA_COLUMNS = {
    "source_batch",
    "source_result_dir",
    "compiled_dir",
    "idf_path",
    "epw_path",
    "schedule_path",
    "event_schedule_path",
    "weather_path",
    "source_path",
    "resolved_path",
    "source_sha256",
    "source_hash",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def table_shape(path: Path) -> tuple[int, list[str]]:
    if path.suffix.lower() == ".parquet":
        metadata = pq.ParquetFile(path).metadata
        return metadata.num_rows, pq.ParquetFile(path).schema.names
    frame = pd.read_csv(path, nrows=0)
    rows = sum(1 for _ in path.open("rb")) - 1
    return max(rows, 0), list(frame.columns)


def compare_frames(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        return {"content_equal": False, "max_abs_numeric_difference": math.nan}
    max_abs = 0.0
    equal = True
    for column in left.columns:
        a = left[column]
        b = right[column]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
            bv = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
            same = np.isclose(av, bv, rtol=1e-12, atol=1e-12, equal_nan=True)
            equal = equal and bool(same.all())
            if (~same).any():
                finite = np.isfinite(av) & np.isfinite(bv)
                if finite.any():
                    max_abs = max(max_abs, float(np.max(np.abs(av[finite] - bv[finite]))))
        else:
            same = a.astype("string").fillna("<NA>").equals(
                b.astype("string").fillna("<NA>")
            )
            equal = equal and same
    return {"content_equal": bool(equal), "max_abs_numeric_difference": max_abs}


def compare_csv(
    left: Path,
    right: Path,
    chunksize: int = 100_000,
    ignore_columns: set[str] | None = None,
) -> dict:
    equal = True
    max_abs = 0.0
    left_reader = pd.read_csv(left, chunksize=chunksize, low_memory=False)
    right_reader = pd.read_csv(right, chunksize=chunksize, low_memory=False)
    while True:
        try:
            a = next(left_reader)
        except StopIteration:
            a = None
        try:
            b = next(right_reader)
        except StopIteration:
            b = None
        if a is None or b is None:
            equal = equal and a is None and b is None
            break
        if ignore_columns:
            keep = [column for column in a.columns if column not in ignore_columns]
            a = a[keep]
            b = b[keep]
        result = compare_frames(a, b)
        equal = equal and result["content_equal"]
        max_abs = max(max_abs, result["max_abs_numeric_difference"])
    return {"content_equal": bool(equal), "max_abs_numeric_difference": max_abs}


def compare_pair(generated: Path, accepted: Path) -> dict:
    generated_hash = sha256(generated)
    accepted_hash = sha256(accepted)
    generated_rows, generated_columns = table_shape(generated)
    accepted_rows, accepted_columns = table_shape(accepted)
    result = {
        "generated_rows": generated_rows,
        "accepted_rows": accepted_rows,
        "row_count_equal": generated_rows == accepted_rows,
        "generated_columns": "|".join(generated_columns),
        "accepted_columns": "|".join(accepted_columns),
        "columns_equal": generated_columns == accepted_columns,
        "generated_sha256": generated_hash,
        "accepted_sha256": accepted_hash,
        "sha256_equal": generated_hash == accepted_hash,
        "content_equal": generated_hash == accepted_hash,
        "max_abs_numeric_difference": 0.0,
    }
    if generated_hash == accepted_hash:
        return result
    if not result["row_count_equal"] or not result["columns_equal"]:
        result["content_equal"] = False
        return result
    if generated.suffix.lower() == ".parquet":
        generated_frame = pd.read_parquet(generated)
        accepted_frame = pd.read_parquet(accepted)
        content = compare_frames(generated_frame, accepted_frame)
    else:
        content = compare_csv(generated, accepted)
    result.update(content)
    if result["content_equal"]:
        result["scientific_content_equal"] = True
    else:
        ignored = PORTABILITY_METADATA_COLUMNS.intersection(generated_columns)
        if generated.suffix.lower() == ".parquet":
            keep = [column for column in generated_columns if column not in ignored]
            scientific = compare_frames(generated_frame[keep], accepted_frame[keep])
        else:
            scientific = compare_csv(generated, accepted, ignore_columns=ignored)
        result["scientific_content_equal"] = bool(scientific["content_equal"])
        result["ignored_portability_columns"] = "|".join(sorted(ignored))
    return result


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, default=root / "result/accepted")
    parser.add_argument(
        "--report-dir", type=Path, default=root / "result/verification"
    )
    args = parser.parse_args(argv)
    generated = args.generated.resolve()
    accepted = args.accepted.resolve()
    extensions = {".csv", ".parquet"}
    left = {
        path.relative_to(generated).as_posix(): path
        for path in generated.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    }
    right = {
        path.relative_to(accepted).as_posix(): path
        for path in accepted.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    }
    rows = []
    for relative in sorted(set(left) | set(right)):
        if relative not in left or relative not in right:
            rows.append(
                {
                    "relative_path": relative,
                    "status": "missing_generated" if relative not in left else "additional_generated",
                }
            )
            continue
        comparison = compare_pair(left[relative], right[relative])
        status = (
            "byte_identical"
            if comparison["sha256_equal"]
            else "content_equal_hash_differs"
            if comparison["content_equal"]
            else "content_differs"
        )
        rows.append({"relative_path": relative, "status": status, **comparison})
    report = pd.DataFrame(rows)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.report_dir / "table_comparison.csv", index=False, encoding="utf-8-sig")
    required = report.loc[report["status"].ne("additional_generated")].copy()
    scientific_equal = required.get(
        "scientific_content_equal", pd.Series(False, index=required.index)
    ).fillna(required["status"].isin(["byte_identical", "content_equal_hash_differs"]))
    summary = {
        "generated_root": os.path.relpath(generated, root).replace("\\", "/"),
        "accepted_root": os.path.relpath(accepted, root).replace("\\", "/"),
        "tables": len(report),
        "accepted_tables": len(right),
        "additional_generated_tables": int(report["status"].eq("additional_generated").sum()),
        "status_counts": report["status"].value_counts().to_dict(),
        "all_rows_and_columns_equal": bool(
            required.get("row_count_equal", pd.Series(False, index=required.index)).fillna(False).all()
            and required.get("columns_equal", pd.Series(False, index=required.index)).fillna(False).all()
        ),
        "all_content_equal": bool(
            required["status"].isin(["byte_identical", "content_equal_hash_differs"]).all()
        ),
        "all_scientific_content_equal": bool(scientific_equal.all()),
    }
    temporary = args.report_dir / "comparison_summary.json.writing"
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, args.report_dir / "comparison_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["all_scientific_content_equal"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
