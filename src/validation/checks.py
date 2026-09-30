from __future__ import annotations

from pathlib import Path

import pandas as pd


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def require_columns(path: Path, columns: set[str]) -> None:
    if path.suffix.lower() == ".csv":
        available = set(pd.read_csv(path, nrows=0).columns)
    elif path.suffix.lower() == ".parquet":
        available = set(pd.read_parquet(path).columns)
    else:
        raise ValueError(f"Unsupported table type: {path}")
    missing = sorted(columns - available)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")


def require_png_outputs(directory: Path, expected_names: tuple[str, ...]) -> None:
    require_files([directory / name for name in expected_names])
    unexpected_pdf = sorted(directory.glob("*.pdf"))
    if unexpected_pdf:
        raise ValueError(f"PDF outputs are not part of this repository: {unexpected_pdf}")
