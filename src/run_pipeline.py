from __future__ import annotations

import argparse
import subprocess
import sys

from data_preparation.paths import project_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the project analysis and figure workflow.")
    parser.add_argument("--kernel", default="hk-heatwave", help="Jupyter kernel name")
    parser.add_argument("--check-only", action="store_true", help="Validate inputs without drawing figures")
    args = parser.parse_args()

    paths = project_paths()
    subprocess.run([sys.executable, str(paths.root / "src" / "check_inputs.py")], check=True)
    if args.check_only:
        return 0

    paths.figures.mkdir(parents=True, exist_ok=True)
    for existing_figure in paths.figures.glob("*.png"):
        existing_figure.unlink()
    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(paths.notebook),
        "--output",
        "plot.executed.ipynb",
        "--output-dir",
        str(paths.intermediate),
        "--ExecutePreprocessor.timeout=1800",
        f"--ExecutePreprocessor.kernel_name={args.kernel}",
    ]
    subprocess.run(command, cwd=paths.root, check=True)
    png_files = sorted(paths.figures.glob("*.png"))
    if len(png_files) != 23:
        raise RuntimeError(f"Expected 23 PNG figures, found {len(png_files)}.")
    subprocess.run(
        [sys.executable, str(paths.root / "src" / "data_preparation" / "build_data_dictionary.py")],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(paths.root / "src" / "data_preparation" / "build_checksums.py")],
        check=True,
    )
    print(f"Generated {len(png_files)} PNG figures in {paths.figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
