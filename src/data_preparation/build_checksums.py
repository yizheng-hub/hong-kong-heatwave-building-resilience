from __future__ import annotations

import hashlib
from pathlib import Path

from paths import project_paths


# SHA256SUMS.txt is the integrity manifest for the redistributable GitHub
# mirror.  The complete local root additionally contains provider-controlled
# inputs and large solver artefacts whose inventory is validated separately.
LOCAL_ONLY_TREES = (
    Path("data"),
    Path("intermediate/compiled_simulation_inputs"),
    Path("intermediate/simulation_outputs"),
    Path("intermediate/source_derived"),
    Path("result/accepted"),
    Path("result/production_run"),
    Path("software/energyplus"),
)

TRACKED_DATA_DOCUMENTATION = {
    Path("data/README.md"),
    Path("data/SOURCE_MANIFEST.csv"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    paths = project_paths()
    destination = paths.root / "SHA256SUMS.txt"
    excluded = {destination, paths.intermediate / "plot.executed.ipynb"}
    files = [
        path
        for path in paths.root.rglob("*")
        if path.is_file()
        and path not in excluded
        and ".git" not in path.parts
        and (
            path.relative_to(paths.root) in TRACKED_DATA_DOCUMENTATION
            or not any(
                path.relative_to(paths.root).is_relative_to(tree)
                for tree in LOCAL_ONLY_TREES
            )
        )
    ]
    lines = [f"{sha256(path)}  {path.relative_to(paths.root).as_posix()}" for path in sorted(files)]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote checksums for {len(lines)} files to {destination}")


if __name__ == "__main__":
    main()
