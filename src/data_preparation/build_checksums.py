from __future__ import annotations

import hashlib
from pathlib import Path

from paths import project_paths


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
        if path.is_file() and path not in excluded and ".git" not in path.parts
    ]
    lines = [f"{sha256(path)}  {path.relative_to(paths.root).as_posix()}" for path in sorted(files)]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote checksums for {len(lines)} files to {destination}")


if __name__ == "__main__":
    main()
