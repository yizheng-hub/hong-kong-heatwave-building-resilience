from __future__ import annotations

import hashlib
import subprocess
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
    # The manifest covers the GitHub release, including hydrated Git LFS files.
    tracked = subprocess.check_output(
        ["git", "-C", str(paths.root), "ls-files", "-z"]
    ).decode("utf-8").split("\0")
    files = sorted(name for name in tracked if name and name != destination.name)
    lines = [f"{sha256(paths.root / name)}  {name}" for name in files]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote checksums for {len(lines)} files to {destination}")


if __name__ == "__main__":
    main()
