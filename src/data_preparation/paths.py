from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def intermediate(self) -> Path:
        return self.root / "intermediate"

    @property
    def result(self) -> Path:
        return self.root / "result"

    @property
    def figure_source_data(self) -> Path:
        return self.result / "figure_source_data"

    @property
    def notebook(self) -> Path:
        return self.root / "plot" / "plot.ipynb"

    @property
    def figures(self) -> Path:
        return self.root / "output" / "figures"


def project_paths() -> ProjectPaths:
    return ProjectPaths(Path(__file__).resolve().parents[2])
