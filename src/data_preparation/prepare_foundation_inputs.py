"""Regenerate solver-free foundation inputs used by the preserved simulations.

This command does not call EnergyPlus.  It recreates the deterministic prototype
and 297-design tables from the production code, verifies their identities against
the preserved simulation manifest, and stages the canonical event catalogue.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.project_root.resolve()
    sys.path.insert(0, str(root / "src"))

    from workflows.stages import build_building_response_and_exposure as production

    hazard = root / "intermediate/foundation_pipeline/01_hazard"
    response = root / "intermediate/foundation_pipeline/02_building_response"
    hazard.mkdir(parents=True, exist_ok=True)
    response.mkdir(parents=True, exist_ok=True)

    parameters = production.prototype_parameter_table()
    provenance = production.prototype_parameter_provenance(parameters)
    design = production.design_matrix(parameters, 297, production.RANDOM_SEED)
    if len(design) != 297 or design["base_case_id"].nunique() != 297:
        raise RuntimeError("Production design function did not produce 297 unique base cases")
    if design.groupby(["prototype_id", "floor_position"]).ngroups != 56:
        raise RuntimeError("Production design function did not reproduce 56 response units")

    preserved = pd.read_csv(
        root / "intermediate/simulation_outputs/upstream/02_runs/g4_297_design_event_manifest.csv",
        usecols=["base_case_id", "prototype_id", "floor_role"],
    ).drop_duplicates()
    check = design[["base_case_id", "prototype_id", "floor_position"]].merge(
        preserved,
        on="base_case_id",
        how="outer",
        suffixes=("_generated", "_preserved"),
        indicator=True,
    )
    identity_ok = bool(
        len(check) == 297
        and check["_merge"].eq("both").all()
        and check["prototype_id_generated"].eq(check["prototype_id_preserved"]).all()
        and check["floor_position"].eq(check["floor_role"]).all()
    )
    if not identity_ok:
        raise RuntimeError("Regenerated design identities differ from preserved simulation inputs")

    outputs = {
        "prototype_physics_parameters.csv": parameters,
        "prototype_parameter_provenance.csv": provenance,
        "thermal_experiment_design.csv": design,
    }
    for name, frame in outputs.items():
        frame.to_csv(response / name, index=False, encoding="utf-8-sig")

    catalog_source = root / "config/events/event_catalog_65.csv"
    catalog_target = hazard / "heatwave_event_catalog.csv"
    shutil.copy2(catalog_source, catalog_target)

    prepared_stock = root / "intermediate/prepared_stock/all_sector_building_model_inputs.parquet"
    if prepared_stock.exists():
        link_or_copy(prepared_stock, response / prepared_stock.name)

    files = [catalog_target, *(response / name for name in outputs)]
    receipt = {
        "status": "complete",
        "energyplus_calls": 0,
        "design_rows": int(len(design)),
        "response_units": int(design.groupby(["prototype_id", "floor_position"]).ngroups),
        "preserved_manifest_identity_match": identity_ok,
        "files": {
            str(path.relative_to(root)).replace("\\", "/"): sha256(path)
            for path in files
        },
    }
    (root / "intermediate/foundation_pipeline/foundation_input_receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
