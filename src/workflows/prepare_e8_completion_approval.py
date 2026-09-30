"""Create the one-time production approval after explicit local confirmation.

This command never invokes EnergyPlus. Without the acceptance flag it only
prints help and cannot modify the formal analysis contract or approval receipt.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from utils.e8_completion_common import (
    Blocked, approval_contract, atomic_json, load_config, validate_approval,
)
from utils.e8_completion_manifests import audit_targeted_c70_migration
from utils.e8_completion_required_analyses import validate_analysis_decisions


def prepare_approval(config_path: str | Path, *, accepted: bool,
                     output: str | Path | None = None) -> Path:
    if not accepted:
        raise Blocked("Explicit --accept-recommended confirmation is required")
    c = load_config(config_path)
    workspace = Path(c["workspace_root"])
    proposal_path = Path(c["analysis_decision_proposal"])
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    decisions = proposal.get("decision_parameters")
    if not isinstance(decisions, dict):
        raise Blocked("Recommended analysis decision file has no decision_parameters")
    approved_decisions = json.loads(json.dumps(decisions))
    for name, value in approved_decisions.items():
        if not isinstance(value, dict):
            raise Blocked(f"Recommended decision is malformed: {name}")
        value["approval_status"] = "approved"
    validate_analysis_decisions({"decision_parameters": approved_decisions})

    migration = audit_targeted_c70_migration(
        Path(c["upstream_project_root"]),
        Path(c["results_root"]) / "00_protocol/targeted_c70_migration",
        c["upstream"].get("candidate_version_lock_sha256"),
        c["upstream"].get("expected_c70_replacements"),
    )
    if not migration.get("evidence_complete"):
        raise Blocked("targeted C70 migration evidence is incomplete: " + "; ".join(migration.get("issues", [])))
    from utils.e8_completion_execution import doctor
    readiness = doctor(c)
    resources = readiness["physical_manifest_and_resources"]
    if resources.get("budget_shortfall_before_actual_dedup"):
        raise Blocked("configured solver-call budget is below the conservative pre-dedup requirement")
    if resources.get("storage_budget_shortfall_before_actual_dedup"):
        raise Blocked(
            "configured storage budget is below the conservative measured estimate; "
            "review DECISIONS_REQUIRED_FINAL.md before approval"
        )

    contract_path = workspace / "config/completion_analysis_contract.json"
    current_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    current_contract["decision_parameters"] = approved_decisions
    current_contract["decision_approval"] = {
        "identity": "explicit_local_cli_acceptance_of_reviewed_recommendations",
        "proposal_path": str(proposal_path),
        "accepted_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    snapshot = workspace / "snapshots/preapproval/completion_analysis_contract.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        shutil.copy2(contract_path, snapshot)
    atomic_json(contract_path, current_contract)

    target = Path(output).resolve() if output else workspace / "config/completion_approval.user.json"
    receipt = approval_contract(c)
    receipt.update({
        "approval_status": "approved",
        "approved_by": "local_user_explicit_cli",
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "lockbox_release": "released_after_prediction_freeze",
        "mechanism_levels_status": "approved",
        "future_building_results_decision": "exclude_incompatible_old_results",
        "results_root_confirmed": True,
        "targeted_c70_scientific_migration_acceptance": "accepted_after_evidence_audit",
        "approval_scope": "E8 completion task plan and recommended contracts only",
    })
    atomic_json(target, receipt)
    validate_approval(c, target)
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Explicitly accept the reviewed E8 completion recommendations; no solver is called")
    parser.add_argument("--config", default=str(ROOT / "config/e8_completion.json"))
    parser.add_argument("--accept-recommended", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if not args.accept_recommended:
        parser.print_help()
        return 0
    try:
        path = prepare_approval(args.config, accepted=True, output=args.output)
        print(json.dumps({"status": "APPROVED_BY_EXPLICIT_LOCAL_COMMAND",
                          "approval_file": str(path), "energyplus_calls": 0},
                         ensure_ascii=False, indent=2))
        return 0
    except (Blocked, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
