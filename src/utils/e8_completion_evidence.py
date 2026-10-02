"""Source-specific compatibility audit for retained manuscript evidence."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from utils.e8_completion_common import Blocked, atomic_csv, sha256

ALLOWED_DISPOSITIONS = {"verified_reuse", "reextract", "recompute", "approved_new_simulation",
                        "excluded_by_explicit_scope_decision", "blocked_with_reason"}

def audit_external_evidence(workspace: Path, upstream: Path, output_dir: Path, future_decision: str,
                            registry_path: Path | None = None) -> dict:
    registry_path = Path(registry_path) if registry_path is not None else workspace / "config/external_evidence_registry.csv"
    registry = pd.read_csv(registry_path).fillna("")
    required = {"source_id", "dataset_match", "validation_layer", "source_path", "method_identity",
                "claim_boundary", "disposition", "required_for_final"}
    if missing := sorted(required - set(registry)): raise Blocked(f"external evidence registry missing columns: {missing}")
    if registry.source_id.duplicated().any(): raise Blocked("external evidence source_id is not unique")
    if not set(registry.disposition) <= ALLOWED_DISPOSITIONS: raise Blocked("external evidence registry has an unsupported disposition")
    rows = []
    for item in registry.to_dict("records"):
        source = upstream / str(item["source_path"]) if item["source_path"] else None
        exists = bool(source and source.is_file()); disposition = str(item["disposition"])
        if item["source_id"] in {"future_energy", "future_thermal"}:
            if future_decision == "exclude_incompatible_old_results": disposition = "excluded_by_explicit_scope_decision"
            elif future_decision == "reuse_only_if_hash_compatible": disposition = "blocked_with_reason"
        rows.append({**item, "resolved_path": str(source) if source else "", "source_exists": exists,
                     "source_sha256": sha256(source) if exists else "", "compatibility_decision": disposition,
                     "source_status": "available" if exists else "missing_or_not_applicable",
                     "evidence_identity": "source-specific; legacy pass labels are not compatibility decisions"})
    audit = pd.DataFrame(rows)
    old = pd.read_csv(workspace / "intermediate/protocol/evidence/validation_evidence_matrix.csv").fillna("")
    compatibility = old.merge(audit[["dataset_match", "compatibility_decision", "resolved_path", "source_exists",
                                     "source_sha256", "method_identity"]],
                              left_on="dataset", right_on="dataset_match", how="left", validate="many_to_one")
    compatibility["compatibility_decision"] = compatibility.compatibility_decision.fillna("blocked_with_reason")
    compatibility["compatibility_reason"] = compatibility.apply(
        lambda row: ("registry source and method audited" if row.compatibility_decision in {"verified_reuse", "reextract", "recompute"}
                     and bool(row.source_exists) else "claim retained only within explicit limitation or scope decision"), axis=1)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(output_dir / "external_source_compatibility.csv", audit)
    atomic_csv(output_dir / "validation_evidence_compatibility.csv", compatibility)
    unresolved = audit[(audit.required_for_final.astype(str).str.lower().isin(["true", "1", "yes"])) &
                       audit.compatibility_decision.isin(["verified_reuse", "reextract", "recompute"]) & ~audit.source_exists]
    return {"sources": len(audit), "missing_required_available_sources": unresolved.source_id.astype(str).tolist(),
            "registry_sha256": sha256(registry_path)}
