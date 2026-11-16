#!/usr/bin/env python3
"""Offline, evidence-bound coverage accounting. No framework imports or network calls."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

REQUIRED_EVIDENCE = ("configuration", "execution", "export", "validation", "diagnostics", "lesson", "example", "ui", "persistence", "failure_handling")
DISPOSITIONS = {"mapped", "alias", "unsupported", "infrastructure", "excluded"}

def _path(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("Evidence/inventory paths must be nonempty repository-relative paths")
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Evidence/inventory path escapes repository")
    return path

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _unique(items, field, label):
    keys = [item[field] for item in items]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate {label}")

def meets_threshold(complete: int, total: int, threshold: float) -> bool:
    return total > 0 and complete * 100 >= threshold * total

def report_catalog(catalog: dict, repo_root: Path) -> dict:
    root = Path(repo_root)
    # A report uses one local filesystem snapshot per canonical file. These
    # caches are discarded on return so later reports observe edits/deletions.
    resolved_paths = {}
    file_hashes = {}
    def resolve_path(value):
        if not isinstance(value, str):
            return _path(root, value)  # Preserve invalid-path validation.
        if value not in resolved_paths:
            resolved_paths[value] = _path(root, value)
        return resolved_paths[value]
    def file_hash(path):
        if path not in file_hashes:
            file_hashes[path] = _sha(path) if path.is_file() else None
        return file_hashes[path]
    if catalog.get("schema_version") != 1:
        raise ValueError("Unsupported catalog schema")
    for field in ("framework", "framework_version", "documentation_version", "catalog_version"):
        if not isinstance(catalog.get(field), str) or not catalog[field]:
            raise ValueError(f"Missing {field}")
    status = catalog.get("denominator_status")
    if status not in {"proposed", "frozen", "approved-consumer-scope"}:
        raise ValueError("Unknown denominator status")
    if status in {"frozen", "approved-consumer-scope"} and not (catalog.get("approval") or {}).get("reference"):
        raise ValueError("Frozen denominator requires an approval record")
    inventory = catalog["inventory"]
    inventory_path = resolve_path(inventory["path"])
    if (inventory_digest := file_hash(inventory_path)) is None or inventory_digest != inventory["sha256"]:
        raise ValueError("Inventory snapshot missing or hash mismatch")
    rows = json.loads(inventory_path.read_text())
    _unique(rows, "symbol", "API symbol")
    symbols = {row["symbol"]: row for row in rows}
    capabilities = catalog["capabilities"]
    _unique(capabilities, "id", "capability ID")
    ids = {unit["id"] for unit in capabilities}
    for row in rows:
        if row.get("disposition") not in DISPOSITIONS or not row.get("reason", "").strip():
            raise ValueError(f"Unexplained disposition: {row['symbol']}")
        canonical = row.get("canonical_symbol")
        if canonical not in symbols:
            raise ValueError(f"Unknown canonical symbol: {canonical}")
        if row["disposition"] == "alias":
            if canonical == row["symbol"] or symbols[canonical]["disposition"] == "alias":
                raise ValueError("Aliases must resolve directly to a distinct canonical symbol")
        elif canonical != row["symbol"]:
            raise ValueError("Nonalias cannot redirect canonical identity")
        if not set(row.get("capability_ids", [])) <= ids:
            raise ValueError("Inventory refers to an unknown capability")
    declared_sources = {unit["id"]: {member["canonical_symbol"] for member in unit.get("source_members", [{"canonical_symbol":unit["canonical_symbol"]}])} for unit in capabilities}
    for row in rows:
        for key in row.get("capability_ids", []):
            if row["canonical_symbol"] not in declared_sources[key]:
                raise ValueError("Inventory mapping is not reciprocal with capability source membership")
    complete, incomplete = [], []
    families = defaultdict(lambda: {"selected": 0, "complete": 0})
    operation_modes = set()
    for unit in capabilities:
        key, canonical = unit["id"], unit["canonical_symbol"]
        modes = unit.get("mode_scope", [])
        if not modes or not all(isinstance(mode, str) and mode for mode in modes) or len(modes) != len(set(modes)):
            raise ValueError(f"Explicit unique mode scope required: {key}")
        members = unit.get("source_members", [{"canonical_symbol": canonical, "required_modes": modes}])
        if not members or canonical not in [m.get("canonical_symbol") for m in members]:
            raise ValueError("Source members must include primary canonical identity")
        _unique(members, "canonical_symbol", "source member")
        member_modes = set()
        for member in members:
            symbol = member["canonical_symbol"]
            if symbol not in symbols or symbols[symbol]["disposition"] == "alias":
                raise ValueError(f"Capability must name canonical inventoried APIs: {key}")
            if symbols[symbol]["disposition"] != "mapped" or key not in symbols[symbol].get("capability_ids", []):
                raise ValueError("Selected capability lacks reciprocal inventory mapping")
            required = member.get("required_modes", [])
            if not required or len(required) != len(set(required)) or not set(required) <= set(modes):
                raise ValueError("Source member required modes must be explicit and within unit modes")
            member_modes.update(required)
            for mode in required:
                if (symbol, mode) in operation_modes:
                    raise ValueError("Duplicate API/mode creates an inflated denominator")
                operation_modes.add((symbol, mode))
        if member_modes != set(modes):
            raise ValueError("All unit modes require source member provenance")
        family = unit["family"]
        if not isinstance(family, str) or not family:
            raise ValueError("Capability family required")
        families[family]["selected"] += 1
        reasons = []
        if unit.get("mode_scope_status") != "reviewed":
            reasons.append("operation-specific modes not reviewed")
        if unit.get("configuration_contract") != "verified":
            reasons.append("configuration contract not verified")
        matrix = unit.get("backend_matrix", {})
        if matrix.get("status") != "verified" or not matrix.get("cases"):
            reasons.append("backend/dtype matrix not verified")
        for dimension in REQUIRED_EVIDENCE:
            proven = set()
            for item in unit.get("evidence", {}).get(dimension, []):
                path = resolve_path(item["path"])
                if item.get("status") == "verified":
                    if (digest := file_hash(path)) is not None and item.get("sha256") == digest:
                        proven.add(item.get("scope"))
                    else:
                        # Every asserted verified dependency must remain current.
                        # An unchanged narrative receipt must not mask changed code.
                        reasons.append(f"{dimension}: stale verified evidence {item['path']}")
            missing = sorted({f"{key}:{mode}" for mode in modes} - proven)
            if missing:
                reasons.append(f"{dimension}: no current verified evidence for {', '.join(missing)}")
        if reasons:
            incomplete.append({"id": key, "family": family, "reasons": reasons})
        else:
            complete.append(key)
            families[family]["complete"] += 1
    for family in catalog["required_families"]:
        families[family]  # Empty required families remain visible.
    for values in families.values():
        values["percent"] = round(100 * values["complete"] / values["selected"], 2) if values["selected"] and status != "proposed" else None
    count = len(capabilities)
    percent = round(100 * len(complete) / count, 2) if count and status != "proposed" else None
    thresholds = catalog["thresholds"]
    minimum, target = thresholds["minimum_percent"], thresholds["target_percent"]
    if isinstance(minimum, bool) or isinstance(target, bool) or not (0 <= minimum <= target <= 100):
        raise ValueError("Invalid release thresholds")
    scope_decision = catalog.get("scope_decision", {})
    approved_count = scope_decision.get("approved_unit_count") if status == "approved-consumer-scope" else None
    if status == "approved-consumer-scope" and approved_count != count:
        raise ValueError("Approved consumer count must match selected units")
    deferrals = catalog.get("deferrals")
    deferrals_pass = deferrals is None and status != "approved-consumer-scope"
    if deferrals is not None:
        entries = deferrals.get("units", [])
        deferred_ids = [item["id"] for item in entries]
        if len(deferred_ids) != len(set(deferred_ids)) or not set(deferred_ids) <= ids:
            raise ValueError("Deferred units must uniquely reference selected capabilities")
        if any(not item.get("reason", "").strip() for item in entries):
            raise ValueError("Every deferral requires rationale")
        if len(entries) > count - math.ceil(count * minimum / 100):
            raise ValueError("Deferral count exceeds minimum coverage allowance")
        deferrals_pass = deferrals.get("status") == "owner-approved" and bool(deferrals.get("reference"))
        if deferrals_pass and set(deferred_ids) != ids - set(complete):
            deferrals_pass = False  # An accepted list must name every incomplete unit.
    boundary_pass = catalog.get("boundary_review", {}).get("status", "reviewed" if status == "frozen" else "pending") == "reviewed"
    mandatory_pass = all(families[family]["complete"] > 0 for family in catalog["required_families"])
    return {
        "framework": catalog["framework"], "framework_version": catalog["framework_version"],
        "documentation_version": catalog["documentation_version"], "catalog_version": catalog["catalog_version"],
        "denominator_status": status, "capability_count": count, "complete_count": len(complete),
        "capability_percent": percent, "approved_consumer_unit_count": approved_count,
        "minimum_complete_units": math.ceil(count * minimum / 100), "target_complete_units": math.ceil(count * target / 100),
        "boundary_review": catalog.get("boundary_review"), "deferrals": deferrals,
        "release_review_ready": boundary_pass and deferrals_pass,
        "source_members": {unit["id"]: unit.get("source_members", [{"canonical_symbol": unit["canonical_symbol"], "required_modes": unit["mode_scope"]}]) for unit in capabilities}, "complete_ids": sorted(complete), "incomplete": incomplete,
        "families": dict(sorted(families.items())), "thresholds": thresholds,
        "release_gate_passed": bool(percent is not None and meets_threshold(len(complete), count, minimum) and mandatory_pass and boundary_pass and deferrals_pass),
        "target_reached": bool(percent is not None and meets_threshold(len(complete), count, target) and mandatory_pass and boundary_pass and deferrals_pass),
        "poc_components": catalog.get("poc_components", []),
        "api": {"symbol_count": len(rows), "canonical_count": sum(row["disposition"] != "alias" for row in rows),
                "dispositions": dict(sorted(Counter(row["disposition"] for row in rows).items())),
                "mapped_canonical_count": sum(row["disposition"] == "mapped" for row in rows),
                "framework_wide_percent": None,
                "framework_completeness": inventory["framework_completeness"], "scope": inventory["scope"],
                "note": "Symbol disposition is inventory accounting, not implemented API coverage. Framework-wide implementation percentage is not established."},
    }

def report_directory(directory: Path, root: Path) -> dict:
    files = sorted(Path(directory).glob("*.json"))
    reports = [report_catalog(json.loads(path.read_text()), root) for path in files]
    _unique(reports, "framework", "framework report")
    return {"report_schema_version": 1, "frameworks": reports,
            "all_release_gates_passed": bool(reports) and all(r["release_gate_passed"] for r in reports),
            "combined_percent": None,
            "note": "No aggregate percentage masks an unsupported framework. Proposed denominators withhold percentage claims."}

def markdown(report: dict) -> str:
    lines = ["# Capability coverage report", "", report["note"], "", "| Framework | Version | Listed API symbols | Proposed/selected units | Evidence-complete units | Capability coverage | Release gate |", "|---|---|---:|---:|---:|---|---|"]
    for row in report["frameworks"]:
        percent = "Not established (proposed)" if row["capability_percent"] is None else f"{row['capability_percent']}%"
        lines.append(f"| {row['framework']} | {row['framework_version']} | {row['api']['symbol_count']} | {row['capability_count']} | {row['complete_count']} | {percent} | {'Pass' if row['release_gate_passed'] else 'Not passed'} |")
    for row in report["frameworks"]:
        if row["approved_consumer_unit_count"] is not None:
            lines += ["", f"{row['framework']}: **{row['approved_consumer_unit_count']} owner-approved consumer units**; minimum **{row['minimum_complete_units']}**, target **{row['target_complete_units']}**. Evidence-complete numerator is {row['complete_count']}; source presence is not acceptance.",
                      f"Mode/backend boundary review: {(row.get('boundary_review') or {}).get('status', 'unknown')}. Final deferral review: {(row.get('deferrals') or {}).get('status', 'unknown')}.", "",
                      "| Required family | Selected | Evidence complete |", "|---|---:|---:|"]
            for family, values in row['families'].items():
                lines.append(f"| {family} | {values['selected']} | {values['complete']} |")
            for item in (row.get('deferrals') or {}).get('units', []):
                lines.append(f"Deferred: {item['id']} — {item['reason']}")
    lines += ["", "Framework-wide API implementation coverage is not established. Listed-symbol dispositions are not feature implementation. See JSON for per-family counts, grouped source membership, pending mode/backend boundaries, named deferrals and incomplete evidence reasons.", ""]
    return "\n".join(lines)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--catalog-dir", type=Path, default=Path("capabilities/catalogs/v1"))
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-release", action="store_true")
    args = parser.parse_args(argv)
    try:
        directory = args.catalog_dir if args.catalog_dir.is_absolute() else args.root / args.catalog_dir
        result = report_directory(directory, args.root)
        payload = json.dumps(result, indent=2, sort_keys=True) + "\n" if args.format == "json" else markdown(result)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload)
        else:
            print(payload, end="")
        return 1 if args.require_release and not result["all_release_gates_passed"] else 0
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"Invalid capability catalog: {error}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
