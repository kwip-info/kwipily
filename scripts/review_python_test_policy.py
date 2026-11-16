#!/usr/bin/env python3
"""Refresh explicitly reviewed test sources, not automatic parallel approval.

Before accepting a file, review hidden auto-device use, time/memory measurements,
process lifetime and shared external resources. Add serial exceptions first.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
TEST_ROOTS = ("runner/tests/", "scripts/tests/", "scripts/runtime_tests/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=ROOT / "scripts/python_test_policy.json")
    parser.add_argument("--costs", type=Path, help="Optional measured JUnit scheduling weights; never acceptance evidence.")
    parser.add_argument("reviewed", nargs="*", help="Exact test/helper files reviewed for lane safety.")
    args = parser.parse_args()
    if not args.reviewed and not args.costs:
        parser.error("Name reviewed files or a measured cost receipt.")
    policy = json.loads(args.policy.read_text())
    for name in args.reviewed:
        path = ROOT / name
        try:
            normalized = str(path.resolve().relative_to(ROOT))
        except ValueError:
            normalized = None
        if normalized != name or not name.startswith(TEST_ROOTS) or not name.endswith(".py") or path.is_symlink() or not path.is_file():
            parser.error(f"Expected an existing local test/helper source: {name}")
        policy["reviewedSources"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if args.costs:
        weights = {}
        for case in ET.parse(args.costs).iter("testcase"):
            parts = case.get("classname", "").split(".")
            index = next((i for i, part in enumerate(parts) if part.startswith("test_")), None)
            if index is None:
                continue
            path = "/".join(parts[:index + 1]) + ".py"
            weights[path] = weights.get(path, 0) + float(case.get("time", 0))
        policy["fileSeconds"] = weights
    policy["reviewedAt"] = datetime.now(timezone.utc).date().isoformat()
    args.policy.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")
    print(f"Updated {len(args.reviewed)} explicitly reviewed source fingerprints.")


if __name__ == "__main__":
    main()
