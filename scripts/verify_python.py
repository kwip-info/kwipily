#!/usr/bin/env python3
"""Complete local Python verification: two CPU workers, then serial contracts.

No app/runtime dependencies, test assertions, devices or hosted runners change.
Every collected node must execute exactly once before PASSED.json is written.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ("scripts/tests", "runner/tests", "scripts/runtime_tests")


def source_identity():
    value = {}
    for directory in ("runner", "scripts"):
        for path in sorted((ROOT / directory).rglob("*.py")):
            value[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    value["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    value["trackedDiff"] = hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=ROOT)).hexdigest()
    return value


def audit_coverage(expected, phases):
    """Refuse missing/duplicate cases, changed collection, or phase failures."""
    identities = expected["collection"]["all"]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate identities in full collection.")
    assignments = expected["collection"]["assignments"]
    selections, terminal, outcomes = [], [], Counter()
    for phase in phases:
        if phase["exitStatus"] != 0:
            raise ValueError("A verification phase failed.")
        collection = phase["collection"]
        if collection["all"] != identities or collection["assignments"] != assignments:
            raise ValueError("Collection or lane assignment changed between phases.")
        allowed = {node for node, assignment in assignments.items()
                   if phase["lane"] == "all" or assignment[0] == phase["lane"]}
        if set(collection["selected"]) != allowed:
            raise ValueError("A case was routed to the wrong verification lane.")
        selections.extend(collection["selected"])
        phase_nodes = set()
        for report in phase["reports"]:
            if report["outcome"] == "failed":
                raise ValueError("A test phase failed despite its process exit status.")
            # Inner assertions retain failure authority but are not collected parent identities.
            if report.get("subtest", False):
                continue
            if report["phase"] == "call" or (report["phase"] == "setup" and report["outcome"] == "skipped"):
                terminal.append(report["nodeid"])
                phase_nodes.add(report["nodeid"])
                outcomes[report["outcome"]] += 1
        if phase_nodes != set(collection["selected"]):
            raise ValueError("Selected cases did not all reach a terminal result.")
        if phase["collectionSkips"] != expected["collectionSkips"]:
            raise ValueError("Collection prerequisite skips changed between phases.")
    for label, values in (("selected", selections), ("executed", terminal)):
        if Counter(values) != Counter(identities):
            raise ValueError(f"Missing or duplicate {label} test identities.")
    return {"collected": len(identities), "passed": outcomes["passed"],
            "skippedCases": outcomes["skipped"],
            "collectionSkips": len(expected["collectionSkips"])}


def combine_junit(paths, destination):
    suite = ET.Element("testsuite", name="complete-local-python")
    seen = set()
    for path in paths:
        for case in ET.parse(path).iter("testcase"):
            key = (case.get("classname"), case.get("name"))
            if key in seen:
                skipped = case.find("skipped")
                if skipped is not None and skipped.get("message") == "collection skipped":
                    continue
                raise ValueError("Duplicate executed identity in JUnit receipts.")
            seen.add(key)
            suite.append(case)
    suite.set("tests", str(len(suite)))
    suite.set("failures", str(sum(case.find("failure") is not None for case in suite)))
    suite.set("errors", str(sum(case.find("error") is not None for case in suite)))
    suite.set("skipped", str(sum(case.find("skipped") is not None for case in suite)))
    suite.set("time", str(sum(float(case.get("time", 0)) for case in suite)))
    root = ET.Element("testsuites")
    root.append(suite)
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)


def execute(args):
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source = source_identity()
    policy = args.policy.resolve()
    policy_sha = hashlib.sha256(policy.read_bytes()).hexdigest()
    base = [sys.executable, "-m", "pytest", "-p", "scripts.python_test_lanes",
            "--kwip-policy", str(policy)]
    durations = {}

    def run(label, flags, paths=DIRECTORIES):
        command = [*base, "--kwip-receipt", str(folder / label), *flags, *paths]
        print(f"{label}: starting", flush=True)
        before = time.perf_counter()
        with (folder / (label + ".log")).open("w") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        durations[label] = time.perf_counter() - before
        print(f"{label}: exit {result.returncode}, {durations[label]:.2f}s", flush=True)
        if result.returncode != 0:
            raise RuntimeError(f"{label} failed; see {folder / (label + '.log')}")
        return json.loads((folder / (label + "-master.json")).read_text())

    paths = tuple(args.paths) if args.paths else DIRECTORIES
    expected = run("collection", ["--collect-only", "-q"], paths)
    if args.serial:
        phase = run("serial", ["-q", "--tb=short", "--durations=50",
                    "--junitxml", str(folder / "serial.xml")], paths)
        phases = [phase]
        xml_paths = [folder / "serial.xml"]
    else:
        def empty_phase(lane):
            durations[lane] = 0.0
            print(f"{lane}: no cases assigned", flush=True)
            return {"lane": lane, "exitStatus": 0, "reports": [],
                    "collection": {**expected["collection"], "selected": []},
                    "collectionSkips": expected["collectionSkips"]}

        parallel_nodes = [node for node, value in expected["collection"]["assignments"].items()
                          if value[0] == "parallel"]
        serial_nodes = [node for node, value in expected["collection"]["assignments"].items()
                        if value[0] == "serial"]
        parallel = run("parallel", ["--kwip-lane=parallel", "-n", str(args.workers),
                       "--dist=loadfile", "--no-loadscope-reorder", "--max-worker-restart=0",
                       "-q", "--tb=short", "--durations=50", "--junitxml", str(folder / "parallel.xml")], paths) if parallel_nodes else empty_phase("parallel")
        worker_receipts = [json.loads(path.read_text()) for path in sorted(folder.glob("parallel-gw*.json"))]
        if parallel_nodes and len(worker_receipts) != args.workers:
            raise ValueError("Missing worker collection receipts.")
        # xdist controller receives results; workers own identical collection.
        for worker in worker_receipts:
            if worker["collection"] != worker_receipts[0]["collection"]:
                raise ValueError("Workers disagree on collection or assignments.")
        if worker_receipts:
            parallel["collection"] = worker_receipts[0]["collection"]
        serial = run("serial", ["--kwip-lane=serial", "-q", "--tb=short", "--durations=50",
                     "--junitxml", str(folder / "serial.xml")], paths) if serial_nodes else empty_phase("serial")
        phases = [parallel, serial]
        xml_paths = [path for path in (folder / "parallel.xml", folder / "serial.xml") if path.is_file()]
    summary = audit_coverage(expected, phases)
    if source_identity() != source or hashlib.sha256(policy.read_bytes()).hexdigest() != policy_sha:
        raise ValueError("Source or policy changed during verification; refusing success receipt.")
    combined = folder / "python-tests.xml"
    combine_junit(xml_paths, combined)
    if args.junitxml:
        args.junitxml.write_bytes(combined.read_bytes())
    receipt = {"schemaVersion": 1, "scope": "local Python only; not Swift/native/release acceptance",
               "completeDefaultScope": not bool(args.paths), "directories": list(paths),
               "source": source, "policySHA256": policy_sha, "workers": 0 if args.serial else args.workers,
               "testToolVersions": {name: version(name) for name in ("pytest", "pytest-xdist", "execnet")},
               "summary": summary, "phaseSeconds": durations,
               "wallSeconds": time.perf_counter() - started,
               "completedAt": datetime.now(timezone.utc).isoformat()}
    (folder / "PASSED.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"passed": summary, "wallSeconds": receipt["wallSeconds"], "receipt": str(folder / "PASSED.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--serial", action="store_true", help="Run the complete selection in one ordinary process.")
    parser.add_argument("--policy", type=Path, default=ROOT / "scripts/python_test_policy.json")
    parser.add_argument("--output", type=Path, default=ROOT / ".local/python-verification" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--junitxml", type=Path)
    parser.add_argument("paths", nargs="*", help="Optional focused scope; omit for the complete three directories.")
    args = parser.parse_args()
    (ROOT / ".local").mkdir(exist_ok=True)
    with (ROOT / ".local/python-verification.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another maintained Python verification is already running.")
        try:
            execute(args)
        except (ValueError, RuntimeError) as error:
            print(str(error), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
