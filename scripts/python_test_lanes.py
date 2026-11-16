"""Opt-in, complete-coverage local pytest lanes; ordinary pytest is unchanged.

Reviewed private-path CPU tests may run in two processes. Actual devices,
measurements and process-lifetime contracts run afterward in a serial process.
New/edited test sources or imported test helpers default to serial.
"""
import ast
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from time import perf_counter

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEST_ROOTS = ("runner/tests/", "scripts/tests/", "scripts/runtime_tests/")
_open = open
_dumps = json.dumps
_reports = []
_collection_skips = {}
_collection = None
_config = None
_times = {"pluginLoaded": perf_counter()}


def test_source(path):
    return str(path).startswith(TEST_ROOTS) and str(path).endswith(".py")


@lru_cache(maxsize=None)
def dependencies(relative):
    """Transitive imports of local test helpers, including imports inside tests."""
    found = set()
    tree = ast.parse((ROOT / relative).read_text())
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package = relative.removesuffix(".py").split("/")[:-1]
                ascend = node.level - 1
                if ascend >= len(package):
                    continue
                base = package[:len(package) - ascend]
                if node.module:
                    base.extend(node.module.split("."))
                module = ".".join(base)
            else:
                module = node.module or ""
            names = [module, *(module + "." + a.name for a in node.names)]
        for name in names:
            path = name.replace(".", "/") + ".py"
            if test_source(path) and (ROOT / path).is_file():
                found.add(path)
    return found


def source_digest(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def source_is_reviewed(relative, reviewed, fixture_dependencies=None):
    pending = [relative]
    seen = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        if reviewed.get(path) != source_digest(path):
            return False
        declared = set((fixture_dependencies or {}).get(path, []))
        pending.extend((dependencies(path) | declared) - seen)
    return True


def classify(nodeid, policy, reviewed_status=None):
    path = nodeid.split("::", 1)[0]
    selector = nodeid.split("[", 1)[0]
    if path in policy["serialFiles"]:
        return "serial", policy["serialFiles"][path]
    if selector in policy["serialTests"]:
        return "serial", policy["serialTests"][selector]
    if "mps" in nodeid.lower() or "cuda" in nodeid.lower():
        return "serial", "accelerator case (including conservative refusal cases)"
    reviewed = (source_is_reviewed(path, policy["reviewedSources"], policy.get("fixtureDependencies", {})) if reviewed_status is None
                else reviewed_status.get(path, False))
    if not reviewed:
        return "serial", "new or changed test source/helper requires review"
    return "parallel", "reviewed private-path CPU correctness case"


def pytest_addoption(parser):
    group = parser.getgroup("kwip-local-verification")
    group.addoption("--kwip-lane", choices=("all", "parallel", "serial"), default="all")
    group.addoption("--kwip-policy", type=Path)
    group.addoption("--kwip-receipt", type=Path)


def pytest_configure(config):
    global _config
    _config = config
    if config.getoption("kwip_policy") is None:
        raise pytest.UsageError("The local lane plugin requires --kwip-policy.")
    filters = ("keyword", "markexpr", "deselect", "ignore", "ignore_glob",
               "lf", "stepwise", "stepwise_skip", "stepwise_reset", "pyargs")
    active = [name for name in filters if config.getoption(name, default=None)]
    if active:
        raise pytest.UsageError("Local verification requires explicit paths without pytest selection filters: "
                                + ", ".join(active))


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    global _collection
    policy = json.loads(config.getoption("kwip_policy").read_text())
    reviewed_status = {path: source_is_reviewed(path, policy["reviewedSources"], policy.get("fixtureDependencies", {}))
                       for path in {item.nodeid.split("::", 1)[0] for item in items}}
    assignments = {item.nodeid: classify(item.nodeid, policy, reviewed_status) for item in items}
    if len(assignments) != len(items):
        raise pytest.UsageError("Duplicate test identities cannot establish coverage.")
    paths = {item.nodeid.split("::", 1)[0] for item in items}
    selectors = {item.nodeid.split("[", 1)[0] for item in items}
    stale = {rule for rule in policy["serialTests"]
             if rule.split("::", 1)[0] in paths and rule not in selectors}
    if stale:
        raise pytest.UsageError("Stale serial test selectors require review: " + ", ".join(sorted(stale)))
    lane = config.getoption("kwip_lane")
    selected = [item for item in items if lane == "all" or assignments[item.nodeid][0] == lane]
    selected_ids = {item.nodeid for item in selected}
    deselected = [item for item in items if item.nodeid not in selected_ids]
    _collection = {"all": [item.nodeid for item in items],
                   "selected": [item.nodeid for item in selected],
                   "assignments": assignments}
    # Start expensive files early. xdist loadfile keeps each module's fixtures
    # in one worker; --no-loadscope-reorder preserves this measured ordering.
    if lane == "parallel":
        selected.sort(key=lambda item: -policy["fileSeconds"].get(item.nodeid.split("::", 1)[0], 0))
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)


def pytest_collection_finish(session):
    _times["collectionFinished"] = perf_counter()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtestloop(session):
    _times["loopStarted"] = perf_counter()
    yield
    _times["loopFinished"] = perf_counter()


def pytest_collectreport(report):
    if report.skipped:
        _collection_skips[report.nodeid] = str(report.longrepr)


def pytest_runtest_logreport(report):
    _reports.append({"nodeid": report.nodeid, "phase": report.when,
                     "outcome": report.outcome, "seconds": report.duration,
                     "subtest": hasattr(report, "context")})


def pytest_sessionfinish(session, exitstatus):
    prefix = session.config.getoption("kwip_receipt")
    if prefix is None:
        return
    worker = getattr(session.config, "workerinput", {}).get("workerid", "master")
    path = prefix.with_name(prefix.name + "-" + worker + ".json")
    value = {"schemaVersion": 1, "lane": session.config.getoption("kwip_lane"),
             "exitStatus": int(exitstatus), "collection": _collection,
             "collectionSkips": _collection_skips, "reports": _reports,
             "timingSeconds": {name: stamp - _times["pluginLoaded"] for name, stamp in _times.items()}}
    with _open(path, "w") as stream:
        stream.write(_dumps(value) + "\n")
