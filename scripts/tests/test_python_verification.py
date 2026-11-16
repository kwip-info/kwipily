"""Local scheduling never turns missing, duplicated or misrouted tests into a pass."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
from scripts import python_test_lanes as lanes
from scripts.verify_python import audit_coverage, combine_junit


@pytest.fixture
def reviewed(tmp_path, monkeypatch):
    monkeypatch.setattr(lanes, "ROOT", tmp_path)
    lanes.dependencies.cache_clear()
    directory = tmp_path / "runner/tests"
    directory.mkdir(parents=True)
    source = directory / "test_cpu.py"
    helper = directory / "fixture.py"
    source.write_text("from runner.tests.fixture import make\ndef test_value(): pass\n")
    helper.write_text("def make(): return 1\n")
    policy = {"serialFiles": {}, "serialTests": {}, "reviewedSources": {
        str(path.relative_to(tmp_path)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (source, helper)}}
    yield tmp_path, policy
    lanes.dependencies.cache_clear()


def test_reviewed_cpu_case_and_new_source_default(reviewed):
    root, policy = reviewed
    assert lanes.classify("runner/tests/test_cpu.py::test_value", policy)[0] == "parallel"
    new = root / "runner/tests/test_new.py"
    new.write_text("def test_new(): pass\n")
    assert lanes.classify("runner/tests/test_new.py::test_new", policy)[0] == "serial"


@pytest.mark.parametrize("path", ["test_cpu.py", "fixture.py"])
def test_changed_test_or_imported_fixture_defaults_to_serial(reviewed, path):
    root, policy = reviewed
    (root / "runner/tests" / path).write_text("# changed execution requirements\n")
    assert lanes.classify("runner/tests/test_cpu.py::test_value", policy)[0] == "serial"


def test_transitive_fixture_changes_cannot_keep_parallel_approval(reviewed):
    root, policy = reviewed
    helper = root / "runner/tests/fixture.py"
    helper.write_text("from runner.tests.nested import make\n")
    nested = root / "runner/tests/nested.py"
    nested.write_text("def make(): return 1\n")
    for path in (helper, nested):
        policy["reviewedSources"][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert lanes.classify("runner/tests/test_cpu.py::test_value", policy)[0] == "parallel"
    nested.write_text("def make(): return 'new resource use'\n")
    assert lanes.classify("runner/tests/test_cpu.py::test_value", policy)[0] == "serial"


@pytest.mark.parametrize("suffix", ["[mps]", "[cuda]", "[mps-refusal]"])
def test_accelerator_parameters_remain_serial(reviewed, suffix):
    _, policy = reviewed
    assert lanes.classify("runner/tests/test_cpu.py::test_value" + suffix, policy)[0] == "serial"


def test_implicit_device_and_resource_exceptions_override_cpu_approval(reviewed):
    _, policy = reviewed
    node = "runner/tests/test_cpu.py::test_value"
    policy["serialTests"][node] = "implicit auto-device math"
    assert lanes.classify(node, policy) == ("serial", "implicit auto-device math")
    policy["serialTests"].clear()
    policy["serialFiles"]["runner/tests/test_cpu.py"] = "measured lifetime"
    assert lanes.classify(node, policy) == ("serial", "measured lifetime")


def records():
    identities = ["file.py::cpu", "file.py::gpu", "file.py::optional"]
    assignments = {node: ("parallel" if node.endswith("cpu") else "serial", "reviewed") for node in identities}
    collection = {"all": identities, "selected": identities, "assignments": assignments}
    expected = {"collection": collection, "collectionSkips": {"optional_module.py": "prerequisite absent"}}
    phases = []
    for lane in ("parallel", "serial"):
        selected = [node for node in identities if assignments[node][0] == lane]
        reports = [{"nodeid": node, "phase": "setup" if node.endswith("optional") else "call",
                    "outcome": "skipped" if node.endswith("optional") else "passed"} for node in selected]
        phases.append({"lane": lane, "exitStatus": 0,
                       "collection": {**collection, "selected": selected},
                       "reports": reports, "collectionSkips": expected["collectionSkips"]})
    return expected, phases


def test_exact_union_counts_optional_prerequisites_without_hiding_cases():
    expected, phases = records()
    assert audit_coverage(expected, phases) == {"collected": 3, "passed": 2,
                                               "skippedCases": 1, "collectionSkips": 1}


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "misroute", "changed", "failed", "skips"])
def test_receipt_refuses_incomplete_or_unsafe_verification(corruption):
    expected, phases = copy.deepcopy(records())
    if corruption == "missing":
        phases[1]["reports"].pop()
    elif corruption == "duplicate":
        phases[0]["reports"].append(dict(phases[0]["reports"][0]))
    elif corruption == "misroute":
        phases[0]["collection"]["selected"].append("file.py::gpu")
    elif corruption == "changed":
        phases[1]["collection"]["all"] = [*expected["collection"]["all"], "new.py::case"]
    elif corruption == "failed":
        phases[0]["reports"][0]["outcome"] = "failed"
    else:
        phases[1]["collectionSkips"] = {}
    with pytest.raises(ValueError):
        audit_coverage(expected, phases)


def test_junit_deduplicates_only_collection_skips(tmp_path):
    first = tmp_path / "first.xml"
    second = tmp_path / "second.xml"
    skip = '<testcase classname="" name="missing"><skipped message="collection skipped"/></testcase>'
    first.write_text('<testsuite><testcase classname="file" name="cpu"/>' + skip + '</testsuite>')
    second.write_text('<testsuite><testcase classname="file" name="gpu"/>' + skip + '</testsuite>')
    target = tmp_path / "combined.xml"
    combine_junit([first, second], target)
    assert target.read_text().count('<testcase ') == 3
    second.write_text(first.read_text())
    with pytest.raises(ValueError, match="Duplicate executed"):
        combine_junit([first, second], target)


@pytest.mark.parametrize("option,value", [
    ("keyword", "test_value"), ("markexpr", "cpu"),
    ("deselect", ["file.py::case"]), ("ignore", ["file.py"]),
    ("ignore_glob", ["*gpu*"]), ("lf", True),
    ("stepwise", True), ("stepwise_skip", True), ("stepwise_reset", True),
    ("pyargs", True),
])
def test_lane_plugin_refuses_effective_selection_filters(option, value):
    class Config:
        def getoption(self, name, default=None):
            return {"kwip_policy": Path("policy.json"), option: value}.get(name, default)
    with pytest.raises(pytest.UsageError, match=option):
        lanes.pytest_configure(Config())


def test_lane_plugin_accepts_explicit_path_selection():
    class Config:
        def getoption(self, name, default=None):
            return Path("policy.json") if name == "kwip_policy" else default
    lanes.pytest_configure(Config())


@pytest.mark.parametrize("statement", ["from .fixture import make", "from . import fixture",
                                        "from ..tests.fixture import make"])
def test_relative_helper_changes_default_to_serial(reviewed, statement):
    root, policy = reviewed
    source = root / "runner/tests/test_cpu.py"
    source.write_text(statement + "\ndef test_value(): pass\n")
    policy["reviewedSources"]["runner/tests/test_cpu.py"] = hashlib.sha256(source.read_bytes()).hexdigest()
    assert lanes.classify("runner/tests/test_cpu.py::test_value", policy)[0] == "parallel"
    (root / "runner/tests/fixture.py").write_text("# changed resource requirements\n")
    assert lanes.classify("runner/tests/test_cpu.py::test_value", policy)[0] == "serial"


def test_declared_dynamic_helper_changes_default_to_serial(reviewed):
    root, policy = reviewed
    source = root / "runner/tests/test_cpu.py"
    source.write_text("# fixture is loaded dynamically\ndef test_value(): pass\n")
    policy["reviewedSources"]["runner/tests/test_cpu.py"] = hashlib.sha256(source.read_bytes()).hexdigest()
    policy["fixtureDependencies"] = {"runner/tests/test_cpu.py": ["runner/tests/fixture.py"]}
    assert lanes.classify("runner/tests/test_cpu.py::test_value", policy)[0] == "parallel"
    (root / "runner/tests/fixture.py").write_text("# changed dynamic resource requirements\n")
    assert lanes.classify("runner/tests/test_cpu.py::test_value", policy)[0] == "serial"


def test_subtest_reports_do_not_duplicate_parent_but_failed_subtest_refuses_success():
    expected, phases = records()
    subtest = {**phases[0]["reports"][0], "subtest": True}
    phases[0]["reports"].insert(0, subtest)
    assert audit_coverage(expected, phases)["collected"] == 3
    subtest["outcome"] = "failed"
    with pytest.raises(ValueError, match="failed"):
        audit_coverage(expected, phases)


def test_lane_receipt_distinguishes_subtests_from_their_parent(monkeypatch):
    from types import SimpleNamespace
    reports = []
    with monkeypatch.context() as patch:
        patch.setattr(lanes, "_reports", reports)
        for extra in ({}, {"context": SimpleNamespace(msg="inner", kwargs={})}):
            lanes.pytest_runtest_logreport(SimpleNamespace(nodeid="case", when="call", outcome="passed", duration=0.1, **extra))
    assert reports[0].get("subtest", False) is False
    assert reports[1].get("subtest", False) is True
