import copy
import importlib.util
import json
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("tensor_layout_bundles", ROOT / "scripts/component_bundles.py")
bundles = importlib.util.module_from_spec(spec); spec.loader.exec_module(bundles)
IDS = ["reshape", "transpose", "flatten"]
def source():
    definitions = json.loads((ROOT / "Sources/KwipMLCore/Resources/components.json").read_text())["components"]
    template = json.loads((ROOT / "components/layers/bundle.json").read_text())["components"][0]
    rows = []
    for ident in IDS:
        row = copy.deepcopy(template)
        row["definition"] = next(x for x in definitions if x["id"] == ident)
        row["adapterID"] = "kwip.runner.graph-v2." + ident
        row["shapeRuleID"] = "kwip.shape." + ident + ".v1"
        row["presentation"].update(theme="violet", icon=ident+"-v2", shortTitle=ident.capitalize())
        row["runtimeContract"] = {"version":{"major":1,"minor":0,"patch":0},"adapterID":"kwip.runner.graph-v2.tensor-layout","graphSchemaVersions":[2],"operations":IDS,"maximumRank":8,"preservesBatch":True}
        row["sources"] = {"native":"Sources/KwipMLCore/TensorTransformConfiguration.swift","worker":"runner/tensor_layout.py","tests":"runner/tests/test_tensor_layout.py","evidence":"docs/production/TORCH_TENSOR_LAYOUT.md"}
        row["learning"]["exampleID"] = "prepared-tensor-layout-classifier"
        row["learning"]["exampleSource"] = "Sources/KwipMLCore/WorkflowExamples.swift"
        rows.append(row)
    return {"schemaVersion":1,"id":"kwip.tensor-layout","version":1,"backend":"pytorch","components":rows}

def test_exact_family_validation_generation_and_all_three_launchers():
    value = source(); bundles.validate(value, ROOT)
    rendered = bundles.render_swift(value, "tensor-layout")
    assert "public enum BundledTensorLayoutComponents" in rendered
    assert rendered.count("WorkflowExamples.preparedTensorLayoutClassifier(device: $0)") == 3
    for ident in IDS: assert "BuiltInCapabilityBridge."+ident+"Declaration" in rendered
    assert "components/tensor-layout/bundle.json" in rendered

@pytest.mark.parametrize("mutation", ["schema","version","definition","default","port","configuration","adapter","shapeRule","runtimeVersion","runtimeOps","rank","batch","theme","icon","reference","example","resource","date","paragraph","missing","duplicate"])
def test_family_refuses_types_configuration_discovery_metadata_and_reference_drift(mutation):
    value=source(); row=value["components"][0]
    if mutation=="schema": value["schemaVersion"]=True
    elif mutation=="version": value["version"]=1.0
    elif mutation=="definition": row["definition"]["fields"].append({"key":"extra"})
    elif mutation=="default": row["definition"]["fields"][0]["defaultValue"]=[-1.0]
    elif mutation=="port": row["definition"]["outputs"].append("extra")
    elif mutation=="configuration": row["configurationVersion"]=1
    elif mutation=="adapter": row["adapterID"]="wrong"
    elif mutation=="shapeRule": row["shapeRuleID"]="wrong"
    elif mutation=="runtimeVersion": row["runtimeContract"]["version"]["major"]=1.0
    elif mutation=="runtimeOps": row["runtimeContract"]["operations"]=["reshape"]
    elif mutation=="rank": row["runtimeContract"]["maximumRank"]=9
    elif mutation=="batch": row["runtimeContract"]["preservesBatch"]=1
    elif mutation=="theme": row["presentation"]["theme"]="rose"
    elif mutation=="icon": row["presentation"]["icon"]="transpose-v2"
    elif mutation=="reference": row["sources"]["worker"]="../outside.py"
    elif mutation=="example": row["learning"]["exampleID"]="invented"
    elif mutation=="resource": row["learning"]["sourceResource"]="missing"
    elif mutation=="date": row["learning"]["updatedAt"]="not-a-date"
    elif mutation=="paragraph": row["learning"]["paragraphs"]=[""]
    elif mutation=="missing": value["components"].pop()
    elif mutation=="duplicate": value["components"].append(copy.deepcopy(row))
    with pytest.raises(ValueError): bundles.validate(value, ROOT)
