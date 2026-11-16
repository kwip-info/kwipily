import copy
import importlib.util
import json
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("layer_bundles", ROOT / "scripts/component_bundles.py")
bundles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundles)

def source():
    row = copy.deepcopy(json.loads((ROOT / "components/normalization/bundle.json").read_text())["components"][0])
    row["definition"] = {"id":"linear","title":"Linear","category":"Layers","inputs":["in"],"outputs":["out"],"fields":[{"key":"in_features","title":"Input features","defaultValue":2,"min":1,"max":512,"kind":"integer"},{"key":"out_features","title":"Output features","defaultValue":8,"min":1,"max":512,"kind":"integer"},{"key":"bias","title":"Bias","kind":"boolean","defaultValue":True}]}
    row["adapterID"] = "kwip.runner.graph-v2.linear"
    row["shapeRuleID"] = "kwip.shape.linear.v1"
    row["presentation"] = {"theme":"rose","icon":"linear-v2","shortTitle":"Linear","purpose":"Combine input numbers with learned weights, then optionally add a learned offset."}
    row["runtimeContract"] = {"version":{"major":1,"minor":2,"patch":0},"adapterID":"kwip.runner.graph-v2.linear","graphSchemaVersions":[1,2],"maximumRank":8,"leadingAxesMinimum":1,"leadingAxesMaximum":7,"biasModes":[False,True],"legacySchema1Bias":True,"widthMinimum":1,"widthMaximum":512,"preservesBatch":True}
    row["sources"] = {"native":"Sources/KwipMLCore/BuiltInCapabilityBridge.swift","worker":"runner/engine.py","tests":"runner/tests/test_linear_acceptance_matrix.py","evidence":"docs/capability/LINEAR_NUMERICAL_MATRIX.md"}
    row["learning"]["exampleID"] = "prepared-parallel-classifier"
    row["learning"]["exampleSource"] = "Sources/KwipMLCore/WorkflowExamples.swift"
    return {"schemaVersion":1,"id":"kwip.layers","version":1,"backend":"pytorch","components":[row]}

def test_reviewed_layer_authoring_and_generated_launch_contract():
    value = source()
    bundles.validate(value, ROOT)
    rendered = bundles.render_swift(value, "layers")
    assert "public enum BundledLayerComponents" in rendered
    assert "BuiltInCapabilityBridge.linearDeclaration" in rendered
    assert "WorkflowExamples.preparedParallelClassifier(device: $0)" in rendered
    assert "components/layers/bundle.json" in rendered
    assert "public static func definition" in rendered

@pytest.mark.parametrize("mutation", ["schema","backend","typedVersion","config","inputDefault","biasDefault","width","rank","leading","legacy","schemas","biasModes","batch","adapter","unknownRuntime","unknownField","icon","theme","source","missingSource","resource","example","lesson","contentVersion","date","paragraphs","duplicate"])
def test_layer_refuses_contract_type_reference_and_metadata_drift(mutation):
    value = source(); row = value["components"][0]
    if mutation == "schema": value["schemaVersion"] = True
    elif mutation == "backend": value["backend"] = "tensorflow"
    elif mutation == "typedVersion": row["runtimeContract"]["version"]["major"] = 1.0
    elif mutation == "config": row["configurationVersion"] = 1
    elif mutation == "inputDefault": row["definition"]["fields"][0]["defaultValue"] = 2.0
    elif mutation == "biasDefault": row["definition"]["fields"][2]["defaultValue"] = 1
    elif mutation == "width": row["runtimeContract"]["widthMaximum"] = 513
    elif mutation == "rank": row["runtimeContract"]["maximumRank"] = 9
    elif mutation == "leading": row["runtimeContract"]["leadingAxesMinimum"] = 0
    elif mutation == "legacy": row["runtimeContract"]["legacySchema1Bias"] = False
    elif mutation == "schemas": row["runtimeContract"]["graphSchemaVersions"] = [2]
    elif mutation == "biasModes": row["runtimeContract"]["biasModes"] = [True]
    elif mutation == "batch": row["runtimeContract"]["preservesBatch"] = 1
    elif mutation == "adapter": row["adapterID"] = "wrong"
    elif mutation == "unknownRuntime": row["runtimeContract"]["extra"] = True
    elif mutation == "unknownField": row["definition"]["fields"].append({"key":"temperature"})
    elif mutation == "icon": row["presentation"]["icon"] = "linear-v1"
    elif mutation == "theme": row["presentation"]["theme"] = "amber"
    elif mutation == "source": row["sources"]["worker"] = "../outside.py"
    elif mutation == "missingSource": row["sources"]["native"] = "Sources/missing.swift"
    elif mutation == "resource": row["learning"]["sourceResource"] = "../missing"
    elif mutation == "example": row["learning"]["exampleID"] = "invented"
    elif mutation == "lesson": row["learning"] = None
    elif mutation == "contentVersion": row["learning"]["contentVersion"] = True
    elif mutation == "date": row["learning"]["updatedAt"] = "not-a-date"
    elif mutation == "paragraphs": row["learning"]["paragraphs"] = [""]
    elif mutation == "duplicate": value["components"].append(copy.deepcopy(row))
    with pytest.raises(ValueError): bundles.validate(value, ROOT)
