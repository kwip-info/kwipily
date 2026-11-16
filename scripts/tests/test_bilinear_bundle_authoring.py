import copy
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("bilinear_bundle",ROOT/"scripts/component_bundles.py")
bundles=importlib.util.module_from_spec(spec);spec.loader.exec_module(bundles)
def source():
    value=json.loads((ROOT/"components/layers/bundle.json").read_text())
    value["components"]=[r for r in value["components"] if r["definition"]["id"]=="linear"]
    row=copy.deepcopy(value["components"][0])
    row["definition"]=next(x for x in json.loads((ROOT/"Sources/KwipMLCore/Resources/components.json").read_text())["components"] if x["id"]=="bilinear")
    row["adapterID"]="kwip.runner.graph-v2.bilinear";row["shapeRuleID"]="kwip.shape.bilinear.v1"
    row["presentation"].update(icon="bilinear-v2",shortTitle="Bilinear")
    row["runtimeContract"]={"version":{"major":1,"minor":0,"patch":0},"adapterID":"kwip.runner.graph-v2.bilinear","graphSchemaVersions":[2],"maximumRank":8,"preservesBatch":True,"minimumSampleRank":1,"sampleBroadcasting":False,"maximumSampleElements":100000000,"maximumParameters":2000000,"featureMinimum":1,"featureMaximum":512,"biasModes":[False,True],"graphDTypes":["float32"]}
    row["sources"]={"native":"Sources/KwipMLCore/BilinearComponent.swift","worker":"runner/bilinear_configuration.py","tests":"runner/tests/test_bilinear_graph.py","evidence":"docs/production/TORCH_BILINEAR_CLOSURE.md"}
    row["learning"].update(exampleID="prepared-bilinear-classifier",exampleSource="Sources/KwipMLCore/BilinearExample.swift")
    value["components"].append(row)
    return value

def test_two_authored_layers_preserve_linear_and_generate_actual_bilinear_factory():
    value=source(); bundles.validate(value,ROOT)
    generated=bundles.render_swift(value,"layers")
    assert "BuiltInCapabilityBridge.linearDeclaration" in generated
    assert "BuiltInCapabilityBridge.bilinearDeclaration" in generated
    assert "WorkflowExamples.preparedBilinearClassifier(device: $0)" in generated
    assert "WorkflowExamples.preparedParallelClassifier(device: $0)" in generated
    bundles.validate({**value,"components":[value["components"][0]]},ROOT)

@pytest.mark.parametrize("mutation",["field","bias","ports","schema","version","floatVersion","broadcast","rank","budget","dtype","feature","config","icon","source","example","resource","duplicate","unknown"])
def test_bilinear_drift_is_refused_without_weakening_linear(mutation):
    value=source();row=value["components"][1]
    if mutation=="field":row["definition"]["fields"][0]["defaultValue"]=3
    elif mutation=="bias":row["definition"]["fields"][3]["defaultValue"]=1
    elif mutation=="ports":row["definition"]["inputs"]=["a"]
    elif mutation=="schema":row["runtimeContract"]["graphSchemaVersions"]=[1,2]
    elif mutation=="version":row["runtimeContract"]["version"]["minor"]=1
    elif mutation=="floatVersion":row["runtimeContract"]["version"]["major"]=1.0
    elif mutation=="broadcast":row["runtimeContract"]["sampleBroadcasting"]=True
    elif mutation=="rank":row["runtimeContract"]["maximumRank"]=9
    elif mutation=="budget":row["runtimeContract"]["maximumParameters"]=2000001
    elif mutation=="dtype":row["runtimeContract"]["graphDTypes"].append("float64")
    elif mutation=="feature":row["runtimeContract"]["featureMaximum"]=513
    elif mutation=="config":row["configurationVersion"]=1
    elif mutation=="icon":row["presentation"]["icon"]="linear-v2"
    elif mutation=="source":row["sources"]["worker"]="../outside.py"
    elif mutation=="example":row["learning"]["exampleID"]="prepared-parallel-classifier"
    elif mutation=="resource":row["learning"]["sourceResource"]="missing"
    elif mutation=="duplicate":value["components"].append(copy.deepcopy(row))
    elif mutation=="unknown":row["definition"]["id"]="invented"
    with pytest.raises(ValueError):bundles.validate(value,ROOT)
