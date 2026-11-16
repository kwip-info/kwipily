import copy
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("lazy_layers",ROOT/"scripts/component_bundles.py")
bundles=importlib.util.module_from_spec(spec);spec.loader.exec_module(bundles)
def source():
    value=json.loads((ROOT/"components/layers/bundle.json").read_text())
    value["components"]=[r for r in value["components"] if r["definition"]["id"]!="lazy_linear"]
    row=copy.deepcopy(value["components"][0])
    row["definition"]=next(x for x in json.loads((ROOT/"Sources/KwipMLCore/Resources/components.json").read_text())["components"] if x["id"]=="lazy_linear")
    row["adapterID"]="kwip.runner.graph-v2.lazy-linear";row["shapeRuleID"]="kwip.shape.lazy-linear.v1"
    row["presentation"].update(icon="lazy_linear-v2",shortTitle="Lazy Linear")
    row["runtimeContract"]={"version":{"major":1,"minor":0,"patch":0},"adapterID":"kwip.runner.graph-v2.lazy-linear","graphSchemaVersions":[2],"maximumRank":8,"preservesBatch":True,"minimumSampleRank":1,"maximumSampleElements":100000000,"maximumParameters":2000000,"inputFeatureMinimum":1,"inputFeatureMaximum":4096,"outputFeatureMinimum":1,"outputFeatureMaximum":512,"biasModes":[False,True],"graphDTypes":["float32"],"inputWidthInferred":True,"materializationPolicy":"validatedShapeBeforeOptimizer"}
    row["sources"]={"native":"Sources/KwipMLCore/LazyLinearComponent.swift","worker":"runner/lazy_linear_configuration.py","tests":"runner/tests/test_lazy_linear_graph.py","evidence":"docs/production/TORCH_LAZY_LINEAR.md"}
    row["learning"].update(exampleID="prepared-lazy-linear-classifier",exampleSource="Sources/KwipMLCore/LazyLinearWorkflowExample.swift")
    value["components"].append(row);return value

def test_three_layers_preserve_existing_contracts_and_actual_lazy_factory():
    value=source();bundles.validate(value,ROOT);rendered=bundles.render_swift(value,"layers")
    for declaration in ["linearDeclaration","bilinearDeclaration","lazyLinearDeclaration"]:assert "BuiltInCapabilityBridge."+declaration in rendered
    assert "WorkflowExamples.preparedLazyLinearClassifier(device: $0)" in rendered
    bundles.validate({**value,"components":value["components"][:2]},ROOT)

@pytest.mark.parametrize("mutation",["inputField","output","bias","infer","inputLimit","outputLimit","policy","budget","schema","typedVersion","dtype","rank","icon","adapter","source","example","unknown"])
def test_lazy_configuration_runtime_and_provenance_drift_refused(mutation):
    value=source();row=value["components"][-1]
    if mutation=="inputField":row["definition"]["fields"].append({"key":"in_features"})
    elif mutation=="output":row["definition"]["fields"][0]["max"]=513
    elif mutation=="bias":row["runtimeContract"]["biasModes"]=[True]
    elif mutation=="infer":row["runtimeContract"]["inputWidthInferred"]=False
    elif mutation=="inputLimit":row["runtimeContract"]["inputFeatureMaximum"]=4097
    elif mutation=="outputLimit":row["runtimeContract"]["outputFeatureMaximum"]=513
    elif mutation=="policy":row["runtimeContract"]["materializationPolicy"]="firstUnvalidatedForward"
    elif mutation=="budget":row["runtimeContract"]["maximumParameters"]=2000001
    elif mutation=="schema":row["runtimeContract"]["graphSchemaVersions"]=[1,2]
    elif mutation=="typedVersion":row["runtimeContract"]["version"]["major"]=1.0
    elif mutation=="dtype":row["runtimeContract"]["graphDTypes"]=["float64"]
    elif mutation=="rank":row["runtimeContract"]["maximumRank"]=9
    elif mutation=="icon":row["presentation"]["icon"]="linear-v2"
    elif mutation=="adapter":row["adapterID"]="kwip.runner.graph-v2.lazy_linear"
    elif mutation=="source":row["sources"]["worker"]="../outside.py"
    elif mutation=="example":row["learning"]["exampleID"]="prepared-parallel-classifier"
    elif mutation=="unknown":row["definition"]["id"]="unreviewed_lazy"
    with pytest.raises(ValueError):bundles.validate(value,ROOT)
