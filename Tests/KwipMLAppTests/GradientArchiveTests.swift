import Foundation
import XCTest
import KwipMLCore
@testable import KwipMLApp

final class GradientArchiveTests:XCTestCase {
    private let pin=String(repeating:"a",count:64)
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    private func plan(selected:Bool=true,steps:Int=2,initialize:Bool=false)throws->WorkflowExecutionPlan {
        var document=try WorkflowExamples.preparedParallelClassifier(device:"cpu");document.schemaVersion=8
        var p=try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:.load(from:repository.appendingPathComponent("runner/dataflow-operators.json")),contractVersion:7)
        if selected {
            let input=try XCTUnwrap(p.model.graph.nodes.first {$0.type == "input"}),output=try XCTUnwrap(p.model.graph.nodes.first {$0.type == "output"}),owner=try XCTUnwrap(p.model.graph.nodes.first {$0.type == "linear"})
            let request:JSONValue = .object(["id":.string("effect"),"output":.object(["node":.string(output.id),"port":.string("out")]),"partition":.string("validation"),"row_start":0,"row_count":2,"reduction":.string("mean"),"upstream":.null,"targets":.array([.object(["kind":.string("input"),"node":.string(input.id),"port":.string("out")]),.object(["kind":.string("parameter"),"node":.string(owner.id),"parameter":.string("weight")])]),"unavailable":.string("reportUnavailable")])
            p.gradientPolicyRecipe = .init(configuration:try .init(values:["accumulation_steps":.integer(Int64(steps)),"requests":.array([request])]),source:.init(domain:.control,localID:"gradient-policy"))
            p.trainabilityRecipe = .init(configuration:try .init(values:["owners":.array([.object(["node":.string(owner.id),"parameter":.string("weight"),"mode":.string("frozen")])])]),source:.init(domain:.control,localID:"trainability"))
            if initialize {p.initializationRecipe = .init(configuration:try .init(values:["seed":42,"rules":.array([.object(["owner":.object(["node":.string(owner.id),"parameter":.string("weight")]),"algorithm":.string("normal"),"settings":.object(["mean":0,"std":0.2])])])]),source:.init(domain:.control,localID:"initialization"))}
        }
        return p
    }
    private func capture(_ p:WorkflowExecutionPlan,schema:Int=10,compiler:Int=9)throws->WorkflowArchiveCapture {
        func file(_ name:String)->WorkflowArchiveFile {.init(path:name,bytes:10,sha256:pin)}
        var captured = WorkflowArchiveCapture(schemaVersion:schema,compilerContractVersion:compiler,projectID:p.projectID,directoryID:UUID().uuidString,runID:UUID().uuidString,requestIdentity:pin,revision:p.revision,train:p.train,workflow:file("workflow.json"),modelRegistry:file("model-registry.json"),dataRegistry:file("data-registry.json"),modelRegistrySHA256:pin,dataRegistrySHA256:pin,runtimeEvidence:file("runtime-evidence.json"),sources:p.sourceAddresses.enumerated().map {.init(address:$0.element,path:"inputs/source-\($0.offset).csv",bytes:10,sha256:pin)},plan:try WorkflowArchivePlan(p,compilerContractVersion:compiler))
        if p.gradientPolicyRecipe != nil {
            captured.captureFormatVersion=5
            captured.executionBudget=try ExecutionPreferences.factory.resolve().budget
            captured.derivativeCapture = .init(values:true)
        }
        return captured
    }
    private func object<T:Encodable>(_ value:T)throws->[String:Any] {try XCTUnwrap(JSONSerialization.jsonObject(with:WorkflowArchiveCodec.encode(value)) as? [String:Any])}
    func testSchemaTenRetainsBothQualifiedPoliciesAndInitialization()throws {
        let p=try plan(initialize:true),archived=try WorkflowArchivePlan(p,compilerContractVersion:9)
        let fields=try object(archived)
        for key in ["gradientPolicyRecipe","trainabilityRecipe","initializationRecipe"] {
            let wrapper=try XCTUnwrap(fields[key] as? [String:Any]);XCTAssertEqual(Set(wrapper.keys),["source","configuration"])
            XCTAssertEqual((wrapper["source"] as? [String:Any])?["domain"] as? String,"control")
        }
        let restored=try JSONDecoder().decode(WorkflowArchivePlan.self,from:WorkflowArchiveCodec.encode(archived))
        XCTAssertTrue(try restored.matches(p,compilerContractVersion:9));XCTAssertFalse(try restored.matches(plan(steps:3,initialize:true),compilerContractVersion:9))
        let value=try capture(p);XCTAssertEqual(try WorkflowArchiveCodec.decodeCapture(WorkflowArchiveCodec.encodeCapture(value)),value)
    }
    func testLegacyCompilersCannotSilentlyDropSelectedPolicies()throws {
        let selected=try plan()
        for version in 1...8 {XCTAssertThrowsError(try WorkflowArchivePlan(selected,compilerContractVersion:version),"compiler \(version)")}
    }
    func testAbsentLegacyFieldsRemainOmittedAndCanonicalBytesRoundTrip()throws {
        let implicit=try plan(selected:false)
        for pair in [(5,4),(6,5),(7,6),(8,7),(9,8)] {
            let value=try capture(implicit,schema:pair.0,compiler:pair.1),data=try WorkflowArchiveCodec.encodeCapture(value)
            let text=String(decoding:data,as:UTF8.self)
            XCTAssertFalse(text.contains("gradientPolicyRecipe"));XCTAssertFalse(text.contains("trainabilityRecipe"))
            XCTAssertEqual(try WorkflowArchiveCodec.encodeCapture(WorkflowArchiveCodec.decodeCapture(data)),data)
        }
    }
    func testWrongSourcesAndCrossControlAliasesRefuse()throws {
        let p=try plan()
        for domain in [WorkflowDomain.model,.data] {
            var bad=p;bad.gradientPolicyRecipe = .init(configuration:p.gradientPolicyRecipe!.configuration,source:.init(domain:domain,localID:"wrong"))
            XCTAssertThrowsError(try WorkflowArchivePlan(bad,compilerContractVersion:9))
        }
        for source in [p.train,p.trainabilityRecipe!.source] {
            var bad=p;bad.gradientPolicyRecipe = .init(configuration:p.gradientPolicyRecipe!.configuration,source:source)
            XCTAssertThrowsError(try WorkflowArchivePlan(bad,compilerContractVersion:9))
        }
    }
    func testTamperedWrappersConfigurationOwnersAndVersionsRefuse()throws {
        let valid=try object(capture(plan()))
        for mutation in ["foreignWrapper","sourceField","wrongKind","missingControl","foreignOwner","lower","future","duplicateSource","blankPort","wrongInputKind"] {
            var raw=valid,plan=try XCTUnwrap(raw["plan"] as? [String:Any]),recipe=try XCTUnwrap(plan["gradientPolicyRecipe"] as? [String:Any])
            if mutation=="foreignWrapper" {recipe["foreign"]=true}
            if mutation=="sourceField" {var source=recipe["source"] as! [String:Any];source["foreign"]=true;recipe["source"]=source}
            if mutation=="wrongKind" {var c=recipe["configuration"] as! [String:Any];c["componentID"]="trainability";recipe["configuration"]=c}
            if mutation=="missingControl" {var c=recipe["configuration"] as! [String:Any],v=c["config"] as! [String:Any];v.removeValue(forKey:"gradient_scale");c["config"]=v;recipe["configuration"]=c}
            if mutation=="foreignOwner" {var c=recipe["configuration"] as! [String:Any],v=c["config"] as! [String:Any],requests=v["requests"] as! [[String:Any]],targets=requests[0]["targets"] as! [[String:Any]];targets[1]["node"]="outside-captured-model";requests[0]["targets"]=targets;v["requests"]=requests;c["config"]=v;recipe["configuration"]=c}
            if mutation=="blankPort" || mutation=="wrongInputKind" {
                var c=recipe["configuration"] as! [String:Any],v=c["config"] as! [String:Any],requests=v["requests"] as! [[String:Any]]
                if mutation=="blankPort" {var output=requests[0]["output"] as! [String:Any];output["port"]="";requests[0]["output"]=output}
                else {var targets=requests[0]["targets"] as! [[String:Any]];targets[0]["node"]=(requests[0]["output"] as! [String:Any])["node"];requests[0]["targets"]=targets}
                v["requests"]=requests;c["config"]=v;recipe["configuration"]=c
            }
            if mutation=="lower" {raw["schemaVersion"]=9;raw["compilerContractVersion"]=8}
            if mutation=="future" {raw["schemaVersion"]=11;raw["compilerContractVersion"]=10}
            if mutation=="duplicateSource" {recipe["source"]=(plan["trainabilityRecipe"] as! [String:Any])["source"]}
            plan["gradientPolicyRecipe"]=recipe;raw["plan"]=plan
            XCTAssertThrowsError(try WorkflowArchiveCodec.decodeCapture(JSONSerialization.data(withJSONObject:raw)),mutation)
        }
    }
}
