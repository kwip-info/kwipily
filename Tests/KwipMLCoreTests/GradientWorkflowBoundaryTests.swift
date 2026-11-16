import Foundation
import XCTest
@testable import KwipMLCore

/// Exercise the saved public wire, rather than needing new payload cases just to RED.
final class GradientWorkflowBoundaryTests: XCTestCase {
    private func registries() throws -> (ComponentRegistry, DataFlowOperatorRegistry) {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try (.builtIn(), .load(from:root.appendingPathComponent("runner/dataflow-operators.json")))
    }
    private func compile(_ document: WorkflowDocument, version: Int) throws -> WorkflowExecutionPlan {
        let (model,data)=try registries()
        return try WorkflowCompiler.compileHistoricalEvidence(document,train:WorkflowExamples.trainAddress,modelRegistry:model,dataRegistry:data,contractVersion:version)
    }
    private func wire(schema: Int = 10, wrongKind: Bool = false, duplicate: Bool = false) throws -> Data {
        let old=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        let plan=try compile(old,version:2)
        let owner=try XCTUnwrap(plan.model.graph.nodes.first {$0.type == "linear"})
        var root=try XCTUnwrap(JSONSerialization.jsonObject(with:WorkflowCodec.encode(old)) as? [String:Any])
        root["schemaVersion"]=schema
        var nodes=try XCTUnwrap(root["nodes"] as? [[String:Any]])
        var edges=try XCTUnwrap(root["edges"] as? [[String:Any]])
        let specifications: [(String,String,[String:Any],String)] = [
            ("gradient-policy","gradientPolicyRecipe",["accumulation_steps":2,"gradient_scale":"mean","requests":[]],"gradientPolicy"),
            ("trainability-policy","trainabilityRecipe",["owners":[["node":owner.id,"parameter":"weight","mode":"frozen"]]],"trainability")
        ]
        for (index,item) in specifications.enumerated() {
            let type=index == 0 ? "gradient_policy" : "trainability"
            let address:[String:Any]=["domain":"control","localID":item.0]
            nodes.append(["address":address,"name":"User labelled policy","x":0,"y":0,"payload":["kind":item.1,"type":type,"configurationVersion":1,"config":item.2]])
            let edge:[String:Any]=["id":"policy-wire-\(index)","from":["address":address,"port":"out"],"to":["address":["domain":WorkflowExamples.trainAddress.domain.rawValue,"localID":WorkflowExamples.trainAddress.localID],"port":item.3],"kind":wrongKind && index == 0 ? "optimizerRecipe" : item.1]
            edges.append(edge)
            if duplicate && index == 0 {var second=edge;second["id"]="duplicate-policy-wire";edges.append(second)}
        }
        root["nodes"]=nodes;root["edges"]=edges
        return try JSONSerialization.data(withJSONObject:root,options:[.sortedKeys])
    }
    func testSchema10RetainsBothControlRecipesThroughCodecAndCompilation() throws {
        let document=try WorkflowCodec.decode(wire())
        XCTAssertEqual(document.schemaVersion,10)
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(document)),document)
        let plan=try compile(document,version:9)
        let original=try compile(WorkflowExamples.preparedParallelClassifier(device:"cpu"),version:2)
        XCTAssertEqual(plan.model.graph,original.model.graph)
        XCTAssertEqual(plan.settings,original.settings)
        let saved=String(decoding:try WorkflowCodec.encode(document),as:UTF8.self)
        XCTAssertTrue(saved.contains("gradientPolicyRecipe"));XCTAssertTrue(saved.contains("trainabilityRecipe"))
        XCTAssertFalse(plan.model.graph.nodes.contains {$0.type == "gradient_policy" || $0.type == "trainability"})
    }
    func testPolicySchemaCompilerAndTypedInputBoundariesRefuse() throws {
        XCTAssertThrowsError(try WorkflowCodec.decode(wire(schema:9)))
        XCTAssertThrowsError(try WorkflowCodec.decode(wire(schema:18)))
        let document=try WorkflowCodec.decode(wire())
        for version in [2,6,7,8,17] {XCTAssertThrowsError(try compile(document,version:version))}
        XCTAssertThrowsError(try compile(WorkflowCodec.decode(wire(wrongKind:true)),version:9))
        XCTAssertThrowsError(try WorkflowCodec.decode(wire(duplicate:true)))
    }
    func testSupportedCompositionVersionsRetainPoliciesWithoutChangingTheModel() throws {
        let document = try WorkflowCodec.decode(wire())
        let baseline = try compile(document, version: 9)
        let schema11 = try WorkflowCodec.decode(wire(schema: 11))
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(schema11)), schema11)
        XCTAssertThrowsError(try compile(schema11, version: 9))
        for supported in [document, schema11] {
            let plan = try compile(supported, version: 10)
            XCTAssertEqual(plan.gradientPolicyRecipe, baseline.gradientPolicyRecipe)
            XCTAssertEqual(plan.trainabilityRecipe, baseline.trainabilityRecipe)
            XCTAssertEqual(plan.model.graph, baseline.model.graph)
            XCTAssertEqual(plan.settings, baseline.settings)
            for currentVersion in [11,12,13,14,15,16] {
                let current=try compile(supported,version:currentVersion)
                XCTAssertEqual(current.gradientPolicyRecipe,baseline.gradientPolicyRecipe)
                XCTAssertEqual(current.trainabilityRecipe,baseline.trainabilityRecipe)
                XCTAssertEqual(current.model.graph,baseline.model.graph)
                XCTAssertEqual(current.settings,baseline.settings)
            }
            XCTAssertThrowsError(try compile(supported, version: 17))
        }
    }
    func testAbsentRecipesKeepHistoricalWorkflowBytesAndLegacyCompiler() throws {
        let old=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        let bytes=try WorkflowCodec.encode(old)
        let decoded=try WorkflowCodec.decode(bytes)
        _ = try compile(decoded,version:2)
        XCTAssertEqual(try WorkflowCodec.encode(decoded),bytes)
        let text=String(decoding:bytes,as:UTF8.self)
        XCTAssertFalse(text.contains("gradientPolicyRecipe"));XCTAssertFalse(text.contains("trainabilityRecipe"))
    }
}
