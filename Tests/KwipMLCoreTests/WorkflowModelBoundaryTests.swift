import XCTest
@testable import KwipMLCore

final class WorkflowModelBoundaryTests:XCTestCase {
    func testDirectBoundaryPreservesCompiledArchitectureAndInputOwnership() throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let data=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        let registry=try ComponentRegistry.builtIn()
        let old=try WorkflowExamples.preparedEmbeddingClassifier()
        let carrier=WorkflowNodeAddress(domain:.control,localID:"model")
        let before=try WorkflowCompiler.modelSnapshot(old,definition:carrier,modelRegistry:registry,dataRegistry:data)
        let next=try WorkflowModelBoundary.consolidate(old)
        XCTAssertFalse(next.nodes.contains {$0.address==carrier})
        let boundary=try XCTUnwrap(next.nodes.first {WorkflowModelBoundary.isDefinition($0)})
        XCTAssertEqual(try WorkflowCompiler.modelSnapshot(next,definition:boundary.address,modelRegistry:registry,dataRegistry:data).graph,before.graph)
        XCTAssertEqual(try WorkflowModelInput.boundary(for:boundary.address,in:next)?.address,try WorkflowModelInput.boundary(for:carrier,in:old)?.address)
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(next)),next)
        XCTAssertEqual(try WorkflowModelBoundary.consolidate(next),next)
        XCTAssertEqual(try WorkflowCodePlan.compile(next,scope:.model,selectedTargets:[],modelRegistry:registry,dataRegistry:data).models.count,1)
        XCTAssertNoThrow(try WorkflowCompiler.compile(next,train:.init(domain:.control,localID:"train"),modelRegistry:registry,dataRegistry:data,contractVersion:15))
    }
    func testBrokenBridgeIsNotSilentlyDropped() throws {
        var source=try WorkflowExamples.preparedEmbeddingClassifier()
        source.edges.removeAll {$0.to.address == .init(domain:.control,localID:"model")}
        XCTAssertThrowsError(try WorkflowModelBoundary.consolidate(source))
    }
    func testSavedVersionThreeGuidedCourseStillHasItsOriginalStages()throws {
        let old=try WorkflowGuidedBuildPlan.parallelClassifier(version:3)
        XCTAssertEqual(old.steps.count,17)
        XCTAssertEqual(old.version,3)
        XCTAssertTrue(old.completeDocument.nodes.contains {if case .modelDefinition=$0.payload {return true};return false})
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(old.document(completedSteps:14))).guidedBuild?.version,3)
    }

    func testEditableOldGuidedPrefixesUpgradeStageWithoutReplacingLearnerEdits()throws {
        let old=try WorkflowGuidedBuildPlan.parallelClassifier(version:3)
        for count in 0...old.steps.count {
            var source=try old.document(completedSteps:count)
            if !source.nodes.isEmpty {source.nodes[0].name="Learner label";source.nodes[0].x=777}
            let next=try WorkflowModelBoundary.consolidate(source)
            XCTAssertEqual(next.guidedBuild?.version,4)
            XCTAssertEqual(next.guidedBuild?.completedSteps,count-(count>=15 ? 1:0))
            XCTAssertFalse(next.nodes.contains {if case .modelDefinition=$0.payload {return true};return false})
            if !next.nodes.isEmpty {XCTAssertEqual(next.nodes[0].name,"Learner label");XCTAssertEqual(next.nodes[0].x,777)}
            XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(next)),next)
        }
    }

}
