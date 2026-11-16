import XCTest
@testable import KwipMLCore

final class WorkflowGuidedRunSetupProvenanceTests:XCTestCase {
    func testAllGuidedAdvancesRetainPinnedModelSemanticsAndLearnerEdits()throws {
        let plan=try WorkflowGuidedBuildPlan.parallelClassifier()
        XCTAssertEqual(plan.steps.count,15)
        XCTAssertEqual(plan.completeDocument.schemaVersion,18)
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let models=try ComponentRegistry.builtIn(),data=try DataFlowOperatorRegistry.load(from:repo.appendingPathComponent("runner/dataflow-operators.json"))
        var current=try plan.document(completedSteps:0)
        for stage in plan.steps.indices {
            let before=current
            current=try WorkflowEditing.apply(.advanceGuidedBuild(expected:XCTUnwrap(current.guidedBuild)),to:current,modelRegistry:models,dataRegistry:data).document
            XCTAssertEqual(current.revision,before.revision+1)
            XCTAssertEqual(current.guidedBuild?.completedSteps,stage+1)
            XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(current)),current)
            if stage==0 {current.nodes[0].name="Learner source";current.nodes[0].x=777}
        }
        XCTAssertEqual(current.nodes[0].name,"Learner source");XCTAssertEqual(current.nodes[0].x,777)
        XCTAssertEqual(current.modelSourceVersions,plan.completeDocument.modelSourceVersions)
        XCTAssertEqual(current.modelInputs,plan.completeDocument.modelInputs)
        XCTAssertEqual(current.modelSourceVersions?["output"],2)
        let completed=try WorkflowCompiler.compile(current,train:WorkflowExamples.trainAddress,modelRegistry:models,dataRegistry:data,contractVersion:15)
        let pinned=try WorkflowCompiler.compile(plan.completeDocument,train:WorkflowExamples.trainAddress,modelRegistry:models,dataRegistry:data,contractVersion:15)
        var actual=completed.model.graph,expected=pinned.model.graph
        for index in actual.nodes.indices {actual.nodes[index].x=0;actual.nodes[index].y=0}
        for index in expected.nodes.indices {expected.nodes[index].x=0;expected.nodes[index].y=0}
        actual.edges.sort {$0.id<$1.id};expected.edges.sort {$0.id<$1.id}
        XCTAssertEqual(actual,expected,"Placement may avoid overlap; model structure/configuration remains exact")
        XCTAssertEqual(completed.model.semanticSHA256,pinned.model.semanticSHA256)
        XCTAssertEqual(completed.settings,pinned.settings)
        XCTAssertEqual(completed.dataflow,pinned.dataflow)
        XCTAssertEqual(completed.featureColumns,["signal","context"])
        XCTAssertEqual(completed.settings.epochs,3)
        XCTAssertEqual(completed.model.graph.nodes.count,8)
        XCTAssertFalse(current.edges.contains {$0.kind == .runConfiguration})
    }
}
