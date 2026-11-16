import XCTest
@testable import KwipMLCore

final class WorkflowGuidedBuildTests:XCTestCase {
    func testProgressRoundTripsAndStaleAdvanceCannotAddASecondStage()throws {
        let plan=try WorkflowGuidedBuildPlan.parallelClassifier()
        let initial=try plan.document(completedSteps:0)
        let progress=try XCTUnwrap(initial.guidedBuild)
        XCTAssertEqual(progress.completedSteps,0)
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let models=try ComponentRegistry.builtIn(),data=try DataFlowOperatorRegistry.load(from:repo.appendingPathComponent("runner/dataflow-operators.json"))
        let next=try WorkflowEditing.apply(.advanceGuidedBuild(expected:progress),to:initial,modelRegistry:models,dataRegistry:data).document
        XCTAssertEqual(next.nodes.count,1);XCTAssertEqual(next.revision,1)
        XCTAssertEqual(next.guidedBuild?.completedSteps,1)
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(next)),next)
        XCTAssertThrowsError(try WorkflowEditing.apply(.advanceGuidedBuild(expected:progress),to:next,modelRegistry:models,dataRegistry:data))
        var invalid=next;invalid.guidedBuild?.completedSteps = -1
        XCTAssertThrowsError(try WorkflowCodec.encode(invalid))
        var legacy=initial;legacy.guidedBuild=nil
        XCTAssertNil(try WorkflowCodec.decode(WorkflowCodec.encode(legacy)).guidedBuild)
        var future=initial;future.guidedBuild?.version=plan.version+1
        XCTAssertThrowsError(try WorkflowEditing.apply(.advanceGuidedBuild(expected:future.guidedBuild!),to:future,modelRegistry:models,dataRegistry:data))
    }
    func testStageInsertionIsAtomicPreservesEditsAndRejectsReplayingAnOccupiedStage() throws {
        let plan=try WorkflowGuidedBuildPlan.parallelClassifier()
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let models=try ComponentRegistry.builtIn(),data=try DataFlowOperatorRegistry.load(from:repo.appendingPathComponent("runner/dataflow-operators.json"))
        var current=try plan.document(completedSteps:0)
        for stage in plan.steps.indices {
            let before=current
            let result=try WorkflowEditing.apply(plan.insertion(at:stage),to:current,modelRegistry:models,dataRegistry:data)
            XCTAssertEqual(result.document.revision,before.revision+1)
            XCTAssertEqual(result.selection.nodeAddresses,[plan.steps[stage].address])
            current=result.document
            if stage == 0 {current.nodes[0].name="My observations";current.nodes[0].x=999}
        }
        XCTAssertEqual(current.nodes[0].name,"My observations");XCTAssertEqual(current.nodes[0].x,999)
        XCTAssertThrowsError(try WorkflowEditing.apply(plan.insertion(at:0),to:current,modelRegistry:models,dataRegistry:data))
        let empty=try plan.document(completedSteps:0)
        XCTAssertThrowsError(try WorkflowEditing.apply(plan.insertion(at:1),to:empty,modelRegistry:models,dataRegistry:data))
        XCTAssertTrue(empty.nodes.isEmpty,"Failed insertion must not publish a partial stage")
        XCTAssertThrowsError(try plan.insertion(at:18))
    }
    func testEveryStageRoundTripsWithoutDanglingConnectionsAndFinalStageCompiles() throws {
        let plan=try WorkflowGuidedBuildPlan.parallelClassifier()
        XCTAssertEqual(plan.steps.count,15)
        XCTAssertTrue(try plan.document(completedSteps:0).nodes.isEmpty)
        for count in 1...plan.steps.count {
            let draft=try plan.document(completedSteps:count)
            XCTAssertEqual(draft.nodes.count,count)
            XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(draft)),draft)
            let addresses=Set(draft.nodes.map(\.address))
            XCTAssertTrue(draft.edges.allSatisfy {addresses.contains($0.from.address) && addresses.contains($0.to.address)})
            XCTAssertFalse(plan.steps[count-1].explanation.isEmpty)
        }
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let result=try WorkflowCompiler.compile(plan.document(completedSteps:plan.steps.count),train:WorkflowExamples.trainAddress,
            modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")),contractVersion:15)
        XCTAssertEqual(result.model.graph.nodes.count,8)
        XCTAssertEqual(result.model.graph.nodes.filter{$0.type == "linear"}.count,3)
        XCTAssertEqual(result.featureColumns,["signal","context"])
        XCTAssertEqual(result.settings.epochs,3)
        XCTAssertEqual(result.training,.init(node:"fit",port:"training"))
        XCTAssertEqual(result.validation,.init(node:"fit",port:"validation"))
        let views=Set(plan.completeDocument.nodes.compactMap {node->WorkflowNodeAddress? in
            guard case .dataOperator(let id,_)=node.payload,id=="kwip.data.view" else {return nil};return node.address
        })
        XCTAssertFalse(plan.completeDocument.edges.contains {views.contains($0.from.address)},"Guided inspection stages never become data providers")

        XCTAssertThrowsError(try plan.document(completedSteps:-1))
        XCTAssertThrowsError(try plan.document(completedSteps:19))
    }
}
