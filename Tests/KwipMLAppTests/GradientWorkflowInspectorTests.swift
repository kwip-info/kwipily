import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class GradientWorkflowInspectorTests:XCTestCase {
    private func fixture(_ task:GradientLearningTask,role:String="gradientPolicy",mutate:(inout WorkflowDocument)->Void={_ in})throws->(WorkflowAuthoringStore,WorkflowNodeEditingAuthority,TrainInlineRecipeDraft) {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock {try? FileManager.default.removeItem(at:root)}
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        var document=try GradientWorkflowExamples.document(task:task,device:"cpu");mutate(&document)
        let library=ProjectLibrary(directory:root);_=try library.saveSource(.workflow(document))
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        XCTAssertTrue(owner.select(.init(nodeAddresses:[WorkflowExamples.trainAddress]),lease:owner.lease()))
        let authority=try WorkflowNodeEditingAuthority(owner:owner,address:WorkflowExamples.trainAddress)
        let draft=TrainInlineRecipeDraft(session:.init(authority:authority),role:role)
        if draft.componentID == nil {XCTAssertTrue(draft.choose(role == "gradientPolicy" ? "gradient_policy":"trainability"))}
        return(owner,authority,draft)
    }
    func testActualCapturedModelEndpointsAndNativeOwnersFeedInlineApplyUndoAndPending()throws {
        let(owner,authority,draft)=try fixture(.selectedInfluence)
        let resolved=draft.session.modelChoices(),context=try XCTUnwrap(draft.gradient)
        XCTAssertTrue(resolved.endpoints.contains {$0.node=="input" && $0.port=="out" && $0.isModelInput && $0.shape==[2]})
        XCTAssertTrue(resolved.endpoints.contains {$0.node=="second" && $0.port=="out" && !$0.isModelInput && $0.shape==[1]})
        XCTAssertTrue(resolved.owners.contains {$0.node=="first" && $0.parameter=="weight" && $0.shape==[2,2]})
        let original=owner.document
        XCTAssertNil(context.validationMessage);XCTAssertTrue(context.setAccumulationSteps(2));XCTAssertTrue(draft.canApply)
        XCTAssertTrue(owner.hasPendingInspectorEdits);XCTAssertEqual(owner.document,original)
        XCTAssertFalse(context.apply(),"Child must never commit independently of the Train role")
        XCTAssertTrue(draft.apply());XCTAssertFalse(owner.hasPendingInspectorEdits)
        XCTAssertEqual(draft.committedConfiguration["accumulation_steps"],2)
        XCTAssertTrue(owner.undo(lease:owner.lease()))
        XCTAssertEqual(owner.document.nodes.first {$0.address==authority.address}?.payload,original.nodes.first {$0.address==authority.address}?.payload)
    }
    func testLayerTrainabilityUsesActualParameterOwnersUndoAndAuthorityGuards()throws {
        let(owner,_,_)=try fixture(.keepSettings)
        let layer=try XCTUnwrap(owner.document.nodes.first {$0.address.localID=="first" && $0.address.domain == .model})
        XCTAssertTrue(owner.select(.init(nodeAddresses:[layer.address]),lease:owner.lease()))
        let context=try WorkflowModelNodeContext(owner:owner,address:layer.address)
        let editor=LayerParametersInspectorContext(context:context)
        let bias=try XCTUnwrap(editor.choices.first {$0.id=="bias"})
        XCTAssertFalse(bias.shape.isEmpty);XCTAssertTrue(editor.choices.contains {$0.id=="weight"})
        let before=owner.document,geometry=context.configuration
        XCTAssertFalse(editor.setMode("frozen",parameter:"foreign"))
        XCTAssertTrue(editor.setMode("trainable",parameter:"bias"))
        XCTAssertNil(editor.validationMessage);XCTAssertTrue(owner.hasPendingInspectorEdits)
        editor.revert();XCTAssertEqual(owner.document,before);XCTAssertFalse(owner.hasPendingInspectorEdits)
        owner.setExecutionBusy(true)
        XCTAssertFalse(editor.setMode("trainable",parameter:"bias"));XCTAssertFalse(editor.apply())
        owner.setExecutionBusy(false)
        XCTAssertTrue(editor.setMode("trainable",parameter:"bias"));XCTAssertTrue(editor.apply())
        XCTAssertEqual(context.configuration,geometry);XCTAssertFalse(owner.hasPendingInspectorEdits)
        XCTAssertEqual(owner.document.nodes.first {$0.address==layer.address}?.parameterSettings?["bias"],.object(["trainability":.string("trainable")]))
        XCTAssertTrue(owner.undo(lease:owner.lease()))
        XCTAssertEqual(owner.document.nodes.first {$0.address==layer.address}?.parameterSettings,layer.parameterSettings)
        XCTAssertFalse(context.isCurrent);XCTAssertFalse(editor.setMode("frozen",parameter:"bias"));XCTAssertFalse(editor.apply())
    }
    func testMissingOrMultipleModelConnectionsNeverFallBackToAnotherModel()throws {
        let(_,_,detached)=try fixture(.selectedInfluence) {doc in doc.edges.removeAll {$0.to.address==WorkflowExamples.trainAddress && $0.to.port=="model"}}
        XCTAssertTrue(detached.session.modelChoices().endpoints.isEmpty)
        XCTAssertNotNil(try XCTUnwrap(detached.gradient).validationMessage)
        var ambiguous=try GradientWorkflowExamples.document(task:.selectedInfluence,device:"cpu")
        let edge=try XCTUnwrap(ambiguous.edges.first {$0.to.address==WorkflowExamples.trainAddress && $0.to.port=="model"})
        ambiguous.edges.append(.init(from:edge.from,to:edge.to,kind:edge.kind))
        XCTAssertThrowsError(try WorkflowCodec.encode(ambiguous)) {error in
            XCTAssertTrue(error.localizedDescription.contains("Duplicate workflow edge identity or input occupancy"))
        }

    }
    func testCompilerRefusesTypedInlineScalarizationMismatchAndRetiredAuthorityCannotApply()throws {
        let(owner,authority,draft)=try fixture(.selectedInfluence)
        let context=try XCTUnwrap(draft.gradient)
        XCTAssertTrue(context.setGradientScale("sum"));XCTAssertTrue(draft.apply())
        // Typed drafts may be stored; complete compiler validation remains authoritative.
        XCTAssertThrowsError(try WorkflowCompiler.compile(owner.document,train:authority.address,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry,contractVersion:13))
        let current=try XCTUnwrap(draft.gradient)
        XCTAssertTrue(current.setGradientScale("mean"));XCTAssertTrue(draft.apply())
        XCTAssertNoThrow(try WorkflowCompiler.compile(owner.document,train:authority.address,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry,contractVersion:13))
        let latest=try XCTUnwrap(draft.gradient);XCTAssertTrue(latest.setAccumulationSteps(2))
        authority.retire();XCTAssertFalse(draft.apply());XCTAssertFalse(draft.revert())
    }
}
