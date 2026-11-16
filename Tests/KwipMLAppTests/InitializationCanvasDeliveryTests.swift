import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class InitializationCanvasDeliveryTests:XCTestCase {
    func testGeneratedArtworkAndAuthoredPresentationAreShared()throws {
        XCTAssertEqual(ComponentArtworkCatalog.assetName(for:"initialization"),"initialization-v1")
        XCTAssertNotNil(ComponentArtworkCatalog.image(for:"initialization"))
        XCTAssertEqual(ComponentArtworkCatalog.assetName(for:"train"),ComponentArtworkCatalog.assetName(for:"workflow.train"))
        XCTAssertNotNil(ComponentArtworkCatalog.image(for:"train"))
        let presentation=try XCTUnwrap(BundledInitializationComponents.entries["initialization"]?.presentation)
        XCTAssertEqual(ComponentStyle.shortTitle("initialization",fallback:"Unknown"),presentation.shortTitle)
        XCTAssertEqual(ComponentStyle.purpose("initialization"),presentation.purpose)
    }
    func testLayerStartingValuesUseRealParametersAndUndoRestoresDocument()throws {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let models=try ComponentRegistry.builtIn()
        let data=try DataFlowOperatorRegistry.load(from:repository.appendingPathComponent("runner/dataflow-operators.json"))
        let document=try WorkflowExamples.preparedParallelClassifier(device:"cpu"),library=ProjectLibrary(directory:directory)
        _=try library.saveSource(.workflow(document))
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:models,dataRegistry:data)
        let canvas=try WorkflowCanvasAdapter(owner:owner,modelRegistry:models,dataRegistry:data)
        XCTAssertFalse(canvas.creationEntries.contains {$0.iconType=="initialization"},"Starting values belong to the layer, not a new configuration node.")
        let address=try XCTUnwrap(owner.document.nodes.first {if case .modelPrimitive("linear",_,_)=$0.payload {return true};return false}).address
        XCTAssertTrue(owner.select(.init(nodeAddresses:[address]),lease:owner.lease()))
        let context=try WorkflowModelNodeContext(owner:owner,address:address)
        let editor=LayerParametersInspectorContext(context:context)
        XCTAssertEqual(editor.choices.map(\.id),["weight","bias"])
        let before=owner.document
        let baseline=try WorkflowCompiler.compile(before,train:WorkflowExamples.trainAddress,modelRegistry:models,dataRegistry:data,contractVersion:13)
        XCTAssertTrue(editor.setAlgorithm("zeros",parameter:"weight"))
        XCTAssertTrue(editor.setMode("frozen",parameter:"bias"))
        XCTAssertFalse(editor.setAlgorithm("xavier_uniform",parameter:"bias"),"Fan initialization cannot invent a second bias dimension.")
        XCTAssertTrue(owner.hasPendingInspectorEdits)
        XCTAssertTrue(editor.apply(),editor.rejection ?? "")
        XCTAssertFalse(owner.hasPendingInspectorEdits)
        XCTAssertEqual(owner.document.nodes.map(\.address),before.nodes.map(\.address))
        XCTAssertEqual(owner.document.edges,before.edges)
        for node in owner.document.nodes {
            let previous=try XCTUnwrap(before.nodes.first {$0.address==node.address})
            XCTAssertEqual(node.x,previous.x);XCTAssertEqual(node.y,previous.y)
            XCTAssertEqual(node.payload,previous.payload)
        }
        XCTAssertEqual(owner.document.revision,before.revision+1)
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(owner.document)),owner.document)
        let plan=try WorkflowCompiler.compile(owner.document,train:WorkflowExamples.trainAddress,modelRegistry:models,dataRegistry:data,contractVersion:13)
        XCTAssertEqual(plan.model.graph,baseline.model.graph)
        XCTAssertEqual(plan.model.semanticSHA256,baseline.model.semanticSHA256)
        XCTAssertEqual(plan.model.origins,baseline.model.origins)
        let rule=try XCTUnwrap(plan.initializationRecipe?.configuration.rules.first)
        XCTAssertEqual(rule.owner.node,address.localID);XCTAssertEqual(rule.owner.parameter,"weight");XCTAssertEqual(rule.algorithm,"zeros")
        XCTAssertEqual(plan.trainabilityRecipe?.configuration.owners,[.object(["node":.string(address.localID),"parameter":.string("bias"),"mode":.string("frozen")])])
        XCTAssertTrue(owner.undo(lease:owner.lease()))
        var restored=before;restored.revision=before.revision+2
        XCTAssertEqual(owner.document,restored)
        XCTAssertFalse(context.isCurrent)
        XCTAssertFalse(editor.setAlgorithm("normal",parameter:"weight"),"Undo retires the old layer editor.")
    }
    func testNoModelOwnerRefusesStartingValuesWithoutInventingSource()throws {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let models=try ComponentRegistry.builtIn()
        let data=try DataFlowOperatorRegistry.load(from:repository.appendingPathComponent("runner/dataflow-operators.json"))
        let document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:WorkflowExamples.trainAddress,x:0,y:0,payload:.train(config:[:]))]),library=ProjectLibrary(directory:directory)
        _=try library.saveSource(.workflow(document))
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:models,dataRegistry:data)
        let canvas=try WorkflowCanvasAdapter(owner:owner,modelRegistry:models,dataRegistry:data)
        XCTAssertFalse(canvas.creationEntries.contains {$0.iconType=="initialization"})
        XCTAssertTrue(owner.select(.init(nodeAddresses:[WorkflowExamples.trainAddress]),lease:owner.lease()))
        let session=TrainInlineRecipesSession(authority:try WorkflowNodeEditingAuthority(owner:owner,address:WorkflowExamples.trainAddress))
        XCTAssertTrue(session.modelChoices().owners.isEmpty)
        let draft=TrainInlineRecipeDraft(session:session,role:"initialization")
        let before=owner.document
        XCTAssertTrue(draft.choose("initialization"))
        XCTAssertFalse(try XCTUnwrap(draft.initialization).addOwner())
        XCTAssertFalse(draft.canApply)
        XCTAssertFalse(draft.apply())
        draft.releasePendingEdits()
        XCTAssertEqual(owner.document,before)
        XCTAssertFalse(owner.canUndo)
    }
}
