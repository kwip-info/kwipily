import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowEvaluationCanvasTests:XCTestCase {
    private let evaluation=WorkflowNodeAddress(domain:.control,localID:"evaluation")
    private let frozen=WorkflowNodeAddress(domain:.control,localID:"frozen")
    private func setup()throws->(WorkflowAuthoringStore,WorkflowCanvasAdapter) {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root)
        var document=WorkflowDocument(projectID:UUID().uuidString,nodes:[
            .init(address:evaluation,payload:.evaluateTest(config:[:])),.init(address:frozen,payload:.frozenModel(config:[:]))])
        document.schemaVersion=5
        _=try library.saveSource(.workflow(document))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let models=try ComponentRegistry.builtIn(),data=try DataFlowOperatorRegistry.load(from:repo.appendingPathComponent("runner/dataflow-operators.json"))
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:models,dataRegistry:data)
        return (owner,try WorkflowCanvasAdapter(owner:owner,modelRegistry:models,dataRegistry:data))
    }
    func testEvaluationCatalogUsesTypedPayloadsAndDistinctCategory()throws {
        let (_,adapter)=try setup()
        let entries=adapter.creationEntries.filter {$0.component.category=="Evaluation"}
        XCTAssertEqual(entries.count,3)
        XCTAssertEqual(Set(entries.map(\.iconType)),["workflow.evaluateTest","workflow.generateText","workflow.gameSession"])
        XCTAssertTrue(adapter.creationEntries.contains {$0.component.category=="Boundaries" && $0.component.title=="Load model" && $0.payload == .frozenModel(config:[:])})
        XCTAssertTrue(entries.contains {$0.payload == .evaluateTest(config:[:])})
        XCTAssertTrue(entries.contains {$0.payload == .generateText(config:[:])})
        XCTAssertTrue(entries.contains {$0.payload == .gameSession(config:[:])})
    }
    func testSettingsValidateAndUndoWithoutAddingTrainingFields()throws {
        let (owner,adapter)=try setup()
        _=owner.select(.init(nodeAddresses:[evaluation]),lease:owner.lease())
        let context=try WorkflowControlNodeContext(owner:owner,address:evaluation)
        XCTAssertTrue(context.isEvaluateTest);XCTAssertFalse(context.isFrozenModel)
        XCTAssertEqual(context.evaluationSettings,.init())
        XCTAssertEqual(context.configuration,[:])
        XCTAssertFalse(context.setEvaluationValue("epochs",value:3,expected:[:]))
        XCTAssertFalse(context.setEvaluationValue("batchSize",value:0,expected:[:]))
        XCTAssertFalse(context.setEvaluationValue("batchSize",value:.number(1.5),expected:[:]))
        XCTAssertFalse(context.setEvaluationValue("previewLimit",value:101,expected:[:]))
        XCTAssertFalse(context.setEvaluationValue("device",value:.string("cuda"),expected:[:]))
        XCTAssertTrue(context.setEvaluationValue("batchSize",value:17,expected:[:]))
        XCTAssertEqual(context.configuration,["batchSize":17])
        XCTAssertFalse(context.setEvaluationValue("previewLimit",value:0,expected:[:]))
        adapter.undo();XCTAssertEqual(owner.document.nodes.first {$0.address==evaluation}?.payload,.evaluateTest(config:[:]))
    }
    func testBusyPendingAndRetiredContextsCannotBypassAuthority()throws {
        let (owner,_)=try setup()
        _=owner.select(.init(nodeAddresses:[evaluation]),lease:owner.lease())
        let context=try WorkflowControlNodeContext(owner:owner,address:evaluation)
        let origin=context.numericOrigin(key:"batchSize")
        context.setNumericPending(origin:origin,key:"batchSize",pending:true)
        XCTAssertTrue(owner.hasPendingInspectorEdits)
        XCTAssertFalse(context.setEvaluationValue("batchSize",value:0,expected:[:]))
        XCTAssertTrue(owner.hasPendingInspectorEdits)
        owner.setExecutionBusy(true)
        XCTAssertFalse(context.setEvaluationValue("batchSize",value:8,expected:[:]))
        owner.setExecutionBusy(false)
        _=owner.select(.init(nodeAddresses:[frozen]),lease:owner.lease())
        XCTAssertFalse(context.setEvaluationValue("batchSize",value:8,expected:[:]))
        XCTAssertFalse(owner.hasPendingInspectorEdits)
        let source=try WorkflowControlNodeContext(owner:owner,address:frozen)
        XCTAssertTrue(source.isFrozenModel);XCTAssertFalse(source.isEvaluateTest)
        XCTAssertNil(source.evaluationSettings)
        XCTAssertFalse(source.setEvaluationValue("batchSize",value:8,expected:[:]))
        XCTAssertEqual(source.configuration,[:])
    }
}
