import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class TrainOwnedMappingInspectorTests:XCTestCase {
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    private func fixture(_ document:WorkflowDocument)throws->(WorkflowAuthoringStore,WorkflowControlNodeContext) {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock {try? FileManager.default.removeItem(at:directory)}
        let library=ProjectLibrary(directory:directory)
        let record=try library.saveSource(.workflow(document)),original=try Data(contentsOf:URL(fileURLWithPath:record.path))
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repository.appendingPathComponent("runner/dataflow-operators.json")))
        XCTAssertTrue(try owner.migrateRetiredToolsForAuthoring())
        XCTAssertEqual(try Data(contentsOf:URL(fileURLWithPath:record.path).appendingPathExtension("before-retired-tools-r\(document.revision)")),original)
        XCTAssertFalse(try owner.migrateRetiredToolsForAuthoring())
        XCTAssertTrue(owner.select(.init(nodeAddresses:[WorkflowExamples.trainAddress]),lease:owner.lease()))
        return(owner,try WorkflowControlNodeContext(owner:owner,address:WorkflowExamples.trainAddress))
    }
    func testInlineDenseApplyUndoInvalidAndStaleAuthorityPreserveRunSettings()throws {
        let(owner,context)=try fixture(WorkflowExamples.preparedParallelClassifier(device:"cpu"))
        let expected=context.configuration,run=expected["runSetup"]
        XCTAssertTrue(context.isTrain);XCTAssertTrue(context.isMapping)
        XCTAssertFalse(context.applyMapping(features:["signal","signal"],target:"label",expected:expected))
        XCTAssertEqual(context.configuration,expected)
        XCTAssertTrue(context.applyMapping(features:["context","signal"],target:"label",expected:expected))
        XCTAssertEqual(context.mappingConfiguration["featureColumns"],.array([.string("context"),.string("signal")]))
        XCTAssertEqual(context.configuration["runSetup"],run)
        XCTAssertFalse(context.applyMapping(features:["signal","context"],target:"label",expected:expected))
        XCTAssertTrue(owner.undo(lease:owner.lease()))
        guard case .train(let restored)=owner.document.nodes.first(where:{$0.address==WorkflowExamples.trainAddress})?.payload else {return XCTFail()}
        XCTAssertEqual(restored,expected)
        let canvas=try WorkflowCanvasAdapter(owner:owner,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry)
        XCTAssertFalse(canvas.creationEntries.contains {$0.iconType=="workflow.trainingData"})
        let entry=try XCTUnwrap(canvas.creationEntries.first {$0.iconType=="workflow.train"})
        XCTAssertEqual(entry.component.inputs,["model","training","validation"])
    }
    func testAllObjectiveEditorsResolveOnTrainWithoutMappingNode()throws {
        for task in LossLearningTask.allCases {
            let(_,context)=try fixture(WorkflowExamples.preparedLossObjective(task:task))
            let editor=try LossTargetMappingInspectorContext(context:context)
            XCTAssertTrue(context.hasObjectiveMapping);XCTAssertEqual(editor.componentID,task.componentID)
            XCTAssertNil(editor.validationMessage);XCTAssertFalse(editor.hasChanges)
        }
    }
    func testBagEditorPreservesBindingsInsideTrain()throws {
        let(_,context)=try fixture(WorkflowExamples.preparedEmbeddingBagClassifier(device:"cpu"))
        XCTAssertTrue(context.requiresBagMapping)
        let editor=try BagTrainingMappingInspectorContext(context:context)
        XCTAssertNil(editor.validationMessage)
        let before=context.configuration
        XCTAssertTrue(editor.update {$0.rows.swapAt(0,1)})
        XCTAssertTrue(editor.apply(),context.statusMessage)
        XCTAssertEqual(context.configuration["runSetup"],before["runSetup"])
        XCTAssertNotEqual(context.mappingConfiguration["bagInput"],{if case .object(let mapping)=before["dataMapping"] {return mapping["bagInput"]};return nil}())
    }
    func testObjectiveApplyPreservesOtherTrainAuthority()throws {
        let(_,context)=try fixture(WorkflowExamples.preparedLossObjective(task:.mse))
        let before=context.configuration,editor=try LossTargetMappingInspectorContext(context:context)
        XCTAssertTrue(editor.update {$0.inputs.move($0.inputs.rows[1].id,by:-1)})
        XCTAssertTrue(editor.apply(),context.statusMessage)
        XCTAssertNotEqual(context.configuration["dataMapping"],before["dataMapping"])
        XCTAssertEqual(context.configuration["objective"],before["objective"])
        XCTAssertEqual(context.configuration["runSetup"],before["runSetup"])
    }
}
