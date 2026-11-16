import XCTest
import KwipMLCore
@testable import KwipMLApp

final class DataProgramLearningControlsTests:XCTestCase {
    @MainActor private func context()->(WorkspaceStore,TensorLearningStore,URL) {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false)
        return (workspace,TensorLearningStore(workspace:workspace),storage)
    }
    @MainActor func testDependentDatasetAndLoaderEditsApplyAtomicallyWithUndoAndColdDraft()throws {
        let (workspace,lesson,storage)=context();defer {withExtendedLifetime(workspace) {};try? FileManager.default.removeItem(at:storage)}
        XCTAssertTrue(lesson.start(operation:DataLoadingLearningTask.indexed.rawValue))
        let canvas=try XCTUnwrap(lesson.canvas)
        canvas.addNode(type:"dataset_source",x:30,y:30);canvas.addNode(type:"data_loader",x:30,y:180)
        let source=try XCTUnwrap(canvas.graph.nodes.first {$0.type=="dataset_source"})
        let original=source.config
        let editor=DataProgramLearningControls(canvas:canvas,nodeID:source.id)
        XCTAssertTrue(editor.setField("kind",value:.string("subset")))
        XCTAssertFalse(editor.canApply)
        XCTAssertEqual(canvas.graph.nodes.first {$0.id==source.id}?.config,original)
        XCTAssertFalse(try XCTUnwrap(lesson.draft).pendingFields.isEmpty)
        XCTAssertEqual(try TensorLearningDraft.decode(try XCTUnwrap(lesson.draft).encoded()),lesson.draft)
        let restored=DataProgramLearningControls(canvas:canvas,nodeID:source.id)
        XCTAssertEqual(restored.values["kind"],.string("subset"))
        XCTAssertTrue(restored.setField("indices",value:.array([4,1,0])))
        XCTAssertTrue(restored.canApply);XCTAssertTrue(restored.apply())
        XCTAssertEqual(canvas.graph.nodes.first {$0.id==source.id}?.config["indices"],.array([4,1,0]))
        XCTAssertTrue(try XCTUnwrap(lesson.draft).pendingFields.isEmpty)
        canvas.undo()
        XCTAssertEqual(canvas.graph.nodes.first {$0.id==source.id}?.config,original)
        let loader=try XCTUnwrap(canvas.graph.nodes.first {$0.type=="data_loader"})
        let groups=DataProgramLearningControls(canvas:canvas,nodeID:loader.id)
        XCTAssertTrue(groups.setField("num_workers",value:2));XCTAssertFalse(groups.canApply)
        XCTAssertTrue(groups.setField("prefetch_factor",value:1));XCTAssertFalse(groups.canApply)
        XCTAssertTrue(groups.setField("timeout",value:.number(5)))
        XCTAssertTrue(groups.apply())
        XCTAssertEqual(canvas.graph.nodes.first {$0.id==loader.id}?.config["num_workers"],2)
        let wide=DataProgramLearningControls(canvas:canvas,nodeID:source.id)
        XCTAssertTrue(wide.setField("kind",value:.string("subset")))
        let positions=JSONValue.array((0..<200).map {JSONValue.integer(Int64($0))})
        XCTAssertTrue(wide.setField("indices",value:positions),"A valid ordered selection must survive the draft text bound")
        XCTAssertTrue(wide.apply())
        XCTAssertEqual(try TensorLearningDraft.decode(try XCTUnwrap(lesson.draft).encoded()),lesson.draft)
        XCTAssertFalse(groups.setField("num_workers",value:3))
        canvas.assessmentIsRunning={true}
        XCTAssertFalse(groups.setField("batch_size",value:8))
        XCTAssertFalse(groups.apply());XCTAssertFalse(groups.discard())
        canvas.assessmentIsRunning={false}
    }
    @MainActor func testProgramPolicyTransitionDiscardStaleSourceAndBusyRefuse()throws {
        let (workspace,lesson,storage)=context();defer {withExtendedLifetime(workspace) {};try? FileManager.default.removeItem(at:storage)}
        XCTAssertTrue(lesson.start(operation:ProgramExportLearningTask.computation.rawValue))
        let editor=DataProgramLearningControls(lesson:lesson)
        XCTAssertTrue(editor.setField("batch_mode",value:.string("bounded")))
        XCTAssertFalse(editor.canApply)
        XCTAssertTrue(editor.setField("minimum_batch_size",value:1))
        XCTAssertTrue(editor.setField("maximum_batch_size",value:4))
        XCTAssertTrue(editor.canApply);XCTAssertTrue(editor.apply())
        XCTAssertEqual(try ProgramExportConfiguration.decodeRecipe(XCTUnwrap(lesson.draft?.programExportRecipe)).values["batch_mode"],.string("bounded"))
        let stale=DataProgramLearningControls(lesson:lesson)
        XCTAssertTrue(stale.setField("strict",value:false))
        XCTAssertTrue(lesson.drafts.update {$0.programExportRecipe=try! ProgramExportConfiguration(values:ProgramExportConfiguration.defaults).recipeValue})
        XCTAssertFalse(stale.apply(),"Stale retained editor cannot overwrite another source revision")
        let fresh=DataProgramLearningControls(lesson:lesson)
        XCTAssertTrue(fresh.setField("strict",value:false));XCTAssertTrue(fresh.discard())
        XCTAssertEqual(fresh.values["strict"],true)
        XCTAssertTrue(try XCTUnwrap(lesson.draft).pendingFields.isEmpty)
    }
}
