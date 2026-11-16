import XCTest
import KwipMLCore
@testable import KwipMLApp

final class NormalizationLearningStoreTests:XCTestCase {
    @MainActor private func workspace(_ directory:URL)->WorkspaceStore {
        WorkspaceStore(rootOverride:URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent(),storageOverride:directory,detectCapabilities:false)
    }
    @MainActor func testSourceOnlyIncompleteDraftRoundTripAndModePrivacy() throws {
        let dir=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:dir)}
        let workspace=workspace(dir),lesson=NormalizationLearningStore(workspace:workspace)
        lesson.start();let id=try XCTUnwrap(workspace.activeProjectID),editor=try XCTUnwrap(lesson.flowEditor)
        XCTAssertEqual(lesson.draft?.assessmentVersion,NormalizationLessonTask.currentVersion)
        XCTAssertEqual(editor.project.graph.nodes.map(\.operatorID),["kwip.data.sourceCSV"])
        XCTAssertEqual(editor.project.sourceBindings,[:]);XCTAssertEqual(lesson.draft?.submission.outputs,[:])
        XCTAssertThrowsError(try XCTUnwrap(lesson.draft).executableSubmission())
        lesson.selectOutput(role:"training",endpoint:DataFlowEndpoint(node:"not-built-yet",port:"table"))
        lesson.setNotes("A training-only fit");lesson.revealHint("mean");lesson.setMode(.assessment);lesson.revealHint("hidden")
        XCTAssertEqual(lesson.draft?.hintIDs,["mean"])
        editor.project.editor["source"] = DataFlowEditorNode(name:"My fixture",x:220,y:140)
        XCTAssertTrue(lesson.save());let graph=workspace.graph
        workspace.beginProject(.init(name: "Untitled flow"));XCTAssertNil(lesson.draft);XCTAssertNil(lesson.flowEditor)
        XCTAssertTrue(workspace.beginProject(graph,id:id))
        XCTAssertTrue(lesson.isActive);XCTAssertEqual(lesson.mode,.assessment)
        XCTAssertEqual(lesson.draft?.notes,"A training-only fit")
        XCTAssertEqual(lesson.draft?.submission.outputs["training"]?.node,"not-built-yet")
        XCTAssertEqual(lesson.editor["source"]?.name,"My fixture")
        XCTAssertEqual(lesson.flowEditor?.project.editor["source"]?.x,220)
    }
    @MainActor func testSaveFailureVetoAndOldEditorCannotWriteDestination() throws {
        let dir=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:dir)}
        let workspace=workspace(dir),lesson=NormalizationLearningStore(workspace:workspace)
        lesson.start();let id=try XCTUnwrap(workspace.activeProjectID),oldEditor=try XCTUnwrap(lesson.flowEditor)
        let marker=dir.appendingPathComponent("learning/projects/\(id)/normalization.json")
        try FileManager.default.removeItem(at:marker);try FileManager.default.createDirectory(at:marker,withIntermediateDirectories:false)
        lesson.setNotes("Keep this draft")
        XCTAssertFalse(lesson.save());XCTAssertFalse(workspace.beginProject(GraphDocument(name:"Blocked")))
        XCTAssertEqual(workspace.activeProjectID,id);XCTAssertEqual(lesson.draft?.notes,"Keep this draft")
        try FileManager.default.removeItem(at:marker);XCTAssertTrue(lesson.save())
        lesson.start(mode:.practice);let next=try XCTUnwrap(lesson.draft?.projectID)
        XCTAssertNotEqual(next,id)
        oldEditor.project.editor["source"]?.name="Late old project"
        XCTAssertNotEqual(lesson.editor["source"]?.name,"Late old project")
    }
    @MainActor func testCorruptRestoreAndBusyEditsRemainProtected() throws {
        let dir=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:dir)}
        let workspace=workspace(dir),lesson=NormalizationLearningStore(workspace:workspace)
        lesson.start();let id=try XCTUnwrap(workspace.activeProjectID),graph=workspace.graph
        workspace.setLearningAssessmentActive(true)
        lesson.setMode(.assessment);lesson.setNotes("Blocked");lesson.selectOutput(role:"state",endpoint:.init(node:"source",port:"table"))
        XCTAssertEqual(lesson.mode,.guided);XCTAssertEqual(lesson.draft?.notes,"");XCTAssertTrue(lesson.draft?.submission.outputs.isEmpty == true)
        workspace.setLearningAssessmentActive(false);workspace.beginProject(.init(name: "Untitled flow"))
        try Data("corrupt".utf8).write(to:dir.appendingPathComponent("learning/projects/\(id)/normalization.json"))
        XCTAssertTrue(workspace.beginProject(graph,id:id))
        XCTAssertTrue(lesson.isRestoreFailure);XCTAssertNil(lesson.flowEditor);XCTAssertTrue(workspace.isLearningLessonActive)
        XCTAssertFalse(workspace.allowOrdinaryExecution())
    }
}


extension NormalizationLearningStoreTests {
    @MainActor func testReturnReopenAndStrictDocumentIdentity() throws {
        let dir=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:dir)}
        let workspace=workspace(dir),lesson=NormalizationLearningStore(workspace:workspace)
        XCTAssertTrue(workspace.saveManagedProject());let previous=try XCTUnwrap(workspace.activeProjectID)
        lesson.start(mode:.practice);let id=try XCTUnwrap(workspace.activeProjectID),graph=workspace.graph
        lesson.setNotes(String(repeating:"a",count:8100));XCTAssertEqual(lesson.draft?.notes.count,8000)
        lesson.leave(returnToPreviousProject:true)
        XCTAssertEqual(workspace.activeProjectID,previous);XCTAssertNil(lesson.draft)
        XCTAssertTrue(workspace.beginProject(graph,id:id));XCTAssertEqual(lesson.mode,.practice)
        workspace.beginProject(.init(name: "Untitled flow"))
        let marker=dir.appendingPathComponent("learning/projects/\(id)/normalization.json")
        var raw=try JSONSerialization.jsonObject(with:Data(contentsOf:marker)) as! [String:Any]
        var draft=raw["draft"] as! [String:Any];draft["projectID"]=UUID().uuidString;raw["draft"]=draft
        try JSONSerialization.data(withJSONObject:raw).write(to:marker)
        XCTAssertTrue(workspace.beginProject(graph,id:id));XCTAssertTrue(lesson.isRestoreFailure)
        XCTAssertNil(lesson.flowEditor);XCTAssertTrue(workspace.isLearningLessonActive)
    }
    @MainActor func testUnknownDocumentFieldsAndInvalidEditorAreRejectedWithoutOverwriting() throws {
        let dir=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:dir)}
        let workspace=workspace(dir),lesson=NormalizationLearningStore(workspace:workspace)
        lesson.start();let id=try XCTUnwrap(workspace.activeProjectID),graph=workspace.graph
        let marker=dir.appendingPathComponent("learning/projects/\(id)/normalization.json"),original=try Data(contentsOf:marker)
        let source=try XCTUnwrap(lesson.draft).submission.graph
        XCTAssertFalse(lesson.updateConstruction(graph:source,editor:["missing":DataFlowEditorNode(name:"No such node")]))
        XCTAssertEqual(try Data(contentsOf:marker),original)
        workspace.beginProject(.init(name: "Untitled flow"))
        var raw=try JSONSerialization.jsonObject(with:original) as! [String:Any];raw["externalSource"]="private.csv"
        let corrupted=try JSONSerialization.data(withJSONObject:raw);try corrupted.write(to:marker)
        XCTAssertTrue(workspace.beginProject(graph,id:id));XCTAssertTrue(lesson.isRestoreFailure)
        XCTAssertEqual(try Data(contentsOf:marker),corrupted)
    }
}

extension NormalizationLearningStoreTests {
    @MainActor func testExplicitSaveAndTransitionRefreshManagedRecencyAndVetoManagedFailure() throws {
        let dir=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer{try? FileManager.default.removeItem(at:dir)}
        let workspace=workspace(dir),lesson=NormalizationLearningStore(workspace:workspace)
        lesson.start();let id=try XCTUnwrap(lesson.draft?.projectID)
        let before=try XCTUnwrap(workspace.projects.first{$0.id == id}).updatedAt
        lesson.setNotes("Saved with project recency")
        XCTAssertTrue(lesson.save())
        let saved=try XCTUnwrap(workspace.projects.first{$0.id == id}).updatedAt
        XCTAssertGreaterThan(saved,before)
        let managed=try workspace.projectLibrary.documentURL(id:id),original=try Data(contentsOf:managed)
        try FileManager.default.removeItem(at:managed);try FileManager.default.createDirectory(at:managed,withIntermediateDirectories:false)
        XCTAssertFalse(lesson.save());XCTAssertEqual(lesson.statusMessage,workspace.statusMessage)
        XCTAssertTrue(lesson.statusMessage.contains("Save failed"))
        XCTAssertFalse(workspace.beginProject(GraphDocument(name:"Must not replace lesson")))
        XCTAssertEqual(workspace.activeProjectID,id);XCTAssertEqual(lesson.draft?.notes,"Saved with project recency")
        let marker=dir.appendingPathComponent("learning/projects/\(id)/normalization.json")
        XCTAssertTrue(String(decoding:try Data(contentsOf:marker),as:UTF8.self).contains("Saved with project recency"))
        try FileManager.default.removeItem(at:managed);try original.write(to:managed)
        XCTAssertTrue(workspace.beginProject(GraphDocument(name:"After repair")))
        XCTAssertGreaterThan(try XCTUnwrap(workspace.projects.first{$0.id == id}).updatedAt,saved)
    }
}
