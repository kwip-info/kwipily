import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ModelWorkspaceActionsTests: XCTestCase {
    @MainActor func testOwnedProjectUsesDerivedGraphAndCanonicalMetadataWithoutFlattening() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:root,detectCapabilities:false),id=UUID().uuidString
        let source=ModelAuthoringDocument(projectID:id,source:CompositeExamples.parallelClassifier)
        let record=try workspace.projectLibrary.saveDocument(.blockModel(source),id:id)
        workspace.refreshProjects();XCTAssertTrue(workspace.projects.contains{$0.id==id})
        XCTAssertTrue(workspace.beginOwnedProject(id:id))
        let owner=try XCTUnwrap(workspace.modelAuthoring)
        XCTAssertNotNil(workspace.modelCanvas)
        XCTAssertEqual(workspace.graph,try owner.executionSnapshot().graph)
        workspace.graph=GraphDocument(name:"Must not replace owner")
        XCTAssertEqual(owner.document,.blockModel(source))
        XCTAssertFalse(workspace.applyCommand(.move(nodeIDs:["input"],offset:.init(x:20,y:0))))
        XCTAssertFalse(workspace.history.canUndo)
        workspace.renameProject("Owned model");workspace.setExecution{$0.epochs=7}
        XCTAssertEqual(workspace.graph.name,"Owned model");XCTAssertEqual(workspace.execution.epochs,7)
        XCTAssertTrue(workspace.saveManagedProject())
        XCTAssertEqual(try workspace.projectLibrary.loadDocument(id:id),owner.document)
        XCTAssertEqual(try ModelProjectCodec.decode(Data(contentsOf:URL(fileURLWithPath:record.path))),owner.document)
    }
    @MainActor func testDraftBlocksExecutionAndSwitchRetiresChildWithoutDiscardingDraft() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:root,detectCapabilities:false),id=UUID().uuidString
        let source=ModelAuthoringDocument(projectID:id,source:CompositeExamples.parallelClassifier)
        _=try workspace.projectLibrary.saveDocument(.blockModel(source),id:id)
        XCTAssertTrue(workspace.beginOwnedProject(id:id))
        let owner=try XCTUnwrap(workspace.modelAuthoring),adapter=try XCTUnwrap(workspace.modelCanvas)
        XCTAssertTrue(owner.beginDraft(path:["features"],lease:owner.lease()))
        XCTAssertFalse(workspace.validation.isValid)
        XCTAssertTrue(workspace.graph.nodes.isEmpty)
        XCTAssertTrue(workspace.validation.issues.contains{$0.message.contains("draft")})
        workspace.isRunning=true
        XCTAssertTrue(owner.isBusy)
        XCTAssertFalse(owner.discardDraft(lease:owner.lease(path:["features"])))
        workspace.isRunning=false;XCTAssertFalse(owner.isBusy)
        XCTAssertTrue(workspace.beginProject(GraphDocument(name:"New flat")))
        XCTAssertNil(workspace.modelAuthoring);XCTAssertNil(workspace.modelCanvas)
        XCTAssertTrue(adapter.isBusy)
        guard case .blockModel(let saved)=try workspace.projectLibrary.loadDocument(id:id) else{return XCTFail("Draft source was flattened")}
        XCTAssertEqual(saved.drafts.count,1)
        XCTAssertTrue(workspace.beginOwnedProject(id:id));XCTAssertFalse(workspace.validation.isValid)
    }
    @MainActor func testExplicitFlatOwnerAndLessonReturnPreserveCanonicalDocument() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        XCTAssertTrue(workspace.saveManagedProject())
        let id=try XCTUnwrap(workspace.activeProjectID)
        XCTAssertTrue(workspace.beginOwnedProject(id:id))
        let owner=try XCTUnwrap(workspace.modelAuthoring)
        guard case .legacyFlat = owner.document else{return XCTFail("Explicit ownership must not promote flat source")}
        XCTAssertTrue(owner.replaceSource(CompositeExamples.parallelClassifier,lease:owner.lease()))
        let canonical=owner.document
        let exported=root.appendingPathComponent("portable.kwipml")
        try workspace.exportModelProject(to:exported)
        XCTAssertEqual(try ModelProjectCodec.decode(Data(contentsOf:exported)),canonical)
        let lesson=AffineLearningStore(workspace:workspace);lesson.start()
        XCTAssertNil(workspace.modelAuthoring);XCTAssertTrue(workspace.isAffineLessonActive)
        lesson.leave(returnToPreviousProject:true)
        XCTAssertEqual(workspace.activeProjectID,id)
        XCTAssertEqual(workspace.modelAuthoring?.document,canonical)
        XCTAssertFalse(workspace.isLearningLessonActive)
        let normalization=NormalizationLearningStore(workspace:workspace);normalization.start()
        XCTAssertNil(workspace.modelAuthoring);XCTAssertTrue(workspace.isNormalizationLessonActive)
        normalization.leave(returnToPreviousProject:true)
        XCTAssertEqual(workspace.activeProjectID,id)
        XCTAssertEqual(workspace.modelAuthoring?.document,canonical)
        XCTAssertFalse(workspace.isLearningLessonActive)
    }
    @MainActor func testFailedOwnedOpenPreservesCurrentOwnerAndLegacySaveVeto() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:root,detectCapabilities:false),id=UUID().uuidString
        _=try workspace.projectLibrary.saveDocument(.blockModel(.init(projectID:id,source:CompositeExamples.parallelClassifier)),id:id)
        XCTAssertTrue(workspace.beginOwnedProject(id:id));let owner=try XCTUnwrap(workspace.modelAuthoring)
        XCTAssertFalse(workspace.beginOwnedProject(id:UUID().uuidString))
        XCTAssertTrue(workspace.modelAuthoring === owner)
        XCTAssertFalse(workspace.beginProject(GraphDocument(name:"Invalid flat overwrite"),id:id))
        XCTAssertTrue(workspace.modelAuthoring === owner)
        XCTAssertTrue(workspace.beginProject(GraphDocument(name:"Flat")))
        let flatID=try XCTUnwrap(workspace.activeProjectID),url=try workspace.projectLibrary.documentURL(id:flatID)
        try FileManager.default.removeItem(at:url);try FileManager.default.createDirectory(at:url,withIntermediateDirectories:false)
        workspace.hasUnsavedChanges=true
        XCTAssertFalse(workspace.beginOwnedProject(id:id))
        XCTAssertEqual(workspace.activeProjectID,flatID);XCTAssertNil(workspace.modelAuthoring)
    }
}
