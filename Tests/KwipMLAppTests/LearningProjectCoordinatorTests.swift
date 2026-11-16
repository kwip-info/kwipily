import XCTest
import KwipMLCore
@testable import KwipMLApp

final class LearningProjectCoordinatorTests: XCTestCase {
    @MainActor func testProjectNavigationSeparatesLessonStorageWithoutFilteringNames() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:root) }
        let workspace=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        workspace.renameProject("Learn · My ordinary model")
        XCTAssertTrue(workspace.saveManagedProject())
        let ordinary=try XCTUnwrap(workspace.activeProjectID)
        let lesson=AffineLearningStore(workspace:workspace);lesson.start()
        let lessonID=try XCTUnwrap(workspace.activeProjectID)
        workspace.refreshProjects()
        XCTAssertEqual(workspace.modelProjects.map(\.id),[ordinary])
        XCTAssertEqual(workspace.lessonProjects.map(\.id),[lessonID])
        let marker=workspace.workspaceDirectory.appendingPathComponent("learning/projects/\(lessonID)/draft.json")
        try Data("invalid".utf8).write(to:marker)
        workspace.refreshProjects()
        XCTAssertEqual(workspace.modelProjects.map(\.id),[ordinary])
        XCTAssertEqual(workspace.lessonProjects.map(\.id),[lessonID])
        XCTAssertEqual(workspace.projects.count,2)
    }
    @MainActor func testSynchronousLegacyReopenAndReturnHaveNoDeferredRestore() async throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:root) }
        let workspace=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        XCTAssertTrue(workspace.saveManagedProject());let previous=try XCTUnwrap(workspace.activeProjectID)
        let lesson=AffineLearningStore(workspace:workspace);lesson.start()
        let id=try XCTUnwrap(workspace.activeProjectID);let graph=workspace.graph
        XCTAssertEqual(workspace.lessonContext?.projectID,id)
        lesson.leave(returnToPreviousProject:true)
        XCTAssertEqual(workspace.activeProjectID,previous);XCTAssertNil(workspace.lessonContext)
        XCTAssertTrue(workspace.beginProject(graph,id:id))
        XCTAssertTrue(lesson.isActive);XCTAssertTrue(workspace.isLearningLessonActive)
        workspace.beginProject(.init(name: "Untitled flow"));let after=workspace.activeProjectID
        await Task.yield()
        XCTAssertEqual(workspace.activeProjectID,after);XCTAssertNil(workspace.lessonContext);XCTAssertNil(lesson.draft)
    }
    @MainActor func testFailedDraftSaveVetoesTransitionWithoutLosingContext() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:root) }
        let workspace=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        let lesson=AffineLearningStore(workspace:workspace);lesson.start()
        let id=try XCTUnwrap(workspace.activeProjectID),draft=try XCTUnwrap(lesson.draft),graph=workspace.graph
        let path=workspace.workspaceDirectory.appendingPathComponent("learning/projects/\(id)/draft.json")
        try FileManager.default.removeItem(at:path);try FileManager.default.createDirectory(at:path,withIntermediateDirectories:false)
        XCTAssertFalse(workspace.beginProject(GraphDocument(name:"Must not open")))
        XCTAssertEqual(workspace.activeProjectID,id);XCTAssertEqual(workspace.graph,graph);XCTAssertEqual(lesson.draft?.projectID,draft.projectID)
        XCTAssertTrue(workspace.isLearningLessonActive);XCTAssertTrue(workspace.statusMessage.contains("save failed"))
    }
    @MainActor func testMalformedLegacyProjectProtectedAndAllOrdinaryExecutionBlocked() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:root) }
        let workspace=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        let lesson=AffineLearningStore(workspace:workspace);lesson.start();let id=try XCTUnwrap(workspace.activeProjectID);let graph=workspace.graph
        workspace.beginProject(.init(name: "Untitled flow"))
        try Data("invalid".utf8).write(to:workspace.workspaceDirectory.appendingPathComponent("learning/projects/\(id)/draft.json"))
        XCTAssertTrue(workspace.beginProject(graph,id:id))
        XCTAssertTrue(lesson.isRestoreFailure);XCTAssertEqual(workspace.lessonContext?.restoreState,.failed)
        workspace.run();workspace.previewSelected();workspace.refreshDraftCode();workspace.saveProject();workspace.savePythonCode("print('not exported')")
        XCTAssertFalse(workspace.isRunning);XCTAssertFalse(workspace.isPreviewing);XCTAssertFalse(workspace.isGeneratingCode)
        XCTAssertTrue(workspace.statusMessage.contains("lesson"))
    }
    @MainActor func testReplacementRegistrationAndAssessmentTokenOwnership() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:root) }
        let workspace=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        let first=AffineLearningStore(workspace:workspace);first.start()
        let second=AffineLearningStore(workspace:workspace);second.reopenCurrentProject()
        XCTAssertFalse(first.isActive);XCTAssertTrue(second.isActive)
        workspace.learningProjects.attachAffine(first);first.reopenCurrentProject()
        XCTAssertTrue(first.isActive);XCTAssertFalse(second.isActive)
        var cancellations=0
        let a=try XCTUnwrap(workspace.learningProjects.beginAssessment(projectID:try XCTUnwrap(workspace.activeProjectID),cancel:{cancellations+=1}))
        workspace.cancelRun();XCTAssertEqual(cancellations,1)
        workspace.learningProjects.finishAssessment(token:a)
        let b=try XCTUnwrap(workspace.learningProjects.beginAssessment(projectID:try XCTUnwrap(workspace.activeProjectID),cancel:{}))
        workspace.learningProjects.finishAssessment(token:a);XCTAssertTrue(workspace.isAssessingLearning)
        workspace.learningProjects.finishAssessment(token:b);XCTAssertFalse(workspace.isAssessingLearning)
    }
}

extension LearningProjectCoordinatorTests {
    @MainActor func testFailedDraftSaveDoesNotStartAssessmentOrClaimSaved() async throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        guard FileManager.default.fileExists(atPath:repository.appendingPathComponent(".venv/bin/python").path) else { throw XCTSkip("Runtime unavailable") }
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:root) }
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        let lesson=AffineLearningStore(workspace:workspace);lesson.start()
        workspace.edit { $0=GraphDocument(nodes:[GraphNode(id:"i",type:"input",config:["features":2]),GraphNode(id:"l",type:"linear",config:["in_features":2,"out_features":2]),GraphNode(id:"o",type:"output")],edges:[GraphEdge(source:"i",target:"l"),GraphEdge(source:"l",target:"o")]) }
        workspace.selectedNodeID="l";lesson.resetParameters(nodeID:"l")
        XCTAssertTrue(lesson.canSubmit)
        let id=try XCTUnwrap(workspace.activeProjectID)
        let path=root.appendingPathComponent("learning/projects/\(id)/draft.json")
        try FileManager.default.removeItem(at:path);try FileManager.default.createDirectory(at:path,withIntermediateDirectories:false)
        lesson.submit()
        XCTAssertFalse(workspace.isAssessingLearning)
        XCTAssertFalse(lesson.resultSaved);XCTAssertTrue(lesson.statusMessage.contains("save failed"))
        // Clean up any worker launched by the regression's pre-fix implementation.
        if workspace.isAssessingLearning { workspace.cancelRun() }
        let deadline=Date().addingTimeInterval(10)
        while workspace.isAssessingLearning && Date()<deadline { try await Task.sleep(nanoseconds:20_000_000) }
        XCTAssertEqual(workspace.activeProjectID,id)
    }
}

extension LearningProjectCoordinatorTests {
    @MainActor func testTensorMarkersReserveLessonOwnershipBeforeEditorIsAvailable() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        XCTAssertTrue(workspace.saveManagedProject())
        let id=try XCTUnwrap(workspace.activeProjectID),graph=workspace.graph
        let repository=TensorLearningDraftRepository(directory:workspace.workspaceDirectory.appendingPathComponent("learning/projects"))
        try repository.save(.init(projectID:id,operation:"transpose",mode:.guided),expectedRevision:nil)
        workspace.refreshProjects()
        XCTAssertEqual(workspace.lessonProjects.map(\.id),[id])
        XCTAssertTrue(workspace.modelProjects.isEmpty)
        workspace.learningProjects.restoreCurrentProject()
        XCTAssertTrue(workspace.isLearningLessonActive)
        XCTAssertEqual(workspace.lessonContext?.restoreState,.failed)
        XCTAssertEqual(workspace.activeSection,"Learn")
        XCTAssertFalse(workspace.canRunCurrentFlow)
        XCTAssertNil(workspace.learningProjects.beginAssessment(projectID:id,cancel:{}))
        workspace.run();workspace.previewSelected();workspace.refreshDraftCode()
        XCTAssertFalse(workspace.isRunning);XCTAssertFalse(workspace.isPreviewing);XCTAssertFalse(workspace.isGeneratingCode)
        let marker=repository.directory.appendingPathComponent("\(id)/tensor.json")
        let original=try Data(contentsOf:marker)
        XCTAssertTrue(workspace.beginProject(.init(name:"Ordinary project")))
        XCTAssertNil(workspace.lessonContext)
        XCTAssertEqual(try Data(contentsOf:marker),original)
        for malformed in ["file","directory","dangling link"] {
            try FileManager.default.removeItem(at:marker)
            switch malformed {
            case "directory":try FileManager.default.createDirectory(at:marker,withIntermediateDirectories:false)
            case "dangling link":try FileManager.default.createSymbolicLink(at:marker,withDestinationURL:root.appendingPathComponent("missing"))
            default:try Data("invalid".utf8).write(to:marker)
            }
            XCTAssertTrue(workspace.learningProjects.hasLessonMarker(projectID:id),malformed)
            XCTAssertTrue(workspace.beginProject(graph,id:id))
            XCTAssertTrue(workspace.isLearningLessonActive,malformed)
            XCTAssertEqual(workspace.lessonContext?.restoreState,.failed,malformed)
            XCTAssertFalse(workspace.canRunCurrentFlow,malformed)
            XCTAssertTrue(workspace.beginProject(.init(name:"Ordinary project")))
        }
    }
    @MainActor func testTensorAndAffineMarkersAreAConflictNotAnAffineLesson() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        let affine=AffineLearningStore(workspace:workspace);affine.start()
        let id=try XCTUnwrap(workspace.activeProjectID)
        let repository=TensorLearningDraftRepository(directory:workspace.workspaceDirectory.appendingPathComponent("learning/projects"))
        try repository.save(.init(projectID:id,operation:"reshape",mode:.guided),expectedRevision:nil)
        workspace.learningProjects.restoreCurrentProject()
        XCTAssertEqual(workspace.lessonContext?.restoreState,.failed)
        XCTAssertTrue(workspace.statusMessage.contains("Multiple lesson markers"))
        XCTAssertNil(affine.draft)
        XCTAssertFalse(workspace.canRunCurrentFlow)
    }
}

extension LearningProjectCoordinatorTests {
    @MainActor func testTensorSessionFollowsProjectNavigationAndStaleEditsVetoLeaving() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        XCTAssertTrue(workspace.saveManagedProject())
        let id=try XCTUnwrap(workspace.activeProjectID),graph=workspace.graph
        let repository=TensorLearningDraftRepository(directory:workspace.workspaceDirectory.appendingPathComponent("learning/projects"))
        try repository.save(.init(projectID:id,operation:"transpose",mode:.practice),expectedRevision:nil)
        workspace.learningProjects.restoreCurrentProject()
        let session=workspace.learningProjects.tensorDraftSession
        XCTAssertEqual(session.draft?.projectID,id)
        XCTAssertTrue(session.update {$0.notes="Retained during navigation";$0.pendingFields["axis"]="-"})
        XCTAssertTrue(workspace.beginProject(.init(name:"Ordinary")))
        XCTAssertNil(session.draft)
        XCTAssertEqual(try repository.load(projectID:id).draft.notes,"Retained during navigation")
        XCTAssertTrue(workspace.beginProject(graph,id:id))
        XCTAssertEqual(session.draft?.pendingFields["axis"],"-")
        let disk=try repository.load(projectID:id)
        var replacement=disk.draft;replacement.notes="Changed elsewhere"
        try repository.save(replacement,expectedRevision:disk.revision)
        XCTAssertTrue(session.update {$0.notes="Unsaved local edits"})
        // A restore request must not silently reload over edits in the active project.
        workspace.learningProjects.restoreCurrentProject()
        XCTAssertEqual(session.draft?.notes,"Unsaved local edits")
        XCTAssertFalse(workspace.beginProject(.init(name:"Must not replace current lesson")))
        XCTAssertEqual(workspace.activeProjectID,id)
        XCTAssertEqual(session.draft?.notes,"Unsaved local edits")
        XCTAssertTrue(workspace.statusMessage.contains("changed since"))
        let conflicting=repository.directory.appendingPathComponent("\(id)/normalization.json")
        try Data("invalid".utf8).write(to:conflicting)
        workspace.learningProjects.restoreCurrentProject()
        XCTAssertFalse(workspace.beginProject(.init(name:"Conflict must retain dirty edits")))
        XCTAssertEqual(session.draft?.notes,"Unsaved local edits")
        session.discardLocalChanges()
        XCTAssertTrue(workspace.beginProject(.init(name:"Ordinary")))
        XCTAssertNil(workspace.lessonContext)
        XCTAssertEqual(try repository.load(projectID:id).draft.notes,"Changed elsewhere")
    }
}
