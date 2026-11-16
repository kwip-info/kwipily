import XCTest
import KwipMLCore
@testable import KwipMLApp

final class NormalizationCoordinatorTests:XCTestCase {
    @MainActor func testNormalizationMarkerWithoutControllerIsProtectedAndRoutesToLearn() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        XCTAssertTrue(workspace.saveManagedProject())
        let record=try XCTUnwrap(workspace.projects.first)
        let marker=workspace.workspaceDirectory.appendingPathComponent("learning/projects/\(record.id)/normalization.json")
        try FileManager.default.createDirectory(at:marker.deletingLastPathComponent(),withIntermediateDirectories:true)
        try Data("malformed".utf8).write(to:marker)
        workspace.beginProject(.init(name: "Untitled flow"))
        XCTAssertTrue(workspace.isManagedLesson(record))
        workspace.openManagedProject(record,section:"Build")
        XCTAssertEqual(workspace.lessonContext?.kind,.normalization)
        XCTAssertEqual(workspace.lessonContext?.restoreState,.failed)
        XCTAssertEqual(workspace.activeSection,"Learn")
        XCTAssertFalse(workspace.allowOrdinaryExecution())
        XCTAssertFalse(workspace.isRunning)
    }
    @MainActor func testAmbiguousMarkersNeverRestoreAffineOrPermitOrdinaryExecution() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        let affine=AffineLearningStore(workspace:workspace);affine.start()
        let id=try XCTUnwrap(workspace.activeProjectID),graph=workspace.graph
        workspace.beginProject(.init(name: "Untitled flow"))
        let marker=workspace.workspaceDirectory.appendingPathComponent("learning/projects/\(id)/normalization.json")
        try Data("malformed".utf8).write(to:marker)
        XCTAssertTrue(workspace.beginProject(graph,id:id))
        XCTAssertNil(affine.draft)
        XCTAssertEqual(workspace.lessonContext?.restoreState,.failed)
        XCTAssertTrue(workspace.statusMessage.lowercased().contains("multiple lesson"))
        XCTAssertFalse(workspace.allowOrdinaryExecution())
    }
    @MainActor func testDanglingNormalizationMarkerRemainsProtected() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        XCTAssertTrue(workspace.saveManagedProject())
        let id=try XCTUnwrap(workspace.activeProjectID),graph=workspace.graph
        workspace.beginProject(.init(name: "Untitled flow"))
        let marker=workspace.workspaceDirectory.appendingPathComponent("learning/projects/\(id)/normalization.json")
        try FileManager.default.createDirectory(at:marker.deletingLastPathComponent(),withIntermediateDirectories:true)
        try FileManager.default.createSymbolicLink(at:marker,withDestinationURL:root.appendingPathComponent("missing"))
        XCTAssertTrue(workspace.beginProject(graph,id:id))
        XCTAssertTrue(workspace.isNormalizationLessonActive)
        XCTAssertEqual(workspace.lessonContext?.restoreState,.failed)
        XCTAssertFalse(workspace.allowOrdinaryExecution())
    }

    @MainActor func testTaskSwitchRestoresOnlyMatchingControllerAndPreviousLesson() async throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        let affine=AffineLearningStore(workspace:workspace)
        let normalization=NormalizationLearningStore(workspace:workspace)
        affine.start();let affineID=try XCTUnwrap(workspace.activeProjectID)
        normalization.start();let normalizationID=try XCTUnwrap(workspace.activeProjectID)
        XCTAssertNotEqual(affineID,normalizationID)
        XCTAssertNil(affine.draft)
        XCTAssertTrue(normalization.isActive)
        XCTAssertEqual(workspace.lessonContext?.kind,.normalization)
        normalization.leave(returnToPreviousProject:true)
        XCTAssertEqual(workspace.activeProjectID,affineID)
        XCTAssertTrue(affine.isActive);XCTAssertNil(normalization.draft)
        XCTAssertEqual(workspace.lessonContext?.kind,.affine)
        let graph=try workspace.projectLibrary.load(id:normalizationID)
        XCTAssertTrue(workspace.beginProject(graph,id:normalizationID))
        XCTAssertTrue(normalization.isActive);XCTAssertNil(affine.draft)
        workspace.beginProject(.init(name: "Untitled flow"));await Task.yield()
        XCTAssertNil(workspace.lessonContext);XCTAssertNil(affine.draft);XCTAssertNil(normalization.draft)
    }

    @MainActor func testMalformedControllerRestorePublishesRecoveryCauseToWorkspace() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        let lesson=NormalizationLearningStore(workspace:workspace);lesson.start()
        let id=try XCTUnwrap(workspace.activeProjectID),graph=workspace.graph
        workspace.beginProject(.init(name: "Untitled flow"))
        try Data("broken-json".utf8).write(to:workspace.workspaceDirectory.appendingPathComponent("learning/projects/\(id)/normalization.json"))
        XCTAssertTrue(workspace.beginProject(graph,id:id))
        XCTAssertEqual(workspace.lessonContext?.restoreState,.failed)
        XCTAssertTrue(workspace.statusMessage.contains("could not be restored"))
        XCTAssertEqual(workspace.statusMessage,lesson.statusMessage)
    }

}
