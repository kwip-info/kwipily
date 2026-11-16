import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkspaceExecutionAdmissionTests:XCTestCase {
    func testFreshWorkspaceRunsWithoutAccountOrLicense() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let store=WorkspaceStore(storageOverride:root,detectCapabilities:false)
        XCTAssertNoThrow(try store.executionAdmission())
        XCTAssertTrue(store.allowExecution())
        XCTAssertFalse(SettingsPage.allCases.map(\.rawValue).contains("account"))
    }
    func testDeniedLegacyRunAndPreviewDoNotStartOrChangeBusyState() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer{try? FileManager.default.removeItem(at:root)}
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        let store=WorkspaceStore(storageOverride:root,detectCapabilities:false)
        store.executionAdmission={throw NSError(domain:"test.execution",code:1,userInfo:[NSLocalizedDescriptionKey:"Execution unavailable"])}
        let graph=store.graph
        store.run()
        XCTAssertFalse(store.isBusy);XCTAssertTrue(store.statusMessage.contains("Execution unavailable"))
        store.previewSelected()
        XCTAssertFalse(store.isBusy);XCTAssertTrue(store.statusMessage.contains("Execution unavailable"))
        XCTAssertEqual(store.graph,graph)
        XCTAssertNoThrow(try store.executionPolicy(for:nil)) // Shared read/export resource policy remains available.
        XCTAssertTrue(store.allowOrdinaryExecution())
        let exported=root.appendingPathComponent("recovery.kwipml")
        XCTAssertNoThrow(try store.exportModelProject(to:exported))
        XCTAssertTrue(FileManager.default.fileExists(atPath:exported.path))
    }
    func testDeniedPreparationAndLessonDoNotReserveWork() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer{try? FileManager.default.removeItem(at:root)}
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        let store=WorkspaceStore(storageOverride:root,detectCapabilities:false)
        store.executionAdmission={throw NSError(domain:"test.execution",code:1,userInfo:[NSLocalizedDescriptionKey:"Execution unavailable"])}
        let prep=DataFlowWorkspaceStore(workspace:store)
        prep.runPreparation()
        XCTAssertFalse(prep.isBusy);XCTAssertTrue(prep.statusMessage.contains("Execution unavailable"))
        let id=UUID().uuidString
        store.activeProjectID=id;store.updateLearningContext(.init(projectID:id,kind:.affine,restoreState:.ready))
        XCTAssertNil(store.learningProjects.beginAssessment(projectID:id,cancel:{XCTFail("No existing work to cancel")}))
        XCTAssertFalse(store.isAssessingLearning);XCTAssertTrue(store.statusMessage.contains("Execution unavailable"))
    }
}
