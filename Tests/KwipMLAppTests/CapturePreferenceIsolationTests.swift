import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class CapturePreferenceIsolationTests: XCTestCase {
    func testResourceDraftAndPersistedResetsPreserveCapture() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let hidden = TrainingSampleCapturePolicy(activationValues: false, predictionRows: false)
        let global = try ExecutionPreferences(maximumRunSeconds: 90, cpuThreads: 3, sampleCapture: hidden)
        let project = try ProjectExecutionOverrides(cpuThreads: 2, sampleCapture: hidden)
        var draft = ExecutionSettingsDraft(global: global, project: project)
        draft.threads = "4"
        XCTAssertEqual(try draft.project().sampleCapture, hidden)
        var appDraft = ExecutionSettingsDraft(global: global)
        appDraft.threads = "4"
        XCTAssertEqual(try appDraft.application().sampleCapture, hidden)
        let appStore = AppPreferencesStore(fileURL: root.appendingPathComponent("preferences.json"))
        var app = try appStore.load(); app.execution = global
        try appStore.save(app)
        XCTAssertEqual(try appStore.resetExecution().execution.sampleCapture, hidden)
        XCTAssertEqual(try appStore.load().execution.cpuThreads, 1)
        let projectStore = ProjectExecutionPreferencesStore(workspace: root)
        let id = UUID().uuidString
        try projectStore.save(project, projectID: id)
        XCTAssertEqual(try projectStore.reset(projectID: id).sampleCapture, hidden)
        XCTAssertNil(try projectStore.load(projectID: id).cpuThreads)
    }
    func testCaptureActionsAndStaleResourceDraftsRemainIndependentAcrossReopen() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(storageOverride: root, detectCapabilities: false)
        let id = UUID().uuidString
        let hidden = TrainingSampleCapturePolicy(activationValues: false, predictionRows: false)
        XCTAssertTrue(store.setExecutionPreferences(try .init(maximumRunSeconds: 90, cpuThreads: 3)))
        XCTAssertTrue(store.setProjectExecutionOverrides(try .init(cpuThreads: 2), projectID: id))
        let stale = ExecutionSettingsDraft(global: store.appPreferences.execution)
        XCTAssertTrue(store.setSampleCapturePreferences(hidden))
        XCTAssertTrue(store.setExecutionPreferences(try stale.application()))
        XCTAssertEqual(try store.executionPolicy(for: nil).sampleCapture, hidden)
        XCTAssertTrue(store.setProjectSampleCapture(.factory, projectID: id))
        let captured = try store.executionPolicy(for: id)
        XCTAssertEqual(captured.sampleCapture, .factory)
        XCTAssertEqual(captured.sampleCaptureSource, .project)
        XCTAssertTrue(store.setProjectExecutionOverrides(try .init(cpuThreads: 4), projectID: id))
        XCTAssertEqual(try store.executionPolicy(for: id).sampleCaptureSource, .project)
        XCTAssertTrue(store.resetProjectSampleCapture(projectID: id))
        XCTAssertEqual(try store.executionPolicy(for: id).sampleCapture, hidden)
        XCTAssertEqual(try store.executionPolicy(for: id).budget.resources?.cpuThreads, 4)
        XCTAssertEqual(captured.sampleCapture, .factory)
        let reopened = WorkspaceStore(storageOverride: root, detectCapabilities: false)
        XCTAssertEqual(try reopened.executionPolicy(for: id), try store.executionPolicy(for: id))
        XCTAssertTrue(reopened.resetSampleCapturePreferences())
        XCTAssertEqual(try reopened.executionPolicy(for: nil).sampleCapture, .factory)
        XCTAssertEqual(try reopened.executionPolicy(for: nil).budget.resources?.cpuThreads, 3)
    }

}
