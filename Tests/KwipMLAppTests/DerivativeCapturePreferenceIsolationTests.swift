import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class DerivativeCapturePreferenceIsolationTests: XCTestCase {
    func testResourceFormsPreserveIndependentDerivativeChoices() throws {
        let enabled = DerivativeCapturePolicy(values: true)
        let global = try ExecutionPreferences(maximumRunSeconds: 90, cpuThreads: 3, derivativeCapture: enabled)
        let project = try ProjectExecutionOverrides(cpuThreads: 2, derivativeCapture: .factory)
        var app = ExecutionSettingsDraft(global: global)
        app.threads = "4"
        XCTAssertEqual(try app.application().derivativeCapture, enabled)
        XCTAssertEqual(try app.application(merging: global, scope: .execution).derivativeCapture, enabled)
        XCTAssertEqual(try app.application(merging: global, scope: .storage).derivativeCapture, enabled)
        var local = ExecutionSettingsDraft(global: global, project: project)
        local.threads = "4"
        XCTAssertEqual(try local.project().derivativeCapture, DerivativeCapturePolicy.factory)
    }
    func testStaleResourceActionsCannotOverwritePersistedDerivativePermission() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(storageOverride: root, detectCapabilities: false)
        let id = UUID().uuidString
        let stale = ExecutionSettingsDraft(global: store.appPreferences.execution)
        var saved = try store.appPreferenceStore.load()
        saved.execution = try saved.execution.replacingDerivativeCapture(.init(values: true))
        try store.appPreferenceStore.save(saved)
        try store.projectExecutionPreferenceStore.save(.init(cpuThreads: 2, derivativeCapture: .factory), projectID: id)
        XCTAssertTrue(store.setExecutionPreferences(try stale.application()))
        XCTAssertTrue(try store.executionPolicy(for: nil).derivativeCapture.values)
        XCTAssertTrue(store.setProjectExecutionOverrides(try .init(cpuThreads: 4), projectID: id))
        XCTAssertEqual(try store.executionPolicy(for: id).derivativeCapture, .factory)
        XCTAssertEqual(try store.executionPolicy(for: id).derivativeCaptureSource, .project)
        XCTAssertTrue(store.setSampleCapturePreferences(.init(activationValues: false, predictionRows: false)))
        XCTAssertTrue(try store.executionPolicy(for: nil).derivativeCapture.values)
        XCTAssertTrue(store.resetExecutionPreferences())
        XCTAssertTrue(try store.executionPolicy(for: nil).derivativeCapture.values)
        XCTAssertTrue(store.resetProjectExecutionOverrides(projectID: id))
        XCTAssertEqual(try store.executionPolicy(for: id).derivativeCaptureSource, .project)
    }
}
