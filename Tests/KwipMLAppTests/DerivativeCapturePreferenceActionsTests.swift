import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class DerivativeCapturePreferenceActionsTests: XCTestCase {
    func testGlobalApplyPatchesLatestOtherGroupsAndColdResetOnlyOwnPermission() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(storageOverride: root, detectCapabilities: false)
        XCTAssertFalse(try store.executionPolicy(for: nil).derivativeCapture.values)
        // Resource/sample edits happen after the capture form was opened.
        XCTAssertTrue(store.setExecutionPreferences(try .init(maximumRunSeconds: 90, cpuThreads: 3, managedStorageLimitBytes: 3_000_000, preparationCacheLimitBytes: 10)))
        XCTAssertTrue(store.setSampleCapturePreferences(.init(activationValues: false, predictionRows: true)))
        XCTAssertTrue(store.setDerivativeCapturePreferences(.init(values: true)))
        let saved = try store.appPreferenceStore.load().execution
        XCTAssertTrue(saved.derivativeCapture.values)
        XCTAssertEqual(saved.maximumRunSeconds, 90); XCTAssertEqual(saved.cpuThreads, 3)
        XCTAssertEqual(saved.managedStorageLimitBytes, 3_000_000); XCTAssertEqual(saved.preparationCacheLimitBytes, 10)
        XCTAssertEqual(saved.sampleCapture, .init(activationValues: false, predictionRows: true))
        let captured = try store.executionPolicy(for: nil)
        let reopened = WorkspaceStore(storageOverride: root, detectCapabilities: false)
        XCTAssertTrue(try reopened.executionPolicy(for: nil).derivativeCapture.values)
        XCTAssertTrue(reopened.resetDerivativeCapturePreferences())
        let reset = try reopened.appPreferenceStore.load().execution
        XCTAssertFalse(reset.derivativeCapture.values)
        XCTAssertEqual(reset, try saved.replacingDerivativeCapture(.factory))
        XCTAssertTrue(captured.derivativeCapture.values, "An accepted run keeps its immutable policy.")
    }
    func testProjectInheritanceExplicitFalseAndResetPreserveLatestResourceAndSampleGroups() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(storageOverride: root, detectCapabilities: false)
        let id = UUID().uuidString; store.activeProjectID = id
        XCTAssertTrue(store.setDerivativeCapturePreferences(.init(values: true)))
        XCTAssertTrue(store.setProjectExecutionOverrides(try .init(maximumRunSeconds: 80, cpuThreads: 4, memory: .unlimited), projectID: id))
        XCTAssertTrue(store.setProjectSampleCapture(.init(activationValues: true, predictionRows: false), projectID: id))
        let original = try store.projectExecutionPreferenceStore.load(projectID: id)
        XCTAssertTrue(store.setProjectDerivativeCapture(.factory, projectID: id))
        XCTAssertEqual(try store.projectExecutionPreferenceStore.load(projectID: id), try original.replacingDerivativeCapture(.factory))
        let disabled = try store.executionPolicy(for: id)
        XCTAssertFalse(disabled.derivativeCapture.values); XCTAssertEqual(disabled.derivativeCaptureSource, .project)
        XCTAssertTrue(store.setProjectDerivativeCapture(.init(values: true), projectID: id))
        XCTAssertTrue(try store.executionPolicy(for: id).derivativeCapture.values)
        XCTAssertTrue(store.resetProjectDerivativeCapture(projectID: id))
        XCTAssertEqual(try store.projectExecutionPreferenceStore.load(projectID: id), original)
        XCTAssertEqual(try store.executionPolicy(for: id).derivativeCaptureSource, .application)
        XCTAssertTrue(try store.executionPolicy(for: id).derivativeCapture.values)
    }
    func testStaleProjectApplyAndResetRefuseWithoutTouchingEitherProject() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(storageOverride: root, detectCapabilities: false)
        let first = UUID().uuidString, second = UUID().uuidString
        try store.projectExecutionPreferenceStore.save(.init(cpuThreads: 2, derivativeCapture: .init(values: true)), projectID: first)
        try store.projectExecutionPreferenceStore.save(.init(cpuThreads: 3, derivativeCapture: .factory), projectID: second)
        let beforeFirst = try store.projectExecutionPreferenceStore.load(projectID: first)
        let beforeSecond = try store.projectExecutionPreferenceStore.load(projectID: second)
        store.activeProjectID = second
        XCTAssertFalse(store.setProjectDerivativeCapture(.factory, projectID: first))
        XCTAssertNotNil(store.appPreferencesError)
        XCTAssertFalse(store.resetProjectDerivativeCapture(projectID: first))
        XCTAssertEqual(try store.projectExecutionPreferenceStore.load(projectID: first), beforeFirst)
        XCTAssertEqual(try store.projectExecutionPreferenceStore.load(projectID: second), beforeSecond)
        store.activeProjectID = nil
        XCTAssertFalse(store.setProjectDerivativeCapture(.init(values: true), projectID: second))
        XCTAssertEqual(try store.projectExecutionPreferenceStore.load(projectID: second), beforeSecond)
    }
    func testUnreadableSavedDefaultsRefuseApplyAndResetWithoutDiscardingData() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(storageOverride: root, detectCapabilities: false)
        try FileManager.default.createDirectory(at: store.workspaceDirectory, withIntermediateDirectories: true)
        let corrupt = Data("not preferences".utf8)
        try corrupt.write(to: store.appPreferenceStore.fileURL)
        XCTAssertFalse(store.setDerivativeCapturePreferences(.init(values: true)))
        XCTAssertFalse(store.resetDerivativeCapturePreferences())
        XCTAssertNotNil(store.appPreferencesError)
        XCTAssertEqual(try Data(contentsOf: store.appPreferenceStore.fileURL), corrupt)
    }
}
