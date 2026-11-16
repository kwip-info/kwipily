import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class ExecutionPreferenceActionsTests: XCTestCase {
    func testSavedGlobalAndProjectChoicesResolveAndStayCapturedAfterEdits() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(storageOverride: root, detectCapabilities: false)
        let project = UUID().uuidString
        XCTAssertTrue(store.setExecutionPreferences(try .init(maximumRunSeconds: 120, cpuThreads: 2,
            maximumResidentBytes: 256 * 1024 * 1024, managedStorageLimitBytes: 2_000_000_000)))
        XCTAssertTrue(store.setProjectExecutionOverrides(try .init(maximumRunSeconds: 90, cpuThreads: 4, memory: .unlimited), projectID: project))
        let captured = try store.executionPolicy(for: project, capabilityMaximumSeconds: 60)
        XCTAssertEqual(captured.budget.requestedSeconds, 90)
        XCTAssertEqual(captured.budget.maximumRunSeconds, 60)
        XCTAssertEqual(captured.budget.resources?.cpuThreads, 4)
        XCTAssertNil(captured.budget.resources?.maximumResidentBytes)
        XCTAssertEqual(captured.storageLimitBytes, 2_000_000_000)
        XCTAssertTrue(store.setExecutionPreferences(try .init(maximumRunSeconds: 180, cpuThreads: 8)))
        XCTAssertTrue(store.resetProjectExecutionOverrides(projectID: project))
        let next = try store.executionPolicy(for: project)
        XCTAssertEqual(next.budget.maximumRunSeconds, 180)
        XCTAssertEqual(next.budget.resources?.cpuThreads, 8)
        XCTAssertEqual(captured.budget.resources?.cpuThreads, 4)
        XCTAssertEqual(captured.storageLimitBytes, 2_000_000_000)
        let reopened = WorkspaceStore(storageOverride: root, detectCapabilities: false)
        XCTAssertEqual(try reopened.executionPolicy(for: project), next)
    }
    func testExecutionResetPreservesAppearanceAndOtherProjectOverrides() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(storageOverride: root, detectCapabilities: false)
        let first = UUID().uuidString, second = UUID().uuidString
        XCTAssertTrue(store.setAppearance(.init(mode: .light, palette: .teal)))
        XCTAssertTrue(store.setExecutionPreferences(try .init(cpuThreads: 3)))
        XCTAssertTrue(store.setProjectExecutionOverrides(try .init(cpuThreads: 5), projectID: first))
        XCTAssertTrue(store.setProjectExecutionOverrides(try .init(cpuThreads: 6), projectID: second))
        XCTAssertTrue(store.resetExecutionPreferences())
        XCTAssertEqual(store.appPreferences.appearance.mode, .light)
        XCTAssertEqual(try store.executionPolicy(for: nil).budget.resources?.cpuThreads, 1)
        XCTAssertEqual(try store.executionPolicy(for: first).budget.resources?.cpuThreads, 5)
        XCTAssertTrue(store.resetProjectExecutionOverrides(projectID: first))
        XCTAssertEqual(try store.executionPolicy(for: first).budget.resources?.cpuThreads, 1)
        XCTAssertEqual(try store.executionPolicy(for: second).budget.resources?.cpuThreads, 6)
    }
    func testUnreadablePolicyRefusesResolutionInsteadOfSilentlyDiscardingLimits() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(storageOverride: root, detectCapabilities: false)
        try FileManager.default.createDirectory(at: store.workspaceDirectory, withIntermediateDirectories: true)
        try Data("invalid preferences".utf8).write(to: store.appPreferenceStore.fileURL)
        XCTAssertThrowsError(try store.executionPolicy(for: nil))
        XCTAssertFalse(store.resetExecutionPreferences())
        XCTAssertNotNil(store.appPreferencesError)
    }
}
