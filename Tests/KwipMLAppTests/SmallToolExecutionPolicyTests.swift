import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

@MainActor final class SmallToolExecutionPolicyTests: XCTestCase {
    private func store() throws -> WorkspaceStore {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        try installFixtureExecutionContract(store, version: "0.0.0+fixture")
        XCTAssertTrue(store.saveManagedProject())
        return store
    }

    func testCodeGenerationPinsProjectPolicyAndExistingThirtySecondCap() async throws {
        let store = try store(), project = try XCTUnwrap(store.activeProjectID)
        XCTAssertTrue(store.setExecutionPreferences(try .init(maximumRunSeconds: 120, cpuThreads: 2,
            maximumResidentBytes: 268_435_456, managedStorageLimitBytes: 12_000_000)))
        XCTAssertTrue(store.setProjectExecutionOverrides(try .init(maximumRunSeconds: 90, cpuThreads: 4), projectID: project))
        let launched = expectation(description: "Code generation receives captured policy")
        var captured: WorkflowToolRequest?
        store.codeWorkerLauncher = { _, arguments, _, _, exit in
            let index = try XCTUnwrap(arguments.firstIndex(of: "--kwip-operation-supervise"))
            let directory = URL(fileURLWithPath: arguments[index + 1])
            captured = try JSONDecoder().decode(WorkflowToolRequest.self, from: Data(contentsOf: directory.appendingPathComponent("request.json")))
            launched.fulfill(); exit(1, "test finished")
            return .init(cancel: {})
        }
        store.refreshDraftCode()
        await fulfillment(of: [launched], timeout: 10)
        let request = try XCTUnwrap(captured)
        XCTAssertEqual(request.executionBudget.maximumRunSeconds, 30)
        XCTAssertEqual(request.executionBudget.requestedSeconds, 90)
        XCTAssertEqual(request.executionBudget.resources?.cpuThreads, 4)
        XCTAssertEqual(request.executionBudget.resources?.maximumResidentBytes, 268_435_456)
    }

    func testCodeGenerationCannotLaunchWhenManagedStorageQuotaRefuses() async throws {
        let store = try store()
        XCTAssertTrue(store.setExecutionPreferences(try .init(managedStorageLimitBytes: 1)))
        var launched = false
        store.codeWorkerLauncher = { _, _, _, _, _ in launched = true; return .init(cancel: {}) }
        let finished = expectation(description: "Storage refusal is visible")
        let observation = store.$draftCodeError.compactMap { $0 }.prefix(1).sink { _ in finished.fulfill() }
        store.refreshDraftCode()
        await fulfillment(of: [finished], timeout: 10)
        observation.cancel()
        XCTAssertFalse(launched)
        XCTAssertTrue(store.draftCode.isEmpty)
        XCTAssertTrue(store.draftCodeError?.contains("storage") == true, store.draftCodeError ?? "Missing storage diagnostic")
    }

    func testCapabilityDiscoveryCapturesApplicationPolicyWithoutProjectOverride() throws {
        let store = try store(), project = try XCTUnwrap(store.activeProjectID)
        XCTAssertTrue(store.setExecutionPreferences(try .init(maximumRunSeconds: 47, cpuThreads: 3, maximumResidentBytes: 268_435_456)))
        XCTAssertTrue(store.setProjectExecutionOverrides(try .init(maximumRunSeconds: 5, cpuThreads: 8), projectID: project))
        var budget: ResolvedExecutionBudget?
        store.capabilityLaunch = { _, _, _, window, _, exit in
            budget = window.policy; exit(1, "fixture complete"); return .init(cancel: {})
        }
        store.refreshCapabilities()
        XCTAssertEqual(budget?.maximumRunSeconds, 47)
        XCTAssertEqual(budget?.resources?.cpuThreads, 3)
        XCTAssertEqual(budget?.resources?.maximumResidentBytes, 268_435_456)
    }
    func testManagedCodeCannotEvadeStorageRetainedByExternalSibling() async throws {
        let store = try store()
        XCTAssertTrue(store.setExecutionPreferences(try .init(managedStorageLimitBytes: 12_000_000)))
        let sibling = store.applicationDirectory.appendingPathComponent("external-projects/" + UUID().uuidString + "/runs/" + UUID().uuidString)
        try FileManager.default.createDirectory(at: sibling, withIntermediateDirectories: true)
        let payload = sibling.appendingPathComponent("weights")
        XCTAssertTrue(FileManager.default.createFile(atPath: payload.path, contents: nil))
        let writer = try FileHandle(forWritingTo: payload)
        try writer.truncate(atOffset: 12_000_000); try writer.close()
        var launched = false
        store.codeWorkerLauncher = { _, _, _, _, _ in launched = true; return .init(cancel: {}) }
        let finished = expectation(description: "External sibling quota refusal")
        let observation = store.$draftCodeError.compactMap { $0 }.prefix(1).sink { _ in finished.fulfill() }
        store.refreshDraftCode()
        await fulfillment(of: [finished], timeout: 10)
        observation.cancel()
        XCTAssertFalse(launched)
        XCTAssertTrue(store.draftCodeError?.contains("storage") == true)
        XCTAssertTrue(FileManager.default.fileExists(atPath: payload.path))
    }

}
