import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class CapabilityDiscoveryLifecycleTests: XCTestCase {
    private func store(_ root: URL) -> WorkspaceStore {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
    }
    private func event(_ store: WorkspaceStore) throws -> String {
        let version = store.runtime?.packageVersions["torch"] ?? "2.14.1"
        let contract = try BuiltInCapabilityBridge.executionContract(registry: store.registry, runtimeVersion: version, availableDevices: [.cpu])
        let descriptor = try JSONSerialization.jsonObject(with: JSONEncoder().encode(contract))
        return String(decoding: try JSONSerialization.data(withJSONObject: ["event": "capabilities", "torchVersion": version,
            "devices": ["cpu"], "defaultDevice": "cpu", "executionContract": descriptor]), as: UTF8.self)
    }
    func testRapidRefreshCancelsButRetainsOldWorkerUntilExitAndNeverInstallsStaleResult() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = store(root), valid = try event(store)
        var lines: [(String) -> Void] = [], exits: [(Int32, String) -> Void] = [], scopes: [String] = []
        var cancelled: [Int] = [], completions: [[Bool]] = [[], [], []]
        store.capabilityLaunch = { _, scope, _, window, line, exit in
            XCTAssertNotNil(UUID(uuidString: scope)); XCTAssertGreaterThan(window.policy.maximumRunSeconds, 0)
            let index = exits.count; scopes.append(scope); lines.append(line); exits.append(exit)
            return .init(cancel: { cancelled.append(index) })
        }
        store.refreshCapabilities { completions[0].append($0) }
        lines[0](valid); XCTAssertNil(store.runtimeExecutionContract)
        store.activeProjectID = UUID().uuidString
        store.refreshCapabilities { completions[1].append($0) }
        store.refreshCapabilities { completions[2].append($0) }
        XCTAssertEqual(cancelled, [0]); XCTAssertEqual(exits.count, 1)
        XCTAssertNotNil(store.capabilityOperation)
        XCTAssertEqual(completions[1], [false], "Only the latest queued refresh should launch")
        XCTAssertTrue(completions[0].isEmpty, "Cancellation must join the retained worker exit")
        lines[0](valid); exits[0](0, ""); exits[0](0, "duplicate")
        XCTAssertEqual(completions[0], [false]); XCTAssertNil(store.runtimeExecutionContract)
        XCTAssertEqual(exits.count, 2); XCTAssertEqual(scopes[0], scopes[1])
        lines[1](valid); exits[1](0, ""); exits[1](1, "duplicate")
        XCTAssertEqual(completions[2], [true]); XCTAssertNotNil(store.runtimeExecutionContract)
        XCTAssertNil(store.capabilityOperation)
        XCTAssertTrue(store.capabilityStatus.contains("verified components"))
    }
    func testSynchronousExitAndThrownLaunchCompleteOnceWithoutRetainingExitedHandle() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = store(root), valid = try event(store)
        var results: [Bool] = [], cancellations = 0
        store.capabilityLaunch = { _, _, _, _, line, exit in
            line(valid); exit(0, ""); exit(1, "duplicate")
            return .init(cancel: { cancellations += 1 })
        }
        store.refreshCapabilities { results.append($0) }
        XCTAssertEqual(results, [true]); XCTAssertNil(store.capabilityOperation)
        store.capabilityLaunch = { _, _, _, _, _, _ in throw WorkflowArchiveError("launch rejected") }
        store.refreshCapabilities { results.append($0) }
        XCTAssertEqual(results, [true, false]); XCTAssertNil(store.runtimeExecutionContract)
        XCTAssertNil(store.capabilityOperation); XCTAssertEqual(cancellations, 0)
        XCTAssertTrue(store.capabilityStatus.contains("launch rejected"))
    }
    func testUnexpectedMalformedAndOversizedEventsInvalidateDiscovery() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = store(root), valid = try event(store)
        for invalid in ["not json", "{\"event\":\"unexpected\"}", "{\"event\":\"error\",\"message\":\"failed\"}", String(repeating: "x", count: 1_000_001)] {
            var results: [Bool] = []
            store.capabilityLaunch = { _, _, _, _, line, exit in
                line(invalid); line(valid); exit(0, ""); return .init(cancel: {})
            }
            store.refreshCapabilities { results.append($0) }
            XCTAssertEqual(results, [false]); XCTAssertNil(store.runtimeExecutionContract)
            XCTAssertNil(store.capabilityOperation)
        }
    }
    func testDuplicateCapabilityPayloadAndTimeoutCannotEstablishReadiness() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = store(root), valid = try event(store)
        for code: Int32 in [0, 124] {
            var results: [Bool] = []
            store.capabilityLaunch = { _, _, _, _, line, exit in
                line(valid)
                if code == 0 { line(valid) }
                exit(code, "timeout or duplicate")
                return .init(cancel: {})
            }
            store.refreshCapabilities { results.append($0) }
            XCTAssertEqual(results, [false]); XCTAssertNil(store.runtimeExecutionContract)
            XCTAssertNil(store.capabilityOperation)
        }
    }
}
