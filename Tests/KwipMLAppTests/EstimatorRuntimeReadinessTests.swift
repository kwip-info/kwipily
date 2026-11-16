import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class EstimatorRuntimeReadinessTests: XCTestCase {
    private func workspace(_ root: URL) -> WorkspaceStore {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
    }
    private func event(_ store: WorkspaceStore, optional: [String: Any] = [:], errors: [String: Any] = [:]) throws -> String {
        let version = store.runtime?.packageVersions["torch"] ?? "2.14.1"
        let contract = try BuiltInCapabilityBridge.executionContract(registry: store.registry, runtimeVersion: version, availableDevices: [.cpu])
        return String(decoding: try JSONSerialization.data(withJSONObject: ["event": "capabilities", "torchVersion": version,
            "devices": ["cpu"], "defaultDevice": "cpu", "executionContract": JSONSerialization.jsonObject(with: JSONEncoder().encode(contract)),
            "optionalBackends": optional, "optionalBackendErrors": errors]), as: UTF8.self)
    }
    func testFirstRunWhileDiscoveryIsPendingReportsCheckingThenVerifiedRetryCanStart() async throws {
        let root = try canonicalBackendTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = workspace(root)
        XCTAssertTrue(store.loadBackendWorkflowExample(backendID: "sklearn"), store.statusMessage)
        let host = try XCTUnwrap(store.workflowSession)
        var line: ((String) -> Void)?, exit: ((Int32, String) -> Void)?
        store.capabilityLaunch = { _, _, _, _, onLine, onExit in
            line = onLine; exit = onExit; return .init(cancel: {})
        }
        host.executionLaunch = { _, _, _, _, _ in throw WorkflowArchiveError("Synthetic worker deliberately not launched") }
        store.refreshCapabilities()
        XCTAssertFalse(host.run(train: WorkflowExamples.sklearnEstimatorAddress))
        XCTAssertTrue(host.statusMessage.contains("Checking"), host.statusMessage)
        XCTAssertFalse(host.statusMessage.contains("does not provide"), host.statusMessage)
        XCTAssertNil(host.backendFit)
        XCTAssertFalse(store.isBusy)
        let descriptor = try JSONSerialization.jsonObject(with: JSONEncoder().encode(SklearnRuntimeCapabilities.expected))
        line?(try event(store, optional: ["sklearn": descriptor])); exit?(0, "")
        XCTAssertNotNil(store.sklearnRuntimeCapability)
        XCTAssertTrue(host.run(train: WorkflowExamples.sklearnEstimatorAddress), host.statusMessage)
        host.backendFit?.cancel()
        for _ in 0..<100 where host.backendFit?.isRunning == true { try await Task.sleep(nanoseconds: 10_000_000) }
        XCTAssertFalse(host.backendFit?.isRunning ?? false)
    }
    func testNoDiscoveryOrAbsentOptionalBackendNeverUsesManifestAsPermission() throws {
        let root = try canonicalBackendTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = workspace(root)
        XCTAssertTrue(store.loadBackendWorkflowExample(backendID: "sklearn"), store.statusMessage)
        let host = try XCTUnwrap(store.workflowSession)
        XCTAssertFalse(host.run(train: WorkflowExamples.sklearnEstimatorAddress))
        XCTAssertTrue(host.statusMessage.contains("does not provide verified sklearn support"), host.statusMessage)
        XCTAssertNil(host.backendFit)
        let payload = try event(store)
        store.capabilityLaunch = { _, _, _, _, line, exit in line(payload); exit(0, ""); return .init(cancel: {}) }
        store.refreshCapabilities()
        XCTAssertNotNil(store.runtimeExecutionContract)
        XCTAssertFalse(host.run(train: WorkflowExamples.sklearnEstimatorAddress))
        XCTAssertTrue(host.statusMessage.contains("does not provide verified sklearn support"), host.statusMessage)
        XCTAssertFalse(host.statusMessage.contains("Checking"))
        XCTAssertNil(host.backendFit)
    }
    func testImportFailureAndInvalidDescriptorStayDeniedWithDistinctActionableFeedback() throws {
        let root = try canonicalBackendTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = workspace(root)
        XCTAssertTrue(store.loadBackendWorkflowExample(backendID: "sklearn"), store.statusMessage)
        let host = try XCTUnwrap(store.workflowSession)
        let valid = try JSONSerialization.jsonObject(with: JSONEncoder().encode(SklearnRuntimeCapabilities.expected))
        let fixtures: [([String: Any], [String: Any], String)] = [
            ([:], ["sklearn": "Synthetic optional import failure /private/customer/token=secret"], "could not load"),
            (["sklearn": ["schemaVersion": 999]], [:], "incompatible"),
            (["sklearn": valid], ["sklearn": "Synthetic contradictory import failure"], "could not load")
        ]
        for (optional, errors, expected) in fixtures {
            let payload = try event(store, optional: optional, errors: errors)
            store.capabilityLaunch = { _, _, _, _, line, exit in line(payload); exit(0, ""); return .init(cancel: {}) }
            store.refreshCapabilities()
            XCTAssertNil(store.sklearnRuntimeCapability)
            XCTAssertFalse(host.run(train: WorkflowExamples.sklearnEstimatorAddress))
            XCTAssertTrue(host.statusMessage.contains(expected), host.statusMessage)
            XCTAssertTrue(host.statusMessage.contains("Refresh"), host.statusMessage)
            XCTAssertFalse(host.statusMessage.contains("/private/customer"))
            XCTAssertFalse(host.statusMessage.contains("token=secret"))
            XCTAssertFalse(host.statusMessage.contains("Checking"), host.statusMessage)
            XCTAssertNil(host.backendFit)
        }
    }
}
