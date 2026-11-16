import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class SigmoidRuntimeCapabilityTests: XCTestCase {
    private var descriptor: [String: Any] {
        ["version": ["major": 1, "minor": 0, "patch": 0], "adapterID": "kwip.runner.graph-v2.sigmoid",
         "graphSchemaVersions": [2], "graphDTypes": ["float32"], "configuration": [String: Any](),
         "outputDTypes": ["out": "float32"], "maximumRank": 8, "preservesBatch": true,
         "preservesShape": true, "parameterless": true, "deterministic": true]
    }
    private func data(_ object: Any) throws -> Data { try JSONSerialization.data(withJSONObject: object, options: .sortedKeys) }
    private func store() -> WorkspaceStore {
        WorkspaceStore(rootOverride: FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString), detectCapabilities: false)
    }
    private func install(_ store: WorkspaceStore, sigmoid: Data?) throws {
        let contract = try BuiltInCapabilityBridge.executionContract(registry: store.registry, runtimeVersion: "2.14.1", availableDevices: [.cpu])
        try store.installDiscoveredRuntimeContract(version: "2.14.1", expectedVersion: nil, devices: ["cpu"], descriptor: JSONEncoder().encode(contract), sigmoidDescriptor: sigmoid)
    }
    private var graph: GraphDocument {
        .init(nodes: [.init(id: "i", type: "input", config: ["features": 2]), .init(id: "l", type: "linear", config: ["in_features": 2, "out_features": 2]), .init(id: "s", type: "sigmoid"), .init(id: "o", type: "output")], edges: [.init(source: "i", target: "l"), .init(source: "l", target: "s"), .init(source: "s", target: "o")])
    }
    func testExactDescriptorAndStrictContractRefusal() throws {
        let accepted = try SigmoidRuntimeCapability.decode(data(descriptor))
        XCTAssertEqual(accepted.adapterID, "kwip.runner.graph-v2.sigmoid")
        XCTAssertEqual(accepted.maximumRank, 8)
        XCTAssertEqual(accepted.graphDTypes, ["float32"])
        var cases: [[String: Any]] = []
        for (key, value) in [("unexpected", true as Any), ("maximumRank", 7), ("maximumRank", "8"), ("graphSchemaVersions", [1, 2]), ("graphDTypes", ["float32", "float64"]), ("configuration", ["dim": -1]), ("configuration", [] as [Int]), ("outputDTypes", ["out": "float64"]), ("parameterless", false), ("deterministic", 1), ("preservesBatch", false), ("preservesShape", false), ("adapterID", "sigmoid") ] {
            var changed = descriptor; changed[key] = value; cases.append(changed)
        }
        var changed = descriptor; changed["version"] = ["major": 1, "minor": 0, "patch": 1]; cases.append(changed)
        changed = descriptor; changed["version"] = ["major": 1, "minor": 0, "patch": 0, "unknown": 0]; cases.append(changed)
        changed = descriptor; changed.removeValue(forKey: "parameterless"); cases.append(changed)
        for value in cases { XCTAssertThrowsError(try SigmoidRuntimeCapability.decode(data(value))) }
        XCTAssertThrowsError(try SigmoidRuntimeCapability.decode(Data(repeating: 32, count: 4097)))
    }
    func testMissingMalformedAndResetDescriptorsGateOnlySigmoid() throws {
        let workspace = store()
        try install(workspace, sigmoid: nil)
        let baseCount = workspace.verifiedRuntimeComponentCount
        workspace.graph = graph
        XCTAssertTrue(try XCTUnwrap(workspace.capabilityBoundaryError).contains("Sigmoid"))
        try install(workspace, sigmoid: data(descriptor))
        XCTAssertNotNil(workspace.sigmoidRuntimeCapability)
        XCTAssertNil(workspace.capabilityBoundaryError)
        XCTAssertEqual(workspace.verifiedRuntimeComponentCount, baseCount + 1)
        var changed = descriptor; changed["maximumRank"] = 7
        try install(workspace, sigmoid: data(changed))
        XCTAssertNil(workspace.sigmoidRuntimeCapability)
        XCTAssertEqual(workspace.verifiedRuntimeComponentCount, baseCount)
        try install(workspace, sigmoid: data(descriptor))
        XCTAssertThrowsError(try workspace.installRuntimeContract(version: "unverified", expectedVersion: nil))
        XCTAssertNil(workspace.sigmoidRuntimeCapability)
        XCTAssertEqual(workspace.verifiedRuntimeComponentCount, 0)
    }
    func testClassicRunPreviewAndCodeRefuseUnverifiedSigmoidBeforeLaunch() throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: directory, detectCapabilities: false)
        _ = try XCTUnwrap(workspace.runtime, workspace.runtimeError ?? "Fixture runtime unavailable")
        try install(workspace, sigmoid: nil)
        workspace.graph = graph; workspace.selectedNodeID = "s"
        XCTAssertTrue(workspace.validation.isValid)
        let refusal = try XCTUnwrap(workspace.capabilityBoundaryError)
        workspace.run(); XCTAssertEqual(workspace.runError, refusal); XCTAssertFalse(workspace.isRunning)
        workspace.previewSelected(); XCTAssertEqual(workspace.statusMessage, refusal); XCTAssertFalse(workspace.isPreviewing)
        workspace.refreshDraftCode(); XCTAssertEqual(workspace.draftCodeError, refusal); XCTAssertFalse(workspace.isGeneratingCode)
        try install(workspace, sigmoid: data(descriptor))
        XCTAssertNil(workspace.capabilityBoundaryError)
    }
    func testProbeClearsThenIngestsOptionalDescriptorOnlyAfterWorkerExit() throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: directory, detectCapabilities: false)
        let runtime = try XCTUnwrap(workspace.runtime, workspace.runtimeError ?? "Fixture runtime unavailable")
        let version = runtime.packageVersions["torch"] ?? "2.14.1"
        let contract = try BuiltInCapabilityBridge.executionContract(registry: workspace.registry, runtimeVersion: version, availableDevices: [.cpu])
        let raw = try JSONSerialization.jsonObject(with: JSONEncoder().encode(contract))
        var event: [String: Any] = ["event": "capabilities", "torchVersion": version, "devices": ["cpu"], "defaultDevice": "cpu", "executionContract": raw, "componentCapabilities": ["sigmoid": descriptor]]
        var emit: ((String) -> Void)?, finish: ((Int32, String) -> Void)?
        workspace.capabilityLaunch = { _, _, _, _, line, exit in emit = line; finish = exit; return .init(cancel: {}) }
        try install(workspace, sigmoid: data(descriptor))
        var accepted: Bool?
        workspace.refreshCapabilities { accepted = $0 }
        XCTAssertNil(workspace.sigmoidRuntimeCapability)
        emit?(String(decoding: try data(event), as: UTF8.self))
        XCTAssertNil(accepted)
        XCTAssertNil(workspace.sigmoidRuntimeCapability)
        finish?(0, "")
        XCTAssertEqual(accepted, true)
        XCTAssertNotNil(workspace.sigmoidRuntimeCapability)
        let withSigmoid = workspace.verifiedRuntimeComponentCount
        event.removeValue(forKey: "componentCapabilities")
        workspace.refreshCapabilities { accepted = $0 }
        emit?(String(decoding: try data(event), as: UTF8.self)); finish?(0, "")
        XCTAssertNil(workspace.sigmoidRuntimeCapability)
        XCTAssertEqual(workspace.verifiedRuntimeComponentCount, withSigmoid - 1)
    }
}
