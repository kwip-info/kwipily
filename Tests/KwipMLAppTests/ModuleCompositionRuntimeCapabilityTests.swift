import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class ModuleCompositionRuntimeCapabilityTests: XCTestCase {
    private var exact: Data { Data(#"{"version":"1.0.0","adapterID":"kwip.runner.module-composition","graphSchemaVersions":[2],"planSchemaVersion":1,"modelAuthoringSchemaVersion":2,"compositeSourceSchemaVersion":3,"nativeExpansionContractVersion":2,"workflowSchemaVersion":11,"compilerContractVersion":10,"captureFormatVersion":6,"devices":["cpu","mps"],"modes":["branched-structure","nested-structure","parameter-ownership","train-eval","architecture-export","sequential-structure","ordered-structure","named-structure"],"sourceMembers":["torch.nn.Module","torch.nn.Sequential","torch.nn.ModuleList","torch.nn.ModuleDict"],"containerKinds":["graph","sequential","orderedList","namedMap"],"maximumDefinitions":64,"maximumExpandedNodes":128,"maximumEdges":256,"maximumContainers":129,"maximumChildReferences":256,"maximumBoundaryEndpoints":4096,"maximumDepth":8,"maximumPlanBytes":131072,"maximumSourceBytes":2000000,"maximumModelSourceBytes":2000000,"maximumParameters":200000,"parameterOwnership":"expandedNodeRelativeNativeParameter","stateOwnership":"bijectiveNativePersistentState","sharedParameters":false,"sequentialExecution":"nativeForwardValidatedChain","collectionExecution":"explicitDAG","sourceVerification":"independentCapturedRegistryReExpansion","exactResumeDevices":["cpu"],"fineTuneDevices":["cpu","mps"],"mpsResume":"weightsOnly","portableExport":"selfContainedNativeHierarchy","legacyAbsence":"unchanged"}"#.utf8) }

    private var repository: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private func install(_ workspace: WorkspaceStore, composition: Data?) throws {
        let contract = try BuiltInCapabilityBridge.executionContract(registry: workspace.registry, runtimeVersion: "2.14.1", availableDevices: [.cpu])
        try workspace.installDiscoveredRuntimeContract(version: "2.14.1", expectedVersion: nil, devices: ["cpu"], descriptor: JSONEncoder().encode(contract), moduleCompositionDescriptor: composition)
    }
    func testDefaultAbsenceAndOptionalExactInstallationNeverCreatesAForwardComponent() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        XCTAssertNil(workspace.moduleCompositionRuntimeCapability)
        try install(workspace, composition: nil)
        let count = workspace.verifiedRuntimeComponentCount
        try install(workspace, composition: exact)
        XCTAssertNotNil(workspace.moduleCompositionRuntimeCapability)
        XCTAssertEqual(workspace.verifiedRuntimeComponentCount, count)
        XCTAssertNil(workspace.registry.component(type: "moduleComposition"))
    }
    func testMissingMalformedRefreshAndInvalidRuntimeClearPriorSupport() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        for missing in [nil, Data("{}".utf8), Data("null".utf8), Data(repeating: 32, count: 4097)] as [Data?] {
            try install(workspace, composition: exact)
            XCTAssertNotNil(workspace.moduleCompositionRuntimeCapability)
            try install(workspace, composition: missing)
            XCTAssertNil(workspace.moduleCompositionRuntimeCapability)
        }
        try install(workspace, composition: exact)
        XCTAssertThrowsError(try workspace.installDiscoveredRuntimeContract(version: "2.14.1", expectedVersion: "wrong", devices: ["cpu"], descriptor: Data("{}".utf8), moduleCompositionDescriptor: exact))
        XCTAssertNil(workspace.moduleCompositionRuntimeCapability)
    }
    func testPrimitiveExecutionContractAloneDoesNotGrantComposition() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        try install(workspace, composition: nil)
        XCTAssertNotNil(workspace.runtimeExecutionContract)
        XCTAssertNil(workspace.moduleCompositionRuntimeCapability)
    }
    private func probeEvent(_ workspace: WorkspaceStore, includeComposition: Bool = true) throws -> String {
        let version = workspace.runtime?.packageVersions["torch"] ?? "2.14.1"
        let contract = try BuiltInCapabilityBridge.executionContract(registry: workspace.registry, runtimeVersion: version, availableDevices: [.cpu])
        var event: [String: Any] = ["event": "capabilities", "torchVersion": version, "devices": ["cpu"],
                                  "defaultDevice": "cpu", "executionContract": try JSONSerialization.jsonObject(with: JSONEncoder().encode(contract))]
        if includeComposition { event["compositionCapabilities"] = ["moduleComposition": try JSONSerialization.jsonObject(with: exact)] }
        return String(decoding: try JSONSerialization.data(withJSONObject: event), as: UTF8.self)
    }
    func testOwnedProbeProducesProofBoundToExactRuntimeAndFreshInventory() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        let runtime = try XCTUnwrap(workspace.runtime)
        let before = try WorkflowRuntimeEvidence.capture(runtime)
        let event = try probeEvent(workspace)
        workspace.capabilityLaunch = { _, _, _, _, line, exit in line(event); exit(0, ""); return .init(cancel: {}) }
        var completed: [Bool] = []
        workspace.refreshCapabilities { completed.append($0) }
        XCTAssertEqual(completed, [true])
        XCTAssertNotNil(workspace.moduleCompositionDiscoveryEvidence, "Only an owned completed probe establishes composition proof")
        guard let proof = workspace.moduleCompositionDiscoveryEvidence else { return }
        XCTAssertEqual(proof.runtime, runtime)
        XCTAssertEqual(proof.runtimeEvidence, before)
        XCTAssertEqual(try proof.verifiedDescriptor(for: runtime, evidence: before), try JSONDecoder().decode(JSONValue.self, from: exact))
        var changed = try JSONSerialization.jsonObject(with: JSONEncoder().encode(before)) as! [String: Any]
        changed["workerInventorySHA256"] = String(repeating: "0", count: 64)
        let stale = try JSONDecoder().decode(WorkflowRuntimeEvidence.self, from: JSONSerialization.data(withJSONObject: changed))
        XCTAssertThrowsError(try proof.verifiedDescriptor(for: runtime, evidence: stale))
    }
    func testManualInstallAndMissingOrFailedProbeNeverRetainOwnedProof() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        let valid = try probeEvent(workspace), missing = try probeEvent(workspace, includeComposition: false)
        workspace.capabilityLaunch = { _, _, _, _, line, exit in line(valid); exit(0, ""); return .init(cancel: {}) }
        workspace.refreshCapabilities()
        XCTAssertNotNil(workspace.moduleCompositionDiscoveryEvidence)
        try install(workspace, composition: exact)
        XCTAssertNotNil(workspace.moduleCompositionRuntimeCapability)
        XCTAssertNil(workspace.moduleCompositionDiscoveryEvidence, "Parser-only installation is not actual-worker evidence")
        let outcomes: [(String, Int32)] = [(missing, 0), (valid, 1)]
        for (event, code) in outcomes {
            workspace.capabilityLaunch = { _, _, _, _, line, exit in line(valid); exit(0, ""); return .init(cancel: {}) }
            workspace.refreshCapabilities()
            XCTAssertNotNil(workspace.moduleCompositionDiscoveryEvidence)
            workspace.capabilityLaunch = { _, _, _, _, line, exit in line(event); exit(code, ""); return .init(cancel: {}) }
            workspace.refreshCapabilities()
            XCTAssertNil(workspace.moduleCompositionDiscoveryEvidence)
            XCTAssertNil(workspace.moduleCompositionRuntimeCapability)
        }
    }

}
