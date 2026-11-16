import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class InitializationRuntimeAdmissionTests: XCTestCase {
    private var repository: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private func descriptor() throws -> Data {
        try WorkflowArchiveCodec.encode(XCTUnwrap(BundledInitializationComponents.metadata("initialization")).runtimeContract)
    }
    func testOnlyExactBoundedDescriptorAdmitsOptionalInitializationSupport() throws {
        let data = try descriptor()
        XCTAssertFalse(InitializationRuntimeCapabilities().contains("initialization"))
        XCTAssertTrue(InitializationRuntimeCapabilities(descriptor: data).contains("initialization"))
        XCTAssertEqual(InitializationRuntimeCapabilities(descriptor: data).componentIDs, ["initialization"])
        XCTAssertFalse(InitializationRuntimeCapabilities(descriptor: data).contains("linear"))
        let raw = try JSONDecoder().decode([String: JSONValue].self, from: data)
        for key in raw.keys {
            var missing = raw; missing.removeValue(forKey: key)
            XCTAssertFalse(InitializationRuntimeCapabilities(descriptor: try WorkflowArchiveCodec.encode(missing)).contains("initialization"), key)
            var changed = raw; changed[key] = .null
            XCTAssertFalse(InitializationRuntimeCapabilities(descriptor: try WorkflowArchiveCodec.encode(changed)).contains("initialization"), key)
        }
        for (key, value) in [("seedMaximum", JSONValue.number(Double(Int64.max))), ("maximumRules", .bool(true)), ("initializesBuffers", .integer(0)), ("unchecked", .bool(true))] {
            var changed = raw; changed[key] = value
            XCTAssertFalse(InitializationRuntimeCapabilities(descriptor: try WorkflowArchiveCodec.encode(changed)).contains("initialization"), key)
        }
        XCTAssertFalse(InitializationRuntimeCapabilities(descriptor: Data(repeating: 32, count: 4097)).contains("initialization"))
    }
    private func install(_ workspace: WorkspaceStore, initialization: Data?) throws {
        let execution = try BuiltInCapabilityBridge.executionContract(registry: workspace.registry, runtimeVersion: "2.14.1", availableDevices: [.cpu])
        try fixtureInstallDiscoveredRuntimeContract(workspace,version: "2.14.1", expectedVersion: nil, devices: ["cpu"], descriptor: JSONEncoder().encode(execution), tensorCompositionDescriptors: fixtureReviewedCompositionDescriptors(), initializationDescriptor: initialization, moduleCompositionDescriptor: fixtureReviewedModuleCompositionDescriptor())
    }
    func testDiscoveryInstallsSeparatelyAndMissingOrDriftedRefreshRevokesSupport() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        try install(workspace, initialization: nil)
        let count = workspace.verifiedRuntimeComponentCount
        XCTAssertFalse(workspace.initializationRuntimeCapabilities.contains("initialization"))
        try install(workspace, initialization: descriptor())
        XCTAssertTrue(workspace.initializationRuntimeCapabilities.contains("initialization"))
        XCTAssertEqual(workspace.verifiedRuntimeComponentCount, count + (workspace.registry.component(type:"initialization") == nil ? 0:1), "A published control counts only when its separate exact descriptor is verified.")
        try install(workspace, initialization: nil)
        XCTAssertFalse(workspace.initializationRuntimeCapabilities.contains("initialization"))
        try install(workspace, initialization: descriptor())
        try install(workspace, initialization: Data("{}".utf8))
        XCTAssertFalse(workspace.initializationRuntimeCapabilities.contains("initialization"))
    }
    func testSelectedMissingSupportRefusesBeforeSourceCaptureButAbsentPolicyRemainsRunnable() throws {
        for selected in [true, false] {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: root) }
            let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
            var document = try WorkflowExamples.preparedParallelClassifier(device: "cpu")
            if selected {
                document.schemaVersion = 9
                let owner = try XCTUnwrap(document.nodes.first { if case .modelPrimitive(type: "linear", configurationVersion: _, config: _) = $0.payload { return true }; return false })
                let config = try InitializationConfiguration(values: ["seed": 42, "rules": .array([.object(["owner": .object(["node": .string(owner.address.localID), "parameter": .string("weight")]), "algorithm": .string("zeros"), "settings": .object([:])])])])
                let source = WorkflowNodeAddress(domain: .control, localID: "starting-settings")
                document.nodes.append(.init(address: source, payload: .initializationRecipe(type: "initialization", configurationVersion: 1, config: config.values)))
                document.edges.append(.init(from: .init(address: source, port: "out"), to: .init(address: WorkflowExamples.trainAddress, port: "initialization"), kind: .initializationRecipe))
            }
            _ = try workspace.projectLibrary.saveSource(.workflow(document), id: document.projectID)
            XCTAssertTrue(workspace.beginWorkflowProject(id: document.projectID), workspace.statusMessage)
            let host = try XCTUnwrap(workspace.workflowSession)
            defer { host.cancel(); workspace.retireWorkflowSession() }
            try install(workspace, initialization: nil)
            let csv = root.appendingPathComponent("observations.csv")
            try FileManager.default.copyItem(at: repository.appendingPathComponent("Sources/KwipMLApp/Resources/workflow-observations.csv"), to: csv)
            try workspace.sourceAccess.remember(csv)
            XCTAssertTrue(host.bindings.setPath(csv.path, for: WorkflowExamples.sourceAddress, lease: host.owner.lease(), expectedRevision: host.bindings.revision))
            host.captureRunSources = { _, _, _, _ in throw CancellationError() }
            host.executionLaunch = { _, _, _, _, _ in XCTFail("This admission fixture must stop at source capture"); throw CancellationError() }
            XCTAssertEqual(host.run(), !selected, host.statusMessage)
            if selected {
                XCTAssertTrue(host.statusMessage.localizedCaseInsensitiveContains("initialization"), host.statusMessage)
                XCTAssertFalse(workspace.isBusy)
                XCTAssertNil(host.queuedRequestID)
                XCTAssertFalse(host.isStagingCheckpoint)
            }
        }
    }
}
