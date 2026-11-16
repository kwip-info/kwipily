import XCTest
import KwipMLCore
@testable import KwipMLApp

final class GradientControlRuntimeAdmissionTests: XCTestCase {
    private func descriptor(_ id: String) throws -> Data {
        try WorkflowArchiveCodec.encode(XCTUnwrap(BundledGradientControlComponents.metadata(id)).runtimeContract)
    }
    private func capability(_ id: String, _ data: Data?) -> GradientControlRuntimeCapabilities {
        id == "gradient_policy" ? .init(gradientPolicy: data) : .init(trainability: data)
    }
    func testOnlyExactDescriptorsAdmitTheirSeparateControl() throws {
        XCTAssertTrue(GradientControlRuntimeCapabilities().componentIDs.isEmpty)
        for id in ["gradient_policy", "trainability"] {
            let bytes = try descriptor(id)
            XCTAssertEqual(capability(id, bytes).componentIDs, [id])
            XCTAssertFalse(capability(id, bytes).contains("linear"))
            let other = id == "gradient_policy" ? "trainability" : "gradient_policy"
            XCTAssertFalse(capability(other, bytes).contains(other))
            let raw = try JSONDecoder().decode([String: JSONValue].self, from: bytes)
            for key in raw.keys {
                var missing = raw; missing.removeValue(forKey: key)
                XCTAssertFalse(capability(id, try WorkflowArchiveCodec.encode(missing)).contains(id), key)
                var changed = raw; changed[key] = .null
                XCTAssertFalse(capability(id, try WorkflowArchiveCodec.encode(changed)).contains(id), key)
            }
            var extra = raw; extra["unchecked"] = .bool(true)
            XCTAssertFalse(capability(id, try WorkflowArchiveCodec.encode(extra)).contains(id))
            XCTAssertFalse(capability(id, Data(repeating: 32, count: 4097)).contains(id))
            XCTAssertFalse(capability(id, Data("null".utf8)).contains(id))
        }
        XCTAssertEqual(GradientControlRuntimeCapabilities(gradientPolicy: try descriptor("gradient_policy"), trainability: try descriptor("trainability")).componentIDs, ["gradient_policy", "trainability"])
    }
    @MainActor func testDiscoveryInstallsEachSiblingAndRevokesStaleSupport() throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        let execution = try BuiltInCapabilityBridge.executionContract(registry: workspace.registry, runtimeVersion: "2.14.1", availableDevices: [.cpu])
        func install(_ gradient: Data? = nil, _ trainability: Data? = nil, executionData: Data? = nil) throws {
            try fixtureInstallDiscoveredRuntimeContract(workspace,version: "2.14.1", expectedVersion: nil, devices: ["cpu"],
                descriptor: executionData ?? JSONEncoder().encode(execution), gradientPolicyDescriptor: gradient, trainabilityDescriptor: trainability)
        }
        try install()
        let before = workspace.verifiedRuntimeComponentCount
        try install(descriptor("gradient_policy"), descriptor("trainability"))
        XCTAssertEqual(workspace.gradientControlRuntimeCapabilities.componentIDs, ["gradient_policy", "trainability"])
        XCTAssertEqual(workspace.verifiedRuntimeComponentCount, before + 2, "Each published control requires its own exact runtime descriptor.")
        try install(nil, descriptor("trainability"))
        XCTAssertEqual(workspace.gradientControlRuntimeCapabilities.componentIDs, ["trainability"])
        XCTAssertEqual(workspace.verifiedRuntimeComponentCount, before + 1)
        try install(descriptor("gradient_policy"), Data("{}".utf8))
        XCTAssertEqual(workspace.gradientControlRuntimeCapabilities.componentIDs, ["gradient_policy"])
        XCTAssertEqual(workspace.verifiedRuntimeComponentCount, before + 1)
        XCTAssertThrowsError(try install(descriptor("gradient_policy"), descriptor("trainability"), executionData: Data("{}".utf8)))
        XCTAssertTrue(workspace.gradientControlRuntimeCapabilities.componentIDs.isEmpty)
        try install(descriptor("gradient_policy"), descriptor("trainability"))
        try install()
        XCTAssertTrue(workspace.gradientControlRuntimeCapabilities.componentIDs.isEmpty)
        XCTAssertEqual(workspace.verifiedRuntimeComponentCount, before)
    }

    @MainActor func testSelectedUnsupportedPolicyRefusesBeforeCapturingSources() throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        for selected in ["gradient_policy", "trainability", "absent"] {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: root) }
            let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
            var document = try WorkflowExamples.preparedParallelClassifier(device: "cpu")
            if selected != "absent" {
                document.schemaVersion = 10
                let address = WorkflowNodeAddress(domain: .control, localID: "policy")
                let gradient = selected == "gradient_policy"
                let payload: WorkflowNodePayload = gradient
                    ? .gradientPolicyRecipe(type: selected, configurationVersion: 1, config: try GradientPolicyConfiguration().values)
                    : .trainabilityRecipe(type: selected, configurationVersion: 1, config: try TrainabilityConfiguration().values)
                document.nodes.append(.init(address: address, payload: payload))
                document.edges.append(.init(from: .init(address: address, port: "out"),
                    to: .init(address: WorkflowExamples.trainAddress, port: gradient ? "gradientPolicy" : "trainability"),
                    kind: gradient ? .gradientPolicyRecipe : .trainabilityRecipe))
            }
            _ = try workspace.projectLibrary.saveSource(.workflow(document), id: document.projectID)
            XCTAssertTrue(workspace.beginWorkflowProject(id: document.projectID), workspace.statusMessage)
            let host = try XCTUnwrap(workspace.workflowSession)
            defer { host.cancel(); workspace.retireWorkflowSession() }
            let execution = try BuiltInCapabilityBridge.executionContract(registry: workspace.registry, runtimeVersion: "2.14.1", availableDevices: [.cpu])
            try fixtureInstallDiscoveredRuntimeContract(workspace,version: "2.14.1", expectedVersion: nil, devices: ["cpu"],
                descriptor: JSONEncoder().encode(execution), tensorCompositionDescriptors: fixtureReviewedCompositionDescriptors(), moduleCompositionDescriptor: fixtureReviewedModuleCompositionDescriptor())
            let csv = root.appendingPathComponent("observations.csv")
            try FileManager.default.copyItem(at: repository.appendingPathComponent("Sources/KwipMLApp/Resources/workflow-observations.csv"), to: csv)
            try workspace.sourceAccess.remember(csv)
            XCTAssertTrue(host.bindings.setPath(csv.path, for: WorkflowExamples.sourceAddress, lease: host.owner.lease(), expectedRevision: host.bindings.revision))
            host.captureRunSources = { _, _, _, _ in throw CancellationError() }
            host.executionLaunch = { _, _, _, _, _ in XCTFail("The fixture must stop before execution"); throw CancellationError() }
            XCTAssertEqual(host.run(), selected == "absent", host.statusMessage)
            if selected != "absent" {
                XCTAssertTrue(host.statusMessage.localizedCaseInsensitiveContains(selected == "gradient_policy" ? "gradient policy" : "trainability"), host.statusMessage)
                XCTAssertFalse(workspace.isBusy)
                XCTAssertNil(host.queuedRequestID)
                XCTAssertFalse(host.isStagingCheckpoint)
            }
        }
    }

}
