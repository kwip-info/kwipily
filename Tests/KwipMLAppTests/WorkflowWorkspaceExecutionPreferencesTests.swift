import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowWorkspaceExecutionPreferencesTests: XCTestCase {
    private func fixture(experiments: Bool = false, continuation: Bool = false) throws -> (URL, WorkspaceStore, WorkflowWorkspaceSession) {
        let (root, originalOwner, runtime) = try WorkflowExecutionRequestTests().setup()
        var owner = originalOwner
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let storage = root.appendingPathComponent("storage")
        try FileManager.default.createDirectory(at: storage, withIntermediateDirectories: true)
        let workspace = WorkspaceStore(rootOverride: runtime.workingDirectoryURL, storageOverride: storage, detectCapabilities: false)
        // This resource/cancellation fixture uses a branched model and optional
        // continuation; admit their reviewed capabilities before exercising capture.
        let execution = try BuiltInCapabilityBridge.executionContract(registry: workspace.registry, runtimeVersion: "2.14.1", availableDevices: [.cpu])
        try workspace.installDiscoveredRuntimeContract(version: "2.14.1", expectedVersion: nil, devices: ["cpu"],
            descriptor: JSONEncoder().encode(execution), tensorCompositionDescriptors: fixtureReviewedCompositionDescriptors(),
            lifecycleDescriptor: WorkflowArchiveCodec.encode(BundledLifecycleCapabilities.entry.runtimeContract))
        if experiments || continuation {
            var document = try WorkflowCodec.migrateToVersion4(owner.document)
            let setup = try XCTUnwrap(document.nodes.firstIndex { if case .runSetup = $0.payload { return true }; return false })
            if experiments, case .runSetup(var config) = document.nodes[setup].payload {
                config["experiment"] = .object(["schemaVersion": 1, "axes": .array([.object(["field": .string("seed"), "values": .array([7, 8])])]), "budget": .object(["maximumRuns": 2, "maximumTotalEpochs": 6])])
                document.nodes[setup].payload = .runSetup(config: config)
            }
            if continuation {
                let train = try XCTUnwrap(document.nodes.firstIndex { $0.address == .init(domain: .control, localID: "train") })
                document.nodes[train].payload = .train(config: ["trainingMode": .string("fineTune")])
            }
            _ = try workspace.projectLibrary.saveSource(.workflow(document), id: document.projectID)
            owner = try WorkflowAuthoringStore(library: workspace.projectLibrary, projectID: document.projectID,
                modelRegistry: originalOwner.modelRegistry, dataRegistry: originalOwner.dataRegistry)
        }
        workspace.activeProjectID = owner.projectID
        let host = try WorkflowWorkspaceSession(workspace: workspace, owner: owner)
        workspace.workflowSession = host
        let source = root.appendingPathComponent("source.csv")
        try Data("a,b,label\n1,2,0\n2,3,1\n".utf8).write(to: source)
        XCTAssertTrue(host.bindings.setPath(source.path, for: .init(domain: .data, localID: "input"), lease: owner.lease(), expectedRevision: 0))
        return (root, workspace, host)
    }
    private func waitUntil(_ predicate: () -> Bool) async throws {
        let deadline = Date().addingTimeInterval(5)
        while !predicate(), Date() < deadline { try await Task.sleep(for: .milliseconds(2)) }
        XCTAssertTrue(predicate())
    }
    func testAcceptedDeadlineIncludesSuspendedSourceCaptureDespiteLaterPreferenceIncrease() async throws {
        let (_, workspace, host) = try fixture()
        XCTAssertTrue(workspace.setExecutionPreferences(try .init(maximumRunSeconds: 1)))
        let capture = host.captureRunSources
        var release: CheckedContinuation<Void, Never>?
        host.captureRunSources = { paths, root, permissions, limit in
            await withCheckedContinuation { release = $0 }
            return try await capture(paths, root, permissions, limit)
        }
        var launches = 0
        host.executionLaunch = { _, _, _, _, done in
            launches += 1
            done(1, "Unexpected worker after capture deadline")
            return .init(cancel: {})
        }
        XCTAssertTrue(host.run(), host.statusMessage)
        try await waitUntil { release != nil }
        let settings=try workspace.applicationSettingsSnapshot()
        _ = try workspace.applyApplicationSettings(workspace.previewApplicationSettings(.execution(try .init(maximumRunSeconds:120)),baseline:settings))
        XCTAssertEqual(try workspace.executionPolicy(for:nil).budget.maximumRunSeconds,120,"Future admissions see new defaults")
        try await Task.sleep(for: .milliseconds(1_100))
        XCTAssertTrue(workspace.isBusy, "Native capture remains joined until it returns")
        release?.resume(); release = nil
        try await waitUntil { !workspace.isBusy }
        XCTAssertEqual(launches, 0)
        XCTAssertNil(host.execution)
        XCTAssertTrue(host.statusMessage.localizedCaseInsensitiveContains("time"), host.statusMessage)
        let stage = workspace.workspaceDirectory.appendingPathComponent("workflow-source-staging")
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: stage.path), [])
    }

    func testRunCapturesPolicyBeforeSourceSuspensionAndKeepsReservationThroughExit() async throws {
        let (_, workspace, host) = try fixture()
        XCTAssertTrue(workspace.setExecutionPreferences(try .init(maximumRunSeconds: 120, cpuThreads: 3, managedStorageLimitBytes: 2_000_000_000)))
        XCTAssertTrue(workspace.setProjectExecutionOverrides(try .init(maximumRunSeconds: 75, cpuThreads: 2), projectID: host.owner.projectID))
        let capture = host.captureRunSources
        var release: CheckedContinuation<Void, Never>?, capturedLimit: Int64?, exit: ((Int32, String) -> Void)?
        host.captureRunSources = { paths, root, permissions, limit in
            capturedLimit = limit
            await withCheckedContinuation { release = $0 }
            return try await capture(paths, root, permissions, limit)
        }
        var envelope: WorkflowOperationRequest?
        host.executionLaunch = { _, args, _, _, done in
            let index = try XCTUnwrap(args.firstIndex(of: "--kwip-operation-supervise"))
            envelope = try .decode(Data(contentsOf: URL(fileURLWithPath: args[index + 1]).appendingPathComponent("request.json")))
            exit = done
            return .init(cancel: {})
        }
        XCTAssertTrue(host.run(), host.statusMessage)
        try await waitUntil { release != nil }
        XCTAssertTrue(workspace.isBusy)
        XCTAssertEqual(capturedLimit, 2_000_000_000)
        XCTAssertTrue(workspace.setExecutionPreferences(try .init(maximumRunSeconds: 10, cpuThreads: 8, managedStorageLimitBytes: 1)))
        XCTAssertTrue(workspace.resetProjectExecutionOverrides(projectID: host.owner.projectID))
        release?.resume(); release = nil
        try await waitUntil { exit != nil }
        XCTAssertEqual(envelope?.executionBudget?.requestedSeconds, 75)
        XCTAssertEqual(envelope?.executionBudget?.resources?.cpuThreads, 2)
        XCTAssertTrue(workspace.isBusy)
        host.cancel()
        XCTAssertTrue(workspace.isBusy)
        exit?(143, "")
        XCTAssertFalse(workspace.isBusy)
        XCTAssertEqual(host.outcomes.first?.status, .cancelled)
        let stage = workspace.workspaceDirectory.appendingPathComponent("workflow-source-staging")
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: stage.path), [])
        XCTAssertEqual(try WorkflowStorageAdmission(workspace: workspace.workspaceDirectory).snapshot().reservedGrowthBytes, 0)
    }
    func testCancellationJoinsSuspendedSourceCaptureBeforeReleasingWorkspace() async throws {
        let (_, workspace, host) = try fixture()
        let capture = host.captureRunSources
        var release: CheckedContinuation<Void, Never>?
        host.captureRunSources = { paths, root, permissions, limit in
            await withCheckedContinuation { release = $0 }
            return try await capture(paths, root, permissions, limit)
        }
        host.executionLaunch = { _, _, _, _, _ in XCTFail("Cancelled capture must not launch"); return .init(cancel: {}) }
        XCTAssertTrue(host.run(), host.statusMessage)
        try await waitUntil { release != nil }
        host.cancel()
        XCTAssertTrue(workspace.isBusy)
        XCTAssertFalse(host.run())
        release?.resume(); release = nil
        try await waitUntil { !workspace.isBusy }
        XCTAssertNil(host.execution)
        XCTAssertTrue(host.outcomes.isEmpty)
    }
    func testInsufficientStorageRejectsBeforeSourceStageAndWorkerAndMalformedPolicyRefusesAcceptance() async throws {
        let (_, workspace, host) = try fixture()
        XCTAssertTrue(workspace.setExecutionPreferences(try .init(managedStorageLimitBytes: 1)))
        host.executionLaunch = { _, _, _, _, _ in XCTFail("Quota refusal must not launch"); return .init(cancel: {}) }
        XCTAssertTrue(host.run(), host.statusMessage)
        try await waitUntil { !workspace.isBusy }
        XCTAssertNil(host.execution)
        XCTAssertFalse(FileManager.default.fileExists(atPath: workspace.workspaceDirectory.appendingPathComponent("workflow-source-staging").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: workspace.workspaceDirectory.appendingPathComponent("workflow-runs").path))
        try Data("invalid".utf8).write(to: workspace.appPreferenceStore.fileURL)
        XCTAssertFalse(host.run())
        XCTAssertFalse(workspace.isBusy)
    }
    func testExperimentQuotaRefusesBeforeGroupCreationOrSourceCopy() async throws {
        let (_, workspace, host) = try fixture(experiments: true)
        XCTAssertTrue(workspace.setExecutionPreferences(try .init(managedStorageLimitBytes: 1)))
        var captures = 0
        host.captureExperimentInputs = { _, _ in captures += 1; throw WorkflowArchiveError("Unexpected source copy") }
        XCTAssertTrue(host.runExperiments())
        try await waitUntil { !host.isStagingExperiment }
        XCTAssertEqual(captures, 0)
        XCTAssertFalse(workspace.isBusy)
        XCTAssertFalse(FileManager.default.fileExists(atPath: workspace.workspaceDirectory.appendingPathComponent("workflow-experiments").path))
    }
    func testCheckpointCaptureReceivesAcceptedQuotaAndCancellationJoinsIt() async throws {
        let (_, workspace, host) = try fixture(continuation: true)
        XCTAssertTrue(workspace.setExecutionPreferences(try .init(managedStorageLimitBytes: 800_000_000)))
        let pin = String(repeating: "a", count: 64)
        let selected = WorkflowCheckpointReference(projectID: host.owner.projectID, directoryID: UUID().uuidString,
            runID: UUID().uuidString, requestIdentity: pin, modelManifestSHA256: pin)
        XCTAssertTrue(host.checkpointBindings.setReference(selected, for: .init(domain: .control, localID: "train"),
            lease: host.owner.lease(), expectedRevision: host.checkpointBindings.revision))
        var release: CheckedContinuation<Void, Never>?, capturedLimit: Int64?
        host.captureCheckpoint = { _, _, limit in
            capturedLimit = limit
            await withCheckedContinuation { release = $0 }
            throw CancellationError()
        }
        XCTAssertTrue(host.run(), host.statusMessage)
        try await waitUntil { release != nil }
        XCTAssertTrue(workspace.isBusy)
        XCTAssertEqual(capturedLimit, 800_000_000)
        host.cancel()
        XCTAssertTrue(workspace.isBusy)
        release?.resume(); release = nil
        try await waitUntil { !workspace.isBusy }
        XCTAssertNil(host.execution)
    }
    func testEvaluationCapturesProjectResourcesBeforeAsynchronousStaging() async throws {
        let helper = WorkflowEvaluationRequestTests()
        let (root, owner, bindings, runtime) = try helper.fixture()
        let storage = root.appendingPathComponent("host")
        try FileManager.default.createDirectory(at: storage, withIntermediateDirectories: true)
        let workspace = WorkspaceStore(rootOverride: runtime.workingDirectoryURL, storageOverride: storage, detectCapabilities: false)
        // This resource/cancellation fixture uses a branched model and optional
        // continuation; admit their reviewed capabilities before exercising capture.
        let execution = try BuiltInCapabilityBridge.executionContract(registry: workspace.registry, runtimeVersion: "2.14.1", availableDevices: [.cpu])
        try workspace.installDiscoveredRuntimeContract(version: "2.14.1", expectedVersion: nil, devices: ["cpu"],
            descriptor: JSONEncoder().encode(execution), tensorCompositionDescriptors: fixtureReviewedCompositionDescriptors(),
            lifecycleDescriptor: WorkflowArchiveCodec.encode(BundledLifecycleCapabilities.entry.runtimeContract))
        workspace.activeProjectID = owner.projectID
        let host = try WorkflowWorkspaceSession(workspace: workspace, owner: owner)
        workspace.workflowSession = host; host.evaluationStage = helper.staged
        let frozen = WorkflowNodeAddress(domain: .control, localID: "frozen")
        XCTAssertTrue(host.frozenBindings.setReference(try XCTUnwrap(bindings.reference(for: frozen)), for: frozen,
            lease: owner.lease(), expectedRevision: host.frozenBindings.revision))
        XCTAssertTrue(workspace.setExecutionPreferences(try .init(maximumRunSeconds: 120, cpuThreads: 2)))
        var exit: ((Int32, String) -> Void)?, request: WorkflowEvaluationWorkerRequest?
        host.evaluationLaunch = { _, args, _, _, done in
            let index = try XCTUnwrap(args.firstIndex(of: "--kwip-evaluate-supervise"))
            request = try .decode(Data(contentsOf: URL(fileURLWithPath: args[index + 1]).appendingPathComponent("request.json")))
            exit = done; return .init(cancel: {})
        }
        XCTAssertTrue(host.evaluate(address: .init(domain: .control, localID: "evaluate")))
        XCTAssertTrue(workspace.setExecutionPreferences(try .init(maximumRunSeconds: 10, cpuThreads: 8)))
        try await waitUntil { exit != nil }
        XCTAssertEqual(request?.executionBudget?.requestedSeconds, 120)
        XCTAssertEqual(request?.executionBudget?.resources?.cpuThreads, 2)
        host.cancel(); exit?(143, "")
        try await waitUntil { !workspace.isBusy }
    }

}
