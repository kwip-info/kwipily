import XCTest
import Darwin
import KwipMLCore
@testable import KwipMLApp

final class WorkflowExperimentBudgetTests: XCTestCase {
    @MainActor func testRealBudgetedVariantsPreserveAllowanceAndHistory() async throws {
        try await exercise(seconds: 60, epochs: 3, shouldComplete: true)
    }
    @MainActor func testRealBudgetExhaustionStopsPendingVariantsAndRetainsFailure() async throws {
        try await exercise(seconds: 1, epochs: 10_000, shouldComplete: false)
    }
    @MainActor private func exercise(seconds: Int, epochs: Int, shouldComplete: Bool) async throws {
        guard let development = ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {
            throw XCTSkip("Set KWIPML_TEST_DEVELOPMENT_ROOT for real supervised experiment execution.")
        }
        let canonical = try XCTUnwrap(realpath(FileManager.default.temporaryDirectory.path, nil))
        let root = URL(fileURLWithPath: String(cString: canonical)).appendingPathComponent(UUID().uuidString)
        free(canonical)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: URL(fileURLWithPath: development), storageOverride: root, detectCapabilities: false)
        await verifyRuntimeForIntegration(workspace)
        var document = try WorkflowCodec.migrateToVersion4(WorkflowExamples.preparedParallelClassifier(device: "cpu"))
        let index = try XCTUnwrap(document.nodes.firstIndex { if case .runSetup = $0.payload { return true }; return false })
        guard case .runSetup(var config) = document.nodes[index].payload else { return XCTFail("Missing setup") }
        config["epochs"] = .integer(Int64(epochs))
        config["experiment"] = .object(["schemaVersion": 1, "axes": .array([.object(["field": .string("seed"), "values": .array([7,8])])]), "budget": .object(["maximumRuns": 2, "maximumTotalEpochs": .integer(Int64(epochs * 2))])])
        document.nodes[index].payload = .runSetup(config: config)
        _ = try workspace.projectLibrary.saveSource(.workflow(document), id: document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id: document.projectID))
        let host = try XCTUnwrap(workspace.workflowSession), runtime = try XCTUnwrap(workspace.runtime)
        let budget = try ResolvedExecutionBudget.resolve(global: .init(maximumRunSeconds: seconds), projectMaximumRunSeconds: nil)
        let base = try WorkflowExecutionRequest.capture(from: host.owner, train: WorkflowExamples.trainAddress, runtime: runtime, executionBudget: budget, moduleCompositionEvidence:workspace.moduleCompositionDiscoveryEvidence)
        let csv = URL(fileURLWithPath: development).appendingPathComponent("Sources/KwipMLApp/Resources/workflow-observations.csv")
        let groupID = UUID().uuidString
        let storage = try WorkflowExperimentStorageAdmission(workspace: root, projectID: host.owner.projectID,
            groupID: groupID, sourceBytes: Int64(try Data(contentsOf: csv).count * base.plan.sourceAddresses.count),
            variantCount: 2, limitBytes: 1_000_000_000)
        let owned = try WorkflowGroupOwnership.createGroup(workspace: root, projectID: host.owner.projectID, groupID: groupID)
        try storage.bind(to: owned.owner)
        let inputs = try WorkflowExperimentInputs.captureBytes(addresses: base.plan.sourceAddresses, directory: owned.directory) { _ in try Data(contentsOf: csv) }
        let requests = try base.experimentVariants(inputs: inputs, owner: host.owner)
        let group = try WorkflowExperimentGroupStore.create(requests: requests, directory: owned.directory, groupOwner: owned.owner)
        let done = expectation(description: "Budgeted experiment terminal")
        var acquired = 0, released = 0
        let runner = try WorkflowExperimentRunner(requests: requests, owner: host.owner, group: group, groupOwner: owned.owner,
            workspaceDirectory: root, reservation: .init(acquire: { acquired += 1; return UUID() }, release: { _ in released += 1 }),
            storageAdmission: storage,
            onTerminal: { done.fulfill() })
        XCTAssertTrue(runner.start(), runner.error ?? "Start failed")
        await fulfillment(of: [done], timeout: 90)
        if runner.isRunning { runner.cancel(); return XCTFail("Experiment did not terminate") }
        XCTAssertEqual(try WorkflowStorageAdmission(workspace: root).snapshot().reservedGrowthBytes, 0)
        XCTAssertGreaterThan(try WorkflowStorageAdmission(workspace: root).snapshot().retainedBytes, 0)
        XCTAssertEqual(acquired, released)
        XCTAssertFalse(host.owner.isBusy)
        if shouldComplete {
            XCTAssertEqual(group.record.variants.map(\.status), [.completed, .completed], runner.error ?? "Missing result")
            for result in runner.outcomes {
                let prep = try WorkflowLaunchIntent.decode(Data(contentsOf: owned.directory.appendingPathComponent("phase-\(result.runID.uuidString)-preparation.json")))
                let train = try WorkflowLaunchIntent.decode(Data(contentsOf: owned.directory.appendingPathComponent("phase-\(result.runID.uuidString)-training.json")))
                XCTAssertEqual(prep.schemaVersion, 2)
                XCTAssertEqual(prep.executionBudget, budget)
                XCTAssertEqual(train.executionBudget, budget)
                XCTAssertLessThan(try XCTUnwrap(train.remainingSeconds), try XCTUnwrap(prep.remainingSeconds))
            }
        } else {
            XCTAssertEqual(group.record.variants.map(\.status), [.failed, .cancelled], runner.error ?? "Missing timeout")
            XCTAssertEqual(runner.outcomes.count, 1)
            let result = try XCTUnwrap(runner.outcomes.first)
            let receipt = try WorkflowArchiveCodec.decodeOutcome(Data(contentsOf: result.directory.appendingPathComponent("outcome.json")))
            XCTAssertEqual(receipt.failureCode, .wallTimeLimitExceeded)
            XCTAssertNil(receipt.modelManifestSHA256)
        }
        let context = try WorkflowExperimentHistoryReader.captureContext(directory: owned.directory)
        for (index, result) in runner.outcomes.enumerated() {
            let saved = try WorkflowExperimentHistoryReader.verifiedVariant(context: context, ordinal: index + 1, workspaceDirectory: root, policy: .historyPinned)
            XCTAssertEqual(saved.run.outcome.status, result.status)
            XCTAssertEqual(saved.run.capture.executionBudget, budget)
        }
        let before = try Data(contentsOf: owned.directory.appendingPathComponent("group.json"))
        XCTAssertEqual(try WorkflowExperimentRecoveryCoordinator.recover(directory: owned.directory, workspaceDirectory: root), group.record)
        XCTAssertEqual(try Data(contentsOf: owned.directory.appendingPathComponent("group.json")), before)
    }
}
