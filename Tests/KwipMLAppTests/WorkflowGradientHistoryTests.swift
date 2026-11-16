import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

final class WorkflowGradientHistoryTests: XCTestCase {
    @MainActor func testActualWorkflowRetainsVerifiedGradientCoverageAndRejectsMissingOrCorruptReload() async throws {
        guard let development = ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {
            throw XCTSkip("Set KWIPML_TEST_DEVELOPMENT_ROOT for real workflow gradient capture.")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: URL(fileURLWithPath: development), storageOverride: root, detectCapabilities: false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.loadPreparedWorkflowExample(device: "cpu"), workspace.statusMessage)
        let host = try XCTUnwrap(workspace.workflowSession)
        let setup = try XCTUnwrap(host.owner.document.nodes.first { if case .train = $0.payload { return true }; return false })
        guard case .train(var trainConfig) = setup.payload else { return XCTFail("Missing Train") }
        var config=try XCTUnwrap(WorkflowTrainRunSetup.configuration(trainConfig,schemaVersion:host.owner.document.schemaVersion))
        let configuration = GradientCaptureConfiguration(everyEpochs: 1, maxBatchesPerEpoch: 1, maxCaptures: 2)
        config["gradientCapture"] = try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(configuration))
        trainConfig=try WorkflowTrainRunSetup.setting(config,in:trainConfig,schemaVersion:host.owner.document.schemaVersion)
        XCTAssertNotNil(host.owner.apply(.configure(address: setup.address, config: trainConfig), lease: host.owner.lease()))
        let terminal = expectation(description: "Actual captured-gradient training terminates")
        let observation = host.$outcomes.dropFirst().filter { !$0.isEmpty }.prefix(1).sink { _ in terminal.fulfill() }
        defer { observation.cancel() }
        try await awaitWorkflowIntegrationIdle(workspace,host:host)
        guard host.run(train: WorkflowExamples.trainAddress) else {throw WorkflowArchiveError(host.statusMessage)}
        await fulfillment(of: [terminal], timeout: 60)
        try await awaitWorkflowIntegrationIdle(workspace,host:host)
        let outcome = try XCTUnwrap(host.outcomes.first)
        XCTAssertEqual(outcome.status, .completed, outcome.error ?? host.statusMessage)
        XCTAssertNil(outcome.recordError); XCTAssertFalse(workspace.isBusy)
        let reference = WorkflowHistoryReference(projectID: host.owner.projectID,
            directoryID: try XCTUnwrap(UUID(uuidString: outcome.directory.lastPathComponent)))
        guard case .verified(let verified) = WorkflowHistoryReader.load(reference, root: root) else {
            return XCTFail("Actual gradient workflow did not produce verified History")
        }
        let content = try XCTUnwrap(verified.modelContent)
        let artifact = try XCTUnwrap(content.gradientArtifact)
        XCTAssertEqual(verified.gradientArtifact, artifact)
        XCTAssertEqual(artifact.runID, verified.outcome.workerRunID)
        XCTAssertEqual(artifact.device, "cpu"); XCTAssertEqual(artifact.config, configuration)
        XCTAssertEqual(artifact.initialEpoch, 0); XCTAssertEqual(artifact.finalEpoch, 3)
        XCTAssertEqual(artifact.inventory.parameterCount, 6); XCTAssertEqual(artifact.inventory.elementCount, 82)
        XCTAssertEqual(artifact.summaries.map(\.epoch), [1, 2, 3])
        XCTAssertEqual(artifact.summaries.map(\.status), [.captured, .captured, .budgetExhausted])
        XCTAssertEqual(artifact.summaries.map(\.observedBatchCount), [1, 1, 0])
        XCTAssertEqual(artifact.summaries.map(\.totalTrainingRows), [32, 32, 32])
        XCTAssertEqual(artifact.summaries.map(\.totalBatches), [1, 1, 1])
        XCTAssertEqual(artifact.summaries.map(\.observedTrainingRows), [32, 32, 0])
        XCTAssertEqual(artifact.summaries.map(\.captureBudgetRemaining), [1, 0, 0])
        for summary in artifact.summaries {
            for batch in summary.batches {
                XCTAssertEqual(batch.presentParameterCount, 6); XCTAssertEqual(batch.presentElementCount, 82)
                XCTAssertEqual(batch.missingParameterCount, 0); XCTAssertEqual(batch.nonfiniteElementCount, 0)
                XCTAssertEqual(batch.finiteElementCount, 82)
                XCTAssertTrue(try XCTUnwrap(batch.l2Norm).isFinite)
                XCTAssertGreaterThan(try XCTUnwrap(batch.l2Norm), 0)
                XCTAssertTrue(try XCTUnwrap(batch.maxAbs).isFinite)
                XCTAssertGreaterThan(try XCTUnwrap(batch.maxAbs), 0)
            }
        }
        let generalizationContext = try XCTUnwrap(MeasuredDiagnosticInput.generalizationContext(from: verified))
        XCTAssertEqual(generalizationContext.runID, verified.capture.runID)
        XCTAssertEqual(generalizationContext.requestSHA256, verified.capture.requestIdentity)
        XCTAssertTrue(generalizationContext.completeEpochCapture)
        XCTAssertTrue(generalizationContext.unchangedPartitions)
        XCTAssertTrue(generalizationContext.unchangedEvaluation)
        let currentDiagnostic = try MeasuredDiagnostics.evaluateCurrent(MeasuredDiagnosticInput.from(verified), context: generalizationContext)
        XCTAssertEqual(currentDiagnostic.registryVersion, 2)
        XCTAssertEqual(currentDiagnostic.checks.first { $0.ruleID == "generalization.divergence.v1" }?.status, .withheld, "Three complete epochs are genuine evidence, but insufficient for the twenty-epoch trend")
        XCTAssertEqual(content.metrics.count, 3)
        XCTAssertTrue(content.metrics.allSatisfy { $0.trainLoss.isFinite && $0.validationLoss.isFinite })
        let path = "training/artifacts/gradient-summary.json"
        let gradientURL = outcome.directory.appendingPathComponent(path)
        let originalBytes = try Data(contentsOf: gradientURL)
        let archivedFile = try XCTUnwrap(verified.outcome.artifacts.first { $0.path == path })
        XCTAssertEqual(archivedFile.bytes, originalBytes.count)
        XCTAssertEqual(archivedFile.sha256, WorkflowArchiveCodec.hash(originalBytes))
        // The typed result is a retained value. Its access must not reread mutable disk.
        try FileManager.default.removeItem(at: gradientURL)
        XCTAssertEqual(content.gradientArtifact, artifact)
        XCTAssertEqual(verified.gradientArtifact, artifact)
        if case .verified = WorkflowHistoryReader.load(reference, root: root) { XCTFail("Missing gradient artifact must invalidate a fresh load") }
        XCTAssertThrowsError(try WorkflowHistoryReader.loadModelContent(reference, root: root))
        try Data("corrupt gradient document".utf8).write(to: gradientURL)
        if case .verified = WorkflowHistoryReader.load(reference, root: root) { XCTFail("Corrupt gradient artifact must invalidate a fresh load") }
        XCTAssertThrowsError(try WorkflowHistoryReader.loadModelContent(reference, root: root))
        XCTAssertEqual(content.gradientArtifact, artifact)
        try originalBytes.write(to: gradientURL)
        let restored = try XCTUnwrap(WorkflowHistoryReader.loadModelContent(reference, root: root))
        XCTAssertEqual(restored.gradientArtifact, artifact)
        // A second ordinary run retains nil, rather than inventing zero-gradient evidence.
        config.removeValue(forKey: "gradientCapture")
        trainConfig=try WorkflowTrainRunSetup.setting(config,in:trainConfig,schemaVersion:host.owner.document.schemaVersion)
        XCTAssertNotNil(host.owner.apply(.configure(address: setup.address, config: trainConfig), lease: host.owner.lease()))
        let offDone = expectation(description: "Capture-off workflow terminates")
        let offObservation = host.$outcomes.filter { $0.count == 2 }.prefix(1).sink { _ in offDone.fulfill() }
        defer { offObservation.cancel() }
        try await awaitWorkflowIntegrationIdle(workspace,host:host)
        guard host.run(train: WorkflowExamples.trainAddress) else {throw WorkflowArchiveError(host.statusMessage)}
        await fulfillment(of: [offDone], timeout: 60)
        try await awaitWorkflowIntegrationIdle(workspace,host:host)
        let off = try XCTUnwrap(host.outcomes.first)
        XCTAssertEqual(off.status, .completed, off.error ?? "")
        let offReference = WorkflowHistoryReference(projectID: host.owner.projectID,
            directoryID: try XCTUnwrap(UUID(uuidString: off.directory.lastPathComponent)))
        let offContent = try XCTUnwrap(WorkflowHistoryReader.loadModelContent(offReference, root: root))
        XCTAssertNil(offContent.gradientArtifact)
        XCTAssertFalse(FileManager.default.fileExists(atPath: off.directory.appendingPathComponent(path).path))
        XCTAssertEqual(restored.gradientArtifact, artifact, "A newer off run must not replace previously verified values")
    }
}
