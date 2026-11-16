import XCTest
import KwipMLCore
@testable import KwipMLApp

final class TrainingPresentationTests: XCTestCase {
    @MainActor func testNewRequestDoesNotInheritCompletedLineageAndCancelPreservesParent() async throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        await verifyRuntimeForIntegration(store)
        store.loadExample(); store.setExecution { $0.device = "cpu"; $0.epochs = 1; $0.sampleLimit = 40 }
        store.run(); try await wait(store)
        let parent = try XCTUnwrap(store.verifiedCheckpointParent)
        store.setExecution { $0.epochs = 2 }
        store.run(continuation: CheckpointContinuation(parent: parent, mode: .exactResume))
        XCTAssertEqual(store.activeTrainingRequest?.mode, "exactResume")
        XCTAssertEqual(store.activeTrainingRequest?.parentRunID, parent.runID)
        try await wait(store)
        let resumed = try XCTUnwrap(store.verifiedCheckpointParent)
        XCTAssertEqual(resumed.lineageMode, "exactResume")
        store.setExecution { $0.epochs = 10000 }
        store.run()
        XCTAssertEqual(store.activeTrainingRequest?.mode, "newTraining")
        XCTAssertNil(store.activeTrainingRequest?.parentRunID)
        XCTAssertEqual(store.verifiedCheckpointParent, resumed)
        store.cancelRun(); try await wait(store)
        XCTAssertNil(store.activeTrainingRequest)
        XCTAssertEqual(store.verifiedCheckpointParent, resumed)
        XCTAssertTrue(store.wasCancelled); XCTAssertFalse(store.lastCompleted)
        store.setExecution { $0.epochs = 1; $0.learningRate = 2 }
        store.run(); try await wait(store)
        XCTAssertNotNil(store.runError)
        XCTAssertNil(store.activeTrainingRequest)
        XCTAssertEqual(store.verifiedCheckpointParent, resumed)
        XCTAssertEqual(try store.checkpointParent(at: resumed.directory), resumed)
    }
    @MainActor func testTrainingUsesOwnedCapturedOperationAndReopensVerifiedArtifacts() async throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        await verifyRuntimeForIntegration(store)
        store.loadExample(); store.setExecution { $0.device = "cpu"; $0.epochs = 1; $0.sampleLimit = 40; $0.gradientCapture = .init(everyEpochs: 1, maxBatchesPerEpoch: 1, maxCaptures: 2) }
        store.run()
        let operationID = try XCTUnwrap(store.activeTrainingRequest?.runID)
        let projectID = try XCTUnwrap(store.activeProjectID)
        try await wait(store)
        XCTAssertNil(store.runError)
        XCTAssertTrue(store.lastCompleted)
        XCTAssertEqual(store.gradientCapture.status, .verified)
        let artifacts = try XCTUnwrap(store.outputDirectory)
        let operation = artifacts.deletingLastPathComponent()
        XCTAssertEqual(operation.lastPathComponent, operationID)
        XCTAssertEqual(operation.deletingLastPathComponent().lastPathComponent, projectID)
        XCTAssertEqual(operation.deletingLastPathComponent().deletingLastPathComponent().lastPathComponent, "workflow-operations")
        let request = try WorkflowToolRequest.decode(Data(contentsOf: operation.appendingPathComponent("request.json")))
        XCTAssertEqual(request.toolMode, .modelTraining)
        XCTAssertTrue(request.inputFiles.contains { $0.path == "input/graph.json" })
        XCTAssertFalse(FileManager.default.fileExists(atPath: operation.appendingPathComponent("input").path))
        XCTAssertFalse(store.generatedCode.isEmpty)
        let parent = try store.checkpointParent(at: artifacts)
        XCTAssertEqual(parent.completedEpoch, 1)
        let record = try XCTUnwrap(store.experiments.first)
        XCTAssertEqual(record.status, "completed")
        XCTAssertNil(record.diagnosticError)
        XCTAssertEqual(record.directory, artifacts.path)
        let reopened = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        reopened.refreshProjects()
        let project = try XCTUnwrap(reopened.projects.first { $0.id == projectID })
        reopened.openManagedProject(project)
        XCTAssertTrue(reopened.lastCompleted)
        XCTAssertEqual(reopened.gradientCapture.status, .verified)
        XCTAssertEqual(reopened.outputDirectory?.standardizedFileURL, artifacts.standardizedFileURL)
        XCTAssertEqual(reopened.generatedCode, store.generatedCode)
        XCTAssertEqual(reopened.verifiedCheckpointParent?.runID, parent.runID)
    }
    @MainActor func testHeldTrainingExitKeepsReservationAndRefusesCancelledOrStalePublication() async throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        for cancel in [true, false] {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: root) }
            let store = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
            await verifyRuntimeForIntegration(store)
            store.loadExample(); store.setExecution { $0.device = "cpu"; $0.epochs = 1; $0.sampleLimit = 20 }
            let exited = expectation(description: "held training exit")
            var release: (() -> Void)?
            store.trainingWorkerLauncher = { executable, args, cwd, line, exit in
                try WorkflowExecutionSession.launchWorker(executable, args, cwd, line, { code, message in
                    release = { exit(code, message) }; exited.fulfill()
                })
            }
            store.run()
            await fulfillment(of: [exited], timeout: 30)
            XCTAssertTrue(store.isBusy)
            XCTAssertFalse(store.lastCompleted)
            XCTAssertNil(store.outputDirectory)
            let before = store.execution
            store.setExecution { $0.seed += 1 }
            XCTAssertEqual(store.execution, before)
            if cancel { store.cancelRun() } else { store.historySessionRevision = UUID() }
            XCTAssertTrue(store.isBusy)
            release?()
            try await wait(store)
            XCTAssertFalse(store.lastCompleted)
            XCTAssertNil(store.outputDirectory)
            XCTAssertNil(store.activeTrainingRequest)
            XCTAssertFalse(store.experiments.contains { $0.status == "completed" })
            if cancel { XCTAssertTrue(store.wasCancelled); XCTAssertEqual(store.experiments.first?.status, "cancelled") }
            else { XCTAssertTrue(store.experiments.isEmpty) }
        }
    }
    @MainActor func testAcceptedTrainingRetainsWorkspaceUntilTerminalPublication() async throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        var owner: WorkspaceStore? = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        await verifyRuntimeForIntegration(owner!)
        owner!.loadExample(); owner!.setExecution { $0.device = "cpu"; $0.epochs = 1; $0.sampleLimit = 20 }
        let exited = expectation(description: "retained training exit")
        var release: (() -> Void)?
        owner!.trainingWorkerLauncher = { executable, args, cwd, line, exit in
            try WorkflowExecutionSession.launchWorker(executable, args, cwd, line, { code, message in
                release = { exit(code, message) }; exited.fulfill()
            })
        }
        weak var retained = owner
        let projectID = try XCTUnwrap(owner!.activeProjectID)
        owner!.run()
        owner = nil
        XCTAssertNotNil(retained, "Accepted capture must retain its owner before its Task first runs")
        await fulfillment(of: [exited], timeout: 30)
        XCTAssertEqual(retained?.isRunning, true)
        release?(); release = nil
        let deadline = Date().addingTimeInterval(15)
        while retained?.isRunning == true && Date() < deadline { try await Task.sleep(nanoseconds: 25_000_000) }
        XCTAssertNotEqual(retained?.isRunning, true)
        let data = try Data(contentsOf: root.appendingPathComponent("experiment-index/\(projectID).json"))
        let records = try JSONDecoder().decode([ExperimentRecord].self, from: data)
        XCTAssertEqual(records.first?.status, "completed")
    }
    @MainActor private func wait(_ store: WorkspaceStore) async throws {
        let deadline = Date().addingTimeInterval(20)
        while store.isRunning && Date() < deadline { try await Task.sleep(nanoseconds: 25_000_000) }
        XCTAssertFalse(store.isRunning)
    }
}
