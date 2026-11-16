import XCTest
import Combine
import Darwin
import KwipMLCore
@testable import KwipMLApp

/// Real owned worker execution with a private, faithfully declared prepublication registry.
/// No cached training artifacts or manually constructed discovery proof are used.
@MainActor final class BagWorkflowSessionTests: XCTestCase {
    struct Captured {
        let reference: WorkflowHistoryReference
        let directory: URL
        let plan: WorkflowExecutionPlan
        let workerRunID: String
    }
    private final class Reservation {
        var acquired = 0, released = 0
        var value: WorkflowExecutionReservation {
            .init(acquire: { [self] in acquired += 1; return UUID() },
                  release: { [self] _ in released += 1 })
        }
    }
    /// Caller owns root and its lifetime; also reused by cold History tests.
    func capture(objective: Bool, device: String = "cpu", root: URL) async throws -> Captured {
        try await execute(objective: objective, device: device, root: root)
    }
    func testActualCPUBinaryPacketSessionPublishesNamedCaptures() async throws {
        let root = try temporaryRoot(); defer { try? FileManager.default.removeItem(at: root) }
        _ = try await capture(objective: false, root: root)
    }
    func testActualCPUObjectivePacketSessionBindsLiveAndSavedLedger() async throws {
        let root = try temporaryRoot(); defer { try? FileManager.default.removeItem(at: root) }
        _ = try await capture(objective: true, root: root)
    }
    func testActualMPSPacketSessionsKeepTypedInputAndObjectiveResults() async throws {
        for objective in [false, true] {
            let root = try temporaryRoot(); defer { try? FileManager.default.removeItem(at: root) }
            _ = try await capture(objective: objective, device: "mps", root: root)
        }
    }
    func testActualPacketCapturePermissionsAreIndependent() async throws {
        for activation in [false, true] { for rows in [false, true] {
            let root = try temporaryRoot(); defer { try? FileManager.default.removeItem(at: root) }
            _ = try await execute(objective: true, root: root,
                                 policy: .init(activationValues: activation, predictionRows: rows))
        }}
    }
    func testActualWorkerContradictionsRefusePublicationAndReleaseOwnership() async throws {
        for fault in ["count", "device", "epoch", "metadata"] {
            let root = try temporaryRoot(); defer { try? FileManager.default.removeItem(at: root) }
            _ = try await execute(objective: true, root: root, fault: fault)
        }
    }
    private func temporaryRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return try physicalRoot(root)
    }
    private func physicalRoot(_ root: URL) throws -> URL {
        // WorkflowOperationOwnership requires byte-for-byte realpath identity;
        // Foundation's symlink resolution may preserve macOS /var aliases.
        guard let path = realpath(root.path, nil) else { throw WorkflowFileLeaseError.unsafePath }
        defer { free(path) }
        return URL(fileURLWithPath: String(cString: path), isDirectory: true)
    }
    private func execute(objective: Bool, device: String = "cpu", root: URL,
                         policy: TrainingSampleCapturePolicy = .factory,
                         fault: String? = nil) async throws -> Captured {
        guard let development = ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {
            throw XCTSkip("Enable the genuine development worker for owned packet Session tests")
        }
        let root = try physicalRoot(root)
        let repository = URL(fileURLWithPath: development)
        let registry = try BagWorkflowArchiveTests().registry()
        let runtimeRoot = try BagRuntimeTestFixture.runtime(repository: repository, registry: registry)
        let workspace = WorkspaceStore(rootOverride: runtimeRoot, storageOverride: root, detectCapabilities: false)
        let runtime = try XCTUnwrap(workspace.runtime)
        let before = try WorkflowRuntimeEvidence.capture(runtime)
        await verifyRuntimeForIntegration(workspace)
        let proof = try XCTUnwrap(workspace.moduleCompositionDiscoveryEvidence,
                                  "The copied actual worker must qualify source3/capture7 support")
        if device == "mps" && !workspace.availableDevices.contains("mps") { throw XCTSkip("Actual worker reports MPS unavailable") }
        XCTAssertEqual(try proof.verifiedDescriptor(for: runtime, evidence: before), workspace.moduleCompositionRuntimeCapability?.descriptor)
        var document = try BagWorkflowArchiveTests().fixture(packed: true, weighted: true, objective: objective)
        for index in document.nodes.indices {
            switch document.nodes[index].payload {
            case .runSetup(var config):
                config["device"] = .string(device); config["epochs"] = 2; config["batchSize"] = 8
                document.nodes[index].payload = .runSetup(config: config)
            case .dataOperator(_, _) where document.nodes[index].address == .init(domain: .data, localID: "cast"):
                let columns: [String: JSONValue] = ["i0": .string("int64"), "i1": .string("int64"), "length": .string("int64"),
                    "w0": .string("float64"), "w1": .string("float64"), "label": .string("int64"),
                    "y0": .string("float64"), "y1": .string("float64")]
                document.nodes[index].payload = .dataOperator(operatorID: "kwip.data.columns", config: [
                    "rows": .array(columns.keys.sorted().map { .object(["source": .string($0), "included": .bool(true), "outputName": .string($0), "targetType": columns[$0]!]) }),
                    "newColumns": .string("keep"), "castFailure": .string("error"), "trimText": .bool(false), "missingPolicy": .string("error")])
            default: break
            }
        }
        // New executions use the exact authoring upgrade; historical fixture
        // construction and archived bytes remain unchanged.
        document = try RetiredTrainingRecipeMigration.migrate(document)
        let library = workspace.projectLibrary
        _ = try library.saveSource(.workflow(document), id: document.projectID)
        let data = try DataFlowOperatorRegistry.load(from: runtimeRoot.appendingPathComponent("runner/dataflow-operators.json"))
        let owner = try WorkflowAuthoringStore(library: library, projectID: document.projectID, modelRegistry: registry, dataRegistry: data)
        let budget = try ResolvedExecutionBudget.resolve(global: .init(maximumRunSeconds: 60), projectMaximumRunSeconds: nil,
            resources: .init(cpuThreads: 2, maximumResidentBytes: 1_073_741_824))
        let request = try WorkflowExecutionRequest.capture(from: owner, train: WorkflowExamples.trainAddress, runtime: runtime,
            executionBudget: budget, sampleCapture: policy, moduleCompositionEvidence: proof)
        XCTAssertNotNil(request.plan.bagInput)
        XCTAssertEqual(request.document.schemaVersion, objective ? 13 : 12)
        XCTAssertEqual(request.archiveCompilerContractVersion, objective ? 16 : 11)
        let csv = root.appendingPathComponent("observations.csv")
        var text = "i0,i1,w0,w1,length,label,y0,y1\n"
        for row in 0..<32 { text += "1,2,0.5,1,\(row % 3),\(row % 2),0.25,-0.5\n" }
        try Data(text.utf8).write(to: csv)
        let directoryID = UUID(), directory = root.appendingPathComponent("workflow-runs/\(document.projectID)/\(directoryID.uuidString)")
        let reservation = Reservation(), finished = expectation(description: "Actual packet Session terminates")
        var outcome: WorkflowExecutionOutcome?, injected = 0, workerRunID: String?
        let launcher: WorkflowWorkerLauncher = { executable, arguments, working, line, exit in
            try WorkflowExecutionSession.launchWorker(executable, arguments, working, { raw in
                var delivered = raw
                if let bytes = raw.data(using: .utf8), var frame = (try? JSONSerialization.jsonObject(with: bytes)) as? [String: Any] {
                    let event = frame["event"] as? String
                    if event == "started", frame["trainingSamples"] != nil { workerRunID = frame["runID"] as? String }
                    if let fault, injected == 0 {
                        if event == "started", frame["trainingSamples"] != nil, fault == "count" { frame["trainingSamples"] = 999; injected += 1 }
                        if event == "started", frame["trainingSamples"] != nil, fault == "device" { frame["device"] = "mps"; injected += 1 }
                        if event == "epoch", fault == "epoch", let loss = frame["trainLoss"] as? Double { frame["trainLoss"] = loss + 10; injected += 1 }
                        if event == "completed", frame["predictionInput"] != nil, fault == "metadata" { frame["predictionInput"] = ["kind": "dense", "sampleShape": [2]]; injected += 1 }
                        if injected > 0, let changed = try? JSONSerialization.data(withJSONObject: frame) { delivered = String(decoding: changed, as: UTF8.self) }
                    }
                }
                line(delivered)
            }, exit)
        }
        let run = WorkflowExecutionSession(request: request, owner: owner,
            sources: [WorkflowExamples.sourceAddress: try .capture(csv)], directory: directory,
            reservation: reservation.value, launch: launcher, onTerminal: { outcome = $0; finished.fulfill() })
        var observed: [EpochMetric] = []
        let live = run.$liveMetrics.sink { observed = $0 }
        defer { live.cancel(); if owner.isBusy { run.cancel() } }
        XCTAssertTrue(run.start())
        await fulfillment(of: [finished], timeout: 65)
        if owner.isBusy { run.cancel() }
        XCTAssertEqual(reservation.acquired, 1); XCTAssertEqual(reservation.released, 1); XCTAssertFalse(owner.isBusy)
        XCTAssertEqual(try WorkflowRuntimeEvidence.capture(runtime), before)
        if fault != nil {
            XCTAssertEqual(injected, 1, "The negative must actually contradict a real training frame")
            XCTAssertEqual(run.phase, .failed); XCTAssertNotNil(outcome?.error)
            XCTAssertNil(run.bagPredictionPreview); XCTAssertNil(run.modelCaptures)
            XCTAssertNil(run.objectiveEvaluation); XCTAssertNil(outcome?.modelManifestSHA256)
        } else {
            XCTAssertEqual(run.phase, .completed, run.statusMessage + " / " + (outcome?.error ?? ""))
            XCTAssertNil(outcome?.error); XCTAssertNil(outcome?.recordError)
            XCTAssertNotNil(outcome?.modelManifestSHA256)
            XCTAssertNil(run.predictionPreview); XCTAssertNil(run.objectivePredictionPreview)
            XCTAssertEqual(run.metrics.count, 2)
            XCTAssertEqual(run.metrics.map(\.trainLoss), observed.map(\.trainLoss))
            XCTAssertEqual(run.metrics.map(\.validationLoss), observed.map(\.validationLoss))
            XCTAssertTrue(run.metrics.allSatisfy { $0.trainLoss.isFinite && $0.validationLoss.isFinite })
            let packet = try XCTUnwrap(run.bagPredictionPreview)
            XCTAssertEqual(packet.runID, workerRunID); XCTAssertEqual(packet.evaluationCount, 8)
            XCTAssertEqual(packet.rows.count, policy.predictionRows ? 8 : 0)
            XCTAssertEqual(packet.captureStatus, policy.predictionRows ? .captured : .omittedByPolicy)
            if policy.predictionRows {
                XCTAssertTrue(packet.rows.contains { $0.input.indices.isEmpty }, "Whole empty bags survive named packet capture")
                XCTAssertTrue(packet.rows.allSatisfy { $0.input.perSampleWeights?.count == $0.input.indices.count })
            }
            let captures = try XCTUnwrap(run.modelCaptures)
            let origin = try XCTUnwrap(request.plan.model.origins.first { $0.value.localNodeID == "bags" }?.value)
            let ports = try XCTUnwrap(captures.ports(instancePath: origin.instancePath, localNodeID: origin.localNodeID))
            XCTAssertEqual(Set(ports.keys), Set(["indices", "offsets", "per_sample_weights"]))
            for tensor in ports.values {
                guard case .object(let value) = tensor else { throw WorkflowPhaseError("Named source capture is not a tensor object") }
                XCTAssertEqual(value["captureStatus"], .string(policy.activationValues ? "captured" : "omittedByPolicy"))
                if !policy.activationValues { XCTAssertEqual(value["values"], .array([])) }
            }
            if objective {
                XCTAssertTrue(run.metrics.allSatisfy { $0.accuracy == nil })
                XCTAssertEqual(run.objectiveEvaluation?.componentID, "loss_mse")
                XCTAssertTrue(packet.rows.allSatisfy { $0.label == nil && $0.predicted == nil })
            } else { XCTAssertNil(run.objectiveEvaluation); XCTAssertTrue(run.metrics.allSatisfy { $0.accuracy != nil }) }
        }
        return Captured(reference: .init(projectID: document.projectID, directoryID: directoryID), directory: directory,
                        plan: request.plan, workerRunID: try XCTUnwrap(workerRunID))
    }
}
