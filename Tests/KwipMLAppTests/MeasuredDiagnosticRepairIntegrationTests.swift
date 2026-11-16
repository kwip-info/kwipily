import XCTest
import Combine
@testable import KwipMLCore
@testable import KwipMLApp

/// Real worker reachability, not synthetic diagnostic evidence.
final class MeasuredDiagnosticRepairIntegrationTests: XCTestCase {
    @MainActor func testActualTrainingProducesRepairableFindingAndReversibleCanonicalEdit() async throws {
        guard let development = ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else { throw XCTSkip("Requires actual development worker") }
        let root = try canonicalBackendTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: URL(fileURLWithPath: development), storageOverride: root, detectCapabilities: false)
        await verifyRuntimeForIntegration(workspace)
        // Reuse the real numerical corpus' frozen ordered 48/16 training/validation
        // matrices, plus 16 separately authored reserved test rows. Current time
        // Partition preserves those populations without executing a retired split.
        let csv=root.appendingPathComponent("diagnostic-authored-populations.csv")
        try Self.writeAuthoredSource(to:csv,development:URL(fileURLWithPath:development))
        var found = false
        for depth in [8] {
            for rate in [0.03, 0.3] {
                var document = try WorkflowExamples.preparedClassifier(projectID: UUID().uuidString, device: "cpu", model: Self.model(depth: depth))
                let split=try XCTUnwrap(document.nodes.firstIndex {$0.address == .init(domain:.data,localID:"split")})
                document.nodes[split].payload = .dataOperator(operatorID:"kwip.data.partition",config:[
                    "strategy":.string("time"),"timeColumn":.string("observed_at"),"timeEncoding":.string("iso8601Offset"),
                    "trainingEnd":.string("2026-01-01T00:47:00Z"),"validationEnd":.string("2026-01-01T01:03:00Z"),
                    "embargoMilliseconds":0,"entityColumns":.array([]),"entityPolicy":.string("allowAcrossTime")])
                let setup = try XCTUnwrap(document.nodes.firstIndex { if case .runSetup = $0.payload { return true }; return false })
                guard case .runSetup(var settings) = document.nodes[setup].payload else { return XCTFail("Missing setup") }
                settings["epochs"] = .integer(20)
                settings["learningRate"] = .number(rate)
                settings["gradientCapture"] = try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(GradientCaptureConfiguration(everyEpochs: 1, maxBatchesPerEpoch: 1, maxCaptures: 20)))
                document.nodes[setup].payload = .runSetup(config: settings)
                _ = try workspace.projectLibrary.saveSource(.workflow(document))
                XCTAssertTrue(workspace.beginWorkflowProject(id: document.projectID), workspace.statusMessage)
                let host = try XCTUnwrap(workspace.workflowSession)
                XCTAssertTrue(host.bindings.setPath(csv.path, for: WorkflowExamples.sourceAddress, lease: host.owner.lease(), expectedRevision: host.bindings.revision), host.bindings.error ?? "Binding failed")
                let done = expectation(description: "Actual candidate depth \(depth) rate \(rate)")
                let observation = host.$outcomes.dropFirst().filter { !$0.isEmpty }.prefix(1).sink { _ in done.fulfill() }
                guard host.run(train: WorkflowExamples.trainAddress) else { observation.cancel(); return XCTFail(host.statusMessage) }
                await fulfillment(of: [done], timeout: 60)
                observation.cancel()
                if workspace.isBusy { host.cancel() }
                let outcome = try XCTUnwrap(host.outcomes.first)
                guard outcome.status == .completed else {
                    print("REPAIR_CANDIDATE depth=\(depth) rate=\(rate) failed=\(outcome.error ?? host.statusMessage)")
                    continue
                }
                let reference = WorkflowHistoryReference(projectID: host.owner.projectID, directoryID: try XCTUnwrap(UUID(uuidString: outcome.directory.lastPathComponent)))
                guard case .verified(let verified) = WorkflowHistoryReader.load(reference, root: root) else { return XCTFail("Worker result failed verification") }
                let input = try MeasuredDiagnosticInput.from(verified)
                XCTAssertEqual(verified.modelContent?.partitionEvidence?.trainingCount,48)
                XCTAssertEqual(input.validationRows,16)
                let report = try MeasuredDiagnostics.evaluate(input)
                print("REPAIR_CANDIDATE depth=\(depth) rate=\(rate) norms=\(input.gradients.map { $0.l2Norm ?? -1 }) rules=\(report.findings.map(\.ruleID))")
                XCTAssertEqual(input.gradients.count, 20)
                let growth = report.findings.first { $0.ruleID == "gradient.growth.v1" }
                if rate == 0.03 {
                    XCTAssertNil(growth, "The same architecture at the control rate must not claim gradient growth")
                    XCTAssertFalse(report.findings.contains { $0.ruleID == "gradient.nonfinite.v1" })
                    continue
                }
                let finding = try XCTUnwrap(growth, "The pinned real-training fixture must trigger the unchanged growth rule")
                if let path = ProcessInfo.processInfo.environment["KWIPML_REPAIR_NATIVE_FIXTURE"] {
                    let export = URL(fileURLWithPath: path)
                    XCTAssertFalse(FileManager.default.fileExists(atPath: export.path), "Fixture export must be new")
                    try FileManager.default.createDirectory(at: export, withIntermediateDirectories: true)
                    try WorkflowCodec.encode(host.owner.document).write(to: export.appendingPathComponent("Diagnostic repair.kwipml"), options: .withoutOverwriting)
                    try Data(contentsOf: csv).write(to: export.appendingPathComponent("observations.csv"), options: .withoutOverwriting)
                }
                let before = host.owner.document
                let proposal = try MeasuredDiagnosticRepair.propose(input: input, findingID: finding.id, train: WorkflowExamples.trainAddress, owner: host.owner, journalRoot: root)
                XCTAssertEqual(host.owner.document, before)
                try proposal.apply(to: host.owner)
                XCTAssertNotEqual(host.owner.document, before)
                XCTAssertEqual(proposal.proposedLearningRate, rate / 2)
                try proposal.revert(in: host.owner)
                XCTAssertGreaterThan(host.owner.document.revision, before.revision)
                var restored = host.owner.document
                restored.revision = before.revision
                XCTAssertEqual(restored, before, "Revert restores the whole authored construction; revision stays monotonic")
                let receipts = try FileManager.default.contentsOfDirectory(atPath: root.appendingPathComponent("diagnostic-repairs/\(host.owner.projectID)").path)
                XCTAssertEqual(receipts.count, 3, "Intent, applied and reverted receipts must be durable")
                guard case .verified(let retained) = WorkflowHistoryReader.load(reference, root: root) else { return XCTFail("Repair must not mutate captured training evidence") }
                XCTAssertEqual(try MeasuredDiagnosticInput.from(retained), input)
                found = true
                break
            }
            if found { break }
        }
        XCTAssertTrue(found, "The pinned worker fixture did not establish the repair path.")
    }

    private static func writeAuthoredSource(to destination:URL,development:URL)throws {
        let process=Process(),log=Pipe()
        process.executableURL=development.appendingPathComponent(".venv/bin/python")
        process.currentDirectoryURL=development
        process.arguments=["-B","-c","from pathlib import Path; import sys; from scripts.measured_diagnostic_training_fixtures import authored_source; Path(sys.argv[1]).write_text(authored_source(25))",destination.path]
        process.standardOutput=log;process.standardError=log
        try process.run()
        let deadline=Date().addingTimeInterval(10)
        while process.isRunning && Date()<deadline {Thread.sleep(forTimeInterval:0.01)}
        if process.isRunning {process.terminate();throw WorkflowArchiveError("Authored diagnostic source generation exceeded ten seconds")}
        process.waitUntilExit()
        let output=String(decoding:log.fileHandleForReading.readDataToEndOfFile(),as:UTF8.self)
        guard process.terminationStatus==0 else {throw WorkflowArchiveError("Authored diagnostic source refused: "+String(output.prefix(2000)))}
    }

    static func model(depth: Int) -> GraphDocument {
        var nodes: [GraphNode] = [.init(id: "input", type: "input", name: "Two features", config: ["features": .integer(2)])]
        for i in 0..<depth {
            nodes.append(.init(id: "linear\(i)", type: "linear", name: "Affine \(i + 1)", x: Double(i + 1) * 180, config: ["in_features": .integer(i == 0 ? 2 : 16), "out_features": .integer(i == depth - 1 ? 2 : 16)]))
        }
        for i in nodes.indices where nodes[i].type == "linear" { nodes[i].config.removeValue(forKey: "bias") }
        nodes.append(.init(id: "output", type: "output", name: "Class scores", x: Double(depth + 1) * 180))
        let edges = zip(nodes, nodes.dropFirst()).enumerated().map { i, pair in GraphEdge(id: "deep-edge-\(i)", source: pair.0.id, target: pair.1.id) }
        return GraphDocument(schemaVersion: 1, name: "Deep affine diagnostic fixture", nodes: nodes, edges: edges)
    }
}
