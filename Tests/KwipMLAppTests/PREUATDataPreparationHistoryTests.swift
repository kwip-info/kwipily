import XCTest
import KwipMLCore
@testable import KwipMLApp

/// Direct negative controls for the immutable standalone preparation verifier.
/// This creates its own actual worker evidence; it never borrows a user's run.
@MainActor final class PreUATDataPreparationHistoryTests: XCTestCase {
    func testColdHistoryRefusesChangedMissingInputAndForeignReceiptWithoutRewritingFiles() async throws {
        guard let python = ProcessInfo.processInfo.environment["KWIPML_BACKEND_TEST_PYTHON"] else {
            throw XCTSkip("Select the real isolated runner interpreter.")
        }
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("original.csv")
        try FileManager.default.copyItem(at: repository.appendingPathComponent("runner/dataflow_examples/data02-observations.csv"), to: source)
        let sourceBytes = try Data(contentsOf: source)
        let graph = try DataFlowCodec.decodeGraph(Data(contentsOf: repository.appendingPathComponent("runner/dataflow_examples/data02-output.json")))
        let document = WorkflowDocument(projectID: UUID().uuidString, name: "Immutable preparation evidence", nodes: graph.nodes.enumerated().map { index, node in
            WorkflowNode(address: .init(domain: .data, localID: node.id), name: node.id, x: Double(index) * 200, y: 100, payload: .dataOperator(operatorID: node.operatorID, config: node.config))
        }, edges: graph.edges.enumerated().map { index, edge in
            WorkflowEdge(id: "edge-\(index)", from: .init(address: .init(domain: .data, localID: edge.from.node), port: edge.from.port), to: .init(address: .init(domain: .data, localID: edge.to.node), port: edge.to.port), kind: .table)
        })
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        _ = try workspace.projectLibrary.saveSource(.workflow(document), id: document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id: document.projectID), workspace.statusMessage)
        let host = try XCTUnwrap(workspace.workflowSession)
        XCTAssertTrue(host.bindings.setPath(source.path, for: .init(domain: .data, localID: "source"), lease: host.owner.lease(), expectedRevision: host.bindings.revision))
        let destination = root.appendingPathComponent("saved-output")
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        host.outputDestinations.pickDirectory = { destination }
        host.canvas.selectNode(try XCTUnwrap(host.canvas.projection.displayIDByAddress[.init(domain: .data, localID: "tool")]), extending: false)
        XCTAssertTrue(host.outputDestinations.chooseDestination(for: .init(domain: .data, localID: "tool"), lease: host.owner.lease()))
        host.executionLaunch = { _, arguments, directory, onLine, onExit in
            try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath: python), arguments, directory, onLine, onExit)
        }
        XCTAssertTrue(host.run(), host.statusMessage)
        let deadline = Date().addingTimeInterval(60)
        while host.dataPreparation?.isRunning == true && Date() < deadline { try await Task.sleep(for: .milliseconds(50)) }
        XCTAssertEqual(host.dataPreparation?.phase, .completed, host.dataPreparation?.statusMessage ?? "Missing preparation session")
        let history = WorkflowDataPreparationHistory(root: workspace.workspaceDirectory, projectID: document.projectID)
        await history.reloadAndWait()
        let entry = try XCTUnwrap(history.entries.first)
        _ = try await history.verify(entry)
        let listed=try await WorkflowRetainedHistoryReader.list(root:workspace.workspaceDirectory,projectID:document.projectID,family:.preparation)
        let retained=try XCTUnwrap(listed.records.first {$0.operationID==entry.id})
        let verified=try await WorkflowRetainedHistoryReader.inspect(root:workspace.workspaceDirectory,projectID:document.projectID,metadata:retained)
        XCTAssertEqual(verified.provenance.verification,"verifiedRetainedArtifacts")
        XCTAssertEqual(verified.provenance.artifactCount,entry.artifacts.count)
        XCTAssertEqual(host.owner.document,document)
        let operation = entry.artifactDirectory.deletingLastPathComponent()
        let requestURL = operation.appendingPathComponent("request.json")
        let receiptURL = operation.appendingPathComponent("preparation-history.json")
        let manifestURL = entry.artifactDirectory.appendingPathComponent("manifest.json")
        let outputURL = destination.appendingPathComponent("prepared-orders.csv")
        let requestBytes = try Data(contentsOf: requestURL)
        let receiptBytes = try Data(contentsOf: receiptURL)
        let manifestBytes = try Data(contentsOf: manifestURL)
        let outputBytes = try Data(contentsOf: outputURL)
        let request = try WorkflowToolRequest.decode(requestBytes)
        let captured = operation.appendingPathComponent(try XCTUnwrap(request.inputFiles.first { $0.path.hasPrefix("input/sources/") }).path)
        let capturedBytes = try Data(contentsOf: captured)

        func assertPreserved(receipt expectedReceipt: Data) throws {
            XCTAssertEqual(try Data(contentsOf: source), sourceBytes)
            XCTAssertEqual(try Data(contentsOf: outputURL), outputBytes)
            XCTAssertEqual(try Data(contentsOf: requestURL), requestBytes)
            XCTAssertEqual(try Data(contentsOf: manifestURL), manifestBytes)
            XCTAssertEqual(try Data(contentsOf: receiptURL), expectedReceipt)
        }
        func assertRefused(_ description: String) async {
            do { _ = try await history.verify(entry); XCTFail("Verified \(description)") } catch { }
            do {_ = try await WorkflowRetainedHistoryReader.inspect(root:workspace.workspaceDirectory,projectID:document.projectID,metadata:retained);XCTFail("Retained reader verified \(description)")} catch {}
            await history.reloadAndWait()
            XCTAssertTrue(history.entries.isEmpty, description)
            XCTAssertFalse(history.status.isEmpty, description)
        }

        // Same-size mutation proves the content pin matters, not merely length.
        var changed = capturedBytes
        let index = try XCTUnwrap(changed.indices.last)
        changed[index] ^= 1
        try changed.write(to: captured)
        await assertRefused("changed captured source")
        XCTAssertEqual(try Data(contentsOf: captured), changed)
        try assertPreserved(receipt: receiptBytes)
        try capturedBytes.write(to: captured)
        _ = try await history.verify(entry)

        try FileManager.default.removeItem(at: captured)
        await assertRefused("missing captured source despite an intact original source")
        XCTAssertFalse(FileManager.default.fileExists(atPath: captured.path), "History must not re-fetch the original source.")
        try assertPreserved(receipt: receiptBytes)
        try capturedBytes.write(to: captured)
        _ = try await history.verify(entry)

        var receipt = try XCTUnwrap(JSONSerialization.jsonObject(with: receiptBytes) as? [String: Any])
        receipt["projectID"] = UUID().uuidString
        let foreignReceipt = try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys])
        try foreignReceipt.write(to: receiptURL)
        await assertRefused("foreign receipt project identity")
        try assertPreserved(receipt: foreignReceipt)
        XCTAssertEqual(try Data(contentsOf: captured), capturedBytes)
        try receiptBytes.write(to: receiptURL)
        _ = try await history.verify(entry)
    }
}
