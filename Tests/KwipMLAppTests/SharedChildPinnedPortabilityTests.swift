import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

final class SharedChildPinnedPortabilityTests: XCTestCase {
    @MainActor func testActualPinnedRunAndExportRemainPortableAfterSharedSourceMoves() async throws {
        guard let development = ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {
            throw XCTSkip("Enable actual workflow runtime")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: URL(fileURLWithPath: development), storageOverride: root, detectCapabilities: false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.loadPreparedWorkflowExample(device: "cpu", kind: .separableConv2d))
        XCTAssertTrue(workspace.convertActiveWorkflowToFiles(), workspace.statusMessage)
        let host = try XCTUnwrap(workspace.workflowSession)
        let aURL = try XCTUnwrap(host.owner.flowRootURL), folder = aURL.deletingLastPathComponent()
        let aRoot = try FlowDocumentStore.readHead(at: aURL), aPin = try XCTUnwrap(aRoot.dependencies.first)
        let aBytes = try Data(contentsOf: aURL)
        var bSource = host.owner.document; bSource.projectID = UUID().uuidString; bSource.name = "Explicit adopter B"
        var portable = bSource; portable.definitions = []
        let bRoot = FlowDocument(id: UUID(uuidString: bSource.projectID)!, revision: bSource.revision,
                                 payload: .workflow(portable), dependencies: aRoot.dependencies)
        let bURL = folder.appendingPathComponent("B.kwipflow")
        try FlowDocumentCodec.encode(bRoot).write(to: bURL)
        let bLibrary = ProjectLibrary(directory: root.appendingPathComponent("b-private"), externalFlowRoot: bURL, projectID: bSource.projectID)
        let b = try WorkflowSourcePersistence(library: bLibrary, projectID: bSource.projectID)
        let childURL = folder.appendingPathComponent(aPin.relativePath)
        var child = try FlowDocumentStore.readHead(at: childURL)
        var values = SeparableConv2dBlock.defaultValues; values["depthwise_bias"] = .bool(false)
        child.payload = .modelBlock(try SeparableConv2dBlock.definition(values: values))
        child.revision += 1
        try FlowDocumentStore.save(child, at: childURL, expected: aPin.revision)
        let newPin = try FlowDocumentCodec.reference(child)
        let proposal = try b.prepareAdoption(definition: aPin.definition, expectedHead: newPin)
        _ = try b.commitAdoption(proposal)
        XCTAssertEqual(try FlowDocumentStore.readHead(at: bURL).dependencies.first?.revision, newPin)
        XCTAssertEqual(try Data(contentsOf: aURL), aBytes)
        XCTAssertEqual(try FlowDocumentStore.snapshot(at: aURL).workflow(), host.owner.document)
        let dataRegistry = try workspace.workflowDataRegistry()
        let aPlan = try WorkflowCompiler.compile(host.owner.document, train: WorkflowExamples.trainAddress, modelRegistry: workspace.registry, dataRegistry: dataRegistry, contractVersion: 15)
        let bPlan = try WorkflowCompiler.compile(b.state.document, train: WorkflowExamples.trainAddress, modelRegistry: workspace.registry, dataRegistry: dataRegistry, contractVersion: 15)
        XCTAssertNotEqual(aPlan.model.semanticSHA256, bPlan.model.semanticSHA256)
        let bGraph = root.appendingPathComponent("adopted-b-graph.json")
        try JSONEncoder().encode(bPlan.model.graph).write(to: bGraph)
        let done = expectation(description: "Pinned A real training completes")
        let observing = host.$outcomes.dropFirst().filter { !$0.isEmpty }.prefix(1).sink { _ in done.fulfill() }
        XCTAssertTrue(host.run(train: WorkflowExamples.trainAddress))
        await fulfillment(of: [done], timeout: 90); observing.cancel()
        let outcome = try XCTUnwrap(host.outcomes.first)
        XCTAssertEqual(outcome.status, .completed)
        let reference = WorkflowHistoryReference(projectID: host.owner.projectID, directoryID: try XCTUnwrap(UUID(uuidString: outcome.directory.lastPathComponent)))
        guard case .verified(let before) = WorkflowHistoryReader.load(reference, root: root) else { return XCTFail("Expected verified trained History") }
        XCTAssertEqual(before.plan.model.graph, aPlan.model.graph)
        XCTAssertNotEqual(before.plan.model.graph, bPlan.model.graph)
        let package = try WorkflowHistoryReader.stageTrainedExport(reference, root: root)
        defer { package.remove() }
        let export = root.appendingPathComponent("portable-a")
        try package.publish(to: export)
        let checkpoint = try Data(contentsOf: export.appendingPathComponent("checkpoint.pt"))
        let runtime = try XCTUnwrap(workspace.runtime)
        func runPython(_ arguments: [String]) async throws -> String {
            let completed = expectation(description: "Standalone portability check")
            var output: [String] = [], status: Int32?
            let process = try WorkflowExecutionSession.launchWorker(runtime.executableURL, arguments, root,
                { output.append($0) }, { code, _ in status = code; completed.fulfill() })
            defer { process.cancel() }
            await fulfillment(of: [completed], timeout: 30)
            XCTAssertEqual(status, 0, output.joined(separator: "\n"))
            return output.joined(separator: "\n")
        }
        let predictionBefore = try await runPython([export.appendingPathComponent("inference.py").path, "--example"])
        let moved = root.appendingPathComponent("relocated-source-folder")
        try FileManager.default.moveItem(at: folder, to: moved)
        XCTAssertFalse(FileManager.default.fileExists(atPath: aURL.path))
        XCTAssertThrowsError(try workspace.projectLibrary.loadSource(id: host.owner.projectID))
        guard case .verified(let after) = WorkflowHistoryReader.load(reference, root: root) else { return XCTFail("Retained History must reopen without source files") }
        XCTAssertEqual(after.plan.model.graph, aPlan.model.graph)
        XCTAssertEqual(after.modelContent?.code, before.modelContent?.code)
        let replayPackage = try WorkflowHistoryReader.stageTrainedExport(reference, root: root)
        defer { replayPackage.remove() }
        let replay = root.appendingPathComponent("reopened-history-export")
        try replayPackage.publish(to: replay)
        XCTAssertEqual(try Data(contentsOf: replay.appendingPathComponent("checkpoint.pt")), checkpoint)
        let predictionAfter = try await runPython([replay.appendingPathComponent("inference.py").path, "--example"])
        XCTAssertEqual(predictionAfter, predictionBefore)
        let prediction = try JSONDecoder().decode([String: JSONValue].self, from: Data(predictionAfter.utf8))
        XCTAssertEqual(prediction["inputKind"], .string("synthetic prepared example"))
        guard case .array(let logits) = prediction["logits"], logits.count == 2 else { return XCTFail("Expected two actual inference rows") }
        for row in logits {
            guard case .array(let values) = row else { return XCTFail("Expected logits") }
            XCTAssertEqual(values.count, 2)
        }
        let script = """
        import json,sys,torch
        sys.path.insert(0,sys.argv[1])
        from engine import GraphModel
        state=torch.load(sys.argv[2],map_location='cpu',weights_only=True)['modelState']
        b=GraphModel(json.load(open(sys.argv[3])))
        unexpected=sorted(set(state)-set(b.state_dict()))
        assert unexpected and all(key.endswith('.bias') for key in unexpected),unexpected
        try: b.load_state_dict(state,strict=True)
        except RuntimeError: pass
        else: raise AssertionError('A weights unexpectedly matched adopted B architecture')
        print(json.dumps({'retainedStateKeys':len(state),'adoptedStateKeys':len(b.state_dict()),'oldOnlyKeys':unexpected},sort_keys=True))
        """
        let keys = try await runPython(["-c", script, development + "/runner", replay.appendingPathComponent("checkpoint.pt").path, bGraph.path])
        print("PINNED_PORTABILITY oldPin=\(aPin.revision.sha256) adoptedPin=\(newPin.sha256) oldSemantic=\(aPlan.model.semanticSHA256) adoptedSemantic=\(bPlan.model.semanticSHA256) \(keys) prediction=\(predictionAfter)")
        workspace.retireWorkflowSession()
    }
}
