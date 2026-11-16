import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

@MainActor final class SchedulerHistoryExportTests: XCTestCase {
    func testActualCPUArchiveReopensExportsTraceAndRejectsChangedBytesOrSource() async throws {
        try await verifyActualArchive(device: "cpu", negatives: true)
    }
    func testActualMPSArchiveReopensAndExportsItsVerifiedTrace() async throws {
        try await verifyActualArchive(device: "mps", negatives: false)
    }
    private func verifyActualArchive(device: String, negatives: Bool) async throws {
        guard let development = ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {
            throw XCTSkip("Enable the actual development worker for scheduler History/export checks")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: URL(fileURLWithPath: development), storageOverride: root, detectCapabilities: false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.availableDevices.contains(device), "This acceptance probe requires the actual " + device + " runtime")
        var document = try WorkflowExamples.preparedStepLRClassifier(device: device)
        let setup = try XCTUnwrap(document.nodes.firstIndex { if case .runSetup = $0.payload { return true }; return false })
        if case .runSetup(var config) = document.nodes[setup].payload {
            config["epochs"] = 3; document.nodes[setup].payload = .runSetup(config: config)
        }
        _ = try workspace.projectLibrary.saveSource(.workflow(document), id: document.projectID)
        let data = try DataFlowOperatorRegistry.load(from: URL(fileURLWithPath: development).appendingPathComponent("runner/dataflow-operators.json"))
        let owner = try WorkflowAuthoringStore(library: workspace.projectLibrary, projectID: document.projectID, modelRegistry: workspace.registry, dataRegistry: data)
        let host = try WorkflowWorkspaceSession(workspace: workspace, owner: owner)
        workspace.activeProjectID = document.projectID; workspace.workflowSession = host
        let source = root.appendingPathComponent("private-observations.csv")
        let rows = (0..<64).map { "\($0),\(($0 * 7) % 13),\($0 % 2)" }
        try Data((["signal,context,label"] + rows).joined(separator: "\n").utf8).write(to: source)
        XCTAssertTrue(host.bindings.setPath(source.path, for: WorkflowExamples.sourceAddress, lease: owner.lease(), expectedRevision: 0))
        let finished = expectation(description: "Actual scheduled " + device + " training completes")
        let subscription = host.$outcomes.dropFirst().filter { !$0.isEmpty }.prefix(1).sink { _ in finished.fulfill() }
        defer { subscription.cancel() }
        XCTAssertTrue(host.run(train: WorkflowExamples.trainAddress), host.statusMessage)
        await fulfillment(of: [finished], timeout: 60)
        let outcome = try XCTUnwrap(host.outcomes.first)
        XCTAssertEqual(outcome.status, .completed, outcome.error ?? host.statusMessage)
        XCTAssertNil(outcome.recordError)
        let reference = WorkflowHistoryReference(projectID: document.projectID, directoryID: try XCTUnwrap(UUID(uuidString: outcome.directory.lastPathComponent)))
        let expectedRecipe = try XCTUnwrap(outcome.request.plan.schedulerRecipe)
        host.retire(); workspace.workflowSession = nil
        guard case .verified(let archived) = WorkflowHistoryReader.load(reference, root: root) else {
            return XCTFail("A completed scheduled run must independently reopen after its live host retires")
        }
        XCTAssertEqual(archived.plan.schedulerRecipe, expectedRecipe)
        XCTAssertEqual(archived.capture.plan.schedulerRecipe?.source, expectedRecipe.source)
        let tracePath = "training/artifacts/rate-trace.json"
        let pin = try XCTUnwrap(archived.outcome.artifacts.first { $0.path == tracePath })
        let traceURL = outcome.directory.appendingPathComponent(tracePath)
        let original = try Data(contentsOf: traceURL)
        XCTAssertLessThanOrEqual(pin.bytes, 4_000_000)
        XCTAssertEqual(pin.bytes, original.count); XCTAssertEqual(pin.sha256, WorkflowArchiveCodec.hash(original))
        let trace = try JSONDecoder().decode([String: JSONValue].self, from: original)
        XCTAssertEqual(trace["recipe"], expectedRecipe.configuration.recipeValue)
        guard case .array(let records) = trace["records"] else { return XCTFail("Actual scheduler trace requires records") }
        XCTAssertEqual(records.count, 3)
        XCTAssertNotNil(try WorkflowHistoryReader.loadModelContent(reference, root: root))
        let package = try WorkflowHistoryReader.stageTrainedExport(reference, root: root)
        defer { package.remove() }
        XCTAssertEqual(package.runID, archived.outcome.runID)
        XCTAssertEqual(package.requestIdentity, archived.outcome.requestIdentity)
        let exported = try XCTUnwrap(package.files.first { $0.path == "rate-trace.json" }, "Portable export must preserve the actual scheduler history")
        XCTAssertEqual(exported.bytes, pin.bytes); XCTAssertEqual(exported.sha256, pin.sha256)
        XCTAssertEqual(try Data(contentsOf: package.directory.appendingPathComponent("rate-trace.json")), original)
        XCTAssertTrue(package.files.contains { $0.path == "training_recipe.py" })
        let readme = try String(contentsOf: package.directory.appendingPathComponent("README.md"), encoding: .utf8)
        XCTAssertTrue(readme.contains("rate-trace.json")); XCTAssertTrue(readme.lowercased().contains("scheduler"))
        XCTAssertTrue(readme.lowercased().contains("fresh"))
        let exportManifest = try JSONDecoder().decode([String: JSONValue].self, from: package.manifestBytes)
        XCTAssertEqual(exportManifest["requestIdentity"], .string(archived.outcome.requestIdentity))
        let capturedRecipe = try XCTUnwrap(archived.capture.plan.schedulerRecipe)
        let capturedRecipeValue = try JSONDecoder().decode(JSONValue.self, from: WorkflowArchiveCodec.encode(capturedRecipe))
        XCTAssertEqual(exportManifest["schedulerRecipe"], capturedRecipeValue, "Export retains the canonical settings and qualified canvas source")
        guard case .array(let files) = exportManifest["files"] else { return XCTFail("Export requires a pinned inventory") }
        XCTAssertTrue(files.contains { if case .object(let row) = $0 { return row["path"] == .string("rate-trace.json") && row["sha256"] == .string(pin.sha256) }; return false })
        if !negatives { return }
        // No cached verification may authorize changed or oversized archive bytes.
        for bytes in [Data("{}".utf8), Data(repeating: 32, count: 4_000_001)] {
            try bytes.write(to: traceURL)
            guard case .unavailable = WorkflowHistoryReader.load(reference, root: root) else { return XCTFail("Changed trace must invalidate cold History") }
            XCTAssertThrowsError(try WorkflowHistoryReader.stageTrainedExport(reference, root: root))
        }
        try original.write(to: traceURL)
        // Exact phase inventory must retain this artifact even when every remaining pin is valid.
        let outcomeURL = outcome.directory.appendingPathComponent("outcome.json")
        let outcomeBytes = try Data(contentsOf: outcomeURL)
        var saved = try XCTUnwrap(JSONSerialization.jsonObject(with: outcomeBytes) as? [String: Any])
        let inventory = try XCTUnwrap(saved["artifacts"] as? [[String: Any]])
        saved["artifacts"] = inventory.filter { $0["path"] as? String != tracePath }
        try JSONSerialization.data(withJSONObject: saved, options: [.sortedKeys]).write(to: outcomeURL)
        guard case .unavailable = WorkflowHistoryReader.load(reference, root: root) else { return XCTFail("Omitted trace inventory must refuse") }
        try outcomeBytes.write(to: outcomeURL)
        // Republishing never accepts a later change to the reviewed staged trace.
        try Data("{}".utf8).write(to: package.directory.appendingPathComponent("rate-trace.json"))
        let destination = root.appendingPathComponent("refused-portable")
        XCTAssertThrowsError(try package.publish(to: destination))
        XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))
        // Even a freshly repinned capture cannot silently move the rate-plan source.
        let captureURL = outcome.directory.appendingPathComponent("capture.json")
        let captureBytes = try Data(contentsOf: captureURL)
        var capture = try XCTUnwrap(JSONSerialization.jsonObject(with: captureBytes) as? [String: Any])
        var plan = try XCTUnwrap(capture["plan"] as? [String: Any])
        var recipe = try XCTUnwrap(plan["schedulerRecipe"] as? [String: Any])
        recipe["source"] = ["domain": "control", "localID": "borrowed-schedule"]
        plan["schedulerRecipe"] = recipe; capture["plan"] = plan
        let changedCapture = try JSONSerialization.data(withJSONObject: capture, options: [.sortedKeys])
        try changedCapture.write(to: captureURL)
        saved = try XCTUnwrap(JSONSerialization.jsonObject(with: outcomeBytes) as? [String: Any])
        saved["capture"] = ["path": "capture.json", "bytes": changedCapture.count, "sha256": WorkflowArchiveCodec.hash(changedCapture)]
        try JSONSerialization.data(withJSONObject: saved, options: [.sortedKeys]).write(to: outcomeURL)
        guard case .unavailable = WorkflowHistoryReader.load(reference, root: root) else { return XCTFail("Qualified scheduler source must independently reproduce") }
    }
}
