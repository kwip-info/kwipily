import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

/// One genuine managed CPU run; no manually installed descriptor or synthesized probe proof.
@MainActor final class ModuleCompositionManagedRunTests: XCTestCase {
    func testOwnedSourceThreeDiscoveryCPUTrainingAndColdHistory() async throws {
        guard let development = ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {
            throw XCTSkip("Enable the actual development worker for managed composition training")
        }
        let repository = URL(fileURLWithPath: development)
        var root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        root = root.resolvingSymlinksInPath()
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        let runtime = try XCTUnwrap(workspace.runtime)
        let before = try WorkflowRuntimeEvidence.capture(runtime)
        await verifyRuntimeForIntegration(workspace)
        // Until publication is explicitly released this is the expected semantic refusal.
        let proof = try XCTUnwrap(workspace.moduleCompositionDiscoveryEvidence,
            "The actual selected worker must advertise and qualify native composition")
        XCTAssertEqual(proof.runtime, runtime)
        XCTAssertEqual(proof.runtimeEvidence, before)
        XCTAssertEqual(try proof.verifiedDescriptor(for: runtime, evidence: before), workspace.moduleCompositionRuntimeCapability?.descriptor)

        var document = try WorkflowExamples.preparedParallelClassifier(device: "cpu")
        document.schemaVersion = 11
        let definition = try XCTUnwrap(document.definitions.indices.first)
        let body = document.definitions[definition].body
        document.definitions[definition].composition = .init(kind: .graph,
            children: (body.nodes.map(\.id) + body.instances.map(\.id)).map { .init(localID: $0) })
        _ = try workspace.projectLibrary.saveSource(.workflow(document), id: document.projectID)
        let dataRegistry = try DataFlowOperatorRegistry.load(from: repository.appendingPathComponent("runner/dataflow-operators.json"))
        let owner = try WorkflowAuthoringStore(library: workspace.projectLibrary, projectID: document.projectID,
            modelRegistry: workspace.registry, dataRegistry: dataRegistry)
        let host = try WorkflowWorkspaceSession(workspace: workspace, owner: owner)
        workspace.activeProjectID = document.projectID; workspace.workflowSession = host
        defer { host.retire(); workspace.workflowSession = nil }
        let csv = root.appendingPathComponent("observations.csv")
        try Data(contentsOf: repository.appendingPathComponent("Sources/KwipMLApp/Resources/workflow-observations.csv")).write(to: csv)
        XCTAssertTrue(host.bindings.setPath(csv.path, for: WorkflowExamples.sourceAddress,
            lease: owner.lease(), expectedRevision: 0))
        let done = expectation(description: "Managed composition run reaches terminal")
        let observation = host.$outcomes.dropFirst().filter { !$0.isEmpty }.prefix(1).sink { _ in done.fulfill() }
        defer { observation.cancel() }
        guard host.run(train: WorkflowExamples.trainAddress) else { return XCTFail(host.statusMessage) }
        await fulfillment(of: [done], timeout: 65)
        let outcome = try XCTUnwrap(host.outcomes.first)
        guard outcome.status == .completed else { return XCTFail(outcome.error ?? host.statusMessage) }
        XCTAssertNil(outcome.recordError)
        XCTAssertNotNil(outcome.modelManifestSHA256)
        XCTAssertNotNil(outcome.request.plan.model.graph.compositionPlan)
        XCTAssertEqual(outcome.request.archiveCompilerContractVersion, 10)
        XCTAssertEqual(outcome.request.plan.model.provenance?.nativeExpansionContractVersion, 2)
        XCTAssertEqual(outcome.request.moduleCompositionDescriptor, proof.capability.descriptor)
        XCTAssertEqual(try WorkflowRuntimeEvidence.capture(runtime), before, "The actual worker source must remain frozen across discovery and training")

        let reference = WorkflowHistoryReference(projectID: document.projectID,
            directoryID: try XCTUnwrap(UUID(uuidString: outcome.directory.lastPathComponent)))
        host.retire(); workspace.workflowSession = nil
        guard case .verified(let archived) = WorkflowHistoryReader.load(reference, root: root) else {
            return XCTFail("Cold History must verify source3, its captured registry, native hierarchy, and saved state")
        }
        XCTAssertEqual(archived.capture.captureFormatVersion, 6)
        XCTAssertEqual(archived.capture.compilerContractVersion, 10)
        XCTAssertEqual(archived.capture.moduleCompositionDescriptor, proof.capability.descriptor)
        XCTAssertEqual(archived.plan.model.graph.compositionPlan, outcome.request.plan.model.graph.compositionPlan)
        let captured = try XCTUnwrap(archived.plan.model.provenance)
        XCTAssertEqual(captured.sourceCompositeDocument.schemaVersion, 3)
        XCTAssertEqual(try captured.verifiedExpansion(registry: try XCTUnwrap(outcome.request.capturedModelRegistry), context: .preparedWorkflow).graph,
            archived.plan.model.graph)
        XCTAssertEqual(try WorkflowRuntimeEvidence.capture(runtime), before)
    }
}
