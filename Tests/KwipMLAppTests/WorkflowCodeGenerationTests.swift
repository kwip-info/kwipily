import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowCodeGenerationTests: XCTestCase {
    @MainActor private func fixture() async throws -> (WorkspaceStore, WorkflowWorkspaceSession, URL) {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        guard FileManager.default.isExecutableFile(atPath: repo.appendingPathComponent(".venv/bin/python").path) else { throw XCTSkip("Python runtime absent") }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repo, storageOverride: root, detectCapabilities: false)
        await verifyRuntimeForIntegration(workspace)
        workspace.workflowCodeScope = .model
        let id = UUID().uuidString
        var source = try WorkflowCodec.migrateModel(.legacyFlat(ExampleFlows.parallelClassifier), projectID: id).document
        let definition = WorkflowNodeAddress(domain: .control, localID: "model")
        source.nodes.append(.init(address: definition, payload: .modelDefinition))
        source.nodes.append(.init(address: .init(domain: .data, localID: "input"), payload: .dataOperator(operatorID: "kwip.data.sourceCSV", config: [:])))
        source.edges.append(.init(from: .init(address: .init(domain: .model, localID: "output"), port: "definition"), to: .init(address: definition, port: "source"), kind: .modelDefinition))
        _ = try workspace.projectLibrary.saveSource(.workflow(source))
        XCTAssertTrue(workspace.beginWorkflowProject(id: id), workspace.statusMessage)
        return (workspace, try XCTUnwrap(workspace.workflowSession), root)
    }
    @MainActor private func finish(_ workspace: WorkspaceStore) async throws {
        // Join admission/launch before waiting for the actual worker terminal state.
        await workspace.codeGenerationTask?.value
        let deadline = Date().addingTimeInterval(20)
        while workspace.isGeneratingCode && Date() < deadline { try await Task.sleep(nanoseconds: 20_000_000) }
        XCTAssertFalse(workspace.isGeneratingCode)
    }
    @MainActor func testCurrentWorkflowSourceExportsWithoutDataBindingOrTraining() async throws {
        let (workspace, host, root) = try await fixture()
        let original = host.owner.document
        workspace.refreshDraftCode()
        XCTAssertTrue(workspace.isGeneratingCode, workspace.draftCodeError ?? "No generation started")
        try await finish(workspace)
        XCTAssertNil(workspace.draftCodeError)
        XCTAssertTrue(workspace.currentCodeReady)
        XCTAssertTrue(workspace.draftCode.contains("import torch"))
        XCTAssertTrue(workspace.draftCode.contains("def forward"))
        XCTAssertTrue(workspace.graph.nodes.isEmpty)
        XCTAssertEqual(host.owner.document, original)
        XCTAssertNil(host.execution)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("workflow-runs").path))
    }
    @MainActor func testRenameCannotPublishOldSourceAndMultipleDefinitionsRequireSelection() async throws {
        let (workspace, host, _) = try await fixture()
        workspace.refreshDraftCode()
        await workspace.codeGenerationTask?.value
        XCTAssertTrue(workspace.isGeneratingCode)
        XCTAssertNotNil(host.owner.apply(.rename(address: .init(domain: .model, localID: "branch-a"), name: "Fresh alias"), lease: host.owner.lease()))
        try await finish(workspace)
        XCTAssertTrue(workspace.draftCode.isEmpty)
        XCTAssertFalse(workspace.currentCodeReady)
        let second = WorkflowNodeAddress(domain: .control, localID: "other")
        XCTAssertNotNil(host.owner.apply(.insert(.init(address: second, payload: .modelDefinition)), lease: host.owner.lease()))
        _ = host.owner.select(.init(), lease: host.owner.lease())
        workspace.refreshDraftCode()
        XCTAssertFalse(workspace.isGeneratingCode)
        XCTAssertTrue(workspace.draftCodeError?.lowercased().contains("select") == true)
        _ = host.owner.select(.init(nodeAddresses: [.init(domain: .model, localID: "output")]), lease: host.owner.lease())
        workspace.refreshDraftCode()
        try await finish(workspace)
        XCTAssertNil(workspace.draftCodeError)
        XCTAssertTrue(workspace.currentCodeReady)
        XCTAssertTrue(workspace.draftCode.contains("fresh_alias"))
    }
    @MainActor func testModelOwnedInputGeneratesThroughTheRealWorkerWithoutLosingCanonicalOwnership()async throws {
        let(workspace,host,_)=try await fixture()
        let definition=WorkflowNodeAddress(domain:.model,localID:"output")
        _=host.owner.select(.init(nodeAddresses:[definition]),lease:host.owner.lease())
        let context=try WorkflowControlNodeContext(owner:host.owner,address:definition)
        let input=try XCTUnwrap(context.modelInputBoundary)
        XCTAssertTrue(context.setModelInput(configuration:context.modelInputConfiguration))
        XCTAssertEqual(host.owner.document.modelInputs?[definition.localID],input.address.localID)
        workspace.workflowCodeScope = .workflow
        workspace.refreshDraftCode();try await finish(workspace)
        XCTAssertNil(workspace.draftCodeError);XCTAssertTrue(workspace.currentCodeReady)
        XCTAssertNotNil(workspace.draftCodeFiles["workflow.py"])
        XCTAssertNotNil(workspace.draftCodeFiles["preparation.py"])
        let planBytes=try XCTUnwrap(workspace.draftCodeFiles["workflow-plan.json"]).data(using:.utf8)!
        let captured=try JSONDecoder().decode(WorkflowCodePlan.self,from:planBytes)
        XCTAssertEqual(captured.document,host.owner.document)
        XCTAssertEqual(captured.document.modelInputs?[definition.localID],input.address.localID)
        XCTAssertEqual(captured.document.modelSourceVersions?[definition.localID],2)
        XCTAssertEqual(host.owner.document.modelInputs?[definition.localID],input.address.localID)
    }

    @MainActor func testLayerOwnedSettingsGenerateWithCapturedCompilerAndUnchangedModuleGeometry()async throws {
        let(workspace,host,_)=try await fixture()
        let node=try XCTUnwrap(host.owner.document.nodes.first {if case .modelPrimitive("linear",_,_)=$0.payload {return true};return false})
        XCTAssertNotNil(host.owner.apply(.setLayerParameters(address:node.address,settings:["weight":.object(["trainability":.string("frozen")])]),lease:host.owner.lease()))
        workspace.workflowCodeScope = .workflow
        workspace.refreshDraftCode();try await finish(workspace)
        XCTAssertNil(workspace.draftCodeError);XCTAssertTrue(workspace.currentCodeReady)
        let raw=try XCTUnwrap(workspace.draftCodeFiles["workflow-plan.json"])
        let plan=try JSONDecoder().decode(WorkflowCodePlan.self,from:Data(raw.utf8))
        XCTAssertEqual(plan.compilerContractVersion,13)
        XCTAssertEqual(plan.document,host.owner.document)
        XCTAssertTrue(plan.models.allSatisfy {$0.graph.nodes.allSatisfy {$0.parameterSettings == nil}})
    }

}
