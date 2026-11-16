import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowNavigationTests: XCTestCase {
    func testUniversalNavigationAndRunGateUseCanonicalSourceInsteadOfLegacyPlaceholder() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let workspace = WorkspaceStore(rootOverride: repo, storageOverride: root, detectCapabilities: false)
        workspace.newProject()
        let host = try XCTUnwrap(workspace.workflowSession)
        XCTAssertEqual(workspace.projectSections, ["Build", "History", "Code", "Settings"])
        XCTAssertFalse(workspace.canRunCurrentFlow)
        XCTAssertNotNil(host.owner.apply(.insert(.init(address: .init(domain: .control, localID: "train"), payload: .train(config: [:]))), lease: host.owner.lease()))
        await settle(host.canvas)
        XCTAssertFalse(workspace.validation.isValid, "Legacy placeholder is not the workflow validator")
        XCTAssertEqual(workspace.canRunCurrentFlow, workspace.runtime != nil, "Clicking Run must reach canonical validation and actionable errors")
        workspace.isPreparingData = true
        XCTAssertFalse(workspace.canRunCurrentFlow)
        workspace.isPreparingData = false
        XCTAssertTrue(workspace.beginProject(ExampleFlows.parallelClassifier))
        XCTAssertEqual(workspace.projectSections, ["Data", "Build", "Run", "Code", "Experiments", "Settings"])
        XCTAssertNil(workspace.workflowHistory)
    }
    func testEstimatorExamplesEnableUniversalRunAndRespectBusyState() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let workspace = WorkspaceStore(rootOverride: repo, storageOverride: root, detectCapabilities: false)
        XCTAssertNotNil(workspace.runtime)
        for backend in ["sklearn", "tensorflow"] {
            XCTAssertTrue(workspace.loadBackendWorkflowExample(backendID: backend))
            let host = try XCTUnwrap(workspace.workflowSession)
            await settle(host.canvas)
            XCTAssertTrue(host.trainTerminals.isEmpty)
            XCTAssertTrue(workspace.canRunCurrentFlow, "Estimator-only workflows must reach canonical Run validation")
            host.owner.setExecutionBusy(true)
            XCTAssertFalse(workspace.canRunCurrentFlow)
            host.owner.setExecutionBusy(false)
        }
    }

    func testRunCueRevealsMissingNamedMappingWithoutEditingTheCanvas() async throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let workspace=WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false)
        XCTAssertTrue(workspace.loadComponentWorkflowExample("model-output",device:"cpu"))
        let host=try XCTUnwrap(workspace.workflowSession)
        await settle(host.canvas)
        let input=try XCTUnwrap(WorkflowModelInput.boundary(for:ModelOutputPracticeExample.address,in:host.owner.document))
        XCTAssertTrue(host.canvas.apply(.setModelInput(definition:ModelOutputPracticeExample.address,type:"input",configuration:["features":2],destinations:WorkflowModelInput.destinations(of:input,in:host.owner.document),inputName:"input"),lease:host.owner.lease()))
        await settle(host.canvas)
        let finding=try XCTUnwrap(workspace.currentRunConfigurationFinding)
        XCTAssertFalse(workspace.canRunCurrentFlow)
        let document=host.owner.document
        XCTAssertTrue(workspace.revealRunConfigurationFinding())
        XCTAssertEqual(host.owner.selection.nodeAddresses,[finding.address])
        XCTAssertEqual(host.owner.document,document)
        XCTAssertTrue(workspace.showInspector)
        workspace.isRunning=true
        XCTAssertFalse(workspace.revealRunConfigurationFinding())
        workspace.isRunning=false
    }
    private func settle(_ canvas:WorkflowCanvasAdapter)async {
        for _ in 0..<200 {
            if !canvas.isInferringTensors {return}
            try? await Task.sleep(nanoseconds:10_000_000)
        }
        XCTFail("Configuration inference did not settle")
    }

}
