import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowPreparationCodeTests: XCTestCase {
    @MainActor func testDataOnlyCodeDoesNotRequireModelOrSourceBinding() throws {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repo, storageOverride: root, detectCapabilities: false)
        var document = WorkflowDocument(projectID: UUID().uuidString, name: "Preparation only")
        document.schemaVersion = 13
        document.nodes = [.init(address: .init(domain: .data, localID: "source"), name: "Observations",
            payload: .dataOperator(operatorID: "kwip.data.sourceCSV", config: CSVParsingConfiguration.newDefaults))]
        _ = try workspace.projectLibrary.saveSource(.workflow(document), id: document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id: document.projectID), workspace.statusMessage)
        XCTAssertTrue(workspace.codeValidation.isValid, workspace.codeValidation.issues.map(\.message).joined(separator: "\n"))
        let captured = try workspace.codeGenerationSnapshot()
        XCTAssertEqual(captured.inputName, "input/workflow-code.json")
        XCTAssertFalse(String(decoding: captured.bytes, as: UTF8.self).contains(root.path))
        XCTAssertEqual(workspace.workflowSession?.owner.document, document)
        XCTAssertNil(workspace.workflowSession?.execution)
    }
    @MainActor func testDataOnlyManagedCodeProducesFilesWithoutReadingCSV() async throws {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repo, storageOverride: root, detectCapabilities: false)
        await verifyRuntimeForIntegration(workspace)
        var document = WorkflowDocument(projectID: UUID().uuidString, name: "Data only")
        document.schemaVersion = 13
        document.nodes = [.init(address: .init(domain: .data, localID: "source"), payload: .dataOperator(operatorID: "kwip.data.sourceCSV", config: CSVParsingConfiguration.newDefaults))]
        _ = try workspace.projectLibrary.saveSource(.workflow(document), id: document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id: document.projectID), workspace.statusMessage)
        workspace.refreshDraftCode()
        await workspace.codeGenerationTask?.value
        let deadline = Date().addingTimeInterval(20)
        while workspace.isGeneratingCode && Date() < deadline { try await Task.sleep(nanoseconds: 20_000_000) }
        XCTAssertFalse(workspace.isGeneratingCode)
        XCTAssertNil(workspace.draftCodeError)
        XCTAssertTrue(workspace.currentCodeReady)
        XCTAssertNotNil(workspace.draftCodeFiles["workflow.py"])
        XCTAssertFalse(workspace.draftCodeFiles.keys.contains("model.py"))
        XCTAssertNil(workspace.workflowSession?.execution)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("workflow-runs").path))
        let runtime = try XCTUnwrap(workspace.runtime)
        let evidence = try XCTUnwrap(workspace.codeRuntimeEvidence)
        let contents = try WorkflowCodePackage.contents(files: workspace.draftCodeFiles, runtime: runtime, expectedEvidence: evidence)
        XCTAssertEqual(contents["runner/laya-LICENSE.txt"], try Data(contentsOf: repo.appendingPathComponent("runner/laya-LICENSE.txt")))
        let package = root.appendingPathComponent("Exported Code")
        try WorkflowCodePackage.write(contents, to: package)
        let csv = root.appendingPathComponent("source.csv")
        try "x,y\n1,2\n3,4\n".write(to: csv, atomically: true, encoding: .utf8)
        let bindings = root.appendingPathComponent("bindings.json")
        try JSONSerialization.data(withJSONObject: ["sources": ["source": csv.path]]).write(to: bindings)
        let stdout = root.appendingPathComponent("replay.log")
        FileManager.default.createFile(atPath: stdout.path, contents: nil)
        let status = try await Task.detached(priority: .userInitiated) {
            let process = Process(), log = try FileHandle(forWritingTo: stdout)
            defer { try? log.close() }
            process.executableURL = runtime.executableURL
            process.currentDirectoryURL = package
            process.arguments = [package.appendingPathComponent("workflow.py").path, "--bindings", bindings.path, "--output", root.appendingPathComponent("Replay").path]
            process.standardOutput = log; process.standardError = log
            try process.run(); process.waitUntilExit()
            return process.terminationStatus
        }.value
        let replayLog = try String(contentsOf: stdout, encoding: .utf8)
        XCTAssertEqual(status, 0, replayLog)
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("Replay/preparation/manifest.json").path))
    }
}
