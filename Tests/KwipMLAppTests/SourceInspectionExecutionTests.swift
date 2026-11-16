import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class SourceInspectionExecutionTests: XCTestCase {
    func testResultRejectsUnboundedMalformedAndUnpinnedTables() throws {
        let hash = String(repeating: "a", count: 64)
        let original: [String: Any] = ["event": "dataPreview", "columns": ["x", "y"], "rows": [["1", "2"]], "rowCount": 1, "sourceSHA256": hash]
        let result = try SourceInspectionResult.decode(JSONSerialization.data(withJSONObject: original), expectedSourceSHA256: hash)
        XCTAssertEqual(result.columns, ["x", "y"]); XCTAssertEqual(result.rows, [["1", "2"]]); XCTAssertEqual(result.rowCount, 1)
        let changes: [(inout [String: Any]) -> Void] = [
            { $0["columns"] = ["x", "x"] }, { $0["columns"] = [" ", "y"] },
            { $0["columns"] = [String(repeating: "a", count: 4097), "y"] },
            { $0["rows"] = [["1"]] }, { $0["rows"] = [[String(repeating: "x", count: 201), "2"]] },
            { $0["rows"] = Array(repeating: ["1", "2"], count: 9) }, { $0["rowCount"] = 0 },
            { $0["rowCount"] = 100001 }, { $0["rowCount"] = true },
            { $0["sourceSHA256"] = String(repeating: "b", count: 64) }, { $0["extra"] = true }
        ]
        for change in changes {
            var object = original; change(&object)
            XCTAssertThrowsError(try SourceInspectionResult.decode(JSONSerialization.data(withJSONObject: object), expectedSourceSHA256: hash))
        }
    }
    func testRequestPreservesBasenameAndExplicitSingleCharacterSeparator() throws {
        let root = URL(fileURLWithPath: "/private/tmp/workflow-operations/\(UUID().uuidString)/\(UUID().uuidString)")
        let file = WorkflowArchiveFile(path: "input/data/source table.csv", bytes: 8, sha256: String(repeating: "a", count: 64))
        for separator in [",", "\t", ";", "|", ":", "§"] {
            let args = ["--inspect-data", root.appendingPathComponent(file.path).path, "--csv-separator", separator]
            XCTAssertNoThrow(try WorkflowSourceInspectionToolContract.validate(arguments: args, files: [file], directory: root))
            XCTAssertThrowsError(try WorkflowSourceInspectionToolContract.validate(arguments: args + ["--registry", "/tmp/registry"], files: [file], directory: root))
        }
        XCTAssertEqual(try WorkflowSourceInspectionToolContract.fileLimit(path: file.path), 10_000_000)
        XCTAssertThrowsError(try WorkflowSourceInspectionToolContract.validate(arguments: ["--inspect-data", "/tmp/outside.csv", "--csv-separator", ","], files: [file], directory: root))
        XCTAssertThrowsError(try WorkflowSourceInspectionToolContract.validate(arguments: ["--inspect-data", root.appendingPathComponent(file.path).path, "--csv-separator", "::"], files: [file], directory: root))
    }
    func testParsingOptionsAdmissionRejectsInvalidAndExtraArguments() throws {
        let root = URL(fileURLWithPath: "/private/tmp/workflow-operations/\(UUID().uuidString)/\(UUID().uuidString)")
        let file = WorkflowArchiveFile(path: "input/data/table.csv", bytes: 8, sha256: String(repeating: "a", count: 64))
        let options = try CSVParsingConfiguration.transportJSON(["separator": .string(":"), "quoteChar": .string("'")])
        let prefix = ["--inspect-data", root.appendingPathComponent(file.path).path, "--csv-options"]
        XCTAssertNoThrow(try WorkflowSourceInspectionToolContract.validate(arguments: prefix + [options], files: [file], directory: root))
        for invalid in ["[]", "null", "{", "{\"separator\":\"::\"}", "{\"arbitraryPath\":\"/tmp/secret\"}"] {
            XCTAssertThrowsError(try WorkflowSourceInspectionToolContract.validate(arguments: prefix + [invalid], files: [file], directory: root))
        }
        XCTAssertThrowsError(try WorkflowSourceInspectionToolContract.validate(arguments: prefix + [options, "--csv-separator", ","], files: [file], directory: root))
    }
    func testParsingEvidenceAndNullCoordinatesAreBounded() throws {
        let hash = String(repeating: "a", count: 64)
        let options = try JSONSerialization.jsonObject(with: Data(CSVParsingConfiguration.transportJSON([:]).utf8))
        var original: [String: Any] = ["event": "dataPreview", "columns": ["x"], "rows": [[""]], "rowCount": 1,
            "sourceSHA256": hash, "parsing": ["validatedRows": 1, "previewRows": 1, "sampleOnly": false, "options": options], "nullCells": [[0, 0]]]
        let result = try SourceInspectionResult.decode(JSONSerialization.data(withJSONObject: original), expectedSourceSHA256: hash)
        XCTAssertEqual(result.parsing?.validatedRows, 1)
        XCTAssertEqual(result.nullCells, [[0, 0]])
        for cells in [[[1, 0]], [[0, 1]], [[0]], [[0, 0], [0, 0]]] {
            original["nullCells"] = cells
            XCTAssertThrowsError(try SourceInspectionResult.decode(JSONSerialization.data(withJSONObject: original), expectedSourceSHA256: hash))
        }
        original["rows"] = [["not empty"]]
        original["nullCells"] = [[0, 0]]
        XCTAssertThrowsError(try SourceInspectionResult.decode(JSONSerialization.data(withJSONObject: original), expectedSourceSHA256: hash))
        original["rows"] = [[""]]
        original["nullCells"] = [] as [[Int]]
        original["parsing"] = ["validatedRows": 2, "previewRows": 1, "sampleOnly": false, "options": options]
        XCTAssertThrowsError(try SourceInspectionResult.decode(JSONSerialization.data(withJSONObject: original), expectedSourceSHA256: hash))
    }
    func testActualWorkerUsesSingleSourceAdmissionAndReleasesAfterVerification() async throws {
        let (workspace, _, _, _) = try WorkflowEvaluationRequestTests().fixture()
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let store = WorkspaceStore(rootOverride: repo, storageOverride: workspace, detectCapabilities: false)
        let runtime = try XCTUnwrap(store.runtime, "Actual development worker is required")
        defer { try? FileManager.default.removeItem(at: workspace) }
        let source = workspace.appendingPathComponent("source.csv"), bytes = Data("x;y\n'one;two';02\n".utf8)
        try bytes.write(to: source)
        let projectID = UUID().uuidString
        let window = try ExecutionBudgetWindow(policy: .resolve(global: .factory, projectMaximumRunSeconds: nil),
            startedAtUptime: ProcessInfo.processInfo.systemUptime)
        let capture = try SourceInspectionCapture.capture(path: source.path, workspace: workspace,
            sourcePermissions: workspace.appendingPathComponent("permissions"), window: window,
            projectID: projectID, storageLimitBytes: 1_100_000)
        defer { capture.remove() }
        let admission = try WorkflowStorageAdmission(workspace: workspace)
        let done = expectation(description: "Actual admitted source inspection")
        let handle = try await SourceInspectionExecution.launch(workspace: workspace, projectID: projectID,
            runtime: runtime, window: window, sourceURL: capture.sourceURL, sourceSHA256: capture.sha256,
            separator: ";", configuration: ["separator": .string(";"), "quoteChar": .string("'")],
            storageAdmission: capture.storageReservation, onExit: { code, message, result in
                XCTAssertEqual(code, 0, message)
                XCTAssertEqual(result?.columns, ["x", "y"])
                XCTAssertEqual(result?.rows, [["one;two", "02"]])
                XCTAssertEqual(try? admission.snapshot().reservedGrowthBytes, 0)
                done.fulfill()
            })
        capture.remove()
        await fulfillment(of: [done], timeout: 15)
        withExtendedLifetime(handle) {}
        XCTAssertEqual(try admission.snapshot().reservedGrowthBytes, 0)
    }
    func testOnlyVerifiedSuccessReturnsTypedResultAfterPrivateCleanup() async throws {
        for scenario in ["success", "wrongHash", "tamper", "cancel", "timeout", "optionsMismatch", "parseError"] {
            let (workspace, _, _, runtime) = try WorkflowEvaluationRequestTests().fixture()
            defer { try? FileManager.default.removeItem(at: workspace) }
            let source = workspace.appendingPathComponent("my source.csv"), data = Data("x,y\n1,2\n".utf8)
            try data.write(to: source)
            let digest = WorkflowArchiveCodec.hash(data)
            let window = try ExecutionBudgetWindow(policy: .resolve(global: .factory, projectMaximumRunSeconds: nil), startedAtUptime: ProcessInfo.processInfo.systemUptime)
            var exit: ((Int32, String) -> Void)?, operation: URL?
            let done = expectation(description: scenario)
            let projectID = UUID().uuidString
            let capture = try SourceInspectionCapture.capture(path: source.path, workspace: workspace,
                sourcePermissions: workspace.appendingPathComponent("permissions"), window: window,
                projectID: projectID, storageLimitBytes: 1_100_000)
            defer { capture.remove() }
            let admission = try WorkflowStorageAdmission(workspace: workspace)
            XCTAssertGreaterThan(try admission.snapshot().reservedGrowthBytes, 0)
            let handle = try await SourceInspectionExecution.launch(workspace: workspace, projectID: projectID,
                runtime: runtime, window: window, sourceURL: capture.sourceURL, sourceSHA256: digest, separator: ",",
                configuration: scenario == "optionsMismatch" ? [:] : nil,
                storageAdmission: capture.storageReservation,
                launcher: { _, args, _, line, onExit in
                    let index = try XCTUnwrap(args.firstIndex(of: "--kwip-operation-supervise")), directory = URL(fileURLWithPath: args[index + 1])
                    operation = directory; exit = onExit
                    let request = try WorkflowToolRequest.decode(Data(contentsOf: directory.appendingPathComponent("request.json")))
                    XCTAssertEqual(request.toolMode, .sourceInspection)
                    XCTAssertEqual(request.inputFiles.map(\.path), ["input/data/my source.csv"])
                    let ready: [String: Any] = ["event": "operationReady", "operationID": request.operationID, "toolMode": "sourceInspection", "requestSHA256": args[index + 2], "ownershipSHA256": args[index + 3]]
                    line(String(decoding: try JSONSerialization.data(withJSONObject: ready), as: UTF8.self))
                    var result: [String: Any] = ["event": "dataPreview", "columns": ["x", "y"], "rows": [["1", "2"]], "rowCount": 1,
                        "sourceSHA256": scenario == "wrongHash" ? String(repeating: "f", count: 64) : digest]
                    if scenario == "optionsMismatch" {
                        let options = try JSONSerialization.jsonObject(with: Data(CSVParsingConfiguration.transportJSON(["separator": .string(";")]).utf8))
                        result["parsing"] = ["validatedRows": 1, "previewRows": 1, "sampleOnly": false, "options": options]
                    }
                    line(String(decoding: try JSONSerialization.data(withJSONObject: result), as: UTF8.self))
                    if scenario == "parseError" {
                        let failure: [String: Any] = ["event": "error", "message": "raw failure",
                            "csvDiagnostic": ["code": "rowWidth", "record": 3, "line": 4, "message": "raw source content"]]
                        line(String(decoding: try JSONSerialization.data(withJSONObject: failure), as: UTF8.self))
                    }
                    if scenario == "tamper" { try Data("changed".utf8).write(to: directory.appendingPathComponent(request.inputFiles[0].path)) }
                    return .init(cancel: {})
                }, onExit: { code, message, result in
                    if scenario == "success" { XCTAssertEqual(code, 0, message); XCTAssertEqual(result?.sourceSHA256, digest); XCTAssertEqual(result?.columns, ["x", "y"]) }
                    else { XCTAssertNotEqual(code, 0); XCTAssertNil(result) }
                    if scenario == "parseError" {
                        let status = SourceInspectionStatus(phase: .failed, detail: message)
                        XCTAssertEqual(status.csvDiagnostic?.record, 3)
                        XCTAssertEqual(status.csvDiagnostic?.line, 4)
                        XCTAssertFalse(status.detail.contains("raw source"))
                    }
                    XCTAssertEqual(try? admission.snapshot().reservedGrowthBytes, 0)
                    if let operation { XCTAssertFalse(FileManager.default.fileExists(atPath: operation.appendingPathComponent("input").path)) }
                    done.fulfill()
                })
            let directory = try XCTUnwrap(operation)
            capture.remove()
            XCTAssertGreaterThan(try admission.snapshot().reservedGrowthBytes, 0,
                "Removing the source stage must not release the worker reservation")
            if scenario == "cancel" { handle.cancel() }
            XCTAssertThrowsError(try WorkflowFileLease(directory: directory, name: "owner.lock", mode: .shared))
            XCTAssertTrue(FileManager.default.fileExists(atPath: directory.appendingPathComponent("input").path))
            exit?(scenario == "timeout" ? 124 : scenario == "parseError" ? 1 : 0, "")
            await fulfillment(of: [done], timeout: 5)
            let released = try WorkflowFileLease(directory: directory, name: "owner.lock", mode: .exclusive); released.close()
        }
    }
}
