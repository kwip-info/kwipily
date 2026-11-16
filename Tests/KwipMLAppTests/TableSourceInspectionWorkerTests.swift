import Foundation
import XCTest
import KwipMLCore
@testable import KwipMLApp

final class TableSourceInspectionWorkerTests: XCTestCase {
    @MainActor func testActualWorkerRoundTripsEveryTableFormatThroughVerifiedHostResult() async throws {
        let (workspace, _, _, _) = try WorkflowEvaluationRequestTests().fixture()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let fixtures = workspace.appendingPathComponent("fixtures")
        try FileManager.default.createDirectory(at: fixtures, withIntermediateDirectories: true)
        // API inspection consumes a broker-captured IPC table, never a URL as a file.
        // The HTTP broker itself has independent transport/pagination fixture coverage.
        let apiOptions=APITableSettings.defaults.merging([
            "url":.string("https://api.example.com/observations"),"recordsPath":.string("/items"),
            "mapping":.array([.object(["source":.string("/value"),"target":.string("value"),"type":.string("int64")]),.object(["source":.string("/name"),"target":.string("name"),"type":.string("string")])])
        ]) {_,value in value}
        let generator = Process()
        generator.executableURL = repository.appendingPathComponent(".venv/bin/python")
        generator.arguments = ["-c", """
import json,sys,sqlite3
from pathlib import Path
import polars as pl
import openpyxl
sys.path.insert(0,sys.argv[2])
from runner.database_source import capture
from runner.api_table_snapshot import identity
import pyarrow as pa,pyarrow.ipc as ipc
p=Path(sys.argv[1])
(p/'csv').write_text('value,name\\n1,A\\n2,B\\n')
(p/'json').write_text('[{"value":1,"name":"A"},{"value":2,"name":"B"}]')
(p/'jsonl').write_text('{"value":1,"name":"A"}\\n{"value":2,"name":"B"}\\n')
frame=pl.DataFrame({'value':[1,2],'name':['A','B']})
frame.write_parquet(p/'parquet')
api_options=json.loads(sys.argv[3])
fetch=dict(complete=True,pages=1,rows=2,responseBytes=64,capturedAt='2026-10-19T00:00:00Z')
api_table=pa.table({'value':pa.array([1,2],type=pa.int64()),'name':pa.array(['A','B'],type=pa.string())}).replace_schema_metadata({b'kwip.api.selection.v1':identity(api_options),b'kwip.api.fetch.v1':json.dumps(fetch).encode()})
with (p/'api').open('wb') as sink:
 with ipc.new_file(sink,api_table.schema) as writer:writer.write_table(api_table)
with sqlite3.connect(p/'fixture.sqlite3') as connection:
 connection.execute('CREATE TABLE observations(value INTEGER,name TEXT)')
 connection.executemany('INSERT INTO observations VALUES (?,?)',[(1,'A'),(2,'B')])
with (p/'database').open('wb') as sink:
 capture({'table':'observations'},{},sink,sqlite_path=p/'fixture.sqlite3')
book=openpyxl.Workbook();sheet=book.active
sheet.append(['value','name']);sheet.append([1,'A']);sheet.append([2,'B']);book.save(p/'excel')
(p/'markdown').write_text('| value | name |\\n|---|---|\\n| 1 | A |\\n| 2 | B |\\n')
(p/'xml').write_text('<rows><row><value>1</value><name>A</name></row><row><value>2</value><name>B</name></row></rows>')
(p/'html').write_text('<table><tr><th>value</th><th>name</th></tr><tr><td>1</td><td>A</td></tr><tr><td>2</td><td>B</td></tr></table>')
""", fixtures.path, repository.path, String(decoding:try WorkflowArchiveCodec.encode(apiOptions),as:UTF8.self)]
        let errors = Pipe(); generator.standardError = errors
        try generator.run(); generator.waitUntilExit()
        XCTAssertEqual(generator.terminationStatus, 0, String(decoding: errors.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self))
        let store = WorkspaceStore(rootOverride: repository, storageOverride: workspace, detectCapabilities: false)
        let runtime = try XCTUnwrap(store.runtime)
        for format in TableInputConfiguration.Format.allCases {
            var configuration = TableInputConfiguration.defaults(for: format)
            if format == .api { configuration["options"] = .object(apiOptions) }
            if format == .database { configuration["options"] = .object(["table": .string("observations")]) }
            let expected = try TableInputConfiguration.validate(configuration)
            let window = try ExecutionBudgetWindow(policy: .resolve(global: .factory, projectMaximumRunSeconds: nil), startedAtUptime: ProcessInfo.processInfo.systemUptime)
            let projectID = UUID().uuidString
            let capture = try SourceInspectionCapture.capture(path: fixtures.appendingPathComponent(format.rawValue).path,
                workspace: workspace, sourcePermissions: workspace.appendingPathComponent("permissions"), window: window,
                projectID: projectID, storageLimitBytes: 30_000_000, completeFile: format != .csv)
            let sourcePin = capture.sha256
            let admission = try WorkflowStorageAdmission(workspace: workspace)
            let done = expectation(description: "Verified actual \(format.rawValue) inspection")
            let handle = try await SourceInspectionExecution.launch(workspace: workspace, projectID: projectID,
                runtime: runtime, window: window, sourceURL: capture.sourceURL, sourceSHA256: sourcePin,
                separator: ",", configuration: configuration, storageAdmission: capture.storageReservation,
                onExit: { code, message, result in
                    XCTAssertEqual(code, 0, "\(format.rawValue): \(message)")
                    XCTAssertEqual(result?.columns, ["value", "name"], format.rawValue)
                    XCTAssertEqual(result?.rows, [["1", "A"], ["2", "B"]], format.rawValue)
                    XCTAssertEqual(result?.sourceSHA256, sourcePin, format.rawValue)
                    if format == .api {XCTAssertEqual(result?.apiFetch?["complete"],.bool(true));XCTAssertEqual(result?.apiFetch?["pages"],.integer(1))}
                    if let options = result?.parsing?.options {
                        XCTAssertEqual(try? TableInputConfiguration.validate(options), expected, format.rawValue)
                    } else { XCTFail("Missing verified parsing options for \(format.rawValue)") }
                    XCTAssertEqual(result?.columnSchema?.map(\.dtype), [format == .csv || format == .markdown || format == .xml || format == .html ? "string" : "int64", "string"], format.rawValue)
                    XCTAssertEqual(try? admission.snapshot().reservedGrowthBytes, 0, format.rawValue)
                    done.fulfill()
                })
            capture.remove()
            await fulfillment(of: [done], timeout: 30)
            withExtendedLifetime(handle) {}
        }
    }
}
