import Foundation
import XCTest
import KwipMLCore
@testable import KwipMLApp

final class TableSourceInspectionContractTests: XCTestCase {
    private let sourcePin = String(repeating: "a", count: 64)

    func testGenericInspectionAdmitsOnlyValidatedSingleCapturedSource() throws {
        let root = URL(fileURLWithPath: "/private/tmp/workflow-operations/\(UUID())/\(UUID())")
        let file = WorkflowArchiveFile(path: "input/data/source.json", bytes: 20, sha256: sourcePin)
        let options = try WorkflowSourceInspectionToolContract.configurationJSON([
            "format": .string("json"), "options": .object([:])])
        let args = ["--inspect-data", root.appendingPathComponent(file.path).path, "--table-options", options]
        XCTAssertNoThrow(try WorkflowSourceInspectionToolContract.validate(arguments: args, files: [file], directory: root))
        for invalid in ["{}", "[]", "{\"format\":\"unknown\",\"options\":{}}",
                        "{\"format\":\"json\",\"options\":{\"source\":\"/tmp/secret\"}}",
                        "{\"format\":\"parquet\",\"options\":{\"storage_options\":{}}}"] {
            XCTAssertThrowsError(try WorkflowSourceInspectionToolContract.validate(
                arguments: Array(args.prefix(3)) + [invalid], files: [file], directory: root))
        }
        XCTAssertThrowsError(try WorkflowSourceInspectionToolContract.validate(
            arguments: args + ["--csv-options", "{}"], files: [file], directory: root))
        XCTAssertThrowsError(try WorkflowSourceInspectionToolContract.validate(
            arguments: args, files: [file, file], directory: root))
    }

    func testGenericScalarSampleRetainsParsingAndNullEvidence() throws {
        let options = try JSONSerialization.jsonObject(with: Data(
            WorkflowSourceInspectionToolContract.configurationJSON([
                "format": .string("json"), "options": .object([:])]).utf8))
        var value: [String: Any] = ["event": "dataPreview", "columns": ["count", "enabled", "name"],
            "rows": [["2", "true", ""]], "rowCount": 1, "sourceSHA256": sourcePin,
            "parsing": ["validatedRows": 1, "previewRows": 1, "sampleOnly": true, "options": options],
            "nullCells": [[0, 2]]]
        let result = try SourceInspectionResult.decode(JSONSerialization.data(withJSONObject: value), expectedSourceSHA256: sourcePin)
        XCTAssertEqual(result.rows, [["2", "true", ""]])
        XCTAssertEqual(try TableInputConfiguration.format(of: XCTUnwrap(result.parsing).options), .json)
        value["parsing"] = ["validatedRows": 2, "previewRows": 1, "sampleOnly": true, "options": options]
        XCTAssertThrowsError(try SourceInspectionResult.decode(JSONSerialization.data(withJSONObject: value), expectedSourceSHA256: sourcePin))
    }

    func testEmptyGenericSampleRequiresExplicitValidatedParsingEvidence() throws {
        let options = try JSONSerialization.jsonObject(with: Data(
            WorkflowSourceInspectionToolContract.configurationJSON([
                "format": .string("json"), "options": .object([:])]).utf8))
        var value: [String: Any] = ["event": "dataPreview", "columns": ["value"], "rows": [] as [[String]],
            "rowCount": 0, "sourceSHA256": sourcePin,
            "parsing": ["validatedRows": 0, "previewRows": 0, "sampleOnly": true, "options": options]]
        XCTAssertNoThrow(try SourceInspectionResult.decode(JSONSerialization.data(withJSONObject: value), expectedSourceSHA256: sourcePin))
        value.removeValue(forKey: "parsing")
        XCTAssertThrowsError(try SourceInspectionResult.decode(JSONSerialization.data(withJSONObject: value), expectedSourceSHA256: sourcePin))
    }

    func testOnlyGenericSettingsReceiveTheLargerTransportBudget() throws {
        let root = URL(fileURLWithPath: "/private/tmp/workflow-operations/\(UUID())/\(UUID())")
        let file = WorkflowArchiveFile(path: "input/data/source.json", bytes: 20, sha256: sourcePin)
        let json = try WorkflowSourceInspectionToolContract.configurationJSON([
            "format": .string("json"), "options": .object([:])])
        let args = ["--inspect-data", root.appendingPathComponent(file.path).path, "--table-options"]
        XCTAssertNoThrow(try WorkflowSourceInspectionToolContract.validate(
            arguments: args + [json + String(repeating: " ", count: 5000)], files: [file], directory: root))
        XCTAssertThrowsError(try WorkflowSourceInspectionToolContract.validate(
            arguments: args + [json + String(repeating: " ", count: 32_768)], files: [file], directory: root))
        XCTAssertThrowsError(try WorkflowSourceInspectionToolContract.validate(
            arguments: Array(args.prefix(2)) + ["--csv-options", "{}" + String(repeating: " ", count: 4096)],
            files: [file], directory: root))
    }

    func testTypedColumnEvidencePreservesTypesAndRejectsMismatchedOrUnknownSchema() throws {
        var value: [String: Any] = ["event": "dataPreview", "columns": ["count", "enabled"],
            "rows": [["2", "true"]], "rowCount": 1, "sourceSHA256": sourcePin,
            "columnSchema": [["name": "count", "dtype": "int64", "nullable": NSNull()],
                             ["name": "enabled", "dtype": "bool", "nullable": NSNull()]]]
        let result = try SourceInspectionResult.decode(JSONSerialization.data(withJSONObject: value), expectedSourceSHA256: sourcePin)
        XCTAssertEqual(result.columnSchema, [.init(name: "count", dtype: "int64", nullable: nil), .init(name: "enabled", dtype: "bool", nullable: nil)])
        for invalid in [
            [["name": "other", "dtype": "int64", "nullable": NSNull()], ["name": "enabled", "dtype": "bool", "nullable": NSNull()]],
            [["name": "count", "dtype": "object", "nullable": NSNull()], ["name": "enabled", "dtype": "bool", "nullable": NSNull()]],
            [["name": "count", "dtype": "int64", "nullable": NSNull()]]
        ] {
            value["columnSchema"] = invalid
            XCTAssertThrowsError(try SourceInspectionResult.decode(JSONSerialization.data(withJSONObject: value), expectedSourceSHA256: sourcePin))
        }
    }
}
