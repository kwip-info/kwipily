import XCTest
@testable import KwipMLCore

final class FileInputConfigurationTests: XCTestCase {
    private var root: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    func testBundledReaderCatalogMatchesAuthoritativeJSONAndDefaults() throws {
        let source = try JSONDecoder().decode(FileInputOptionsSchema.self, from: Data(contentsOf: root.appendingPathComponent("runner/file-input-options.json")))
        XCTAssertEqual(try FileInputConfiguration.loadSchema(), source)
        XCTAssertEqual(try FileInputConfiguration.validate([:]), FileInputConfiguration.newDefaults)
        XCTAssertEqual(FileInputConfiguration.readerOrder.count, 7)
        if case .array(let extensions)?=FileInputConfiguration.newDefaults["extensions"] {XCTAssertFalse(extensions.contains(.string("wav")))} else {XCTFail("Missing default extensions")}
        XCTAssertEqual(FileInputConfiguration.newDefaults["maxFiles"], .integer(1000))
        XCTAssertEqual(FileInputConfiguration.newDefaults["maxFileBytes"], .integer(67_108_864))
    }
    func testEveryDeclaredReaderOptionDefaultAndChoiceValidates() throws {
        for (name, reader) in FileInputConfiguration.readerDefinitions {
            for field in reader.fields {
                for value in [field.defaultValue] + (field.choices ?? []).map({ .string($0.value) }) {
                    let result = try FileInputConfiguration.validateReaderOptions(name, [field.key: value])
                    XCTAssertEqual(result[field.key], value)
                }
            }
        }
    }
    func testPrivateUnknownWronglyTypedAndUnboundedSettingsAreRefused() {
        let invalid: [[String: JSONValue]] = [
            ["path": .string("/private/file")], ["sourceMode": .string("folder")], ["sourceMode": .string("url")],
            ["includeHidden": .integer(1)], ["maxFiles": .bool(true)], ["maxFiles": .integer(0)],
            ["maxFiles": .integer(1001)], ["maxDepth": .integer(33)],
            ["maxFileBytes": .integer(67_108_865)], ["maxTotalBytes": .integer(268_435_457)],
            ["maxFileBytes": .integer(20), "maxTotalBytes": .integer(10)],
            ["extensions": .array([.string("TXT")])], ["extensions": .array([.string(".txt")])],
            ["extensions": .array([.string("txt"), .string("txt")])],
            ["metadataFields": .array([.string("path")])], ["readMode": .string("metadata"), "metadataFields": .array([])],
            ["includePatterns": .array([.string("../*")])], ["includePatterns": .array([.string("/absolute/*")])],
            ["includePatterns": .array([.string("C:/file")])], ["includePatterns": .array([.string("https://example.com/*")])],
            ["readers": .object(["unknown": .object([:])])]
        ]
        for config in invalid { XCTAssertThrowsError(try FileInputConfiguration.validate(config)) }
    }
    func testReaderRangePageAndOrientationValidationPreservesDormantOptions() throws {
        XCTAssertThrowsError(try FileInputConfiguration.validateReaderOptions("image", ["frames": .string("range"), "frameStart": .integer(3), "frameEnd": .integer(1)]))
        XCTAssertEqual(try FileInputConfiguration.validateReaderOptions("image", ["frames": .string("first"), "frameStart": .integer(3), "frameEnd": .integer(1)])["frameStart"], .integer(3))
        for options: [String: JSONValue] in [["pages": .array([.integer(0)])], ["pages": .array([.integer(1), .integer(1)])], ["orientations": .array([])], ["orientations": .array([.integer(45)])], ["space_width": .number(.infinity)]] {
            XCTAssertThrowsError(try FileInputConfiguration.validateReaderOptions("pdf", options))
        }
        let pdf = try FileInputConfiguration.validateReaderOptions("pdf", ["pages": .array([.integer(2), .integer(1)]), "orientations": .array([.integer(90), .integer(0)])])
        XCTAssertEqual(pdf["pages"], .array([.integer(2), .integer(1)]))
    }
    func testDirectoryAllCustomExtensionsAndOrderedMetadataRemainExplicit() throws {
        let result = try FileInputConfiguration.validate(["sourceMode": .string("directory"), "extensions": .array([.string("custom"), .string("pdf")]), "metadataFields": .array([.string("content_sha256"), .string("file_id"), .string("name")])])
        XCTAssertEqual(result["extensions"], .array([.string("custom"), .string("pdf")]))
        XCTAssertEqual(result["metadataFields"], .array([.string("content_sha256"), .string("file_id"), .string("name")]))
        XCTAssertEqual(try FileInputConfiguration.validate(["extensions": .array([])])["extensions"], .array([]))
    }
    func testFileIdentityRemainsRequiredInEveryReadMode() throws {
        for mode in ["content", "metadata"] {
            XCTAssertThrowsError(try FileInputConfiguration.validate(["readMode": .string(mode), "metadataFields": .array([])]))
            XCTAssertThrowsError(try FileInputConfiguration.validate(["readMode": .string(mode), "metadataFields": .array([.string("name")])]))
            XCTAssertEqual(try FileInputConfiguration.validate(["readMode": .string(mode), "metadataFields": .array([.string("file_id")])])["metadataFields"], .array([.string("file_id")]))
        }
    }
    func testNormalizedDefaultsMatchPythonWithoutImportingReaders() throws {
        guard let python = ProcessInfo.processInfo.environment["KWIPML_BACKEND_TEST_PYTHON"] else { throw XCTSkip("Set KWIPML_BACKEND_TEST_PYTHON for native/worker option parity.") }
        let process = Process(), input = Pipe(), output = Pipe(), errors = Pipe()
        process.executableURL = URL(fileURLWithPath: python); process.currentDirectoryURL = root
        process.arguments = ["-c", "import json,sys; from runner.file_input_options import validate_configuration; c=json.load(sys.stdin); assert c==validate_configuration({}); assert 'PIL' not in sys.modules and 'pypdf' not in sys.modules; print('verified')"]
        process.standardInput = input; process.standardOutput = output; process.standardError = errors
        try process.run(); try input.fileHandleForWriting.write(contentsOf: JSONEncoder().encode(FileInputConfiguration.newDefaults)); try input.fileHandleForWriting.close(); process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus, 0, String(decoding: errors.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self))
        XCTAssertEqual(String(decoding: output.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines), "verified")
    }
}
