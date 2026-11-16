import XCTest
@testable import KwipMLCore

final class FlowAuthoringRetentionPreferencesTests: XCTestCase {
    func testLegacyDefaultsDoNotRewriteAndCustomPolicySurvivesOtherResets() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("preferences.json"), store = AppPreferencesStore(fileURL: file)
        let bytes = Data(#"{"schemaVersion":1,"canvasDefaults":{"flowDirection":"topToBottom","nodeDetail":"small"}}"#.utf8)
        try bytes.write(to: file)
        var value = try store.load()
        XCTAssertEqual(value.effectiveAuthoringRetention, .factory)
        XCTAssertEqual(try Data(contentsOf: file), bytes)
        value.authoringRetention = try .init(maximumFiles: 100, maximumBytes: 4 * 1024 * 1024)
        try store.save(value)
        XCTAssertEqual(try store.resetExecution().authoringRetention, value.authoringRetention)
        XCTAssertEqual(try store.resetAppearance().authoringRetention, value.authoringRetention)
        XCTAssertEqual(try store.load().effectiveAuthoringRetention.maximumFiles, 100)
    }
    func testMalformedPolicyIsRefusedAndPreserved() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("preferences.json"), store = AppPreferencesStore(fileURL: file)
        XCTAssertThrowsError(try FlowAuthoringRetentionPreferences(maximumFiles: 0, maximumBytes: 1))
        XCTAssertThrowsError(try FlowAuthoringRetentionPreferences(maximumFiles: 100_001, maximumBytes: 1))
        XCTAssertThrowsError(try FlowAuthoringRetentionPreferences(maximumFiles: 1, maximumBytes: 0))
        var fields = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(AppPreferences.factory)) as? [String: Any])
        for policy: Any in [NSNull(), ["maximumFiles": 1, "maximumBytes": -1], ["maximumFiles": 1, "maximumBytes": 20, "unknown": 1]] {
            fields["authoringRetention"] = policy
            let bytes = try JSONSerialization.data(withJSONObject: fields)
            try bytes.write(to: file)
            XCTAssertThrowsError(try store.load())
            XCTAssertThrowsError(try store.resetExecution())
            XCTAssertEqual(try Data(contentsOf: file), bytes)
        }
    }
}
