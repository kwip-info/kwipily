import XCTest
@testable import KwipMLApp

final class DiagnosticRecoveryMessageTests: XCTestCase {
    func testMissingFileAndAncestorExplainRecoveryWithoutExposingAbsolutePaths() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        for parts in [["missing.json"], ["missing-folder", "report.json"]] {
            XCTAssertThrowsError(try DiagnosticFileIO.read(root: root, parts: parts)) { error in
                let message = error.localizedDescription.lowercased()
                XCTAssertTrue(message.contains("missing"))
                XCTAssertTrue(message.contains("restore"))
                XCTAssertTrue(message.contains("retry"))
                XCTAssertFalse(message.contains(root.path.lowercased()))
            }
        }
    }
}
