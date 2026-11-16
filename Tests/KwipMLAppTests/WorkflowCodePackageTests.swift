import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowCodePackageTests: XCTestCase {
    func testSourcePackageNeverOverwritesAndRejectsPathEscape() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let destination = root.appendingPathComponent("Export")
        let original = Data("original source".utf8)
        try WorkflowCodePackage.write(["workflow.py": original, "runner/helper.py": Data("# helper".utf8)], to: destination)
        XCTAssertEqual(try Data(contentsOf: destination.appendingPathComponent("workflow.py")), original)
        XCTAssertThrowsError(try WorkflowCodePackage.write(["workflow.py": Data("replacement".utf8)], to: destination))
        XCTAssertEqual(try Data(contentsOf: destination.appendingPathComponent("workflow.py")), original)
        for path in ["../escape.py", "/absolute.py", "a//b.py", "a/./b.py", "a\\b.py"] {
            XCTAssertThrowsError(try WorkflowCodePackage.write([path: original], to: root.appendingPathComponent(UUID().uuidString)))
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("escape.py").path))
    }
    func testCancelledWritePublishesNoPartialPackage() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let destination = root.appendingPathComponent("Export")
        let task = Task {
            try await Task.sleep(nanoseconds: 10_000_000)
            try WorkflowCodePackage.write(["workflow.py": Data("source".utf8)], to: destination)
        }
        task.cancel()
        do { try await task.value; XCTFail("Cancelled export published") } catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))
    }
}
