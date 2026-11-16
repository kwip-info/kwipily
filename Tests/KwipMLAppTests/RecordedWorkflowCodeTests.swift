import XCTest
import KwipMLCore
@testable import KwipMLApp

final class RecordedWorkflowCodeTests: XCTestCase {
    func testPinnedSourceReadsNeverExecuteOrOpenOriginalSource() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        // Reading code must neither execute this statement nor consult its nonexistent source.
        let source = "raise RuntimeError('must not execute')\nopen('/missing/original.csv')\n"
        let bytes = Data(source.utf8)
        try bytes.write(to: root.appendingPathComponent("preparation.py"))
        let artifact = WorkflowArchiveFile(path: "preparation.py", bytes: bytes.count, sha256: WorkflowArchiveCodec.hash(bytes))
        XCTAssertEqual(try RecordedWorkflowCode.readPinnedSources(root: root, base: [], artifacts: [artifact]), ["preparation.py": source])
        try Data(source.replacingOccurrences(of: "raise", with: "print").utf8).write(to: root.appendingPathComponent("preparation.py"))
        XCTAssertThrowsError(try RecordedWorkflowCode.readPinnedSources(root: root, base: [], artifacts: [artifact]))
    }
    func testSourceReaderRejectsSymlinksAndPathEscape() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let bytes = Data("pass\n".utf8)
        try bytes.write(to: root.appendingPathComponent("actual.py"))
        try FileManager.default.createSymbolicLink(atPath: root.appendingPathComponent("model.py").path, withDestinationPath: root.appendingPathComponent("actual.py").path)
        for path in ["model.py", "../actual.py"] {
            let artifact = WorkflowArchiveFile(path: path, bytes: bytes.count, sha256: WorkflowArchiveCodec.hash(bytes))
            XCTAssertThrowsError(try RecordedWorkflowCode.readPinnedSources(root: root, base: [], artifacts: [artifact]))
        }
    }
    @MainActor func testUnverifiedRunDoesNotExposeLooseCode() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let project = UUID().uuidString, run = UUID()
        let directory = root.appendingPathComponent("workflow-runs/\(project)/\(run.uuidString)/training/artifacts")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data("unverified code\n".utf8).write(to: directory.appendingPathComponent("model.py"))
        do {
            _ = try await RecordedWorkflowCode.load(root: root, projectID: project, record: .training(.init(projectID: project, directoryID: run)))
            XCTFail("Loose source without an authoritative receipt must not be displayed")
        } catch { }
    }
}
