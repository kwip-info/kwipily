import XCTest
import Darwin
import KwipMLCore
@testable import KwipMLApp

final class SelectedSourceFileReadTests: XCTestCase {
    private func fixture() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return root
    }
    func testSelectedFileReadDoesNotRequireOpeningParentDirectory() throws {
        let root = try fixture(), parent = root.appendingPathComponent("source")
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: false)
        let source = parent.appendingPathComponent("selected.csv"), bytes = Data("x,y\n1,2\n".utf8)
        try bytes.write(to: source)
        XCTAssertEqual(chmod(parent.path, 0o111), 0)
        defer { _ = chmod(parent.path, 0o700) }
        let descriptor = Darwin.open(parent.path, O_RDONLY | O_DIRECTORY)
        if descriptor >= 0 { Darwin.close(descriptor); throw XCTSkip("Host permits directory reads despite mode; native sandbox test remains required") }
        XCTAssertThrowsError(try DiagnosticFileIO.read(root: parent, parts: [source.lastPathComponent]))
        XCTAssertEqual(try DiagnosticFileIO.readSelectedFile(source, limit: bytes.count), bytes)
    }
    func testExperimentCaptureUsesExactSelectedFileAuthority() async throws {
        let root = try fixture(), parent = root.appendingPathComponent("source")
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: false)
        let source = parent.appendingPathComponent("selected.csv"), bytes = Data("x,y\n1,2\n".utf8)
        try bytes.write(to: source)
        let access = SourceAccess(directory: root.appendingPathComponent("permissions"))
        XCTAssertEqual(chmod(parent.path, 0o111), 0)
        defer { _ = chmod(parent.path, 0o700) }
        let descriptor = Darwin.open(parent.path, O_RDONLY | O_DIRECTORY)
        if descriptor >= 0 { Darwin.close(descriptor); throw XCTSkip("Host permits directory reads despite mode; native sandbox test remains required") }
        let address = WorkflowNodeAddress(domain: .data, localID: "source")
        let result = try await access.captureExperimentInputs(paths: [address: source.path], parent: root)
        XCTAssertEqual(try Data(contentsOf: XCTUnwrap(result.sources[address]).url), bytes)
        try result.verify()
    }
    func testSelectedFileRejectsLinksDirectoriesFIFOsAndOversizedContent() throws {
        let root = try fixture(), source = root.appendingPathComponent("source.csv")
        try Data("1234".utf8).write(to: source)
        let link = root.appendingPathComponent("link.csv"), fifo = root.appendingPathComponent("fifo")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: source)
        XCTAssertEqual(mkfifo(fifo.path, 0o600), 0)
        XCTAssertThrowsError(try DiagnosticFileIO.readSelectedFile(link, limit: 4))
        XCTAssertThrowsError(try DiagnosticFileIO.readSelectedFile(root, limit: 4))
        XCTAssertThrowsError(try DiagnosticFileIO.readSelectedFile(fifo, limit: 4))
        XCTAssertThrowsError(try DiagnosticFileIO.readSelectedFile(source, limit: 3))
        XCTAssertThrowsError(try DiagnosticFileIO.readSelectedFile(source, limit: -1))
        XCTAssertThrowsError(try DiagnosticFileIO.readSelectedFile(source, limit: Int.max))
        XCTAssertEqual(try DiagnosticFileIO.readSelectedFile(source, limit: 4), Data("1234".utf8))
    }
    func testCaptureReadFailureExplainsReselectionWithoutPrivatePath() async throws {
        let root = try fixture(), source = root.appendingPathComponent("private-source.csv")
        let access = SourceAccess(directory: root.appendingPathComponent("permissions"))
        let address = WorkflowNodeAddress(domain: .data, localID: "source")
        do {
            _ = try await access.captureExperimentInputs(paths: [address: source.path], parent: root)
            XCTFail("Missing selected source must fail")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("selected source"))
            XCTAssertTrue(error.localizedDescription.contains("Choose the source again"))
            XCTAssertFalse(error.localizedDescription.contains(root.path))
            XCTAssertFalse(error.localizedDescription.contains(source.lastPathComponent))
        }
    }
}
