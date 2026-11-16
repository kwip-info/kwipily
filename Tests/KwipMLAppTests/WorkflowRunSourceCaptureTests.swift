import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowRunSourceCaptureTests: XCTestCase {
    private func root() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return root
    }
    private func address(_ id: String = "source") -> WorkflowNodeAddress { .init(domain: .data, localID: id) }

    private final class ProgressRecorder:@unchecked Sendable {
        private let lock=NSLock()
        private var events:[WorkflowSourceCaptureProgress]=[]
        func append(_ event:WorkflowSourceCaptureProgress) {lock.lock();defer {lock.unlock()};events.append(event)}
        func snapshot()->[WorkflowSourceCaptureProgress] {lock.lock();defer {lock.unlock()};return events}
    }
    func testSourceProgressBracketsEachActualCopyAndAdmissionFailureHasNoActivity() async throws {
        let root=try root(),source=root.appendingPathComponent("source.csv")
        try Data("x,y\n1,2\n".utf8).write(to:source)
        let recorder=ProgressRecorder()
        let capture=try await WorkflowRunSourceCapture.capture(paths:[address("a"):source.path,address("b"):source.path],workspace:root,sourcePermissions:root.appendingPathComponent("permissions"),storageLimitBytes:nil,progress:recorder.append)
        let events=recorder.snapshot()
        XCTAssertEqual(events.map(\.sequence),[1,2,3,4])
        XCTAssertEqual(events.map(\.finished),[false,true,false,true])
        XCTAssertEqual(events.map(\.address),[address("a"),address("a"),address("b"),address("b")])
        XCTAssertEqual(capture.sources.count,2)
        let denied=ProgressRecorder()
        do {
            _=try await WorkflowRunSourceCapture.capture(paths:[address():source.path],workspace:root,sourcePermissions:root.appendingPathComponent("permissions"),storageLimitBytes:0,progress:denied.append)
            XCTFail("Storage refusal must precede copy activity")
        } catch {}
        XCTAssertTrue(denied.snapshot().isEmpty)
    }
    func testExactStorageAdmissionCapturesPrivateHashAndDeinitReleasesStage() async throws {
        let root = try root(), source = root.appendingPathComponent("observations.csv")
        let bytes = Data("x,y\n1,2\n".utf8); try bytes.write(to: source)
        var capture: WorkflowRunSourceCapture? = try await WorkflowRunSourceCapture.capture(paths: [address(): source.path],
            workspace: root, sourcePermissions: root.appendingPathComponent("permissions"), storageLimitBytes: Int64(bytes.count))
        let copied = try XCTUnwrap(capture?.sources[address()])
        XCTAssertTrue(copied.url.path.contains("/workflow-source-staging/"))
        XCTAssertEqual(copied.byteCount, bytes.count)
        XCTAssertEqual(copied.sha256, WorkflowArchiveCodec.hash(bytes))
        try Data("changed source".utf8).write(to: source)
        XCTAssertEqual(try Data(contentsOf: copied.url), bytes)
        capture = nil
        XCTAssertFalse(FileManager.default.fileExists(atPath: copied.url.path))
        try bytes.write(to: source)
        let retry = try await WorkflowRunSourceCapture.capture(paths: [address(): source.path], workspace: root,
            sourcePermissions: root.appendingPathComponent("permissions"), storageLimitBytes: Int64(bytes.count))
        retry.remove()
        XCTAssertFalse(FileManager.default.fileExists(atPath: try XCTUnwrap(retry.sources[address()]).url.path))
    }

    func testInsufficientQuotaRefusesBeforeAnySourceCopy() async throws {
        let root = try root(), source = root.appendingPathComponent("observations.csv")
        let bytes = Data("x,y\n1,2\n".utf8); try bytes.write(to: source)
        do {
            _ = try await WorkflowRunSourceCapture.capture(paths: [address(): source.path], workspace: root,
                sourcePermissions: root.appendingPathComponent("permissions"), storageLimitBytes: Int64(bytes.count - 1))
            XCTFail("Expected pre-copy storage refusal")
        } catch {}
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("workflow-source-staging").path))
        XCTAssertEqual(try Data(contentsOf: source), bytes)
    }

    func testMetadataBoundsRejectOversizedAggregateAndUnsafeSourcesBeforeStaging() async throws {
        let root = try root(), source = root.appendingPathComponent("large.csv")
        try Data().write(to: source)
        let handle = try FileHandle(forWritingTo: source)
        try handle.truncate(atOffset: 10_000_000); try handle.close()
        let aggregate = Dictionary(uniqueKeysWithValues: (0..<27).map { (address("source-\($0)"), source.path) })
        let link = root.appendingPathComponent("linked.csv")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: source)
        let tooMany = Dictionary(uniqueKeysWithValues: (0..<129).map { (address("source-\($0)"), source.path) })
        for paths in [aggregate, tooMany, [address(): link.path]] {
            do {
                _ = try await WorkflowRunSourceCapture.capture(paths: paths, workspace: root,
                    sourcePermissions: root.appendingPathComponent("permissions"), storageLimitBytes: nil)
                XCTFail("Expected metadata admission refusal")
            } catch {}
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("workflow-source-staging").path))
    }

    func testNilQuotaStillOwnsManagedStageAndRefusesRevokedSource() async throws {
        let root = try root(), source = root.appendingPathComponent("observations.csv")
        try Data("x\n1\n".utf8).write(to: source)
        let permissions = root.appendingPathComponent("permissions")
        let capture = try await WorkflowRunSourceCapture.capture(paths: [address(): source.path], workspace: root,
            sourcePermissions: permissions, storageLimitBytes: nil)
        let copied = try XCTUnwrap(capture.sources[address()])
        XCTAssertTrue(copied.url.path.contains("/workflow-source-staging/"))
        capture.remove()
        XCTAssertFalse(FileManager.default.fileExists(atPath: copied.url.path))
        try SourceAccess(directory: permissions).revoke(path: source.path)
        do {
            _ = try await WorkflowRunSourceCapture.capture(paths: [address(): source.path], workspace: root,
                sourcePermissions: permissions, storageLimitBytes: nil)
            XCTFail("Expected revoked access refusal")
        } catch {}
        let stages = root.appendingPathComponent("workflow-source-staging")
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: stages.path).isEmpty)
    }
}
