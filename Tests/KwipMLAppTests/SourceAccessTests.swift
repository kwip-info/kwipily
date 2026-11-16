import XCTest
@testable import KwipMLApp

final class SourceAccessTests: XCTestCase {
    func testRevocationBlocksPlainPathAfterRestartAndReselectionRestoresAccess() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        let source=root.appendingPathComponent("source.csv"),other=root.appendingPathComponent("other.csv")
        let bytes=Data("a,b\n1,2\n".utf8)
        try bytes.write(to:source);try bytes.write(to:other)
        let folder=root.appendingPathComponent("permissions"),access=SourceAccess(directory:root.appendingPathComponent("permissions"))
        try access.remember(source);try access.remember(other)
        try access.revoke(path:source.path);try access.revoke(path:source.path)
        let reopened=SourceAccess(directory:folder)
        var invoked=false
        XCTAssertThrowsError(try reopened.withAccess(to:source.path) {_ in invoked=true}) {error in
            XCTAssertTrue(error.localizedDescription.contains("revoked"))
        }
        XCTAssertFalse(invoked)
        XCTAssertEqual(try reopened.withAccess(to:other.path) {try Data(contentsOf:$0)},bytes)
        XCTAssertEqual(try Data(contentsOf:source),bytes)
        try reopened.remember(source)
        XCTAssertEqual(try reopened.withAccess(to:source.path) {try Data(contentsOf:$0)},bytes)
    }
    func testRevokingUnrememberedSourceStillPreventsPathFallback() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let access=SourceAccess(directory:root)
        try access.revoke(path:"/unremembered.csv")
        var invoked=false
        XCTAssertThrowsError(try access.withAccess(to:"/unremembered.csv") {_ in invoked=true})
        XCTAssertFalse(invoked)
    }
    func testSnapshotFailureExplainsRecoveryAndPreservesUnderlyingCause() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let access = SourceAccess(directory: root.appendingPathComponent("permissions"))
        XCTAssertThrowsError(try access.snapshot(to: root.appendingPathComponent("missing.csv").path, directory: root.appendingPathComponent("snapshots"))) { error in
            XCTAssertTrue(error.localizedDescription.contains("Choose local CSV"))
            XCTAssertNotNil((error as NSError).userInfo[NSUnderlyingErrorKey])
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("snapshots").path))
    }
    func testBookmarkCanBeReopenedForLocalReadWithoutChangingSource() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let source = root.appendingPathComponent("source.csv"), data = Data("a,b\n1,2\n".utf8)
        try data.write(to: source)
        let folder = root.appendingPathComponent("private-permissions")
        try SourceAccess(directory: folder).remember(source)
        let restored = SourceAccess(directory: folder)
        XCTAssertEqual(try restored.withAccess(to: source.path) { try Data(contentsOf: $0) }, data)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil).count, 1)
        XCTAssertEqual(try Data(contentsOf: source), data)
    }
    func testCorruptPermissionDoesNotSilentlyFallBackToAnUntrackedRead() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let source = root.appendingPathComponent("source.csv")
        try Data("a,b".utf8).write(to: source)
        let folder = root.appendingPathComponent("private-permissions"), access = SourceAccess(directory: root.appendingPathComponent("private-permissions"))
        try access.remember(source)
        let record = try XCTUnwrap(FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil).first)
        try Data("invalid bookmark".utf8).write(to: record)
        var invoked = false
        XCTAssertThrowsError(try access.withAccess(to: source.path) { _ in invoked = true })
        XCTAssertFalse(invoked)
    }
}
