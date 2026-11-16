import XCTest
@testable import KwipMLCore

final class SourceSnapshotTests: XCTestCase {
    func testSnapshotFreezesBytesAndContentIdentityWithoutChangingSource() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let source = root.appendingPathComponent("measurements.csv")
        let first = Data("x,y,label\n1,2,0\n".utf8)
        try first.write(to: source)
        let snapshot = try SourceSnapshot.prepare(source: source, directory: root.appendingPathComponent("snapshots"))
        XCTAssertEqual(try Data(contentsOf: snapshot.url), first)
        XCTAssertEqual(snapshot.url.lastPathComponent, source.lastPathComponent)
        XCTAssertEqual(snapshot.sha256.count, 64)
        try Data("x,y,label\n3,4,1\n".utf8).write(to: source)
        let changed = try SourceSnapshot.prepare(source: source, directory: root.appendingPathComponent("snapshots"))
        XCTAssertNotEqual(changed.sha256, snapshot.sha256)
        XCTAssertEqual(try Data(contentsOf: snapshot.url), first)
        XCTAssertEqual(try SourceSnapshot.prepare(source: source, directory: root.appendingPathComponent("snapshots")).url, changed.url)
    }
    func testOversizedOrMissingSourcePublishesNoSnapshot() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let source = root.appendingPathComponent("too-large.csv"), output = root.appendingPathComponent("snapshots")
        try Data(repeating: 65, count: 101).write(to: source)
        XCTAssertThrowsError(try SourceSnapshot.prepare(source: source, directory: output, maximumBytes: 100))
        XCTAssertThrowsError(try SourceSnapshot.prepare(source: root.appendingPathComponent("missing.csv"), directory: output))
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.path))
    }
    func testModifiedSnapshotIsRepairedFromAuthorizedSource() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let source = root.appendingPathComponent("data.csv"), bytes = Data("a,b\n1,2\n".utf8)
        try bytes.write(to: source)
        let snapshot = try SourceSnapshot.prepare(source: source, directory: root.appendingPathComponent("snapshots"))
        try Data("modified".utf8).write(to: snapshot.url)
        let repaired = try SourceSnapshot.prepare(source: source, directory: root.appendingPathComponent("snapshots"))
        XCTAssertEqual(try Data(contentsOf: repaired.url), bytes)
    }
}
