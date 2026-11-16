import XCTest
@testable import KwipMLApp

final class CanvasTabSessionStoreTests: XCTestCase {
    private func temporary() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
    func testMixedRootChildAndDetachedGroupsKeepOrderSelectionAndLocalIdentity() throws {
        let directory = try temporary(); defer { try? FileManager.default.removeItem(at: directory) }
        let store = CanvasTabSessionStore(directory: directory)
        XCTAssertNil(try store.load())
        let projectID = UUID(), locationA = UUID(), locationB = UUID(), childID = UUID()
        let a = CanvasTabSession.Target.external(locationA)
        let b = CanvasTabSession.Target.external(locationB)
        let snapshot = CanvasTabSession(groups: [
            .init(tabs: [.library, .managed(projectID), a], selected: 2),
            .init(tabs: [.subflow(root: .external(locationA), path: "blocks/Child.kwipflow", documentID: childID), b], selected: 0)
        ], activeGroup: 1)
        try store.save(snapshot)
        XCTAssertEqual(try CanvasTabSessionStore(directory: directory).load(), snapshot)
        let bytes = try Data(contentsOf: directory.appendingPathComponent("canvas-session.json"))
        XCTAssertFalse(String(decoding: bytes, as: UTF8.self).contains(directory.path))
    }
    func testInvalidPathsDuplicatesAndSelectionsCannotReplaceSavedSession() throws {
        let directory = try temporary(); defer { try? FileManager.default.removeItem(at: directory) }
        let store = CanvasTabSessionStore(directory: directory)
        let root = CanvasTabSession.Root.managed(UUID())
        let good = CanvasTabSession(groups: [.init(tabs: [.library], selected: 0)], activeGroup: 0)
        try store.save(good)
        for path in ["../Child.kwipflow", "/tmp/Child.kwipflow", "blocks/../Child.kwipflow", "blocks//Child.kwipflow", "Child.json", "blocks\\Child.kwipflow"] {
            let invalid = CanvasTabSession(groups: [.init(tabs: [.subflow(root: root, path: path, documentID: UUID())], selected: 0)], activeGroup: 0)
            XCTAssertThrowsError(try store.save(invalid), path)
            XCTAssertEqual(try store.load(), good)
        }
        XCTAssertThrowsError(try store.save(.init(groups: [.init(tabs: [.library, .library], selected: 0)], activeGroup: 0)))
        XCTAssertThrowsError(try store.save(.init(groups: [.init(tabs: [.library], selected: 1)], activeGroup: 0)))
        XCTAssertThrowsError(try store.save(.init(groups: [.init(tabs: [.library], selected: 0)], activeGroup: 1)))
        XCTAssertEqual(try store.load(), good)
    }
    func testBoundsAndConflictingChildIdentityAreRejected() throws {
        let directory = try temporary(); defer { try? FileManager.default.removeItem(at: directory) }
        let store = CanvasTabSessionStore(directory: directory)
        let root = CanvasTabSession.Root.external(UUID())
        let duplicates = CanvasTabSession(groups: [.init(tabs: [
            .subflow(root: root, path: "Child.kwipflow", documentID: UUID()),
            .subflow(root: root, path: "Child.kwipflow", documentID: UUID())
        ], selected: 0)], activeGroup: 0)
        XCTAssertThrowsError(try store.save(duplicates))
        let tooMany = CanvasTabSession(groups: [.init(tabs: (0..<129).map { _ in .managed(UUID()) }, selected: 0)], activeGroup: 0)
        XCTAssertThrowsError(try store.save(tooMany))
        var group = CanvasTabSession.Group(tabs: [.library], selected: 0)
        group.frame = .init(x: 0, y: 0, width: .infinity, height: 700)
        XCTAssertThrowsError(try store.save(.init(groups: [group], activeGroup: 0)))
        XCTAssertNil(try store.load())
    }
    func testUnknownVersionCorruptionAndSymlinkRefuseWithoutOverwriting() throws {
        let directory = try temporary(); defer { try? FileManager.default.removeItem(at: directory) }
        let store = CanvasTabSessionStore(directory: directory)
        let snapshot = CanvasTabSession(groups: [], activeGroup: nil)
        try store.save(snapshot)
        let file = directory.appendingPathComponent("canvas-session.json")
        let original = try Data(contentsOf: file)
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: original) as? [String: Any])
        object["schemaVersion"] = 999
        let future = try JSONSerialization.data(withJSONObject: object)
        try future.write(to: file)
        XCTAssertThrowsError(try store.load())
        XCTAssertEqual(try Data(contentsOf: file), future)
        XCTAssertThrowsError(try store.save(snapshot))
        XCTAssertEqual(try Data(contentsOf: file), future)
        try FileManager.default.removeItem(at: file)
        let outside = directory.appendingPathComponent("unrelated.json")
        try original.write(to: outside)
        try FileManager.default.createSymbolicLink(at: file, withDestinationURL: outside)
        XCTAssertThrowsError(try store.load())
        XCTAssertThrowsError(try store.save(snapshot))
        XCTAssertEqual(try Data(contentsOf: outside), original)
    }
}
