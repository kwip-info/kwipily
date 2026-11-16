import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class ExternalFlowCatalogTests: XCTestCase {
    private final class ScopeProbe {
        var starts = 0; var stops = 0; var stale = false
        var resolved: URL?
        var hooks: FlowFolderAccess.Hooks {
            .init(createBookmark: { Data($0.path.utf8) }, resolveBookmark: { [self] data in
                (resolved ?? URL(fileURLWithPath: String(decoding: data, as: UTF8.self)), stale)
            }, start: { [self] _ in starts += 1; return true }, stop: { [self] _ in stops += 1 })
        }
    }
    private func fixture(_ directory: URL) throws -> URL {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return try FlowDocumentStore.commitMigration(FlowDocumentMigration.plan(WorkflowExamples.preparedSeparableConv2dClassifier()), to: directory.appendingPathComponent("Flow"))
    }
    func testAttachIsReadOnlyDedupePersistsAndSharedLeaseOutlivesOriginalHolder() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let root = try fixture(directory)
        let before = try Data(contentsOf: root)
        let probe = ScopeProbe()
        let catalog = ExternalFlowCatalog(directory: directory.appendingPathComponent("Private"), access: FlowFolderAccess(hooks: probe.hooks))
        let record = try catalog.attach(folder: root.deletingLastPathComponent(), rootURL: root)
        XCTAssertEqual(try catalog.attach(folder: root.deletingLastPathComponent(), rootURL: root).id, record.id)
        XCTAssertEqual(try Data(contentsOf: root), before)
        XCTAssertEqual(try catalog.list(), [record])
        var first: FlowFolderAccess.Lease? = try catalog.open(record)
        var child = first
        let stops = probe.stops
        first = nil
        XCTAssertEqual(probe.stops, stops)
        XCTAssertEqual(child?.rootURL, root.resolvingSymlinksInPath())
        XCTAssertEqual(child?.locationID, record.id)
        child = nil
        XCTAssertEqual(probe.stops, stops + 1)
    }
    func testMissingMovedStaleAndWrongIdentityRefuseButCachedEntrySurvives() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let root = try fixture(directory)
        let probe = ScopeProbe()
        let catalog = ExternalFlowCatalog(directory: directory.appendingPathComponent("Private"), access: FlowFolderAccess(hooks: probe.hooks))
        let record = try catalog.attach(folder: root.deletingLastPathComponent(), rootURL: root)
        probe.stale = true
        XCTAssertThrowsError(try catalog.open(record))
        probe.stale = false; probe.resolved = directory
        XCTAssertThrowsError(try catalog.open(record))
        probe.resolved = nil
        let original = try Data(contentsOf: root)
        var wrong = try FlowDocumentCodec.decode(original)
        wrong.id = UUID()
        if case .workflow(var workflow) = wrong.payload { workflow.projectID = wrong.id.uuidString; wrong.payload = .workflow(workflow) }
        try FlowDocumentCodec.encode(wrong).write(to: root)
        XCTAssertThrowsError(try catalog.open(record))
        XCTAssertThrowsError(try catalog.attach(folder: root.deletingLastPathComponent(), rootURL: root))
        try FileManager.default.removeItem(at: root)
        XCTAssertThrowsError(try catalog.open(record))
        XCTAssertEqual(try catalog.list(), [record])
        XCTAssertEqual(probe.starts, probe.stops)
    }
    func testCopiesKeepDistinctLocalIdentityAndFolderMustContainRootDirectly() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let root = try fixture(directory)
        let copy = directory.appendingPathComponent("Copy")
        try FileManager.default.copyItem(at: root.deletingLastPathComponent(), to: copy)
        let probe = ScopeProbe()
        let catalog = ExternalFlowCatalog(directory: directory.appendingPathComponent("Private"), access: FlowFolderAccess(hooks: probe.hooks))
        let a = try catalog.attach(folder: root.deletingLastPathComponent(), rootURL: root)
        let b = try catalog.attach(folder: copy, rootURL: copy.appendingPathComponent("Main.kwipflow"))
        XCTAssertNotEqual(a.id, b.id); XCTAssertEqual(a.projectID, b.projectID)
        XCTAssertThrowsError(try catalog.attach(folder: directory, rootURL: root))
        XCTAssertEqual(try catalog.list().count, 2)
    }
}
