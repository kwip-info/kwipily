import XCTest
import AppKit
import KwipMLCore
@testable import KwipMLApp

final class RestorationCatalogSnapshotTests: XCTestCase {
    @MainActor func testOneIndependentlyCountedLoadPerCanonicalLibraryAndNormalReadsRemainLive() throws {
        let snapshot = RestorationCatalogSnapshot()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let alias = root.appendingPathComponent("alias")
        let library = root.appendingPathComponent("projects")
        try FileManager.default.createDirectory(at: library, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: library)
        var reads = 0
        func load() -> [ProjectRecord] { reads += 1; return [] }
        try snapshot.withBatch {
            for index in 0..<26 { _ = try snapshot.records(identity: index.isMultiple(of: 2) ? library : alias, load: load) }
            XCTAssertEqual(reads, 1)
            _ = try snapshot.records(identity: root.appendingPathComponent("another-library"), load: load)
            XCTAssertEqual(reads, 2)
        }
        _ = try snapshot.records(identity: library, load: load)
        _ = try snapshot.records(identity: library, load: load)
        XCTAssertEqual(reads, 4, "Outside restoration no cached list may hide changes.")
    }
    @MainActor func testNestedBatchFailureCleanupFreshRetryAndExplicitInvalidation() throws {
        let snapshot = RestorationCatalogSnapshot()
        let identity = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        var reads = 0
        var records: [ProjectRecord] = []
        func load() -> [ProjectRecord] { reads += 1; return records }
        enum Stop: Error { case interrupted }
        XCTAssertThrowsError(try snapshot.withBatch {
            _ = try snapshot.records(identity: identity, load: load)
            try snapshot.withBatch { _ = try snapshot.records(identity: identity, load: load) }
            XCTAssertEqual(reads, 1)
            throw Stop.interrupted
        })
        records = [.init(id: UUID().uuidString, name: "Added after failure", path: identity.appendingPathComponent("new.kwipml").path, updatedAt: .distantPast)]
        try snapshot.withBatch {
            XCTAssertEqual(try snapshot.records(identity: identity, load: load).count, 1)
            XCTAssertEqual(reads, 2)
            snapshot.invalidate(identity: identity)
            _ = try snapshot.records(identity: identity, load: load)
            XCTAssertEqual(reads, 3)
        }
    }
    @MainActor func testFailedLoaderIsBoundedToBatchAndRetriedFresh() throws {
        let snapshot = RestorationCatalogSnapshot()
        let identity = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        var reads = 0
        enum Missing: Error { case library }
        func load() throws -> [ProjectRecord] { reads += 1; throw Missing.library }
        try snapshot.withBatch {
            for _ in 0..<26 { XCTAssertThrowsError(try snapshot.records(identity: identity, load: load)) }
            XCTAssertEqual(reads, 1)
        }
        try snapshot.withBatch { XCTAssertThrowsError(try snapshot.records(identity: identity, load: load)) }
        XCTAssertEqual(reads, 2)
    }
    @MainActor func testCoordinatorFailureRetainsTargetsAndRetryStartsFreshCatalogBatch() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = CanvasTabSessionStore(directory: root)
        let target = CanvasTabSession.Target.managed(UUID())
        try store.save(.init(groups: [.init(tabs: [target], selected: 0)], activeGroup: 0))
        let snapshot = RestorationCatalogSnapshot()
        var reads = 0, denied = true
        let window = NSWindow(contentRect: .init(x: 0, y: 0, width: 400, height: 300), styleMask: [.titled], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        defer { window.close() }
        let coordinator = CanvasTabSessionCoordinator(store: store, bindings: { [] }, resolve: { _ in
            _ = try snapshot.records(identity: root) { reads += 1; return [] }
            if denied { throw CanvasTabSession.Failure("The saved source is unavailable.") }
            return window
        }, presentsWindows: false, beginRestorationBatch: { snapshot.begin() }, endRestorationBatch: { snapshot.end() })
        coordinator.restoreOnce(enabled: true)
        XCTAssertEqual(coordinator.failures.map(\.target), [target])
        XCTAssertEqual(reads, 1)
        denied = false
        coordinator.retry()
        XCTAssertTrue(coordinator.failures.isEmpty)
        XCTAssertEqual(reads, 2)
        _ = try snapshot.records(identity: root) { reads += 1; return [] }
        XCTAssertEqual(reads, 3, "Coordinator must leave no active cache after either failure or successful retry.")
        XCTAssertEqual(try store.load()?.groups.first?.tabs, [target])
    }

}
