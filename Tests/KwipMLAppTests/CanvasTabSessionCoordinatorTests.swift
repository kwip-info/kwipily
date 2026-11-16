import XCTest
import AppKit
import KwipMLCore
@testable import KwipMLApp

@MainActor final class CanvasTabSessionCoordinatorTests: XCTestCase {
    private func fixture() throws -> CanvasTabSessionStore {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return CanvasTabSessionStore(directory: url)
    }
    private func window() -> NSWindow {
        let result = NSWindow(contentRect: .init(x: 0, y: 0, width: 800, height: 600), styleMask: [.titled, .closable, .resizable], backing: .buffered, defer: false)
        result.isReleasedWhenClosed = false
        addTeardownBlock { MainActor.assumeIsolated { result.close() } }
        return result
    }
    func testFailedPreparationNeverCapturesAndActualTabOrderSelectionRestoresOnce() throws {
        let store = try fixture(), a = window(), b = window(), detached = window()
        a.addTabbedWindow(b, ordered: .above); a.tabGroup?.selectedWindow = b
        let first = CanvasTabSession.Target.managed(UUID()), second = CanvasTabSession.Target.external(UUID())
        let standalone = CanvasTabSession.Target.standalone(locationID: UUID(), path: "Child.kwipflow", documentID: UUID())
        var calls = 0
        let coordinator = CanvasTabSessionCoordinator(store: store, bindings: { [.init(target: first, window: a), .init(target: second, window: b), .init(target: standalone, window: detached)] }, resolve: { target in
            calls += 1; return target == first ? a : target == second ? b : detached
        }, screens: { [.init(x: 0, y: 0, width: 1200, height: 800)] }, presentsWindows: false)
        XCTAssertFalse(try coordinator.captureAfterPreparation { false }); XCTAssertNil(try store.load())
        XCTAssertTrue(try coordinator.captureAfterPreparation { true })
        let saved = try XCTUnwrap(store.load())
        XCTAssertTrue(saved.groups.contains { $0.tabs == [first, second] && $0.selected == 1 })
        coordinator.restoreOnce(enabled: true); coordinator.restoreOnce(enabled: true)
        XCTAssertEqual(calls, 3); XCTAssertTrue(coordinator.failures.isEmpty)
        XCTAssertTrue(a.tabGroup?.selectedWindow === b)
    }
    func testMissingChildSurvivesQuitAndRetryUsesExistingOwnerWithoutOpeningRoot() throws {
        let store = try fixture(), existing = window(), root = CanvasTabSession.Root.external(UUID())
        let child = CanvasTabSession.Target.subflow(root: root, path: "blocks/Child.kwipflow", documentID: UUID())
        try store.save(.init(groups: [.init(tabs: [child], selected: 0)], activeGroup: 0))
        var denied = true, calls: [CanvasTabSession.Target] = []
        let coordinator = CanvasTabSessionCoordinator(store: store, bindings: { [] }, resolve: { target in
            calls.append(target)
            if denied { throw CanvasTabSession.Failure("Folder access needs renewal") }
            return existing
        }, presentsWindows: false)
        coordinator.restoreOnce(enabled: true)
        XCTAssertEqual(coordinator.failures.count, 1); XCTAssertTrue(coordinator.failures[0].reason.contains("renewal"))
        XCTAssertTrue(try coordinator.captureAfterPreparation { true })
        XCTAssertEqual(try store.load()?.groups.first?.tabs, [child])
        denied = false; coordinator.retry(); coordinator.retry()
        XCTAssertTrue(coordinator.failures.isEmpty); XCTAssertEqual(calls, [child, child, child])
    }
    func testDisabledLaunchPreservesManualRetryAndUnknownSessionBytes() throws {
        let store = try fixture(), a = window()
        try store.save(.init(groups: [.init(tabs: [.library], selected: 0)], activeGroup: 0))
        var opened = 0
        let coordinator = CanvasTabSessionCoordinator(store: store, bindings: { [] }, resolve: { _ in opened += 1; return a }, presentsWindows: false)
        coordinator.restoreOnce(enabled: false); XCTAssertEqual(opened, 0)
        coordinator.retry(); XCTAssertEqual(opened, 1)
        let corrupt = Data("{\"schemaVersion\":999}".utf8)
        try corrupt.write(to: store.directory.appendingPathComponent("canvas-session.json"))
        XCTAssertThrowsError(try coordinator.captureAfterPreparation { true })
        XCTAssertEqual(try Data(contentsOf: store.directory.appendingPathComponent("canvas-session.json")), corrupt)
    }
    func testFramesAreClampedToCurrentScreenWithoutInvalidGeometry() {
        let frame = CanvasTabSession.Frame(x: -8000, y: 8000, width: 32000, height: 32000)
        let screen = NSRect(x: 0, y: 25, width: 1200, height: 775)
        let restored = CanvasTabSessionCoordinator.safeFrame(frame, screens: [screen])
        XCTAssertEqual(restored, screen)
    }
}

extension CanvasTabSessionCoordinatorTests {
    func testRestorePreferenceMigratesAbsentFieldAndRejectsUnknownWithoutRewrite() throws {
        let sessionStore = try fixture(), url = sessionStore.directory.appendingPathComponent("preferences.json")
        let store = KwipMLCore.AppPreferencesStore(fileURL: url)
        try store.save(.factory)
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        object.removeValue(forKey: "restoreCanvasSessionAtLaunch")
        let legacy = try JSONSerialization.data(withJSONObject: object)
        try legacy.write(to: url)
        XCTAssertNotEqual(try store.load().restoreCanvasSessionAtLaunch, false)
        XCTAssertEqual(try Data(contentsOf: url), legacy)
        var next = try store.load(); next.restoreCanvasSessionAtLaunch = false
        try store.save(next); XCTAssertEqual(try store.load().restoreCanvasSessionAtLaunch, false)
        object["schemaVersion"] = 999
        let future = try JSONSerialization.data(withJSONObject: object); try future.write(to: url)
        XCTAssertThrowsError(try store.load()); XCTAssertEqual(try Data(contentsOf: url), future)
    }
}

extension CanvasTabSessionCoordinatorTests {
    func testRetryUnavailableDoesNotResolveOrRegroupSuccessfulTabsAgain() throws {
        let store = try fixture(), good = window(), missing = window(), other = window()
        let a = CanvasTabSession.Target.managed(UUID()), b = CanvasTabSession.Target.external(UUID())
        try store.save(.init(groups: [.init(tabs: [a, b], selected: 0)], activeGroup: 0))
        var denied = true, calls: [CanvasTabSession.Target] = []
        let coordinator = CanvasTabSessionCoordinator(store: store, bindings: { [.init(target: a, window: good)] }, resolve: { target in
            calls.append(target)
            if target == b && denied { throw CanvasTabSession.Failure("Missing file") }
            return target == a ? good : missing
        }, presentsWindows: false)
        coordinator.restoreOnce(enabled: true)
        good.addTabbedWindow(other, ordered: .above); good.tabGroup?.selectedWindow = other
        denied = false; coordinator.retry()
        XCTAssertEqual(calls, [a, b, b])
        XCTAssertTrue(good.tabbedWindows?.contains(where: { $0 === other }) == true)
    }
    func testTrackedOpenOwnerIsCapturedEvenWhenItsWindowIsNotVisible() throws {
        let store = try fixture(), minimizedOrHidden = window(), target = CanvasTabSession.Target.managed(UUID())
        let coordinator = CanvasTabSessionCoordinator(store: store, bindings: { [.init(target: target, window: minimizedOrHidden)] }, resolve: { _ in minimizedOrHidden }, presentsWindows: true)
        XCTAssertFalse(minimizedOrHidden.isVisible)
        XCTAssertTrue(try coordinator.captureAfterPreparation { true })
        XCTAssertEqual(try store.load()?.groups.first?.tabs, [target])
    }
}
