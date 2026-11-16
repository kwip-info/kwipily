import XCTest
import AppKit
import KwipMLCore
@testable import KwipMLApp

@MainActor final class ChildDraftReturnFocusTests: XCTestCase {
    func testLegacyModelChildUsesInstalledOwnerFocusForApplyAndDiscard() throws {
        _ = NSApplication.shared
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let library = ProjectLibrary(directory: directory.appendingPathComponent("projects"))
        let model = ModelAuthoringDocument(projectID: UUID().uuidString, source: CompositeExamples.parallelClassifier)
        let aRecord = try library.saveSource(.blockModel(model))
        let bRecord = try library.saveSource(.workflow(WorkflowExamples.preparedSeparableConv2dClassifier()))
        let registry = ProjectWindowRegistry(presentsWindows: false, makeWorkspace: { WorkspaceStore(rootOverride: repo, storageOverride: directory, detectCapabilities: false) })
        let a = try registry.open(aRecord, present: false), b = try registry.open(bRecord, present: false)
        let child = try XCTUnwrap(a.session.workspace.modelChildWindows)
        XCTAssertTrue(child.open(path: ["features"], present: false))
        registry.select(window: b.window)
        XCTAssertTrue(child.applyAndClose())
        XCTAssertTrue(registry.activeSession === a.session)
        XCTAssertTrue(child.open(path: ["features"], present: false))
        registry.select(window: b.window)
        XCTAssertTrue(child.discardAndClose())
        XCTAssertTrue(registry.activeSession === a.session)
        let callback = child.onDraftResolved
        XCTAssertTrue(registry.close(a))
        registry.select(window: b.window)
        callback?()
        XCTAssertTrue(registry.activeSession === b.session)
        XCTAssertTrue(registry.close(b))
    }
    func testResolvedDraftReturnsToExactParentWhileCloseFailureAndRetirementDoNot() throws {
        _ = NSApplication.shared
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let library = ProjectLibrary(directory: directory.appendingPathComponent("projects"))
        let aSource = try WorkflowExamples.preparedSeparableConv2dClassifier()
        var bSource = aSource; bSource.projectID = UUID().uuidString
        let aRecord = try library.saveSource(.workflow(aSource)), bRecord = try library.saveSource(.workflow(bSource))
        let registry = ProjectWindowRegistry(presentsWindows: false, makeWorkspace: { WorkspaceStore(rootOverride: repo, storageOverride: directory, detectCapabilities: false) })
        let a = try registry.open(aRecord, present: false), b = try registry.open(bRecord, present: false)
        let owner = try XCTUnwrap(a.session.workspace.workflowSession).owner
        let authority = try WorkflowBlockCanvasAuthority(owner: owner, root: .init(domain: .model, localID: "convolution"))
        let child = ModelChildWindowController(session: ModelChildEditorSession(owner: authority))
        var callbacks = 0
        child.onDraftResolved = { callbacks += 1; a.session.workspace.focusOwningProject?() }
        XCTAssertTrue(child.open(path: ["convolution"], present: false))
        registry.select(window: b.window)
        XCTAssertTrue(child.applyAndClose())
        XCTAssertTrue(registry.activeSession === a.session); XCTAssertEqual(callbacks, 1)
        XCTAssertTrue(child.open(path: ["convolution"], present: false))
        registry.select(window: b.window)
        XCTAssertTrue(child.discardAndClose())
        XCTAssertTrue(registry.activeSession === a.session); XCTAssertEqual(callbacks, 2)
        XCTAssertFalse(child.applyAndClose())
        XCTAssertEqual(callbacks, 2)
        XCTAssertTrue(child.open(path: ["convolution"], present: false))
        registry.select(window: b.window)
        child.restorationWindow?.close()
        XCTAssertTrue(registry.activeSession === b.session); XCTAssertEqual(callbacks, 2)
        XCTAssertTrue(registry.close(a))
        registry.select(window: b.window)
        a.session.workspace.focusOwningProject?()
        XCTAssertTrue(registry.activeSession === b.session)
        child.invalidate()
        XCTAssertEqual(callbacks, 2)
        XCTAssertTrue(registry.close(b))
    }
}
