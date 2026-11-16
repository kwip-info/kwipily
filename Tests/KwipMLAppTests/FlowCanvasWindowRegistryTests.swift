import XCTest
import AppKit
import KwipMLCore
@testable import KwipMLApp

@MainActor final class FlowCanvasWindowRegistryTests: XCTestCase {
    func testKeyWindowSaveDoesNotLeaveCachedUnsavedFooter() throws {
        _ = NSApplication.shared
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let plan = try FlowDocumentMigration.plan(WorkflowExamples.preparedSeparableConv2dClassifier())
        let root = try FlowDocumentStore.commitMigration(plan, to: directory.appendingPathComponent("Project"))
        let child = root.deletingLastPathComponent().appendingPathComponent(try XCTUnwrap(plan.subflows.keys.first))
        let registry = FlowCanvasWindowRegistry()
        let controller = try registry.open(child, present: false)
        controller.canvas.updateBoundary(context: controller.canvas.boundaryRevision) { $0.title = "Reviewed child" }
        XCTAssertTrue(controller.documentOwner.isDirty)
        XCTAssertTrue(registry.save(window: controller.window))
        XCTAssertFalse(controller.documentOwner.isDirty)
        XCTAssertEqual(controller.canvas.statusMessage, "", "Save state belongs to the live authority, never a cached notice")
        XCTAssertEqual(controller.authority.editStatus, "Subflow document saved. Callers retain their pinned revision.")
        controller.window?.close()
    }
    func testReopenReusesWindowAndDifferentFilesHaveIndependentOwners() throws {
        _ = NSApplication.shared
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let plan = try FlowDocumentMigration.plan(WorkflowExamples.preparedSeparableConv2dClassifier())
        let root = try FlowDocumentStore.commitMigration(plan, to: directory.appendingPathComponent("Project"))
        let firstURL = root.deletingLastPathComponent().appendingPathComponent(try XCTUnwrap(plan.subflows.keys.first))
        var second = try FlowDocumentCodec.decode(Data(contentsOf: firstURL)); second.id = UUID()
        let secondURL = firstURL.deletingLastPathComponent().appendingPathComponent("Second.kwipflow")
        try FlowDocumentCodec.encode(second).write(to: secondURL)
        let registry = FlowCanvasWindowRegistry()
        let first = try registry.open(firstURL, present: false)
        let reopened = try registry.open(firstURL, present: false)
        let other = try registry.open(secondURL, present: false)
        XCTAssertTrue(first === reopened); XCTAssertFalse(first.documentOwner === other.documentOwner)
        XCTAssertTrue(first.documentOwner.edit { doc in if case .modelBlock(var block) = doc.payload { block.title = "Unsaved"; doc.payload = .modelBlock(block) } })
        XCTAssertTrue(first.documentOwner.isDirty); XCTAssertFalse(other.documentOwner.isDirty)
        XCTAssertFalse(first.prepareToClose(.cancel)); XCTAssertTrue(first.documentOwner.isDirty)
        XCTAssertFalse(registry.prepareToTerminate { _ in .cancel })
        XCTAssertTrue(first.documentOwner.isDirty)
        XCTAssertTrue(registry.save(window: other.window!)) // handling a clean peer must not save another document
        XCTAssertTrue(first.documentOwner.isDirty)

        XCTAssertTrue(registry.prepareToTerminate { _ in .save });
        XCTAssertTrue(first.prepareToClose(.save)); XCTAssertFalse(first.documentOwner.isDirty)
        let duplicateURL = firstURL.deletingLastPathComponent().appendingPathComponent("Duplicate.kwipflow")
        try Data(contentsOf: firstURL).write(to: duplicateURL)
        XCTAssertThrowsError(try registry.open(duplicateURL, present: false))
        first.window?.close(); other.window?.close()
    }
    func testFileTabsShareGroupAndKeepIndependentDocumentOwners() throws {
        _ = NSApplication.shared
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let plan = try FlowDocumentMigration.plan(WorkflowExamples.preparedSeparableConv2dClassifier())
        let root = try FlowDocumentStore.commitMigration(plan, to: directory.appendingPathComponent("Project"))
        let childURL = root.deletingLastPathComponent().appendingPathComponent(try XCTUnwrap(plan.subflows.keys.first))
        let registry = FlowCanvasWindowRegistry()
        let main = NSWindow(contentRect: .init(x:0,y:0,width:1100,height:760),styleMask:[.titled,.closable,.resizable],backing:.buffered,defer:false)
        main.isReleasedWhenClosed = false
        registry.registerWorkspaceWindow(main, title: "Main flow")
        let child = try registry.open(childURL, present:false)
        registry.attachAsTab(child, to:main)
        XCTAssertTrue(main.tabbedWindows?.contains(where: { $0 === child.window }) == true)
        XCTAssertEqual(child.window?.tabbingIdentifier,main.tabbingIdentifier)
        XCTAssertEqual(main.tab.title,"Main flow")
        XCTAssertTrue(try registry.open(childURL,present:false) === child)
        XCTAssertTrue(child.documentOwner.edit { doc in if case .modelBlock(var block) = doc.payload {block.title="Tab edit";doc.payload = .modelBlock(block)} })
        XCTAssertTrue(child.documentOwner.isDirty)
        XCTAssertTrue(registry.save(window:child.window))
        XCTAssertFalse(child.documentOwner.isDirty)
        child.window?.close(); main.close()
    }

}
