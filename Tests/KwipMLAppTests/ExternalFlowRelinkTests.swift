import XCTest
import AppKit
import KwipMLCore
@testable import KwipMLApp

@MainActor final class ExternalFlowRelinkTests: XCTestCase {
    private func catalog(_ directory: URL) -> ExternalFlowCatalog {
        ExternalFlowCatalog(directory: directory.appendingPathComponent("Catalog"), access: FlowFolderAccess(hooks: .init(
            createBookmark: { Data($0.path.utf8) },
            resolveBookmark: { (URL(fileURLWithPath: String(decoding: $0, as: UTF8.self)), false) },
            start: { _ in true }, stop: { _ in })))
    }
    private func fixture(_ directory: URL) throws -> URL {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return try FlowDocumentStore.commitMigration(FlowDocumentMigration.plan(WorkflowExamples.preparedSeparableConv2dClassifier()), to: directory.appendingPathComponent("Flow"))
    }
    func testExplicitRelinkPreservesLocalIdentityAndFilesAndRejectsStaleRecord() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let root = try fixture(directory), store = catalog(directory)
        let original = try store.attach(folder: root.deletingLastPathComponent(), rootURL: root)
        let before = try Data(contentsOf: root)
        let moved = directory.appendingPathComponent("Moved")
        try FileManager.default.moveItem(at: root.deletingLastPathComponent(), to: moved)
        let destination = moved.appendingPathComponent("Main.kwipflow")
        let updated = try store.relink(original, folder: moved, rootURL: destination)
        XCTAssertEqual(updated.id, original.id)
        XCTAssertEqual(updated.projectID, original.projectID)
        XCTAssertEqual(try Data(contentsOf: destination), before)
        XCTAssertEqual(try store.list(), [updated])
        XCTAssertEqual(try store.open(updated).rootURL, destination.resolvingSymlinksInPath())
        XCTAssertThrowsError(try store.relink(original, folder: moved, rootURL: destination))
        XCTAssertEqual(try store.list(), [updated])
    }
    func testRelinkRefusesAnotherCatalogLocationOrDifferentProjectWithoutChangingEitherRecord() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let root = try fixture(directory), store = catalog(directory)
        let original = try store.attach(folder: root.deletingLastPathComponent(), rootURL: root)
        let copy = directory.appendingPathComponent("Copy")
        try FileManager.default.copyItem(at: root.deletingLastPathComponent(), to: copy)
        let copyRoot = copy.appendingPathComponent("Main.kwipflow")
        _ = try store.attach(folder: copy, rootURL: copyRoot)
        let other = try fixture(directory.appendingPathComponent("Other"))
        let before = try store.list()
        XCTAssertThrowsError(try store.relink(original, folder: copy, rootURL: copyRoot))
        XCTAssertThrowsError(try store.relink(original, folder: other.deletingLastPathComponent(), rootURL: other))
        XCTAssertEqual(try store.list(), before)
    }
    func testOpenRootAndRetainedChildPreventRelocationUntilBothClose() throws {
        _ = NSApplication.shared
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let root = try fixture(directory), store = catalog(directory)
        let record = try store.attach(folder: root.deletingLastPathComponent(), rootURL: root)
        let copy = directory.appendingPathComponent("Moved")
        try FileManager.default.copyItem(at: root.deletingLastPathComponent(), to: copy)
        let destination = copy.appendingPathComponent("Main.kwipflow")
        let application = directory.appendingPathComponent("App")
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry = ProjectWindowRegistry(presentsWindows: false,
            makeWorkspace: { WorkspaceStore(rootOverride: repository, storageOverride: application, detectCapabilities: false) },
            externalCatalog: store,
            makeExternalWorkspace: { WorkspaceStore(rootOverride: repository, storageOverride: application, detectCapabilities: false, externalLocation: $0) })
        let children = FlowCanvasWindowRegistry()
        let owner = try registry.openExternal(record, present: false)
        let dependency = try XCTUnwrap(FlowDocumentStore.readHead(at: root).dependencies.first)
        let child = try children.open(root.deletingLastPathComponent().appendingPathComponent(dependency.relativePath),
            present: false, accessLease: owner.session.workspace.externalFolderLease)
        defer { child.window?.close(); _ = registry.close(owner) }
        XCTAssertThrowsError(try registry.relinkExternal(record, folder: copy, rootURL: destination, children: children))
        XCTAssertTrue(registry.close(owner))
        XCTAssertThrowsError(try registry.relinkExternal(record, folder: copy, rootURL: destination, children: children))
        XCTAssertEqual(try store.list(), [record])
        child.window?.close()
        let updated = try registry.relinkExternal(record, folder: copy, rootURL: destination, children: children)
        XCTAssertEqual(updated.id, record.id)
        XCTAssertEqual(try store.list(), [updated])
    }
    func testSharedChildBlocksRelinkOfEveryCallerLocationNotOnlyRetainedGrant() throws {
        _ = NSApplication.shared
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let root = try fixture(directory), store = catalog(directory)
        let folder = root.deletingLastPathComponent()
        let secondRoot = folder.appendingPathComponent("Second.kwipflow")
        try FileManager.default.copyItem(at: root, to: secondRoot)
        let first = try store.attach(folder: folder, rootURL: root)
        let second = try store.attach(folder: folder, rootURL: secondRoot)
        XCTAssertNotEqual(first.id, second.id)
        let moved = directory.appendingPathComponent("Moved")
        try FileManager.default.copyItem(at: folder, to: moved)
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry = ProjectWindowRegistry(presentsWindows: false,
            makeWorkspace: { WorkspaceStore(rootOverride: repository, storageOverride: directory.appendingPathComponent("App"), detectCapabilities: false) },
            externalCatalog: store)
        let children = FlowCanvasWindowRegistry()
        let dependency = try XCTUnwrap(FlowDocumentStore.readHead(at: root).dependencies.first)
        let childURL = folder.appendingPathComponent(dependency.relativePath)
        let child = try children.open(childURL, present: false, accessLease: store.open(first))
        defer { child.window?.close() }
        let shared = try children.open(childURL, present: false, accessLease: store.open(second))
        XCTAssertTrue(child === shared)
        XCTAssertEqual(child.accessLease?.locationID, first.id, "The shared editor intentionally retains the first sufficient grant")
        XCTAssertEqual(registry.projectCount, 0, "Both parent editors can be closed while the shared child stays open")
        let before = try store.list()
        XCTAssertThrowsError(try registry.relinkExternal(second, folder: moved,
            rootURL: moved.appendingPathComponent("Second.kwipflow"), children: children))
        XCTAssertEqual(try store.list(), before, "No local history association changes while the old child remains editable")
    }

    func testRelinkRefusesClosedManagedLibraryDestinationAndPreservesFiles() throws {
        _ = NSApplication.shared
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: directory.appendingPathComponent("App"), detectCapabilities: false)
        let source = try WorkflowExamples.preparedSeparableConv2dClassifier()
        let saved = try workspace.projectLibrary.saveSource(.workflow(source))
        let managed = try workspace.projectLibrary.activateFileBackedWorkflow(id: saved.id)
        let managedRoot = URL(fileURLWithPath: managed.path)
        let copy = directory.appendingPathComponent("External")
        try FileManager.default.copyItem(at: managedRoot.deletingLastPathComponent(), to: copy)
        let externalRoot = copy.appendingPathComponent("Main.kwipflow")
        let store = catalog(directory)
        let external = try store.attach(folder: copy, rootURL: externalRoot)
        let registry = ProjectWindowRegistry(presentsWindows: false, makeWorkspace: { workspace }, externalCatalog: store)
        XCTAssertEqual(registry.projectCount, 0, "A closed managed project still owns its local file/history namespace")
        let managedBytes = try Data(contentsOf: managedRoot), externalBytes = try Data(contentsOf: externalRoot)
        XCTAssertThrowsError(try registry.relinkExternal(external, folder: managedRoot.deletingLastPathComponent(), rootURL: managedRoot,
            children: FlowCanvasWindowRegistry()))
        XCTAssertEqual(try store.list(), [external])
        XCTAssertEqual(try Data(contentsOf: managedRoot), managedBytes)
        XCTAssertEqual(try Data(contentsOf: externalRoot), externalBytes)
    }

}
