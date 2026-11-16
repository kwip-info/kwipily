import XCTest
import AppKit
import KwipMLCore
@testable import KwipMLApp

@MainActor final class ExternalProjectWindowTests: XCTestCase {
    func testSameUUIDExternalCopiesAndManagedOriginalOwnIndependentTabs() throws {
        _ = NSApplication.shared
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let application = directory.appendingPathComponent("app")
        let access = FlowFolderAccess(hooks: .init(createBookmark: { Data($0.path.utf8) }, resolveBookmark: { (URL(fileURLWithPath: String(decoding: $0, as: UTF8.self)), false) }, start: { _ in true }, stop: { _ in }))
        let catalog = ExternalFlowCatalog(directory: application.appendingPathComponent("external-flow-catalog"), access: access)
        let source = try WorkflowExamples.preparedSeparableConv2dClassifier()
        let plan = try FlowDocumentMigration.plan(source)
        let aURL = try FlowDocumentStore.commitMigration(plan, to: directory.appendingPathComponent("first"))
        let bURL = try FlowDocumentStore.commitMigration(plan, to: directory.appendingPathComponent("second"))
        let aRecord = try catalog.attach(folder: aURL.deletingLastPathComponent(), rootURL: aURL)
        let bRecord = try catalog.attach(folder: bURL.deletingLastPathComponent(), rootURL: bURL)
        let managed = try ProjectLibrary(directory: application.appendingPathComponent("projects")).saveSource(.workflow(source))
        var allocations = 0
        let registry = ProjectWindowRegistry(presentsWindows: false,
            makeWorkspace: { WorkspaceStore(rootOverride: root, storageOverride: application, detectCapabilities: false) },
            externalCatalog: catalog,
            makeExternalWorkspace: { location in allocations += 1; return WorkspaceStore(rootOverride: root, storageOverride: application, detectCapabilities: false, externalLocation: location) })
        let a = try registry.openExternal(aRecord, present: false)
        let b = try registry.openExternal(bRecord, present: false)
        let original = try registry.open(managed, present: false)
        XCTAssertEqual(registry.projectCount, 3)
        XCTAssertFalse(a === b); XCTAssertFalse(original === a); XCTAssertFalse(original === b)
        XCTAssertNotEqual(a.session.workspace.workspaceDirectory, b.session.workspace.workspaceDirectory)
        a.session.workspace.generatedCode = "Output owned by first copy"
        XCTAssertTrue(try registry.openExternal(aRecord, present: false) === a)
        XCTAssertEqual(allocations, 2)
        XCTAssertTrue(try registry.openFile(aURL, present: false) === a)
        XCTAssertEqual(a.session.workspace.generatedCode, "Output owned by first copy")
        XCTAssertNotNil(a.session.workspace.externalFolderLease)
        XCTAssertTrue(try registry.openExternal(aRecord,present:true) === a)
        XCTAssertNotNil(try RecentProjectHistory(directory:application).load()[aRecord.rootURL.standardizedFileURL.path],"Linked workspaces must update the application-wide recent list")
        XCTAssertTrue(registry.close(a))
        XCTAssertNil(a.session.workspace.externalFolderLease)
        XCTAssertNotNil(b.session.workspace.workflowSession)
        XCTAssertTrue(registry.close(b)); XCTAssertTrue(registry.close(original))
    }
}
