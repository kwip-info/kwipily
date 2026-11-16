import XCTest
import AppKit
import KwipMLCore
@testable import KwipMLApp

@MainActor final class CanvasRegistryRestorationTests: XCTestCase {
    func testExternalChildRestoresWithClosedRootAndRetainsIndependentGrant() throws {
        _ = NSApplication.shared
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let app = directory.appendingPathComponent("app"), repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let access = FlowFolderAccess(hooks: .init(createBookmark: { Data($0.path.utf8) }, resolveBookmark: { (URL(fileURLWithPath: String(decoding: $0, as: UTF8.self)), false) }, start: { _ in true }, stop: { _ in }))
        let catalog = ExternalFlowCatalog(directory: app.appendingPathComponent("external-flow-catalog"), access: access)
        let plan = try FlowDocumentMigration.plan(WorkflowExamples.preparedSeparableConv2dClassifier())
        let mainURL = try FlowDocumentStore.commitMigration(plan, to: directory.appendingPathComponent("external"))
        let record = try catalog.attach(folder: mainURL.deletingLastPathComponent(), rootURL: mainURL)
        let registry = ProjectWindowRegistry(presentsWindows: false,
            makeWorkspace: { WorkspaceStore(rootOverride: repo, storageOverride: app, detectCapabilities: false) }, externalCatalog: catalog,
            makeExternalWorkspace: { WorkspaceStore(rootOverride: repo, storageOverride: app, detectCapabilities: false, externalLocation: $0) })
        let root = try registry.openExternal(record, present: false)
        let dependency = try XCTUnwrap(plan.root.dependencies.first)
        let target = CanvasTabSession.Target.subflow(root: .external(record.id), path: dependency.relativePath, documentID: dependency.revision.documentID)
        let child = try FlowCanvasWindowRegistry.shared.open(mainURL.deletingLastPathComponent().appendingPathComponent(dependency.relativePath), present: false,
            accessLease: root.session.workspace.externalFolderLease, restorationTarget: target)
        XCTAssertTrue(registry.close(root)); XCTAssertEqual(registry.projectCount, 0)
        XCTAssertNotNil(child.accessLease); XCTAssertEqual(child.restorationTarget, target)
        child.window?.close()
        let reopened = try registry.restoreCanvas(target)
        defer { reopened.close() }
        XCTAssertEqual(registry.projectCount, 0, "Restoring a child must not create its root editor")
        XCTAssertTrue(try registry.restoreCanvas(target) === reopened)
        let owner = try XCTUnwrap(FlowCanvasWindowRegistry.shared.openControllers.first { $0.window === reopened })
        XCTAssertEqual(owner.accessLease?.locationID, record.id)
        XCTAssertEqual(owner.documentOwner.document.id, dependency.revision.documentID)
    }
}
