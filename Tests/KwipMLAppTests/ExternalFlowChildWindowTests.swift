import XCTest
import AppKit
import KwipMLCore
@testable import KwipMLApp

@MainActor final class ExternalFlowChildWindowTests: XCTestCase {
    func testCopiesHaveIndependentWindowsAndChildRetainsScopeUntilClose() throws {
        _ = NSApplication.shared
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let plan = try FlowDocumentMigration.plan(WorkflowExamples.preparedSeparableConv2dClassifier())
        let root = try FlowDocumentStore.commitMigration(plan, to: directory.appendingPathComponent("First"))
        let copyFolder = directory.appendingPathComponent("Copy")
        try FileManager.default.copyItem(at: root.deletingLastPathComponent(), to: copyFolder)
        let dependency = try XCTUnwrap(plan.root.dependencies.first)
        let firstURL = root.deletingLastPathComponent().appendingPathComponent(dependency.relativePath)
        let secondURL = copyFolder.appendingPathComponent(dependency.relativePath)
        var stops = 0
        let access = FlowFolderAccess(hooks: .init(createBookmark: { Data($0.path.utf8) }, resolveBookmark: { (URL(fileURLWithPath: String(decoding:$0,as:UTF8.self)),false) }, start: { _ in true }, stop: { _ in stops += 1 }))
        let firstLocation = UUID(), secondLocation = UUID()
        var rootLease: FlowFolderAccess.Lease? = try access.grant(folder: root.deletingLastPathComponent(), rootFilename: "Main.kwipflow", locationID: firstLocation).lease
        var secondLease: FlowFolderAccess.Lease? = try access.grant(folder: copyFolder, rootFilename: "Main.kwipflow", locationID: secondLocation).lease
        let registry = FlowCanvasWindowRegistry()
        let first = try registry.open(firstURL, present: false, accessLease: rootLease)
        let second = try registry.open(secondURL, present: false, accessLease: secondLease)
        XCTAssertFalse(first === second)
        XCTAssertEqual(first.documentOwner.document.id, second.documentOwner.document.id)
        XCTAssertTrue(try registry.open(firstURL, present:false, accessLease:rootLease) === first)
        XCTAssertThrowsError(try registry.open(secondURL, present:false, accessLease:rootLease), "A folder grant cannot cover a sibling copy")
        rootLease = nil; secondLease = nil
        XCTAssertEqual(stops, 0)
        XCTAssertEqual(first.accessLease?.locationID, firstLocation)
        XCTAssertTrue(first.documentOwner.edit { document in
            if case .modelBlock(var block) = document.payload { block.title = "Independent edit"; document.payload = .modelBlock(block) }
        })
        XCTAssertTrue(registry.save(window:first.window))
        XCTAssertNotEqual(first.documentOwner.document, second.documentOwner.document)
        first.window?.close()
        XCTAssertEqual(stops, 1, "Closing must release grant even while controller remains retained")
        XCTAssertNil(first.accessLease)
        second.window?.close()
        XCTAssertEqual(stops, 2)
    }
    func testManagedOpenCanRetainGrantOnSameFileWithoutRelaxingManagedUUIDCollision() throws {
        _ = NSApplication.shared
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        defer { try? FileManager.default.removeItem(at:directory) }
        let plan = try FlowDocumentMigration.plan(WorkflowExamples.preparedSeparableConv2dClassifier())
        let root = try FlowDocumentStore.commitMigration(plan,to:directory.appendingPathComponent("Project"))
        let path = try XCTUnwrap(plan.root.dependencies.first?.relativePath)
        let childURL = root.deletingLastPathComponent().appendingPathComponent(path)
        let duplicate = childURL.deletingLastPathComponent().appendingPathComponent("Duplicate.kwipflow")
        try FileManager.default.copyItem(at:childURL,to:duplicate)
        let registry = FlowCanvasWindowRegistry()
        let original = try registry.open(childURL,present:false)
        XCTAssertThrowsError(try registry.open(duplicate,present:false))
        let access = FlowFolderAccess(hooks: .init(createBookmark: { Data($0.path.utf8) }, resolveBookmark: { (URL(fileURLWithPath:String(decoding:$0,as:UTF8.self)),false) },start:{_ in true},stop:{_ in}))
        let grant = try access.grant(folder:root.deletingLastPathComponent(),rootFilename:"Main.kwipflow",locationID:UUID())
        XCTAssertTrue(try registry.open(childURL,present:false,accessLease:grant.lease) === original)
        XCTAssertTrue(original.accessLease === grant.lease)
        XCTAssertThrowsError(try registry.open(duplicate,present:false,accessLease:grant.lease))
        original.window?.close()
    }
}
