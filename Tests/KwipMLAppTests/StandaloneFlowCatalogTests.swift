import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class StandaloneFlowCatalogTests: XCTestCase {
    func testStandaloneGrantSurvivesRecreationAndCopiesKeepLocalIdentity() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let plan = try FlowDocumentMigration.plan(WorkflowExamples.preparedSeparableConv2dClassifier())
        let main = try FlowDocumentStore.commitMigration(plan, to: directory.appendingPathComponent("source"))
        let path = try XCTUnwrap(plan.root.dependencies.first?.relativePath)
        let block = main.deletingLastPathComponent().appendingPathComponent(path)
        var denied = false
        let access = FlowFolderAccess(hooks: .init(createBookmark: { Data($0.path.utf8) }, resolveBookmark: { bytes in
            if denied { throw FlowFolderAccess.Failure("Saved folder permission needs renewal") }
            return (URL(fileURLWithPath: String(decoding: bytes, as: UTF8.self)), false)
        }, start: { _ in true }, stop: { _ in }))
        let store = StandaloneFlowCatalog(directory: directory.appendingPathComponent("catalog"), access: access)
        let first = try store.attach(folder: block.deletingLastPathComponent(), file: block)
        let duplicate = block.deletingLastPathComponent().appendingPathComponent("Copy.kwipflow")
        try FileManager.default.copyItem(at: block, to: duplicate)
        let second = try store.attach(folder: duplicate.deletingLastPathComponent(), file: duplicate)
        XCTAssertNotEqual(first.id, second.id); XCTAssertEqual(first.documentID, second.documentID)
        let recreated = StandaloneFlowCatalog(directory: directory.appendingPathComponent("catalog"), access: access)
        XCTAssertEqual(try recreated.open(first.id).record, first)
        denied = true; XCTAssertThrowsError(try recreated.open(first.id))
        XCTAssertEqual(try recreated.list().count, 2, "Denied grants retain explicit retry metadata")
        denied = false
        let renewed = try recreated.attach(folder: block.deletingLastPathComponent(), file: block)
        XCTAssertEqual(renewed.id, first.id)
        XCTAssertEqual(try recreated.open(renewed.id).record.documentID, first.documentID)
    }
}
