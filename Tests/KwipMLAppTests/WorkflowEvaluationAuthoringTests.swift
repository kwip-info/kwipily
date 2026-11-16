import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowEvaluationAuthoringTests:XCTestCase {
    @MainActor func testInsertionMigrationPersistsAsOneUndoableTransaction()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root)
        let original=WorkflowDocument(projectID:UUID().uuidString,revision:10,name:"My flow")
        _=try library.saveSource(.workflow(original),id:original.projectID)
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let store=try WorkflowAuthoringStore(library:library,projectID:original.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        let node=WorkflowNode(address:.init(domain:.control,localID:"frozen"),name:"Selected checkpoint",payload:.frozenModel(config:[:]))
        XCTAssertNotNil(store.apply(.insert(node),lease:store.lease()))
        XCTAssertEqual(store.document.schemaVersion,5)
        XCTAssertEqual(store.document.revision,11)
        XCTAssertEqual(store.document.nodes,[node])
        XCTAssertEqual(try library.loadSource(id:store.projectID),.workflow(store.document))
        XCTAssertTrue(store.undo(lease:store.lease()))
        XCTAssertEqual(store.document.schemaVersion,1)
        XCTAssertEqual(store.document.nodes,original.nodes)
        XCTAssertEqual(store.document.revision,12)
        XCTAssertFalse(store.canUndo)
        XCTAssertEqual(try library.loadSource(id:store.projectID),.workflow(store.document))
        XCTAssertTrue(store.redo(lease:store.lease()))
        XCTAssertEqual(store.document.schemaVersion,5)
        XCTAssertEqual(store.document.nodes,[node])
        XCTAssertEqual(store.document.revision,13)
        XCTAssertEqual(try library.loadSource(id:store.projectID),.workflow(store.document))
    }
}
