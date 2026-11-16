import XCTest
@testable import KwipMLApp
import KwipMLCore
final class WorkspaceArrangementActionsTests:XCTestCase {
    @MainActor func testArrangementDoesNotMutateComputeAndFiltersStaleSelection()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer{try? FileManager.default.removeItem(at:root)}
        let store=WorkspaceStore(rootOverride:root,detectCapabilities:false);store.beginProject(ExampleFlows.parallelClassifier)
        let signature=store.semanticSignature(store.graph)
        store.showInspector=false;store.selectedNodeIDs=["input","absent"];store.saveWorkspaceArrangement(name:"Focused",section:"Code")
        store.showInspector=true;store.applyWorkspaceArrangement(store.workspaceArrangements[0])
        XCTAssertFalse(store.showInspector);XCTAssertEqual(store.activeSection,"Code");XCTAssertEqual(store.selectedNodeIDs,["input"]);XCTAssertEqual(store.semanticSignature(store.graph),signature)
    }
    @MainActor func testPendingViewSaveCannotCrossProjectSwitch()async throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer{try? FileManager.default.removeItem(at:root)}
        let store=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        store.beginProject(ExampleFlows.parallelClassifier);let first=try XCTUnwrap(store.activeProjectID)
        store.showLibrary=false;store.activeSection="Data"
        store.beginProject(GraphDocument(name:"Second"));let second=try XCTUnwrap(store.activeProjectID)
        store.showLibrary=true;store.activeSection="Build"
        try await Task.sleep(nanoseconds:500_000_000)
        XCTAssertFalse(try XCTUnwrap(store.arrangementLibrary.projectState(projectID:first)).showLibrary)
        XCTAssertEqual(try store.arrangementLibrary.projectState(projectID:first)?.activeSection,"Data")
        XCTAssertTrue(try XCTUnwrap(store.arrangementLibrary.projectState(projectID:second)).showLibrary)
        store.beginProject(try store.projectLibrary.load(id:first),id:first)
        XCTAssertEqual(store.activeSection,"Data");XCTAssertFalse(store.showLibrary)
    }
}
