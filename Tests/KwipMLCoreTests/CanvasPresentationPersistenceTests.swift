import XCTest
@testable import KwipMLCore

final class CanvasPresentationPersistenceTests:XCTestCase {
    func testLegacyStateDefaultsAndExplicitPresentationRoundTrip() throws {
        let legacy=Data(#"{"zoom":0.7,"scrollX":10,"scrollY":20,"selectedNodeIDs":[],"selectedEdgeIDs":[]}"#.utf8)
        XCTAssertEqual(try JSONDecoder().decode(CanvasViewState.self,from:legacy).presentation,CanvasPresentationSettings())
        var state=WorkspaceViewState();state.modelCanvas.presentation=CanvasPresentationSettings(flowDirection:.topToBottom,nodeDetail:.large)
        state.dataCanvas.presentation.nodeDetail = .small
        state.compositeCanvasStates=["root":state.modelCanvas,"composite:parallel-features@1.0.0":state.dataCanvas]
        XCTAssertEqual(try JSONDecoder().decode(WorkspaceViewState.self,from:JSONEncoder().encode(state)),state)
        let dir=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer { try? FileManager.default.removeItem(at:dir) }
        let library=WorkspaceArrangementLibrary(directory:dir),id=UUID().uuidString
        try library.saveProjectState(state,projectID:id);XCTAssertEqual(try library.projectState(projectID:id),state)
        XCTAssertEqual(try library.save(name:"Vertical detail",state:state).state,state)
    }
    func testOverviewVisibilityDefaultsForOldFilesAndPersistsIndependently() throws {
        let legacy=Data(#"{"zoom":0.7,"scrollX":10,"scrollY":20,"selectedNodeIDs":[],"selectedEdgeIDs":[]}"#.utf8)
        XCTAssertTrue(try JSONDecoder().decode(CanvasViewState.self,from:legacy).overviewVisible)
        var state=WorkspaceViewState()
        state.modelCanvas.overviewVisible=false
        let reopened=try JSONDecoder().decode(WorkspaceViewState.self,from:JSONEncoder().encode(state))
        XCTAssertFalse(reopened.modelCanvas.overviewVisible)
        XCTAssertTrue(reopened.dataCanvas.overviewVisible)
        XCTAssertEqual(reopened.modelCanvas.presentation,CanvasPresentationSettings())
        XCTAssertThrowsError(try JSONDecoder().decode(CanvasViewState.self,from:Data(#"{"zoom":0.7,"scrollX":10,"scrollY":20,"selectedNodeIDs":[],"selectedEdgeIDs":[],"overviewVisible":"false"}"#.utf8)))
    }
    func testCompositeKeysVersionsAndBoundsAreStrict() throws {
        var state=WorkspaceViewState()
        for key in ["", "definition", "composite:x@1", "composite:x@-1.0.0", "composite:x@01.0.0",String(repeating:"a",count:1025)] {
            state.compositeCanvasStates=[key:CanvasViewState()];XCTAssertThrowsError(try state.validate(),key)
        }
        state.compositeCanvasStates=Dictionary(uniqueKeysWithValues:(0..<66).map { ("composite:id\($0)@1.0.0",CanvasViewState()) })
        XCTAssertThrowsError(try state.validate())
        state.compositeCanvasStates=["root":CanvasViewState(zoom:.infinity)];XCTAssertThrowsError(try state.validate())
    }
}
