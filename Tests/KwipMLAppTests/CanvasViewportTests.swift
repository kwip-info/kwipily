import XCTest
import KwipMLCore
@testable import KwipMLApp

final class CanvasViewportTests: XCTestCase {
    func testOnePercentRestoreKeepsPresentationAndClampsOnlyUnsupportedZoom() {
        var saved=CanvasViewState(zoom:0.01)
        saved.presentation=CanvasPresentationSettings(flowDirection:.topToBottom,nodeDetail:.large)
        let restored=CanvasViewportProjection.restored(saved,nodes:[],edges:[])
        XCTAssertEqual(restored.zoom,0.01)
        XCTAssertEqual(restored.presentation,saved.presentation)
        saved.zoom=0.009
        XCTAssertEqual(CanvasViewportProjection.restored(saved,nodes:[],edges:[]).zoom,0.01)
    }
    func testRestoredStateClampsGeometryAndDropsUnavailableSelection() {
        let value = CanvasViewportProjection.restored(CanvasViewState(zoom: 3, scrollX: -10, scrollY: 90, selectedNodeIDs: ["a", "gone"], selectedEdgeIDs: ["wire", "old"]), nodes: ["a"], edges: ["wire"])
        XCTAssertEqual(value.zoom, 1.5)
        XCTAssertEqual(value.scrollX, -10)
        XCTAssertEqual(value.scrollY, 90)
        XCTAssertEqual(value.selectedNodeIDs, ["a"])
        XCTAssertEqual(value.selectedEdgeIDs, ["wire"])
    }
    func testNonFiniteValuesNeverReachNativeScrollBounds() {
        let value = CanvasViewportProjection.restored(CanvasViewState(zoom: .nan, scrollX: .infinity, scrollY: -.infinity), nodes: [], edges: [])
        XCTAssertEqual(value.zoom, 0.65)
        XCTAssertEqual(value.scrollX, 0)
        XCTAssertEqual(value.scrollY, 0)
    }
    @MainActor func testModelAndDataViewportCaptureAreSeparateFromRestoreRevision() {
        let workspace = WorkspaceStore(rootOverride: URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString), detectCapabilities: false)
        let data = DataFlowWorkspaceStore(workspace: workspace)
        let originalRevision = workspace.canvasViewRevision
        workspace.canvasViewState = CanvasViewState(zoom: 1.2, scrollX: 90, scrollY: 40)
        data.canvasViewState = CanvasViewState(zoom: 0.8, scrollX: 20, scrollY: 60)
        XCTAssertEqual(workspace.modelCanvasViewState.zoom, 1.2)
        XCTAssertEqual(workspace.dataCanvasViewState.zoom, 0.8)
        XCTAssertEqual(workspace.canvasViewRevision, originalRevision)
        workspace.installWorkspaceView(WorkspaceViewState(modelCanvas: CanvasViewState(zoom: 0.9), dataCanvas: CanvasViewState(zoom: 1.1)))
        XCTAssertNotEqual(workspace.canvasViewRevision, originalRevision)
        XCTAssertEqual(data.canvasViewState.zoom, 1.1)
        XCTAssertEqual(workspace.canvasViewState.zoom, 0.9)
    }

    @MainActor func testDisappearingDataCanvasCannotOverwriteAnotherProjectsViewport() {
        let workspace = WorkspaceStore(rootOverride: URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString), detectCapabilities: false)
        workspace.activeProjectID = UUID().uuidString
        let oldEditor = DataFlowWorkspaceStore(workspace: workspace)
        workspace.activeProjectID = UUID().uuidString
        workspace.dataCanvasViewState = CanvasViewState(zoom: 1.1)
        oldEditor.canvasViewState = CanvasViewState(zoom: 0.4)
        XCTAssertEqual(workspace.dataCanvasViewState.zoom, 1.1)
    }

}
