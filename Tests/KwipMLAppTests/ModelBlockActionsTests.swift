import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ModelBlockActionsTests:XCTestCase {
    @MainActor func testFirstBlockPromotesPreservesPrimitivesAndReopensSource() throws {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let workspace=WorkspaceStore(rootOverride:directory,detectCapabilities:false)
        XCTAssertTrue(workspace.beginProject(ExampleFlows.parallelClassifier))
        let projectID=try XCTUnwrap(workspace.activeProjectID)
        let original=workspace.graph
        let blockID=try XCTUnwrap(workspace.addModelBlock(x:230,y:100))
        let owner=try XCTUnwrap(workspace.modelAuthoring)
        guard case .blockModel(let model)=owner.document else{return XCTFail("Expected source envelope")}
        XCTAssertEqual(model.source.root.nodes,original.nodes)
        XCTAssertEqual(model.source.root.edges,original.edges)
        XCTAssertEqual(model.source.root.instances.first?.id,blockID)
        XCTAssertEqual(workspace.modelCanvas?.graph.nodes.count,original.nodes.count+1)
        let canvas = try XCTUnwrap(workspace.modelCanvas)
        let restored = CanvasViewportProjection.restored(canvas.canvasViewState,
            nodes: Set(canvas.graph.nodes.map(\.id)), edges: Set(canvas.graph.edges.map(\.id)))
        canvas.clearSelection()
        for id in restored.selectedNodeIDs { canvas.selectNode(id, extending: true) }
        XCTAssertEqual(canvas.selectedNodeID, blockID, "Initial shared-canvas restoration must retain the newly inserted Block inspector")
        XCTAssertEqual(workspace.modelCanvasViewState.selectedNodeIDs, [blockID])
        XCTAssertFalse(workspace.validation.isValid)
        XCTAssertTrue(workspace.saveManagedProject())
        XCTAssertTrue(workspace.beginOwnedProject(id:projectID))
        XCTAssertEqual(workspace.modelAuthoring?.document,owner.document)
        XCTAssertTrue(workspace.modelCanvas?.canOpenBlock(blockID) == true)
    }
}
