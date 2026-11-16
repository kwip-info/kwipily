import XCTest
import SwiftUI
import KwipMLCore
@testable import KwipMLApp

final class CanvasWorkspaceSeamTests: XCTestCase {
    @MainActor func testDefaultSurfaceKeepsExistingIconsAndEligibleRegistryTypes() throws {
        let root = URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let store = WorkspaceStore(rootOverride:root,detectCapabilities:false)
        store.graph.nodes = [.init(id:"layer",type:"linear")]
        XCTAssertEqual(store.canvasIconType(nodeID:"layer"),"linear")
        XCTAssertEqual(store.canvasCatalog,store.registry)
        XCTAssertEqual(store.canvasInsertionComponents(edgeID:"edge").map(\.id),["conv2d","avg_pool1d","relu","dropout","layer_norm","linear","reshape","transpose","embedding","unfold","fold"])
    }
    @MainActor func testDropUsesCatalogCardSizeButExistingCanvasCoordinateProjection() {
        let store = MixedCanvasProbe()
        store.canvasCatalog.components[0].inputs = (0..<8).map { "input-\($0)" }
        let point = CGPoint(x:600,y:600)
        let result = GraphCanvasView<MixedCanvasProbe>.dropOrigin(point,type:"create-linear",registry:store.registry,catalog:store.canvasCatalog,presentation:.init())
        // Eight inputs plus one output occupy a 256 × 388 table. The
        // one-input/one-output scene uses 256/150 × 150/84 spacing.
        // The dense creation catalog must not define the scene projection.
        XCTAssertEqual(result.x,(600-128)*150/256,accuracy:0.001)
        XCTAssertEqual(result.y,(600-194)*84/150,accuracy:0.001)
    }
    @MainActor func testSyntheticNodeRegistryNeverBecomesCreationCatalogAndInsertionCanBeDisabled() {
        let store = MixedCanvasProbe()
        func inspect<W:CanvasWorkspace>(_ workspace:W)->(String,[String]) {
            (workspace.canvasIconType(nodeID:"display-node"),workspace.canvasInsertionComponents(edgeID:"edge").map(\.id))
        }
        XCTAssertEqual(inspect(store).0,"linear")
        XCTAssertEqual(inspect(store).1,["create-linear"])
        XCTAssertNil(store.registry.component(type:"create-linear"))
        XCTAssertNil(store.canvasCatalog.component(type:"synthetic-per-node"))
        store.insertionEnabled = false
        XCTAssertTrue(inspect(store).1.isEmpty)
        // Unknown IDs supplied by an adapter must not become executable UI choices.
        store.insertionEnabled = true; store.eligible = ["synthetic-per-node","missing"]
        XCTAssertTrue(inspect(store).1.isEmpty)
    }
}

@MainActor private final class MixedCanvasProbe: ObservableObject, CanvasWorkspace {
    var graph = GraphDocument(nodes:[.init(id:"display-node",type:"synthetic-per-node")])
    var registry = ComponentRegistry(components:[.init(id:"synthetic-per-node",inputs:["in"],outputs:["out"])])
    var canvasCatalog = ComponentRegistry(components:[.init(id:"create-linear",inputs:["in"],outputs:["out"])])
    var insertionEnabled = true
    var eligible = ["create-linear"]
    func canvasIconType(nodeID:String)->String {"linear"}
    func canvasInsertionTypes(edgeID:String)->[String] {insertionEnabled ? eligible : []}
    var canvasViewState = CanvasViewState()
    var canvasViewRevision = UUID()
    var validation = ValidationResult()
    var isBusy = false
    var selectedNodeIDs: Set<String> = []
    var selectedEdgeID: String?
    var pendingSourceID: String?

    var statusMessage = ""
    var previewWeightLabel = ""
    var canUndo = false
    var canRedo = false
    var historyContextID = "probe"
    var historyContextIsActive = true
    func undo() {}
    func redo() {}
    func previewStatus(for id:String)->String {"Not evaluated"}
    func canvasSummary(_ node:GraphNode)->String {""}
    func canvasPortDescription(nodeID:String,port:String)->String {""}
    func canConnect(source:String,sourcePort:String,target:GraphNode,port:String)->Bool {false}
    func selectNode(_ id:String,extending:Bool) {}
    func selectEdge(_ id:String) {}
    func clearSelection() {}
    func moveNodes(ids:Set<String>,dx:Double,dy:Double) {}
    func duplicateSelection() {}
    func deleteSelection() {}
    func addNode(type:String,x:Double,y:Double) {}
    func connect(source:String,sourcePort:String,target:String,targetPort:String) {}
    func reconnect(edgeID:String,source:String,sourcePort:String,target:String,targetPort:String) {}
    func insertNode(type:String,edgeID:String,x:Double,y:Double) {}
}
