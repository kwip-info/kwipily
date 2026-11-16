import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

final class ActivationInsertionTests:XCTestCase {
    @MainActor func testWireInsertionOffersOneActivationWithActualReLUPorts()throws {
        let registry=try ComponentRegistry.builtIn(),probe=ActivationInsertionProbe(registry:registry)
        probe.eligible=registry.components.filter {$0.inputs.count==1 && $0.outputs.count==1}.map(\.id)
        let offered=probe.canvasInsertionComponents(edgeID:"wire")
        let activation=try XCTUnwrap(offered.first {$0.title=="Activation"})
        let original=try XCTUnwrap(registry.component(type:"relu"))
        XCTAssertEqual(offered.filter {ActivationConfiguration.componentIDs.contains($0.id)}.count,1)
        XCTAssertEqual(activation.id,"relu");XCTAssertEqual(activation.inputs,original.inputs)
        XCTAssertEqual(activation.outputs,original.outputs);XCTAssertEqual(activation.fields,original.fields)
        XCTAssertEqual(offered.first {$0.id=="dropout"},registry.component(type:"dropout"))
        XCTAssertFalse(offered.contains {$0.id==ActivationPalette.type})
    }
    @MainActor func testRestrictedEligibilitySelectsOnlyAnAllowedActualFunction()throws {
        let registry=try ComponentRegistry.builtIn(),probe=ActivationInsertionProbe(registry:registry)
        probe.eligible=["sigmoid","gelu","linear","missing"]
        let offered=probe.canvasInsertionComponents(edgeID:"wire")
        let activation=try XCTUnwrap(offered.first {$0.title=="Activation"})
        XCTAssertEqual(activation.id,"gelu","First allowed function follows the shared function ordering")
        XCTAssertEqual(activation.fields,registry.component(type:"gelu")?.fields)
        XCTAssertEqual(Set(offered.map(\.id)),["gelu","linear"])
        probe.eligible=["sigmoid"]
        XCTAssertEqual(probe.canvasInsertionComponents(edgeID:"wire").map(\.id),["sigmoid"])
        probe.eligible=["linear"]
        XCTAssertEqual(probe.canvasInsertionComponents(edgeID:"wire"),[try XCTUnwrap(registry.component(type:"linear"))])
        probe.eligible=[]
        XCTAssertTrue(probe.canvasInsertionComponents(edgeID:"wire").isEmpty)
    }
}

@MainActor private final class ActivationInsertionProbe:ObservableObject,CanvasWorkspace {
    let registry:ComponentRegistry
    var eligible:[String]=[]
    init(registry:ComponentRegistry) {self.registry=registry}
    var graph=GraphDocument()
    var canvasViewState=CanvasViewState()
    var canvasViewRevision=UUID()
    var validation=ValidationResult()
    var isBusy=false
    var selectedNodeIDs:Set<String>=[]
    var selectedEdgeID:String?
    var pendingSourceID:String?
    var statusMessage=""
    var previewWeightLabel=""
    var canUndo=false
    var canRedo=false
    var historyContextID="activation-insertion-test"
    var historyContextIsActive=true
    func canvasInsertionTypes(edgeID:String)->[String] {eligible}
    func undo() {}
    func redo() {}
    func previewStatus(for id:String)->String {""}
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
