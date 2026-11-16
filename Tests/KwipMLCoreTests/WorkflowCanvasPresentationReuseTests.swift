import XCTest
@testable import KwipMLCore

final class WorkflowCanvasPresentationReuseTests: XCTestCase {
    private func fixture() throws -> (WorkflowDocument,ComponentRegistry,DataFlowOperatorRegistry) {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let models=try ComponentRegistry.builtIn(),data=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        let a=WorkflowNodeAddress(domain:.model,localID:"a"),b=WorkflowNodeAddress(domain:.model,localID:"b")
        let document=WorkflowDocument(projectID:UUID().uuidString,nodes:[
            .init(address:a,name:"Input",x:-20,y:30,payload:.modelPrimitive(type:"input",configurationVersion:1,config:["features":2])),
            .init(address:b,name:"Layer",x:300,y:40,payload:.modelPrimitive(type:"linear",configurationVersion:2,config:["in_features":2,"out_features":2,"bias":true]))],
            edges:[.init(id:"repair",from:.init(address:a,port:"saved-out"),to:.init(address:b,port:"in"),kind:.tensor)])
        return (document,models,data)
    }
    func testPresentationReuseEqualsFreshProjectionAndRetainsRepairContracts() throws {
        let (original,models,data)=try fixture()
        let first=try WorkflowCanvasProjector.project(original,modelRegistry:models,dataRegistry:data)
        var next=original;next.revision=12;next.name="Renamed project";next.nodes[0].name="Renamed input";next.nodes[0].x=640;next.nodes[1].y = -200
        let updated=try XCTUnwrap(first.updatingPresentation(next))
        let fresh=try WorkflowCanvasProjector.project(next,modelRegistry:models,dataRegistry:data)
        XCTAssertEqual(updated.graph,fresh.graph);XCTAssertEqual(updated.registry,fresh.registry)
        XCTAssertEqual(updated.addressByDisplayID,fresh.addressByDisplayID);XCTAssertEqual(updated.displayIDByAddress,fresh.displayIDByAddress)
        XCTAssertEqual(updated.edgeIDByDisplayID,fresh.edgeIDByDisplayID);XCTAssertEqual(updated.displayIDByEdgeID,fresh.displayIDByEdgeID)
        XCTAssertEqual(updated.typeByDisplayID,fresh.typeByDisplayID);XCTAssertEqual(updated.ports,fresh.ports);XCTAssertEqual(updated.issues,fresh.issues)
        for port in fresh.ports {XCTAssertEqual(updated.portDescriptor(address:port.endpoint.address,port:port.endpoint.port,direction:port.direction),port)}
        XCTAssertEqual(try updated.updatingPresentation(original)?.graph,first.graph,"Undo must use the new source identity, preserving all original metadata")
        XCTAssertEqual(first.graph.nodes[0].name,"Input","Previously published snapshots stay immutable")
    }
    func testSemanticOrOwnershipChangesNeverReuseProjection() throws {
        let (original,models,data)=try fixture()
        let first=try WorkflowCanvasProjector.project(original,modelRegistry:models,dataRegistry:data)
        var changes:[WorkflowDocument]=[]
        var changed=original;changed.projectID=UUID().uuidString;changes.append(changed)
        changed=original;changed.schemaVersion=2;changes.append(changed)
        changed=original;changed.nodes[1].payload = .modelPrimitive(type:"linear",configurationVersion:2,config:["in_features":4,"out_features":2]);changes.append(changed)
        changed=original;changed.nodes.reverse();changes.append(changed)
        changed=original;changed.edges[0].from.port="out";changes.append(changed)
        changed=original;changed.guidedBuild = .init(courseID:"course",version:1,completedSteps:1);changes.append(changed)
        for next in changes {XCTAssertNil(try first.updatingPresentation(next))}
    }
    func testInvalidPresentationStillUsesCodecBounds() throws {
        let (original,models,data)=try fixture()
        let first=try WorkflowCanvasProjector.project(original,modelRegistry:models,dataRegistry:data)
        var next=original;next.nodes[0].name=String(repeating:"x",count:501)
        XCTAssertThrowsError(try first.updatingPresentation(next))
        next=original;next.nodes[0].x = .infinity
        XCTAssertThrowsError(try first.updatingPresentation(next))
    }
}
