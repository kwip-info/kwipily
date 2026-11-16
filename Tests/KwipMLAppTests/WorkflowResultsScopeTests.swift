import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowResultsScopeTests: XCTestCase {
    private let data=WorkflowNodeAddress(domain:.data,localID:"same")
    private let model=WorkflowNodeAddress(domain:.model,localID:"same")
    private var document:WorkflowDocument {
        .init(projectID:"scope-project",nodes:[
            .init(address:data,payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:[:])),
            .init(address:model,payload:.modelPrimitive(type:"input",configurationVersion:1,config:[:]))],
            edges:[.init(id:"edge",from:.init(address:data,port:"table"),to:.init(address:model,port:"in"),kind:.table)])
    }
    func testSelectionResolvesOnlyExactSingleOrExplicitGlobalScope() {
        XCTAssertEqual(WorkflowResultsScope.resolve(selection:.init(),document:document),.global)
        for address in [data,model] {
            XCTAssertEqual(WorkflowResultsScope.resolve(selection:.init(nodeAddresses:[address]),document:document),.component(address))
        }
        XCTAssertEqual(WorkflowResultsScope.resolve(selection:.init(edgeIDs:["edge"]),document:document),.connection(document.edges[0]))
        XCTAssertEqual(WorkflowResultsScope.resolve(selection:.init(nodeAddresses:[data,model]),document:document),.multiple)
        XCTAssertEqual(WorkflowResultsScope.resolve(selection:.init(nodeAddresses:[data],edgeIDs:["edge"]),document:document),.multiple)
    }
    func testStaleAndAmbiguousIdentityNeverFallsBackToGlobalOrFirstItem() {
        let missing=WorkflowNodeAddress(domain:.data,localID:"removed")
        for selection in [WorkflowSelection(nodeAddresses:[missing]),.init(nodeAddresses:[data,missing]),.init(edgeIDs:["missing"])] {
            XCTAssertEqual(WorkflowResultsScope.resolve(selection:selection,document:document),.unavailable)
        }
        var broken=document;broken.nodes.removeAll {$0.address == model}
        XCTAssertEqual(WorkflowResultsScope.resolve(selection:.init(edgeIDs:["edge"]),document:broken),.unavailable)
        var duplicate=document;duplicate.nodes.append(duplicate.nodes[0])
        XCTAssertEqual(WorkflowResultsScope.resolve(selection:.init(nodeAddresses:[data]),document:duplicate),.unavailable)
    }
    func testCapturedDependencyMembershipUsesQualifiedAddressesAndCapturedEdges() {
        var captured=document
        let isolated=WorkflowNodeAddress(domain:.control,localID:"unrelated")
        captured.nodes.append(.init(address:isolated,payload:.train(config:[:])))
        XCTAssertEqual(WorkflowResultsScope.capturedDependencies(document:captured,terminal:model),[data,model])
        XCTAssertFalse(WorkflowResultsScope.capturedDependencies(document:captured,terminal:model).contains(isolated))
        var edited=captured;edited.edges=[]
        XCTAssertEqual(WorkflowResultsScope.capturedDependencies(document:edited,terminal:model),[model])
        XCTAssertEqual(WorkflowResultsScope.capturedDependencies(document:captured,terminal:model),[data,model])
        captured.edges.append(.init(id:"cycle",from:.init(address:model,port:"out"),to:.init(address:data,port:"in"),kind:.table))
        XCTAssertTrue(WorkflowResultsScope.capturedDependencies(document:captured,terminal:model).isEmpty)
    }
}
