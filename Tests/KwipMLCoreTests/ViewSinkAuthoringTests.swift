import XCTest
@testable import KwipMLCore

final class ViewSinkAuthoringTests: XCTestCase {
    private let source=WorkflowNodeAddress(domain:.data,localID:"source")
    private let view=WorkflowNodeAddress(domain:.data,localID:"view")
    private let cast=WorkflowNodeAddress(domain:.data,localID:"cast")
    private func endpoint(_ address:WorkflowNodeAddress)->WorkflowEndpoint {.init(address:address,port:"table")}
    private func registry()throws->DataFlowOperatorRegistry {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try .load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
    }
    private func document(legacy:Bool=false)->WorkflowDocument {
        .init(projectID:"B86163F6-F453-4AB8-A2B3-A164A070B478",nodes:[
            .init(address:source,payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:[:])),
            .init(address:view,payload:.dataOperator(operatorID:"kwip.data.view",config:[:])),
            .init(address:cast,payload:.dataOperator(operatorID:"kwip.data.columns",config:["rows":.array([.object(["source":.string("value"),"included":true,"outputName":.string("value"),"targetType":.string("int64")])]),"newColumns":.string("keep")]))],
            edges:[.init(id:"input",from:endpoint(source),to:endpoint(view),kind:.table)] + (legacy ? [.init(id:"legacy",from:endpoint(view),to:endpoint(cast),kind:.table)] : []))
    }
    private func apply(_ command:WorkflowCommand,to document:WorkflowDocument)throws->WorkflowEditResult {
        try WorkflowEditing.apply(command,to:document,modelRegistry:.builtIn(),dataRegistry:registry())
    }
    func testViewSinkSeparatesAuthoringPortsFromRetainedExecutionContract()throws {
        let registry=try registry(),definition=try XCTUnwrap(registry.operatorDefinition(id:"kwip.data.view"))
        XCTAssertEqual(definition.canvasRole,"sink");XCTAssertTrue(definition.isCanvasSink)
        XCTAssertTrue(definition.authoringOutputs.isEmpty)
        var historical=definition;historical.canvasRole=nil
        XCTAssertTrue(historical.isCanvasSink);XCTAssertTrue(historical.authoringOutputs.isEmpty)
        XCTAssertEqual(definition.outputs.map(\.id),["table"])
        let document=document(),node=document.nodes[1]
        XCTAssertEqual(try WorkflowPortContracts.resolve(node,document:document,modelRegistry:.builtIn(),dataRegistry:registry).outputs.map(\.id),["table"])
        XCTAssertTrue(try WorkflowPortContracts.resolveForAuthoring(node,document:document,modelRegistry:.builtIn(),dataRegistry:registry).outputs.isEmpty)
    }
    func testCanvasHidesUnusedViewOutputAndRetainsLegacyWireContract() throws {
        let fresh = try WorkflowCanvasProjector.project(document(), modelRegistry: .builtIn(), dataRegistry: registry())
        XCTAssertFalse(fresh.ports.contains { $0.endpoint.address == view && $0.direction == .output })
        let legacy = try WorkflowCanvasProjector.project(document(legacy: true), modelRegistry: .builtIn(), dataRegistry: registry())
        let port = try XCTUnwrap(legacy.ports.first { $0.endpoint.address == view && $0.direction == .output })
        XCTAssertEqual(port.semanticKind, .table)
        XCTAssertNil(port.unresolvedReason)
    }
    func testNewViewOutboundWireIsRefusedButOrdinaryProducerStillConnects()throws {
        XCTAssertThrowsError(try apply(.connect(from:endpoint(view),to:endpoint(cast)),to:document()))
        let ordinary=try apply(.connect(from:endpoint(source),to:endpoint(cast)),to:document())
        XCTAssertEqual(ordinary.document.edges.count,2)
    }
    func testLegacyChainStillCompilesRoundTripsAndCanMove()throws {
        let original=document(legacy:true),encoded=try WorkflowCodec.encode(original)
        let decoded=try WorkflowCodec.decode(encoded)
        XCTAssertEqual(decoded,original)
        let plan=try WorkflowDataPreparationPlan.compile(decoded,registry:registry())
        XCTAssertEqual(plan.edges.count,2)
        let moved=try apply(.move(addresses:[view],offset:.init(x:12,y:3)),to:decoded)
        XCTAssertEqual(moved.document.edges,original.edges)
        let unchanged=try apply(.reconnect(edgeID:"legacy",from:endpoint(view),to:endpoint(cast)),to:decoded)
        XCTAssertEqual(unchanged.document,decoded)
    }
    func testLegacyChainDuplicatePreservesItsExistingConnections()throws {
        let original=document(legacy:true)
        let duplicated=try apply(.duplicate(addresses:[view,cast],offset:.init(x:10,y:10)),to:original)
        XCTAssertEqual(duplicated.document.nodes.count,5)
        XCTAssertEqual(duplicated.document.edges.count,3)
        XCTAssertEqual(duplicated.copies.edges.count,1)
    }
}
