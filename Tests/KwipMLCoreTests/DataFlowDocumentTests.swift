import XCTest
@testable import KwipMLCore
final class DataFlowDocumentTests:XCTestCase {
    func testTypedConfigurationRoundTripAndPrivateEnvelope() throws {
        let config:[String:JSONValue]=["label":.string("User's label"),"flag":.bool(true),"large":.integer(9007199254740993),"ratio":.number(0.25),"list":.array([.null,.object(["x":.integer(2)])])]
        let graph=DataFlowDocument(nodes:[DataFlowNode(id:"source",operatorID:"kwip.data.sourceCSV",config:config)])
        let project=DataFlowProject(name:"Named flow",graph:graph,editor:["source":DataFlowEditorNode(name:"My values",x:-12,y:42)],sourceBindings:["source":"/private/my.csv"])
        XCTAssertEqual(try DataFlowCodec.decodeProject(DataFlowCodec.encodeProject(project)),project)
        let portable=try DataFlowCodec.encodeGraph(graph)
        XCTAssertEqual(try DataFlowCodec.decodeGraph(portable),graph)
        let text=String(decoding:portable,as:UTF8.self)
        XCTAssertFalse(text.contains("private"));XCTAssertFalse(text.contains("My values"));XCTAssertTrue(text.contains("9007199254740993"))
    }
    func testStrictGraphRejectsUnknownFieldsSchemaAndBadEndpoints() throws {
        for raw in [#"{"schemaVersion":true,"nodes":[],"edges":[]}"#,#"{"schemaVersion":1,"nodes":[],"edges":[],"sourceBindings":{}}"#,#"{"schemaVersion":2,"nodes":[],"edges":[]}"#,#"{"schemaVersion":1,"nodes":[{"id":"a","operator":"x","config":{},"x":1}],"edges":[]}"#] {XCTAssertThrowsError(try DataFlowCodec.decodeGraph(Data(raw.utf8)))}
        let node=DataFlowNode(id:"a",operatorID:"x")
        XCTAssertThrowsError(try DataFlowCodec.encodeGraph(DataFlowDocument(nodes:[node,node])))
        XCTAssertThrowsError(try DataFlowCodec.encodeGraph(DataFlowDocument(nodes:[node],edges:[DataFlowEdge(from:.init(node:"a",port:"table"),to:.init(node:"missing",port:"table"))])))
    }
    func testBoundsNonfiniteAndStaleSidecarRejected() throws {
        var graph=DataFlowDocument(nodes:[DataFlowNode(id:"a",operatorID:"x",config:["bad":.number(.infinity)])])
        XCTAssertThrowsError(try DataFlowCodec.encodeGraph(graph))
        var nested=JSONValue.null;for _ in 0..<34 {nested = .array([nested])};graph.nodes[0].config=["nested":nested]
        XCTAssertThrowsError(try DataFlowCodec.encodeGraph(graph))
        graph.nodes[0].config=[:]
        XCTAssertThrowsError(try DataFlowCodec.encodeProject(DataFlowProject(name:"x",graph:graph,editor:["a":.init(x:.nan)])))
        XCTAssertThrowsError(try DataFlowCodec.encodeProject(DataFlowProject(name:"x",graph:graph,sourceBindings:["missing":"/a"])))
        XCTAssertThrowsError(try DataFlowCodec.decodeGraph(Data(repeating:32,count:2_000_001)))
    }
    func testDeepJSONRejectedByDecoder() throws {
        let raw=String(repeating:"[",count:2000)+"0"+String(repeating:"]",count:2000)
        XCTAssertThrowsError(try DataFlowCodec.decodeGraph(Data(raw.utf8)))
    }
    func testWorkerExampleAndRegistryDecodeWithoutHandwrittenSchemas() throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let graph=try DataFlowCodec.decodeGraph(Data(contentsOf:root.appendingPathComponent("examples/dataflow/scaler.json")))
        let registry=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        let ids=registry.operators.map(\.id)
        XCTAssertEqual(Set(ids).count,ids.count,"Operator IDs must be unique")
        let required:Set<String>=["kwip.data.sourceCSV","kwip.data.columns","kwip.data.partition","kwip.data.preprocess"]
        XCTAssertTrue(required.isSubset(of:Set(ids)),"Current source, column, three-population partition and training-only preparation contracts must remain available")
        XCTAssertTrue(RetiredDataOperatorMigration.retired.isDisjoint(with:Set(ids)))
        for node in graph.nodes {XCTAssertNotNil(registry.operatorDefinition(id:node.operatorID))}
        XCTAssertEqual(registry.operatorDefinition(id:"kwip.data.partition")?.outputs.map(\.id),["training","validation","test","exclusions"])
    }
}
