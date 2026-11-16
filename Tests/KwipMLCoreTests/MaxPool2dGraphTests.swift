import XCTest
@testable import KwipMLCore

final class MaxPool2dGraphTests:XCTestCase {
    func testTypedSpatialOutputsNullableStrideAndBridge() throws {
        let registry=try ComponentRegistry.builtIn()
        var graph=ExampleFlows.maxPool1dClassifier()
        graph.nodes[graph.nodes.firstIndex{$0.id=="projection"}!].config["out_features"]=40
        graph.nodes[graph.nodes.firstIndex{$0.id=="reshape"}!].config["shape"] = .array([2,4,5])
        let index=graph.nodes.firstIndex{$0.id=="pool"}!
        graph.nodes[index].type="max_pool2d"
        graph.nodes[index].config=["kernel_size":.array([2,3]),"stride":.null,"padding":.array([0,0]),"dilation":.array([1,1]),"ceil_mode":false,"return_indices":true]
        graph.nodes[graph.nodes.firstIndex{$0.id=="classifier"}!].config["in_features"]=4
        let result=GraphValidator.validate(graph,registry:registry)
        XCTAssertTrue(result.isValid,"\(result.issues)")
        XCTAssertEqual(result.outputTensorContracts[.init("pool","out")],try .init(sample:[2,2,1],dtype:.float32))
        XCTAssertEqual(result.outputTensorContracts[.init("pool","indices")],try .init(sample:[2,2,1],dtype:.int64))
        let capability=try XCTUnwrap(BuiltInCapabilityBridge.declarations(registry:registry).first{$0.id=="max_pool2d"})
        XCTAssertEqual(capability.ports.first{$0.id=="indices"}?.schema.tensor?.dtype,.int64)
        graph.nodes[index].config["return_indices"]=false
        XCTAssertEqual(MaxPool2dComponent.definition.configuredOutputs(graph.nodes[index].config),["out"])
        graph.edges[graph.edges.firstIndex{$0.source=="pool"}!].sourcePort="indices"
        XCTAssertFalse(GraphValidator.validate(graph,registry:registry).isValid)
    }
    func testOptionalPairedFieldEnableAndValidation() throws {
        let field=try XCTUnwrap(MaxPool2dComponent.definition.fields.first{$0.key=="stride"})
        XCTAssertTrue(field.isValidDefinition)
        XCTAssertTrue(field.accepts(.null))
        XCTAssertEqual(field.valueWhenEnabled,.array([1,1]))
        XCTAssertTrue(field.accepts(field.valueWhenEnabled))
        XCTAssertFalse(field.accepts(.array([1])))
        XCTAssertFalse(field.accepts(.array([1,0])))
        XCTAssertFalse(field.accepts(.array([true,1])))
        let decoded=try JSONDecoder().decode(ComponentField.self,from:JSONEncoder().encode(field))
        XCTAssertEqual(decoded,field)
    }
}
