import XCTest
@testable import KwipMLCore

final class MaxPool1dGraphTests:XCTestCase {
    private func registry() throws -> ComponentRegistry {
        return try ComponentRegistry.builtIn()
    }
    private func graph(indices:Bool=true) -> GraphDocument {
        var g=ExampleFlows.conv1dClassifier()
        let i=g.nodes.firstIndex{$0.id=="convolution"}!
        g.nodes[i].type="max_pool1d"
        g.nodes[i].config=["kernel_size":3,"stride":2,"padding":1,"dilation":1,"ceil_mode":true,"return_indices":.bool(indices)]
        g.nodes[g.nodes.firstIndex{$0.id=="classifier"}!].config["in_features"]=8
        return g
    }
    func testTypedValuesAndIndicesPreserveBatchChannelsAndLearnedWeights() throws {
        let g=graph();let r=try registry();let v=GraphValidator.validate(g,registry:r)
        XCTAssertTrue(v.isValid,"\(v.issues)")
        XCTAssertEqual(v.outputTensorContracts[.init("convolution","out")],try .init(sample:[2,4],dtype:.float32))
        XCTAssertEqual(v.outputTensorContracts[.init("convolution","indices")],try .init(sample:[2,4],dtype:.int64))
        XCTAssertEqual(v.tensorLayouts["flatten"]?.sample.dimensions,[8])
        XCTAssertEqual(v.tensorPortDescriptor(nodeID:"convolution",port:"indices",direction:.output,graph:g,registry:r).declared,["[N, 2, 4]","int64"])
        let disabled=GraphValidator.validate(graph(indices:false),registry:r)
        XCTAssertTrue(disabled.isValid)
        XCTAssertNil(disabled.outputTensorContracts[.init("convolution","indices")])
    }
    func testIndicesRefuseFloatInputAndDisabledPortCannotAliasValues() throws {
        for enabled in [false,true] {
            var g=graph(indices:enabled)
            g.edges[g.edges.firstIndex{$0.source=="convolution"}!].sourcePort="indices"
            let v=GraphValidator.validate(g,registry:try registry())
            XCTAssertFalse(v.isValid)
            XCTAssertNil(v.tensorContracts["activation"])
        }
    }
    func testBridgeAndStrictMigrationExposeSixSettings() throws {
        let r=try registry();let definition=MaxPool1dComponent.definition
        XCTAssertEqual(definition.fields.count,6)
        XCTAssertTrue(definition.fields.first{$0.key=="stride"}!.allowsNull)
        let c=try XCTUnwrap(BuiltInCapabilityBridge.declarations(registry:r).first{$0.id=="max_pool1d"})
        XCTAssertEqual(c.ports.first{$0.id=="indices"}?.schema.tensor?.dtype,.int64)
        var g=graph();g.nodes[g.nodes.firstIndex{$0.id=="convolution"}!].config["padding"]=2
        XCTAssertFalse(GraphValidator.validate(g,registry:r).isValid)
    }
    func testConfiguredOutputAvailabilityRefusesDisabledIndicesBeforeWiring() throws {
        let r = try registry(), g = graph(indices:false)
        let node = try XCTUnwrap(g.nodes.first(where:{$0.type == "max_pool1d"}))
        let definition = try XCTUnwrap(r.component(type:node.type))
        XCTAssertEqual(definition.configuredOutputs(node.config),["out"])
        var enabled = node.config;enabled["return_indices"] = true
        XCTAssertEqual(definition.configuredOutputs(enabled),["out","indices"])
        XCTAssertEqual(GraphValidator.validate(g,registry:r).tensorPortDescriptor(nodeID:node.id,port:"indices",direction:.output,graph:g,registry:r).kind,.unknown)
        var draft = g;draft.edges.removeAll(where:{$0.source == node.id})
        XCTAssertThrowsError(try GraphCommandEngine.apply(.connect(source:.init(node.id,"indices"),target:.init("flatten","in")),to:.init(document:draft),registry:r))
    }

}
