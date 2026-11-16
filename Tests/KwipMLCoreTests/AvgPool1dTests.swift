import XCTest
@testable import KwipMLCore

final class AvgPool1dTests: XCTestCase {
    func testStrictConfigurationAndSharedGeometry() throws {
        let values: [String:JSONValue] = ["kernel_size":3,"stride":.null,"padding":0,"ceil_mode":false,"count_include_pad":true]
        let config = try AvgPool1dConfiguration(values:values)
        XCTAssertEqual(config.values,values)
        XCTAssertEqual(config.effectiveStride,3)
        XCTAssertEqual(try config.resolved(input:.init(sample:[2,7])).outputLayout.sample.dimensions,[2,2])
        for changes: [String:JSONValue] in [["kernel_size":true],["padding":2],["stride":0],["count_include_pad":1],["dilation":1]] {
            XCTAssertThrowsError(try AvgPool1dConfiguration(values:values.merging(changes){_,new in new}))
        }
        XCTAssertThrowsError(try config.resolved(input:.init(sample:[2,1])))
    }
    func testPublicBundleBothDivisorModesAndWorkflowRoundTrip() throws {
        let registry = try ComponentRegistry.builtIn()
        let root = URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        XCTAssertEqual(AvgPool1dComponent.definition.fields.count,5)
        let declaration = try XCTUnwrap(BuiltInCapabilityBridge.declarations(registry:registry).first{$0.id == "avg_pool1d"})
        XCTAssertEqual(declaration.ports.map(\.id),["in","out"])
        for include in [false,true] {
            let graph = ExampleFlows.avgPool1dClassifier(countIncludePad:include)
            let validation = GraphValidator.validate(graph,registry:registry)
            XCTAssertTrue(validation.isValid,"\(validation.issues)")
            XCTAssertEqual(validation.tensorLayouts["pool"]?.sample.dimensions,[2,4])
            XCTAssertNil(validation.outputTensorContracts[.init("pool","indices")])
            let workflow = try WorkflowExamples.preparedAvgPool1dClassifier(device:"cpu",countIncludePad:include)
            let plan = try WorkflowCompiler.compile(workflow,train:WorkflowExamples.trainAddress,modelRegistry:registry,
                dataRegistry:.load(from:root.appendingPathComponent("runner/dataflow-operators.json")))
            XCTAssertEqual(plan.model.graph.nodes.first{$0.type == "avg_pool1d"}?.config["count_include_pad"],.bool(include))
            XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(workflow)),workflow)
        }
        XCTAssertNoThrow(try BundledPoolingComponents.example("avg_pool1d",device:"cpu"))
    }
}
