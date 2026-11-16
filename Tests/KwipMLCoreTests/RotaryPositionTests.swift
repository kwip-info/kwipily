import XCTest
@testable import KwipMLCore

final class RotaryPositionTests:XCTestCase {
    func testStrictInlineConfigurationAndExactShapePorts()throws {
        let c=try AttentionConfiguration(componentID:"rotary_position",values:[:])
        XCTAssertEqual(c.configuredInputs,["query","key"]);XCTAssertEqual(c.outputPorts,["query","key"])
        XCTAssertEqual(c.parameterCount,0);XCTAssertFalse(c.stochastic)
        for shapes in [[[3,4],[3,4]],[[2,3,4],[1,3,4]]] {
            let resolved=try c.resolved(layouts:shapes.map {try .init(sample:$0)})
            XCTAssertEqual(resolved.outputs["query"]?.sample.dimensions,shapes[0]);XCTAssertEqual(resolved.outputs["key"]?.sample.dimensions,shapes[1])
        }
        let alternate=try RotaryPositionConfiguration(values:["sequence_axis":1,"rotation":.string("custom"),"rotary_dim":2,"pairing":.string("halves"),"base":.number(1234),"start_position":32,"position_scale":2])
        XCTAssertNoThrow(try alternate.resolved(layouts:[.init(sample:[3,2,4]),.init(sample:[3,1,4])]))
        for bad:[String:JSONValue] in [["position_source":.string("supplied")],["base":1],["base":true],["rotary_dim":3],["position_scale":0],["sequence_axis":0],["start_position":-1]] {XCTAssertThrowsError(try RotaryPositionConfiguration(values:bad))}
        for shapes in [[[3,4],[2,4]],[[3,3],[3,3]],[[2,3],[2,4]]] {XCTAssertThrowsError(try c.resolved(layouts:shapes.map {try .init(sample:$0)}))}
        let entry=try XCTUnwrap(BundledAttentionComponents.entries["rotary_position"])
        XCTAssertNoThrow(try AttentionRuntimeCapabilities.verify(JSONEncoder().encode(entry.runtimeContract),componentID:"rotary_position"))
        let declaration=try AttentionCapabilityBridge.rotaryPositionDeclaration(entry.definition)
        XCTAssertEqual(declaration.ports.map(\.id),["query","key","query","key"])
        XCTAssertTrue(declaration.lifecycle.state.isEmpty)
    }
    func testCausalPracticeCompilesAndPersistsWithTwoDataInputs()throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let document=try RotaryPositionPracticeExample.document(device:"cpu")
        XCTAssertEqual(PreparedWorkflowExample.component("rotary_position").observationsResource,TextEncodingPracticeExample.trainingResource)
        let plan=try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:.load(from:root.appendingPathComponent("runner/dataflow-operators.json")),contractVersion:15)
        let graph=plan.model.graph,validation=GraphValidator.validate(graph,registry:try .builtIn(),context:.explicitObjective)
        XCTAssertTrue(validation.issues.isEmpty,validation.issues.map(\.message).joined(separator:"; "))
        let r=try XCTUnwrap(graph.nodes.first {$0.type=="rotary_position"})
        XCTAssertEqual(Set(graph.edges.filter {$0.target==r.id}.map(\.targetPort)),["query","key"])
        XCTAssertEqual(validation.outputTensorContracts[.init(r.id,"query")]?.layout.sample.dimensions,[1,3,4])
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(document)),document)
    }
}
