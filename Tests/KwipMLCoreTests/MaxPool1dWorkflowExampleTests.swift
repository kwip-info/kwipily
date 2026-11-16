import XCTest
@testable import KwipMLCore

final class MaxPool1dWorkflowExampleTests: XCTestCase {
    func testPreparedPoolingHasTypedIndicesAndCompilesBothModes() throws {
        let registry = try ComponentRegistry.builtIn()
        let root = URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        for indices in [false,true] {
            let graph = ExampleFlows.maxPool1dClassifier(returnIndices:indices)
            let validation = GraphValidator.validate(graph,registry:registry)
            XCTAssertTrue(validation.isValid,"\(validation.issues)")
            XCTAssertEqual(validation.tensorLayouts["pool"]?.sample.dimensions,[2,4])
            XCTAssertEqual(validation.tensorLayouts["flatten"]?.sample.dimensions,[8])
            XCTAssertEqual(validation.outputTensorContracts[.init("pool","indices")]?.dtype,indices ? .int64 : nil)
            let workflow = try WorkflowExamples.preparedMaxPool1dClassifier(device:"cpu",returnIndices:indices)
            let plan = try WorkflowCompiler.compile(workflow,train:WorkflowExamples.trainAddress,modelRegistry:registry,
                dataRegistry:.load(from:root.appendingPathComponent("runner/dataflow-operators.json")))
            XCTAssertEqual(plan.settings.device,"cpu")
            XCTAssertEqual(plan.model.graph.nodes.first(where:{$0.type == "max_pool1d"})?.config["return_indices"],.bool(indices))
            XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(workflow)),workflow)
        }
        let document = try PreparedWorkflowExample.maxPool1d.document()
        for (i,node) in document.nodes.enumerated() {
            for other in document.nodes.dropFirst(i+1) {
                XCTAssertTrue(abs(node.x-other.x)>=200 || abs(node.y-other.y)>=130)
            }
        }
    }
}
