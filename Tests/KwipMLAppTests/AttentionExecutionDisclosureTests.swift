import XCTest
import KwipMLCore
@testable import KwipMLApp
final class AttentionExecutionDisclosureTests:XCTestCase {
    func testExactMPSPositiveDropoutPolicyAndInactivePaths()throws {
        var graph=try ExampleFlows.attentionClassifier(componentID:"scaled_dot_product_attention")
        let index=try XCTUnwrap(graph.nodes.firstIndex {$0.type=="scaled_dot_product_attention"});graph.nodes[index].config["dropout"] = .number(0.5)
        let policy:JSONValue = .array([.object(["nodeID":.string("attention"),"operator":.string("scaled_dot_product_attention"),"computeDevice":.string("mps"),"policy":.string("deviceLocalPublicOps"),"phase":.string("training")])])
        XCTAssertNoThrow(try WorkflowLifecycleVerifier.validateAttentionExecution(["device":.string("mps"),"attentionExecution":policy],graph:graph))
        XCTAssertThrowsError(try WorkflowLifecycleVerifier.validateAttentionExecution(["device":.string("mps")],graph:graph))
        XCTAssertNoThrow(try WorkflowLifecycleVerifier.validateAttentionExecution(["device":.string("cpu")],graph:graph))
        XCTAssertThrowsError(try WorkflowLifecycleVerifier.validateAttentionExecution(["device":.string("cpu"),"attentionExecution":policy],graph:graph))
        graph.nodes[index].config["dropout"] = 0
        XCTAssertNoThrow(try WorkflowLifecycleVerifier.validateAttentionExecution(["device":.string("mps")],graph:graph))
        XCTAssertThrowsError(try WorkflowLifecycleVerifier.validateAttentionExecution(["device":.string("mps"),"attentionExecution":policy],graph:graph))
    }
    func testForeignMissingAndDuplicatePolicyRowsRefuse()throws {
        var graph=try ExampleFlows.attentionClassifier(componentID:"scaled_dot_product_attention")
        let index=try XCTUnwrap(graph.nodes.firstIndex {$0.type=="scaled_dot_product_attention"});graph.nodes[index].config["dropout"] = 0.5
        let fields:[String:JSONValue]=["nodeID":.string("attention"),"operator":.string("scaled_dot_product_attention"),"computeDevice":.string("mps"),"policy":.string("deviceLocalPublicOps"),"phase":.string("training")]
        for key in fields.keys {var altered=fields;altered[key] = .string("foreign");XCTAssertThrowsError(try WorkflowLifecycleVerifier.validateAttentionExecution(["device":.string("mps"),"attentionExecution":.array([.object(altered)])],graph:graph))}
        for rows:[JSONValue] in [[],[.object(fields),.object(fields)],[.object(fields.merging(["extra":true]) {_,new in new})]] {XCTAssertThrowsError(try WorkflowLifecycleVerifier.validateAttentionExecution(["device":.string("mps"),"attentionExecution":.array(rows)],graph:graph))}
    }
}
