import XCTest
@testable import KwipMLCore

final class GraphTests: XCTestCase {
    func result(_ graph: GraphDocument) throws -> ValidationResult {
        GraphValidator.validate(graph, registry: try ComponentRegistry.builtIn())
    }
    func testRegistryAndReferenceFlow() throws {
        let registry = try ComponentRegistry.builtIn()
        let componentIDs=Set(registry.components.map(\.id))
        XCTAssertEqual(componentIDs.count,registry.components.count,"Registry component IDs must be unique")
        XCTAssertTrue(Set(["input", "linear", "relu", "tanh", "gelu", "softmax", "concat", "add", "output", "reshape", "transpose", "flatten", "layer_norm", "index_input", "embedding", "matmul", "lazy_linear", "bilinear", "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d", "depthwise_conv2d", "max_pool1d", "max_pool2d"]).isSubset(of:componentIDs),"Required foundation components must remain available")
        let validation = try result(ExampleFlows.parallelClassifier)
        XCTAssertTrue(validation.isValid, validation.issues.map(\.message).joined(separator: "; "))
        XCTAssertEqual(validation.widths["merge"], 16)
        XCTAssertEqual(validation.widths["output"], 2)
        XCTAssertEqual(validation.tensorLayouts["merge"]?.displayDimensions,["N","16"])
        XCTAssertEqual(validation.tensorLayouts["output"]?.displayDimensions,["N","2"])
        XCTAssertEqual(Set(validation.tensorLayouts.keys),Set(validation.widths.keys))
        XCTAssertEqual(validation.orderedNodeIDs.count, 8)
    }
    func testInvalidLinearShapeDoesNotInventDownstreamLayouts()throws {
        var graph=ExampleFlows.parallelClassifier
        let index=try XCTUnwrap(graph.nodes.firstIndex {$0.id == "branch-a"})
        graph.nodes[index].config["in_features"] = .integer(3)
        let validation=try result(graph)
        XCTAssertFalse(validation.isValid)
        for id in ["branch-a","activation-a","merge","classifier","output"] {
            XCTAssertNil(validation.tensorLayouts[id]);XCTAssertNil(validation.widths[id])
        }
        XCTAssertEqual(validation.tensorLayouts["activation-b"]?.displayDimensions,["N","8"])
    }
    func testRoundTripPreservesNamesAndLayoutIncludingInvalidDraft() throws {
        var graph = ExampleFlows.parallelClassifier
        graph.nodes[0].name = "My measurements"
        graph.nodes[0].x = 123.5
        graph.edges.removeLast()
        XCTAssertEqual(try JSONDecoder().decode(GraphDocument.self, from: JSONEncoder().encode(graph)), graph)
        XCTAssertFalse(try result(graph).isValid)
    }
    func testDuplicateIDsAndUnknownComponentAndVersionRejected() throws {
        var graph = ExampleFlows.parallelClassifier
        graph.nodes.append(graph.nodes[0])
        XCTAssertFalse(try result(graph).isValid)
        graph = ExampleFlows.parallelClassifier
        graph.nodes[0].type = "unknown"
        XCTAssertFalse(try result(graph).isValid)
        graph = ExampleFlows.parallelClassifier
        graph.schemaVersion = 99
        XCTAssertFalse(try result(graph).isValid)
    }
    func testPortsAndMultiplicityRejected() throws {
        var graph = ExampleFlows.parallelClassifier
        graph.edges[0].sourcePort = "missing"
        XCTAssertFalse(try result(graph).isValid)
        graph = ExampleFlows.parallelClassifier
        var edge = graph.edges[0]; edge.id = "duplicate-wire"
        graph.edges.append(edge)
        XCTAssertFalse(try result(graph).isValid)
        graph = ExampleFlows.parallelClassifier
        graph.edges[0].target = "missing"
        XCTAssertFalse(try result(graph).isValid)
    }
    func testDisconnectedAndCyclesRejected() throws {
        var graph = ExampleFlows.parallelClassifier
        graph.nodes.append(GraphNode(id: "orphan", type: "relu"))
        XCTAssertFalse(try result(graph).isValid)
        graph = ExampleFlows.parallelClassifier
        graph.edges.append(GraphEdge(id:"cycle",source:"activation-a",sourcePort:"out",target:"branch-a",targetPort:"in"))
        XCTAssertFalse(try result(graph).isValid)
    }
    func testWidthsAndAddRule() throws {
        var graph = ExampleFlows.parallelClassifier
        graph.nodes[1].config["in_features"] = 3
        XCTAssertFalse(try result(graph).isValid)
        graph = ExampleFlows.parallelClassifier
        graph.nodes[1].config["out_features"] = 1.5
        XCTAssertFalse(try result(graph).isValid)
        graph = ExampleFlows.parallelClassifier
        graph.nodes[5].type = "add"
        graph.nodes[6].config["in_features"] = 8
        XCTAssertTrue(try result(graph).isValid)
        graph.nodes[3].config["out_features"] = 9
        XCTAssertFalse(try result(graph).isValid)
    }
    func testClassificationBoundariesAndUnknownFields() throws {
        var graph = ExampleFlows.parallelClassifier
        graph.nodes[0].config["features"] = 3
        graph.nodes[1].config["in_features"] = 3
        graph.nodes[3].config["in_features"] = 3
        XCTAssertFalse(try result(graph).isValid)
        graph = ExampleFlows.parallelClassifier
        graph.nodes[6].config["out_features"] = 3
        XCTAssertFalse(try result(graph).isValid)
        graph = ExampleFlows.parallelClassifier
        graph.nodes[1].config["typo"] = 1
        XCTAssertFalse(try result(graph).isValid)
    }
}

extension GraphTests {
    func testParameterFreeGraphRejectedBeforeRunning() throws {
        let graph = GraphDocument(nodes: [GraphNode(id: "input", type: "input", config: ["features": 2]), GraphNode(id: "output", type: "output")], edges: [GraphEdge(source: "input", target: "output")])
        let validation = try result(graph)
        XCTAssertFalse(validation.isValid)
        XCTAssertTrue(validation.issues.contains { $0.message.contains("trainable") })
    }
}


extension GraphTests {
    func testRunnerLimitsAreVisibleBeforeExecution() throws {
        var graph = ExampleFlows.parallelClassifier
        graph.nodes[1].id = "bad.id"
        XCTAssertTrue(try result(graph).issues.contains { $0.message.contains("without dots") })
        graph = ExampleFlows.parallelClassifier
        graph.nodes[1].id = String(repeating: "x", count: 129)
        XCTAssertTrue(try result(graph).issues.contains { $0.message.contains("128 characters") })
        graph = ExampleFlows.parallelClassifier
        graph.nodes += (0..<121).map { GraphNode(id: "extra-\($0)", type: "relu") }
        XCTAssertTrue(try result(graph).issues.contains { $0.message.contains("128 nodes") })
        graph = ExampleFlows.parallelClassifier
        graph.edges += (0..<249).map { GraphEdge(id: "extra-\($0)", source: "input", target: "branch-a") }
        XCTAssertTrue(try result(graph).issues.contains { $0.message.contains("256 connections") })
    }
    func testParameterBudgetRejectedBeforeAllocation() throws {
        var nodes = [GraphNode(id: "input", type: "input", config: ["features":2])]
        var edges: [GraphEdge] = []
        var previous = "input"
        for i in 0..<10 {
            let id = "layer-\(i)"
            nodes.append(GraphNode(id:id,type:"linear",config:["in_features":i == 0 ? 2 : 512,"out_features":512]))
            edges.append(GraphEdge(source:previous,target:id)); previous=id
        }
        nodes.append(GraphNode(id:"head",type:"linear",config:["in_features":512,"out_features":2]))
        nodes.append(GraphNode(id:"output",type:"output"))
        edges.append(GraphEdge(source:previous,target:"head")); edges.append(GraphEdge(source:"head",target:"output"))
        XCTAssertTrue(try result(GraphDocument(nodes:nodes,edges:edges)).issues.contains { $0.message.contains("2 million") })
    }
}
