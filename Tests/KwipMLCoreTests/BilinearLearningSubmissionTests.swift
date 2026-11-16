import XCTest
@testable import KwipMLCore

final class BilinearLearningSubmissionTests: XCTestCase {
    static func graph(bias: Bool = true) -> GraphDocument {
        var nodes: [GraphNode] = [
            .init(id: "left", type: "input", name: "My first input", config: ["fixture": .string("left")]),
            .init(id: "right", type: "input", config: ["fixture": .string("right")]),
            .init(id: "weight", type: "input", config: ["fixture": .string("weight")]),
            .init(id: "contract", type: "bilinear", config: ["bias": .bool(bias)]),
            .init(id: "result", type: "output")
        ]
        var edges: [GraphEdge] = [
            .init(source: "left", target: "contract", targetPort: "a"),
            .init(source: "right", target: "contract", targetPort: "b"),
            .init(source: "weight", target: "contract", targetPort: "weight"),
            .init(source: "contract", target: "result")
        ]
        if bias {
            nodes.append(.init(id: "bias", type: "input", config: ["fixture": .string("bias")]))
            edges.append(.init(source: "bias", target: "contract", targetPort: "bias"))
        }
        return GraphDocument(nodes: nodes, edges: edges)
    }

    func testBothModesRoundTripPreservingAuthoredTransformsAndRouting() throws {
        for bias in [false, true] {
            var graph = Self.graph(bias: bias)
            graph.nodes.append(.init(id: "shape", type: "reshape", name: "My shape", config: ["shape": .array([.integer(-1)])]))
            graph.edges[0].source = "shape"
            graph.edges.append(.init(source: "left", target: "shape"))
            let submission = try BilinearLearningSubmission(graph: graph)
            XCTAssertEqual(submission.kind, "bilinearGraph")
            XCTAssertEqual(try BilinearLearningSubmission.decode(submission.encoded()).graph, graph)
            graph.edges[0].targetPort = "b"; graph.edges[1].targetPort = "a"
            XCTAssertEqual(try BilinearLearningSubmission(graph: graph).graph, graph)
        }
    }

    func testRejectsInvalidBiasRolesWidthsTopologyAndUnsupportedNodes() throws {
        for mutation in 0..<10 {
            var graph = Self.graph()
            switch mutation {
            case 0: graph.nodes[1].config["fixture"] = .string("left")
            case 1: graph.nodes[3].config["bias"] = .integer(1)
            case 2: graph.nodes[3].config["in1_features"] = .integer(2)
            case 3: graph.nodes[3].config["bias"] = .bool(false)
            case 4: graph.edges.removeLast()
            case 5: graph.nodes.append(.init(id: "second", type: "bilinear", config: ["bias": .bool(true)]))
            case 6: graph.edges[0].source = "contract"
            case 7: graph.nodes[2].type = "custom_code"
            case 8: graph.edges[2].targetPort = "b"
            default: graph.nodes[2].config["fixture"] = .string("upstream")
            }
            XCTAssertThrowsError(try BilinearLearningSubmission(graph: graph), "mutation \(mutation)")
        }
    }

    func testPartialGraphAllowsEditingButRejectsCyclesAndDuplicateOperators() throws {
        var graph = Self.graph(bias: false)
        graph.edges = []
        XCTAssertEqual(try BilinearLearningSubmission.validatedGraph(graph, complete: false), graph)
        XCTAssertThrowsError(try BilinearLearningSubmission(graph: graph))
        graph.nodes.append(.init(id: "other", type: "bilinear", config: ["bias": .bool(false)]))
        XCTAssertThrowsError(try BilinearLearningSubmission.validatedGraph(graph, complete: false))
        graph = Self.graph(bias: false)
        graph.edges[0].source = "contract"
        XCTAssertThrowsError(try BilinearLearningSubmission.validatedGraph(graph, complete: false))
        XCTAssertNoThrow(try BilinearLearningSubmission.validatedGraph(GraphDocument(nodes: [], edges: []), complete: false))
    }

    func testStrictEnvelopeRejectsAnswersExecutionAndHiddenNodeFields() throws {
        let data = try BilinearLearningSubmission(graph: Self.graph()).encoded()
        let original = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        for mutation in 0..<5 {
            var raw = original
            if mutation == 0 { raw["answers"] = ["output": [1]] }
            if mutation == 1 { raw["kind"] = "matrixGraph" }
            var graph = try XCTUnwrap(raw["graph"] as? [String: Any])
            if mutation == 2 { graph["execution"] = NSNull() }
            if mutation == 3 {
                var nodes = try XCTUnwrap(graph["nodes"] as? [[String: Any]])
                nodes[0]["code"] = "return expected"; graph["nodes"] = nodes
            }
            if mutation == 4 { raw["schemaVersion"] = true }
            raw["graph"] = graph
            XCTAssertThrowsError(try BilinearLearningSubmission.decode(JSONSerialization.data(withJSONObject: raw)))
        }
    }
}
