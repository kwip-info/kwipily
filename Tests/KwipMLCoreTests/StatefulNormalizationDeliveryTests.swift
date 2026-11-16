import XCTest
@testable import KwipMLCore

final class StatefulNormalizationDeliveryTests: XCTestCase {
    func testFiveActualPreparedExamplesAndCanonicalGraphCapture() throws {
        let registry = try ComponentRegistry.builtIn()
        for id in StatefulNormalizationConfiguration.componentIDs.sorted() {
            let graph = try ExampleFlows.statefulNormalizationClassifier(componentID: id)
            let bytes = try JSONEncoder().encode(graph)
            XCTAssertEqual(try JSONDecoder().decode(GraphDocument.self, from: bytes), graph)
            let inspection = GraphValidator.validate(graph, registry: registry)
            XCTAssertTrue(inspection.isValid, "\(id): \(inspection.issues)")
            let normalize = try XCTUnwrap(graph.nodes.first { $0.id == "normalize" })
            XCTAssertEqual(inspection.outputTensorContracts[.init("normalize", "out")]?.layout.sample.dimensions, id == "group_norm" ? [4, 2] : id.hasSuffix("2d") ? [2, 2, 2] : [2, 2])
            XCTAssertEqual(try GraphMigration.effectiveConfiguration(of: normalize, schemaVersion: 2, registry: registry), normalize.config)
            XCTAssertThrowsError(try GraphMigration.effectiveConfiguration(of: normalize, schemaVersion: 1, registry: registry))
            XCTAssertNoThrow(try PreparedWorkflowExample.component(id).document(device: "cpu"))
            // A normalization block can own all trainable state without a Linear block.
            let shape = id == "group_norm" ? [2] : id.hasSuffix("2d") ? [1, 1, 2] : [1, 2]
            let config: [String: JSONValue] = id == "group_norm" ? ["num_channels": 2, "num_groups": 1, "affine": true] : ["num_features": 1, "affine": true]
            let nodes: [GraphNode] = [
                .init(id: "input", type: "input", name: "Measurements", x: 0, y: 0, config: ["features": 2]),
                .init(id: "shape", type: "reshape", name: "Channel shape", x: 100, y: 0, config: ["shape": .array(shape.map { .integer(Int64($0)) })]),
                .init(id: "normalize", type: id, name: "Learn channel adjustments", x: 200, y: 0, config: config),
                .init(id: "flat", type: "flatten", name: "Scores", x: 300, y: 0, config: ["start_dim": 1, "end_dim": -1]),
                .init(id: "output", type: "output", name: "Prediction", x: 400, y: 0)
            ]
            let edges = zip(nodes, nodes.dropFirst()).enumerated().map { index, pair in GraphEdge(id: "wire-\(index)", source: pair.0.id, sourcePort: "out", target: pair.1.id, targetPort: "in") }
            let affineOnly = GraphValidator.validate(GraphDocument(name: "Channel adjustments only", nodes: nodes, edges: edges), registry: registry)
            XCTAssertTrue(affineOnly.isValid, "\(id) affine-only model: \(affineOnly.issues)")
            if let destination = ProcessInfo.processInfo.environment["KWIPML_NORMALIZATION_EXAMPLE_CAPTURE"] {
                let directory = URL(fileURLWithPath: destination, isDirectory: true)
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                try bytes.write(to: directory.appendingPathComponent(id + ".json"), options: .atomic)
            }
        }
    }
}
