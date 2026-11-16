import Foundation

extension ExampleFlows {
    /// Two lazy layers resolve distinct widths, including one applied per sample position.
    /// Explicit reshape/flatten nodes show why inferred width is the final axis, not total size.
    public static func lazyLinearClassifier(bias: Bool = true) -> GraphDocument {
        let nodes: [GraphNode] = [
            .init(id: "input", type: "input", name: "Measurements", x: 40, y: 300, config: ["features": 2]),
            .init(id: "projection", type: "linear", name: "Six learned features", x: 270, y: 300, config: ["in_features": 2, "out_features": 6, "bias": true]),
            .init(id: "reshape", type: "reshape", name: "Two positions, three features", x: 500, y: 300, config: ["shape": .array([2, 3])]),
            .init(id: "local", type: "lazy_linear", name: "Infer three input features", x: 730, y: 300, config: ["out_features": 4, "bias": .bool(bias)]),
            .init(id: "flatten", type: "flatten", name: "Combine eight features", x: 960, y: 300, config: ["start_dim": 1, "end_dim": -1]),
            .init(id: "classifier", type: "lazy_linear", name: "Infer eight input features", x: 1190, y: 300, config: ["out_features": 2, "bias": .bool(bias)]),
            .init(id: "output", type: "output", name: "Class scores", x: 1420, y: 300)
        ]
        return GraphDocument(name: "LazyLinear inferred-width classifier", nodes: nodes,
            edges: zip(nodes, nodes.dropFirst()).enumerated().map {
                .init(id: "lazy-linear-wire-\($0.offset)", source: $0.element.0.id, target: $0.element.1.id)
            }, execution: ExecutionSettings(device: "auto", epochs: 20, batchSize: 128, sampleLimit: 512,
                seed: 42, splitSeed: 42, validationPercent: 20, learningRate: 0.01))
    }
}

extension WorkflowExamples {
    public static func preparedLazyLinearClassifier(projectID: String = UUID().uuidString, device: String = "auto", bias: Bool = true) throws -> WorkflowDocument {
        var document = try preparedClassifier(projectID: projectID, device: device, model: ExampleFlows.lazyLinearClassifier(bias: bias))
        let data: [String: (Int, Double)] = ["observations": (0, 40), "cast": (1, 40), "split": (2, 40),
            "fit": (3, 40), "training": (4, 40), "validation": (4, 210)]
        let model = Dictionary(uniqueKeysWithValues: ["input", "projection", "reshape", "local", "flatten", "classifier", "output"].enumerated().map { ($0.element, ($0.offset, 380.0)) })
        let control: [String: (Int, Double)] = ["mapping": (5, 125), "model": (7, 380), "setup": (7, 550), "train": (8, 300)]
        for index in document.nodes.indices {
            let address = document.nodes[index].address
            let positions = address.domain == .data ? data : address.domain == .model ? model : control
            if let (column, y) = positions[address.localID] {
                document.nodes[index].x = 40 + Double(column) * 230
                document.nodes[index].y = y
            }
        }
        return document
    }
}
