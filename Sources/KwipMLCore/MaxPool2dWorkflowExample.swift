import Foundation

extension ExampleFlows {
    /// Constructed feature positions demonstrate selection; these observations are not an image.
    public static func maxPool2dClassifier(returnIndices: Bool = true) -> GraphDocument {
        let nodes: [GraphNode] = [
            .init(id: "input", type: "input", name: "Two measurements", x: 40, y: 380, config: ["features": 2]),
            .init(id: "projection", type: "linear", name: "Forty learned features", x: 270, y: 380, config: ["in_features": 2, "out_features": 40, "bias": true]),
            .init(id: "reshape", type: "reshape", name: "Two channels, four by five grid", x: 500, y: 380, config: ["shape": .array([2, 4, 5])]),
            .init(id: "pool", type: "max_pool2d", name: "Local maxima and positions", x: 730, y: 380,
                config: ["kernel_size": .array([2, 3]), "stride": .array([1, 2]), "padding": .array([0, 0]), "dilation": .array([1, 1]),
                         "ceil_mode": false, "return_indices": .bool(returnIndices)]),
            .init(id: "activation", type: "relu", name: "Nonlinear responses", x: 960, y: 380),
            .init(id: "flatten", type: "flatten", name: "Twelve selected features", x: 1190, y: 380, config: ["start_dim": 1, "end_dim": -1]),
            .init(id: "classifier", type: "linear", name: "Two class scores", x: 1420, y: 380, config: ["in_features": 12, "out_features": 2, "bias": true]),
            .init(id: "output", type: "output", name: "Class scores", x: 1650, y: 380)
        ]
        return GraphDocument(name: "MaxPool2d feature-selection classifier", nodes: nodes,
            edges: zip(nodes, nodes.dropFirst()).enumerated().map {
                .init(id: "pool2d-wire-\($0.offset)", source: $0.element.0.id, target: $0.element.1.id)
            }, execution: ExecutionSettings(device: "auto", epochs: 20, batchSize: 128, sampleLimit: 512,
                seed: 42, splitSeed: 42, validationPercent: 20, learningRate: 0.01))
    }
}

extension WorkflowExamples {
    public static func preparedMaxPool2dClassifier(projectID: String = UUID().uuidString, device: String = "auto", returnIndices: Bool = true) throws -> WorkflowDocument {
        var document = try preparedClassifier(projectID: projectID, device: device, model: ExampleFlows.maxPool2dClassifier(returnIndices: returnIndices))
        let data: [String: (Int, Double)] = ["observations": (0, 40), "cast": (1, 40), "split": (2, 40),
            "fit": (3, 40), "training": (4, 40), "validation": (4, 210)]
        let model = Dictionary(uniqueKeysWithValues: ["input", "projection", "reshape", "pool", "activation", "flatten", "classifier", "output"].enumerated().map { ($0.element, ($0.offset, 380.0)) })
        let control: [String: (Int, Double)] = ["mapping": (5, 125), "model": (8, 380), "setup": (8, 550), "train": (9, 300)]
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
