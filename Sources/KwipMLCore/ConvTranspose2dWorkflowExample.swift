import Foundation

extension ExampleFlows {
    /// A projected-feature demonstration of grouped, shared kernels; observations
    /// are ordinary two-column measurements, not a image dataset.
    public static func convTranspose2dClassifier(bias: Bool = true) -> GraphDocument {
        let nodes: [GraphNode] = [
            .init(id: "input", type: "input", name: "Two measurements", x: 40, y: 380, config: ["features": 2]),
            .init(id: "projection", type: "linear", name: "Twenty-four learned features", x: 270, y: 380, config: ["in_features": 2, "out_features": 24, "bias": true]),
            .init(id: "reshape", type: "reshape", name: "Two channels, three by four grid", x: 500, y: 380, config: ["shape": .array([2, 3, 4])]),
            .init(id: "convolution", type: "conv_transpose2d", name: "Grouped grid expansion", x: 730, y: 380,
                config: ["in_channels": 2, "out_channels": 4, "kernel_size": .array([3, 3]), "stride": .array([2, 2]),
                         "dilation": .array([1, 1]), "groups": 2, "bias": .bool(bias), "padding": .array([1, 1]), "output_padding": .array([1, 0])]),
            .init(id: "activation", type: "relu", name: "Nonlinear responses", x: 960, y: 380),
            .init(id: "flatten", type: "flatten", name: "One hundred sixty-eight responses", x: 1190, y: 380, config: ["start_dim": 1, "end_dim": -1]),
            .init(id: "classifier", type: "linear", name: "Two class scores", x: 1420, y: 380, config: ["in_features": 168, "out_features": 2, "bias": true]),
            .init(id: "output", type: "output", name: "Class scores", x: 1650, y: 380)
        ]
        return GraphDocument(name: "Transposed-convolution constructed-grid classifier", nodes: nodes,
            edges: zip(nodes, nodes.dropFirst()).enumerated().map {
                .init(id: "conv-transpose2d-wire-\($0.offset)", source: $0.element.0.id, target: $0.element.1.id)
            }, execution: ExecutionSettings(device: "auto", epochs: 20, batchSize: 128, sampleLimit: 512,
                seed: 42, splitSeed: 42, validationPercent: 20, learningRate: 0.01))
    }
}

extension WorkflowExamples {
    public static func preparedConvTranspose2dClassifier(projectID: String = UUID().uuidString, device: String = "auto", bias: Bool = true) throws -> WorkflowDocument {
        var document = try preparedClassifier(projectID: projectID, device: device, model: ExampleFlows.convTranspose2dClassifier(bias: bias))
        let data: [String: (Int, Double)] = ["observations": (0, 40), "cast": (1, 40), "split": (2, 40),
            "fit": (3, 40), "training": (4, 40), "validation": (4, 210)]
        let model = Dictionary(uniqueKeysWithValues: ["input", "projection", "reshape", "convolution", "activation", "flatten", "classifier", "output"].enumerated().map { ($0.element, ($0.offset, 380.0)) })
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
