import Foundation

extension ExampleFlows {
    /// Different learned feature spaces interact through one trainable weight tensor.
    public static var bilinearClassifier: GraphDocument {
        let nodes: [GraphNode] = [
            .init(id: "input", type: "input", name: "Measurements", x: 30, y: 180, config: ["features": 2]),
            .init(id: "left", type: "linear", name: "Three learned features", x: 260, y: 80, config: ["in_features": 2, "out_features": 3, "bias": true]),
            .init(id: "right", type: "linear", name: "Five learned features", x: 260, y: 300, config: ["in_features": 2, "out_features": 5, "bias": true]),
            .init(id: "interaction", type: "bilinear", name: "Learn feature interactions", x: 490, y: 180, config: ["in1_features": 3, "in2_features": 5, "out_features": 2, "bias": true]),
            .init(id: "output", type: "output", name: "Class scores", x: 720, y: 180)
        ]
        let links = [("input", "left", "in"), ("input", "right", "in"), ("left", "interaction", "a"),
                     ("right", "interaction", "b"), ("interaction", "output", "in")]
        return GraphDocument(name: "Bilinear interaction classifier", nodes: nodes, edges: links.enumerated().map {
            .init(id: "bilinear-wire-\($0.offset)", source: $0.element.0, target: $0.element.1, targetPort: $0.element.2)
        }, execution: ExecutionSettings(device: "auto", epochs: 20, batchSize: 128, sampleLimit: 512,
            seed: 42, splitSeed: 42, validationPercent: 20, learningRate: 0.01))
    }
}

extension WorkflowExamples {
    public static func preparedBilinearClassifier(projectID: String = UUID().uuidString, device: String = "auto") throws -> WorkflowDocument {
        var document = try preparedClassifier(projectID: projectID, device: device, model: ExampleFlows.bilinearClassifier)
        let data: [String: (Int, Double)] = ["observations": (0, 40), "cast": (1, 40), "split": (2, 40),
            "fit": (3, 40), "training": (4, 40), "validation": (4, 210)]
        let model: [String: (Int, Double)] = ["input": (0, 380), "left": (1, 300), "right": (1, 470),
            "interaction": (2, 380), "output": (3, 380)]
        let control: [String: (Int, Double)] = ["mapping": (5, 125), "model": (5, 380), "setup": (5, 550), "train": (6, 300)]
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
