import Foundation

extension ExampleFlows {
    /// Runnable two-feature classifier; both approximation modes remain editable graph configuration.
    /// This ordinary project example does not award measured lesson completion.
    public static func geluClassifier(approximateTanh: Bool = false) -> GraphDocument {
        GraphDocument(name: approximateTanh ? "GELU classifier · tanh approximation" : "GELU classifier · exact", nodes: [
            GraphNode(id: "input", type: "input", name: "Measurements", x: 30, y: 180, config: ["features": 2]),
            GraphNode(id: "features", type: "linear", name: "Learn features", x: 235, y: 180, config: ["in_features": 2, "out_features": 8, "bias": true]),
            GraphNode(id: "activation", type: "gelu", name: "Smooth activation", x: 440, y: 180, config: ["approximate_tanh": .bool(approximateTanh)]),
            GraphNode(id: "classifier", type: "linear", name: "Class scores", x: 645, y: 180, config: ["in_features": 8, "out_features": 2, "bias": true]),
            GraphNode(id: "output", type: "output", name: "Prediction", x: 850, y: 180)
        ], edges: [
            GraphEdge(id: "e1", source: "input", target: "features"),
            GraphEdge(id: "e2", source: "features", target: "activation"),
            GraphEdge(id: "e3", source: "activation", target: "classifier"),
            GraphEdge(id: "e4", source: "classifier", target: "output")
        ], execution: ExecutionSettings(device: "auto", epochs: 100, batchSize: 1024, sampleLimit: 0, seed: 42, splitSeed: 42, validationPercent: 20, learningRate: 0.03))
    }
}
