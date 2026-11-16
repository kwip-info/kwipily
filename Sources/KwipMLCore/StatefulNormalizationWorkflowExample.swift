import Foundation

extension ExampleFlows {
    public static func statefulNormalizationClassifier(componentID: String) throws -> GraphDocument {
        guard StatefulNormalizationConfiguration.componentIDs.contains(componentID) else { throw GraphConfigurationError("Unknown channel-normalization example.") }
        let grouped = componentID == "group_norm"
        let shape = grouped ? [4, 2] : componentID.hasSuffix("2d") ? [2, 2, 2] : [2, 2]
        let width = shape.reduce(1, *)
        var config: [String: JSONValue] = grouped ? ["num_channels": 4, "num_groups": 2] : ["num_features": 2]
        config = try StatefulNormalizationConfiguration(componentID: componentID, values: config).values
        let nodes: [GraphNode] = [
            .init(id: "input", type: "input", name: "Measurements", x: 30, y: 180, config: ["features": 2]),
            .init(id: "learn", type: "linear", name: "Learn channel values", x: 250, y: 180, config: ["in_features": 2, "out_features": .integer(Int64(width)), "bias": true]),
            .init(id: "array", type: "reshape", name: "Channel patterns", x: 470, y: 180, config: ["shape": .array(shape.map { .integer(Int64($0)) })]),
            .init(id: "normalize", type: componentID, name: "Normalize channel patterns", x: 690, y: 180, config: config),
            .init(id: "flat", type: "flatten", name: "Classification features", x: 910, y: 180, config: ["start_dim": 1, "end_dim": -1]),
            .init(id: "head", type: "linear", name: "Class scores", x: 1130, y: 180, config: ["in_features": .integer(Int64(width)), "out_features": 2, "bias": true]),
            .init(id: "output", type: "output", name: "Prediction", x: 1350, y: 180)
        ]
        let edges = zip(nodes, nodes.dropFirst()).enumerated().map { index, pair in GraphEdge(id: "normalization-wire-\(index)", source: pair.0.id, sourcePort: "out", target: pair.1.id, targetPort: "in") }
        return GraphDocument(name: "\(componentID) channel-feature classifier", nodes: nodes, edges: edges, execution: ExecutionSettings(device: "auto", epochs: 20, batchSize: 128, sampleLimit: 512, seed: 42, splitSeed: 42, validationPercent: 20, learningRate: 0.01))
    }
}
extension WorkflowExamples {
    public static func preparedBatchNorm1dClassifier(projectID: String = UUID().uuidString, device: String = "auto") throws -> WorkflowDocument { try preparedClassifier(projectID: projectID, device: device, model: ExampleFlows.statefulNormalizationClassifier(componentID: "batch_norm1d")) }
    public static func preparedBatchNorm2dClassifier(projectID: String = UUID().uuidString, device: String = "auto") throws -> WorkflowDocument { try preparedClassifier(projectID: projectID, device: device, model: ExampleFlows.statefulNormalizationClassifier(componentID: "batch_norm2d")) }
    public static func preparedInstanceNorm1dClassifier(projectID: String = UUID().uuidString, device: String = "auto") throws -> WorkflowDocument { try preparedClassifier(projectID: projectID, device: device, model: ExampleFlows.statefulNormalizationClassifier(componentID: "instance_norm1d")) }
    public static func preparedInstanceNorm2dClassifier(projectID: String = UUID().uuidString, device: String = "auto") throws -> WorkflowDocument { try preparedClassifier(projectID: projectID, device: device, model: ExampleFlows.statefulNormalizationClassifier(componentID: "instance_norm2d")) }
    public static func preparedGroupNormClassifier(projectID: String = UUID().uuidString, device: String = "auto") throws -> WorkflowDocument { try preparedClassifier(projectID: projectID, device: device, model: ExampleFlows.statefulNormalizationClassifier(componentID: "group_norm")) }
}
