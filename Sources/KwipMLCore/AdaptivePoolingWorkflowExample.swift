import Foundation

extension ExampleFlows {
    /// Constructed learned feature positions demonstrate pooling; rows are not images or time series.
    public static func adaptivePoolingClassifier(componentID: String, configuration: [String:JSONValue]) throws -> GraphDocument {
        let shape = componentID == "adaptive_avg_pool1d" ? [2,5] : [2,5,4]
        let input = try BatchedTensorLayout(sample: shape)
        let output: BatchedTensorLayout
        if componentID == "avg_pool2d" { output = try AvgPool2dConfiguration(values: configuration).resolved(input: input).outputLayout }
        else { output = try AdaptivePoolingConfiguration(componentID: componentID, values: configuration).resolved(input: input).outputLayout }
        var model = maxPool2dClassifier(returnIndices: false)
        model.name = "\(componentID) feature-summary classifier"
        for index in model.nodes.indices {
            switch model.nodes[index].id {
            case "projection":
                model.nodes[index].name = "Learned feature positions"
                model.nodes[index].config["out_features"] = .integer(Int64(input.sample.elementCount))
            case "reshape":
                model.nodes[index].name = "Two channels of feature positions"
                model.nodes[index].config["shape"] = .array(shape.map { .integer(Int64($0)) })
            case "pool":
                model.nodes[index].type = componentID
                model.nodes[index].name = "Feature summaries"
                model.nodes[index].config = configuration
            case "flatten": model.nodes[index].name = "Summary features"
            case "classifier": model.nodes[index].config["in_features"] = .integer(Int64(output.sample.elementCount))
            default: break
            }
        }
        return model
    }
}

extension WorkflowExamples {
    public static func preparedAvgPool2dClassifier(projectID: String = UUID().uuidString, device: String = "auto") throws -> WorkflowDocument {
        try preparedAdaptivePooling(projectID: projectID, device: device, componentID: "avg_pool2d", configuration:
            ["kernel_size": .array([3,3]), "stride": .array([2,3]), "padding": .array([1,1]), "ceil_mode": true, "count_include_pad": true, "divisor_override": .null])
    }
    public static func preparedAdaptiveAvgPool1dClassifier(projectID: String = UUID().uuidString, device: String = "auto") throws -> WorkflowDocument {
        try preparedAdaptivePooling(projectID: projectID, device: device, componentID: "adaptive_avg_pool1d", configuration: ["output_size": 3])
    }
    public static func preparedAdaptiveAvgPool2dClassifier(projectID: String = UUID().uuidString, device: String = "auto") throws -> WorkflowDocument {
        try preparedAdaptivePooling(projectID: projectID, device: device, componentID: "adaptive_avg_pool2d", configuration: ["output_height": 3, "output_width": .null])
    }
    public static func preparedAdaptiveMaxPool2dClassifier(projectID: String = UUID().uuidString, device: String = "auto") throws -> WorkflowDocument {
        try preparedAdaptivePooling(projectID: projectID, device: device, componentID: "adaptive_max_pool2d", configuration: ["output_height": 3, "output_width": .null, "return_indices": true])
    }
    private static func preparedAdaptivePooling(projectID: String, device: String, componentID: String, configuration: [String:JSONValue]) throws -> WorkflowDocument {
        try preparedClassifier(projectID: projectID, device: device, model: ExampleFlows.adaptivePoolingClassifier(componentID: componentID, configuration: configuration))
    }
}
