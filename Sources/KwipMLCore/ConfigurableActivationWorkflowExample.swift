import Foundation

extension ExampleFlows {
    private static func configurableActivationClassifier(type: String, configuration: [String: JSONValue]) throws -> GraphDocument {
        let values = try ConfigurableActivationConfiguration(componentID: type, values: configuration).values
        var graph = geluClassifier()
        let title = BundledActivationComponents.definition(type).title
        graph.name = title + " hidden-feature classifier"
        for index in graph.nodes.indices where graph.nodes[index].type == "gelu" {
            graph.nodes[index].type = type
            graph.nodes[index].name = title + " feature response"
            graph.nodes[index].config = values
        }
        return graph
    }
    public static func siluClassifier() throws -> GraphDocument {
        try configurableActivationClassifier(type: "silu", configuration: [:])
    }
    public static func leakyReLUClassifier(negativeSlope: Double = 0.01) throws -> GraphDocument {
        try configurableActivationClassifier(type: "leaky_relu", configuration: ["negative_slope": .number(negativeSlope)])
    }
    public static func eluClassifier(alpha: Double = 1) throws -> GraphDocument {
        try configurableActivationClassifier(type: "elu", configuration: ["alpha": .number(alpha)])
    }
}
extension WorkflowExamples {
    public static func preparedSiLUClassifier(projectID: String = UUID().uuidString, device: String = "auto") throws -> WorkflowDocument {
        try preparedClassifier(projectID: projectID, device: device, model: ExampleFlows.siluClassifier())
    }
    public static func preparedLeakyReLUClassifier(projectID: String = UUID().uuidString, device: String = "auto", negativeSlope: Double = 0.01) throws -> WorkflowDocument {
        try preparedClassifier(projectID: projectID, device: device, model: ExampleFlows.leakyReLUClassifier(negativeSlope: negativeSlope))
    }
    public static func preparedELUClassifier(projectID: String = UUID().uuidString, device: String = "auto", alpha: Double = 1) throws -> WorkflowDocument {
        try preparedClassifier(projectID: projectID, device: device, model: ExampleFlows.eluClassifier(alpha: alpha))
    }
}
