import Foundation

extension ExampleFlows {
    public static func sigmoidClassifier() -> GraphDocument {
        var graph = geluClassifier()
        graph.name = "Sigmoid hidden-feature classifier"
        for index in graph.nodes.indices where graph.nodes[index].type == "gelu" {
            graph.nodes[index].type = "sigmoid"
            graph.nodes[index].name = "Smooth independent feature scores"
            graph.nodes[index].config = [:]
        }
        return graph
    }
}
extension WorkflowExamples {
    public static func preparedSigmoidClassifier(projectID: String = UUID().uuidString, device: String = "auto") throws -> WorkflowDocument {
        try preparedClassifier(projectID: projectID, device: device, model: ExampleFlows.sigmoidClassifier())
    }
}
