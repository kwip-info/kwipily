import Foundation

extension ExampleFlows {
    public static func reluClassifier() -> GraphDocument {
        var graph = geluClassifier()
        graph.name = "ReLU hidden-feature classifier"
        for index in graph.nodes.indices where graph.nodes[index].type == "gelu" {
            graph.nodes[index].type = "relu"
            graph.nodes[index].name = "Clipped independent hidden values"
            graph.nodes[index].config = [:]
        }
        return graph
    }
}
extension WorkflowExamples {
    public static func preparedReLUClassifier(projectID: String = UUID().uuidString, device: String = "auto") throws -> WorkflowDocument {
        try preparedClassifier(projectID: projectID, device: device, model: ExampleFlows.reluClassifier())
    }
}
