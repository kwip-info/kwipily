import Foundation

extension ExampleFlows {
    public static func tanhClassifier() -> GraphDocument {
        var graph = geluClassifier()
        graph.name = "Tanh hidden-feature classifier"
        for index in graph.nodes.indices where graph.nodes[index].type == "gelu" {
            graph.nodes[index].type = "tanh"
            graph.nodes[index].name = "Smooth signed hidden values"
            graph.nodes[index].config = [:]
        }
        return graph
    }
}
extension WorkflowExamples {
    public static func preparedTanhClassifier(projectID: String = UUID().uuidString, device: String = "auto") throws -> WorkflowDocument {
        try preparedClassifier(projectID: projectID, device: device, model: ExampleFlows.tanhClassifier())
    }
}
