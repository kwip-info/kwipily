import Foundation

public enum OptimizerLearningCanvas {
    public static func registry(task:OptimizerLearningTask)->ComponentRegistry {
        .init(components:[BundledOptimizerComponents.definition(task.componentID),.init(id:"train",title:"Train",category:"Training",inputs:["optimizer"])])
    }
    public static func inspect(_ graph:GraphDocument,task:OptimizerLearningTask)->ValidationResult {
        var result=ValidationResult()
        do {
            _ = try OptimizerLearningSubmission.validatedGraph(graph,task:task,complete:false)
            _ = try OptimizerLearningSubmission.validatedGraph(graph,task:task,complete:true)
            result.orderedNodeIDs=graph.nodes.sorted {$0.type != "train" && $1.type == "train"}.map(\.id)
        } catch {result.issues.append(.init(id:"optimizer.lesson.connection",message:error.localizedDescription))}
        return result
    }
}
