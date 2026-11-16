import Foundation

public enum SchedulerLearningCanvas {
    public static func registry(task:SchedulerLearningTask)->ComponentRegistry {
        var supplied=BundledOptimizerComponents.definition("optimizer_sgd")
        supplied.title="Supplied SGD"
        return .init(components:[BundledSchedulerComponents.definition(task.componentID),supplied,.init(id:"train",title:"Train",category:"Training",inputs:["optimizer","scheduler"])])
    }
    public static func inspect(_ graph:GraphDocument,task:SchedulerLearningTask)->ValidationResult {
        var result=ValidationResult()
        do {_ = try SchedulerLearningSubmission.validatedGraph(graph,task:task,complete:true);result.orderedNodeIDs=graph.nodes.sorted {$0.type != "train" && $1.type == "train"}.map(\.id)}
        catch {result.issues.append(.init(id:"scheduler.lesson.connection",message:error.localizedDescription))}
        return result
    }
}
