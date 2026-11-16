import Foundation

public enum GradientLearningExamples {
    public static func modelGraph(widths w:[Int]=[2,2,1])throws->GraphDocument {
        guard GradientLearningTask.widths.contains(w) else {throw LearningContractError("Use one of the supplied two-layer model shapes.")}
        return .init(schemaVersion:2,name:"Supplied two-layer model",nodes:[.init(id:"input",type:"input",name:"Supplied observations",config:["features":.integer(Int64(w[0]))]),.init(id:"first",type:"linear",name:"First learned layer",x:180,config:["in_features":.integer(Int64(w[0])),"out_features":.integer(Int64(w[1])),"bias":true]),.init(id:"second",type:"linear",name:"Second learned layer",x:360,config:["in_features":.integer(Int64(w[1])),"out_features":.integer(Int64(w[2])),"bias":true]),.init(id:"output",type:"output",name:"Predictions",x:540)],edges:[.init(id:"input-first",source:"input",target:"first"),.init(id:"first-second",source:"first",target:"second"),.init(id:"second-output",source:"second",target:"output")])
    }
    public static func defaultGraph(task:GradientLearningTask)throws->GraphDocument {
        var graph=try preparedGraph(task:task);graph.edges=[]
        if task == .selectedInfluence {graph.nodes[0].config=try GradientPolicyConfiguration(values:[:]).values}
        if task == .combineSmallGroups {graph.nodes[0].config=try GradientPolicyConfiguration(values:[:]).values}
        if task == .keepSettings {graph.nodes[1].config=try TrainabilityConfiguration(values:[:]).values}
        if task == .nextRun {graph.nodes[1].config=try TrainabilityConfiguration(values:GradientLearningTask.keepSettings.trainabilityValues).values}
        return graph
    }
    public static func preparedGraph(task:GradientLearningTask)throws->GraphDocument {
        let controls=try GradientLearningConfiguration(gradientValues:task.gradientValues,trainabilityValues:task.trainabilityValues,task:task)
        return .init(schemaVersion:10,name:task.title,nodes:[.init(id:"gradient",type:"gradient_policy",name:"Gradient policy",x:80,y:50,config:controls.gradient.values),.init(id:"trainability",type:"trainability",name:"Trainability",x:80,y:250,config:controls.trainability.values),.init(id:"optimizer",type:"optimizer_sgd",name:"Supplied SGD",x:80,y:450,config:try OptimizerConfiguration(componentID:"optimizer_sgd",values:task.optimizerValues).values),.init(id:"train",type:"train",name:"Train",x:460,y:250)],edges:[.init(id:"gradient-train",source:"gradient",target:"train",targetPort:"gradientPolicy"),.init(id:"trainability-train",source:"trainability",target:"train",targetPort:"trainability"),.init(id:"optimizer-train",source:"optimizer",target:"train",targetPort:"optimizer")])
    }
}
public enum GradientLearningCanvas {
    public static func inspect(_ graph:GraphDocument,task:GradientLearningTask)->ValidationResult {
        var result=ValidationResult()
        do {_ = try GradientLearningSubmission.validatedGraph(graph,task:task,complete:true);result.orderedNodeIDs=graph.nodes.filter {$0.type != "train"}.map(\.id)+graph.nodes.filter {$0.type=="train"}.map(\.id)}
        catch {result.issues.append(.init(id:"gradient.lesson.connection",message:error.localizedDescription))}
        return result
    }
}
