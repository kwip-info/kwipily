import Foundation

public enum InitializationLearningExamples {
    public static func modelGraph(widths:[Int] = [2,3,2])throws->GraphDocument {
        guard InitializationLearningTask.widths.contains(widths) else {throw LearningContractError("Choose one of the supplied model shapes.")}
        return .init(schemaVersion:2,name:"Two layers with starting settings",nodes:[
            .init(id:"input",type:"input",name:"Supplied observations",config:["features":.integer(Int64(widths[0]))]),
            .init(id:"first",type:"linear",name:"First learned layer",x:180,config:["in_features":.integer(Int64(widths[0])),"out_features":.integer(Int64(widths[1])),"bias":true]),
            .init(id:"activation",type:"leaky_relu",name:"Keep a small negative response",x:360,config:["negative_slope":0.2]),
            .init(id:"second",type:"linear",name:"Second learned layer",x:540,config:["in_features":.integer(Int64(widths[1])),"out_features":.integer(Int64(widths[2])),"bias":true]),
            .init(id:"output",type:"output",name:"Predictions",x:720)],edges:[
            .init(id:"input-first",source:"input",target:"first"),.init(id:"first-response",source:"first",target:"activation"),.init(id:"response-second",source:"activation",target:"second"),.init(id:"second-output",source:"second",target:"output")])
    }
    public static func defaultGraph(task:InitializationLearningTask)throws->GraphDocument {
        var graph=try preparedGraph(task:task);graph.edges=[]
        graph.nodes[0].config=try InitializationLearningConfiguration(values:["seed":42,"rules":.array([.object(["owner":.object(["node":.string("first"),"parameter":.string("weight")]),"algorithm":.string("normal"),"settings":.object(["mean":0,"std":0.02])])])]).values
        return graph
    }
    public static func preparedGraph(task:InitializationLearningTask)throws->GraphDocument {
        .init(schemaVersion:9,name:task.title,nodes:[.init(id:"initialization",type:"initialization",name:"Starting settings",x:80,y:60,config:try InitializationLearningConfiguration(values:task.configuration).values),.init(id:"optimizer",type:"optimizer_sgd",name:"Supplied SGD",x:80,y:260,config:InitializationLearningTask.suppliedOptimizerValues),.init(id:"train",type:"train",name:"Train",x:460,y:160)],edges:[.init(id:"starting-to-train",source:"initialization",target:"train",targetPort:"initialization"),.init(id:"optimizer-to-train",source:"optimizer",target:"train",targetPort:"optimizer")])
    }
    static func workflow(task:InitializationLearningTask,projectID:String,device:String)throws->WorkflowDocument {
        var document=try WorkflowExamples.preparedClassifier(projectID:projectID,device:device,model:modelGraph())
        document.schemaVersion=max(document.schemaVersion,15);document.name=task.title
        let configuration=try InitializationLearningConfiguration(values:task.configuration).values
        guard case .array(let rules)=configuration["rules"] else {throw DataFlowContractError("Practice starting-value rules are missing.")}
        for value in rules {
            guard case .object(let rule)=value,case .object(let owner)=rule["owner"],let node=owner["node"]?.stringValue,
                  let parameter=owner["parameter"]?.stringValue,let algorithm=rule["algorithm"],let settings=rule["settings"],
                  let index=document.nodes.firstIndex(where:{$0.address == .init(domain:.model,localID:node)}) else {throw DataFlowContractError("Practice layer starting-value owner is missing.")}
            var parameters=document.nodes[index].parameterSettings ?? [:]
            parameters[parameter] = .object(["initialization":.object(["algorithm":algorithm,"settings":settings])])
            document.nodes[index].parameterSettings=parameters
        }
        try CurrentPracticeConfiguration.setTrain("optimizer",value:try OptimizerConfiguration(componentID:"optimizer_sgd",values:InitializationLearningTask.suppliedOptimizerValues).recipeValue,in:&document)
        try WorkflowCodec.validate(document)
        return document
    }
}
extension WorkflowExamples {
    public static func preparedInitializationStartingValues(projectID:String=UUID().uuidString,device:String="auto")throws->WorkflowDocument {try InitializationLearningExamples.workflow(task:.startingValues,projectID:projectID,device:device)}
    public static func preparedInitializationStartingSpread(projectID:String=UUID().uuidString,device:String="auto")throws->WorkflowDocument {try InitializationLearningExamples.workflow(task:.startingSpread,projectID:projectID,device:device)}
}

/// Training controls on the lesson canvas carry recipes, never tensor data.
public enum InitializationLearningCanvas {
    public static func registry(task:InitializationLearningTask)->ComponentRegistry {
        var optimizer=BundledOptimizerComponents.definition("optimizer_sgd")
        optimizer.title="Supplied SGD"
        return .init(components:[BundledInitializationComponents.definition("initialization"),optimizer,.init(id:"train",title:"Train",category:"Training",inputs:["optimizer","initialization"])])
    }
    public static func inspect(_ graph:GraphDocument,task:InitializationLearningTask)->ValidationResult {
        var result=ValidationResult()
        do {
            _ = try InitializationLearningSubmission.validatedGraph(graph,task:task,complete:true)
            result.orderedNodeIDs=graph.nodes.sorted {$0.type != "train" && $1.type == "train"}.map(\.id)
        } catch {result.issues.append(.init(id:"initialization.lesson.connection",message:error.localizedDescription))}
        return result
    }
}
