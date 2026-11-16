import Foundation

/// Genuine source/typed objective/model workflows, separate from the assessed control-only canvas.
public enum GradientWorkflowExamples {
    public static func document(task:GradientLearningTask,projectID:String=UUID().uuidString,device:String="auto",parentPreparation:Bool=false)throws->WorkflowDocument {
        guard !parentPreparation || task == .nextRun else {throw GraphConfigurationError("Only the next-run exercise creates a saved parent first.")}
        var document=try WorkflowExamples.preparedLossObjective(task:.mse,projectID:projectID,device:device)
        document.schemaVersion=max(document.schemaVersion,15);document.name=parentPreparation ? "Create saved parent · "+task.title:task.title
        let columns=["x0","x1"],targets=["y0"]
        let mapping=LossTargetMapping(componentID:"loss_mse",inputColumns:columns,roles:[.init(role:"target",columns:targets,sampleShape:[1],dtype:.float32)])
        try mapping.validate()
        let mappingValue=try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(mapping))
        document.nodes.removeAll {$0.address.domain == .model}
        document.edges.removeAll {$0.from.address.domain == .model || $0.to.address.domain == .model}
        for index in document.nodes.indices {
            let id=document.nodes[index].address.localID
            if id == "cast" {document.nodes[index].payload = .dataOperator(operatorID:"kwip.data.columns",config:CurrentPracticeConfiguration.columns((columns+targets).map {($0,"float64")}))}
            if id == "split" {document.nodes[index].payload = .dataOperator(operatorID:"kwip.data.partition",config:["strategy":.string("random"),"seed":42,"validationPercent":37,"testPercent":25]);document.nodes[index].name="Three training, three validation and two reserved rows"}
            if id == "mapping" {document.nodes[index].payload = .trainingData(config:["lossTargetMapping":mappingValue])}
            if id == "setup" {document.nodes[index].payload = .runSetup(config:["device":.string(device),"epochs":.integer(task == .combineSmallGroups ? 2:1),"batchSize":2,"seed":42,"learningRate":0.05]);document.nodes[index].name="Small groups of two observations"}
            if document.nodes[index].address == WorkflowExamples.trainAddress {document.nodes[index].payload = .train(config:task == .nextRun && !parentPreparation ? ["trainingMode":.string("fineTune")]:[:]);document.nodes[index].name=task == .nextRun && !parentPreparation ? "Fine-tune from the selected real saved parent":"Train and inspect"}
        }
        let model=try GradientLearningExamples.modelGraph()
        for node in model.nodes {
            document.nodes.append(.init(address:.init(domain:.model,localID:node.id),name:node.name,x:node.x,y:260,payload:.modelPrimitive(type:node.type,configurationVersion:2,config:node.config)))
        }
        for edge in model.edges {document.edges.append(.init(id:"gradient-model-"+edge.id,from:.init(address:.init(domain:.model,localID:edge.source),port:edge.sourcePort),to:.init(address:.init(domain:.model,localID:edge.target),port:edge.targetPort),kind:.tensor))}
        document.edges.append(.init(id:"gradient-model-definition",from:.init(address:.init(domain:.model,localID:"output"),port:"definition"),to:.init(address:.init(domain:.control,localID:"model"),port:"source"),kind:.modelDefinition))
        let gradient=try GradientPolicyConfiguration(values:task.gradientValues),flags=try TrainabilityConfiguration(values:parentPreparation ? GradientLearningTask.keepSettings.trainabilityValues:task.trainabilityValues)
        try CurrentPracticeConfiguration.setTrain("objective",value:try LossConfiguration(componentID:"loss_mse",values:["reduction":.string("mean"),"target_broadcast":.bool(false)]).recipeValue,in:&document)
        try CurrentPracticeConfiguration.setTrain("gradientPolicy",value:gradient.recipeValue,in:&document)
        try CurrentPracticeConfiguration.setTrain("optimizer",value:try OptimizerConfiguration(componentID:"optimizer_sgd",values:task.optimizerValues).recipeValue,in:&document)
        for owner in flags.owners {
            guard case .object(let row)=owner,let node=row["node"]?.stringValue,let parameter=row["parameter"]?.stringValue,
                  let index=document.nodes.firstIndex(where:{$0.address == .init(domain:.model,localID:node)}),let mode=row["mode"] else {throw DataFlowContractError("Practice layer trainability owner is missing.")}
            var settings=document.nodes[index].parameterSettings ?? [:];settings[parameter] = .object(["trainability":mode]);document.nodes[index].parameterSettings=settings
        }
        try WorkflowCodec.validate(document)
        return try WorkflowCodec.decode(WorkflowCodec.encode(document))
    }
    public static func sourceCSV(widths w:[Int]=[2,2,1])throws->String {
        guard GradientLearningTask.widths.contains(w) else {throw GraphConfigurationError("Choose a reviewed tiny model shape.")}
        let i=w[0],o=w[2],header=(0..<i).map {"x\($0)"}+(0..<o).map {"y\($0)"}
        var lines=[header.joined(separator:",")]
        for row in 0..<8 {
            let x=(0..<i).map {Double(row-3)/8+Double($0)/16+Double(row)/1024}
            let y=(0..<o).map {out in x.enumerated().reduce(0.125) {$0+Double(($1.offset+1)*(out+1))*$1.element/8}}
            lines.append((x+y).map {String($0)}.joined(separator:","))
        }
        return lines.joined(separator:"\n")+"\n"
    }
}
public extension WorkflowExamples {
    static func preparedGradientSelectedInfluence(projectID:String=UUID().uuidString,device:String="auto")throws->WorkflowDocument {try GradientWorkflowExamples.document(task:.selectedInfluence,projectID:projectID,device:device)}
    static func preparedGradientCombineSmallGroups(projectID:String=UUID().uuidString,device:String="auto")throws->WorkflowDocument {try GradientWorkflowExamples.document(task:.combineSmallGroups,projectID:projectID,device:device)}
    static func preparedTrainabilityKeepSettings(projectID:String=UUID().uuidString,device:String="auto")throws->WorkflowDocument {try GradientWorkflowExamples.document(task:.keepSettings,projectID:projectID,device:device)}
    static func preparedTrainabilityNextRun(projectID:String=UUID().uuidString,device:String="auto")throws->WorkflowDocument {try GradientWorkflowExamples.document(task:.nextRun,projectID:projectID,device:device)}
}
