import Foundation

public enum FineTuningPracticeExample {
    public static let exampleID="fine-tuning"
    public static let source=WorkflowExamples.sourceAddress
    public static let model=WorkflowNodeAddress(domain:.control,localID:"loaded-model")
    public static let partition=WorkflowNodeAddress(domain:.data,localID:"partition")
    public static let train=WorkflowExamples.trainAddress
    public static let generate=WorkflowNodeAddress(domain:.control,localID:"generate")
    public static let save=WorkflowNodeAddress(domain:.control,localID:"save-model")
    public static func document(projectID:String=UUID().uuidString,device:String="cpu")throws->WorkflowDocument {
        let nodes:[WorkflowNode]=[
            .init(address:source,name:"Short training documents",x:60,y:320,payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:CSVParsingConfiguration.newDefaults)),
            .init(address:partition,name:"Keep validation separate",x:360,y:320,payload:.dataOperator(operatorID:"kwip.data.partition",config:["strategy":.string("random"),"seed":42,"validationPercent":25,"testPercent":25])),
            .init(address:model,name:"Choose a causal language model",x:360,y:100,payload:.frozenModel(config:LoadModelConfiguration(source:.localFolder,maximumTokens:8).values)),
            .init(address:train,name:"Adapt the loaded model",x:700,y:220,payload:.train(config:["fineTuning":.object(try FineTuningConfiguration(["maximumTokens":8,"stride":7]).values),"dataMapping":.object([:]),"runSetup":.object(["device":.string(device),"epochs":1,"batchSize":2,"seed":23,"learningRate":0.001])])),
            .init(address:generate,name:"Try the trained model",x:1040,y:110,payload:.generateText(config:["prompt":.string("hello world"),"maxNewTokens":8,"device":.string(device)])),
            .init(address:save,name:"Save the updated model",x:1040,y:340,payload:.saveModel(config:WorkflowSaveModelConfiguration(format:.huggingFace).values))]
        let edges:[WorkflowEdge]=[
            .init(from:.init(address:source,port:"table"),to:.init(address:partition,port:"table"),kind:.table),
            .init(from:.init(address:partition,port:"training"),to:.init(address:train,port:"training"),kind:.table),
            .init(from:.init(address:partition,port:"validation"),to:.init(address:train,port:"validation"),kind:.table),
            .init(from:.init(address:model,port:"model"),to:.init(address:train,port:"model"),kind:.frozenModel),
            .init(from:.init(address:train,port:"model"),to:.init(address:generate,port:"model"),kind:.frozenModel),
            .init(from:.init(address:train,port:"model"),to:.init(address:save,port:"model"),kind:.frozenModel)]
        var result=WorkflowDocument(projectID:projectID,name:"Fine-tune a local language model",nodes:nodes,edges:edges);result.schemaVersion=21
        try WorkflowCodec.validate(result);return result
    }
}
