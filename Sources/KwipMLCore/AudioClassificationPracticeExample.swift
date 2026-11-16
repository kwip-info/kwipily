import Foundation
public enum AudioClassificationPracticeExample {
    public static let exampleID="classify-audio"
    public static let source=WorkflowNodeAddress(domain:.data,localID:"recording")
    public static let classifier=WorkflowNodeAddress(domain:.data,localID:"classifier")
    public static let model=WorkflowNodeAddress(domain:.control,localID:"classifier-model")
    public static func document(projectID:String=UUID().uuidString)throws->WorkflowDocument {
        var file=FileInputConfiguration.newDefaults;file["extensions"] = .array([.string("wav")])
        if case .object(var readers)?=file["readers"] {readers["audio"] = .object(["format":.string("pcmWav")]);file["readers"] = .object(readers)}
        var prepare=try PrepareMediaConfiguration.defaults(kind:"audio");prepare["sampleFormat"] = .string("float32");prepare["channels"] = .string("mono");prepare["targetRate"] = 16000
        let config:[String:JSONValue] = ["inputColumn":.string("prepared"), "outputPrefix":.string("audio_classification"), "windowSeconds":10, "hopSeconds":8, "finalWindow":.string("pad"), "minimumScore":0.5, "reportedLabels":5, "batchWindows":1, "maximumWindows":256, "maximumScoreRows":150000]
        var load=LoadModelConfiguration();load.source = .huggingFace;load.repository="MIT/ast-finetuned-audioset-10-10-0.4593";load.revision="f826b80d28226b62986cc218e5cec390b1096902"
        let prepared=WorkflowNodeAddress(domain:.data,localID:"waveform"),predictions=WorkflowNodeAddress(domain:.data,localID:"predictions"),scores=WorkflowNodeAddress(domain:.data,localID:"scores")
        let nodes:[WorkflowNode]=[
            .init(address:source,name:"Read your recording",x:30,y:100,payload:.dataOperator(operatorID:DataFlowSourceOperators.fileInput,config:file)),
            .init(address:prepared,name:"Prepare mono waveform",x:360,y:100,payload:.dataOperator(operatorID:PrepareMediaConfiguration.operatorID,config:prepare)),
            .init(address:model,name:"Load audio classifier",x:360,y:370,payload:.frozenModel(config:load.values)),
            .init(address:classifier,name:"Classify recording",x:700,y:100,payload:.dataOperator(operatorID:AudioClassificationConfiguration.operatorID,config:config)),
            .init(address:predictions,name:"Inspect window predictions",x:1030,y:40,payload:.dataOperator(operatorID:"kwip.data.view",config:[:])),
            .init(address:scores,name:"Inspect all class scores",x:1030,y:320,payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))]
        func edge(_ id:String,_ a:WorkflowNodeAddress,_ port:String,_ b:WorkflowNodeAddress)->WorkflowEdge {.init(id:id,from:.init(address:a,port:port),to:.init(address:b,port:"table"),kind:.table)}
        var doc=WorkflowDocument(projectID:projectID,name:"Classify sounds in a recording",nodes:nodes,edges:[edge("file-wave",source,"table",prepared),edge("wave-classifier",prepared,"table",classifier),edge("classifier-text",classifier,"predictions",predictions),edge("classifier-scores",classifier,"scores",scores),.init(id:"model-classifier",from:.init(address:model,port:"model"),to:.init(address:classifier,port:"model"),kind:.frozenModel)])
        doc.schemaVersion=21
        try WorkflowCodec.validate(doc);return doc
    }
}
