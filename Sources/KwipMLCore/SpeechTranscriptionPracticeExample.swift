import Foundation
public enum SpeechTranscriptionPracticeExample {
    public static let exampleID="transcribe-audio"
    public static let source=WorkflowNodeAddress(domain:.data,localID:"recording")
    public static let speech=WorkflowNodeAddress(domain:.data,localID:"speech")
    public static let model=WorkflowNodeAddress(domain:.control,localID:"speech-model")
    public static func document(projectID:String=UUID().uuidString)throws->WorkflowDocument {
        var file=FileInputConfiguration.newDefaults;file["extensions"] = .array([.string("wav")])
        if case .object(var readers)?=file["readers"] {readers["audio"] = .object(["format":.string("pcmWav")]);file["readers"] = .object(readers)}
        var prepare=try PrepareMediaConfiguration.defaults(kind:"audio");prepare["sampleFormat"] = .string("float32");prepare["channels"] = .string("mono");prepare["targetRate"] = 16000
        let config:[String:JSONValue] = ["inputColumn":.string("prepared"),"outputColumn":.string("transcript"),"language":.string("auto"),"chunkSeconds":30,"maximumTokens":256,"beamWidth":1,"maximumCharacters":1000000,"maximumSegments":10000]
        var load=LoadModelConfiguration();load.source = .huggingFace;load.repository="openai/whisper-tiny";load.revision="169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
        let prepared=WorkflowNodeAddress(domain:.data,localID:"waveform"),transcripts=WorkflowNodeAddress(domain:.data,localID:"transcripts"),segments=WorkflowNodeAddress(domain:.data,localID:"segments")
        let nodes:[WorkflowNode]=[
            .init(address:source,name:"Read your recording",x:30,y:100,payload:.dataOperator(operatorID:DataFlowSourceOperators.fileInput,config:file)),
            .init(address:prepared,name:"Prepare mono waveform",x:360,y:100,payload:.dataOperator(operatorID:PrepareMediaConfiguration.operatorID,config:prepare)),
            .init(address:model,name:"Load speech model",x:360,y:370,payload:.frozenModel(config:load.values)),
            .init(address:speech,name:"Transcribe recording",x:700,y:100,payload:.dataOperator(operatorID:SpeechTranscriptionConfiguration.operatorID,config:config)),
            .init(address:transcripts,name:"Inspect transcripts",x:1030,y:40,payload:.dataOperator(operatorID:"kwip.data.view",config:[:])),
            .init(address:segments,name:"Inspect source segments",x:1030,y:320,payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))]
        func edge(_ id:String,_ a:WorkflowNodeAddress,_ port:String,_ b:WorkflowNodeAddress)->WorkflowEdge {.init(id:id,from:.init(address:a,port:port),to:.init(address:b,port:"table"),kind:.table)}
        var doc=WorkflowDocument(projectID:projectID,name:"Transcribe a private recording",nodes:nodes,edges:[edge("file-wave",source,"table",prepared),edge("wave-speech",prepared,"table",speech),edge("speech-text",speech,"transcripts",transcripts),edge("speech-segments",speech,"segments",segments),.init(id:"model-speech",from:.init(address:model,port:"model"),to:.init(address:speech,port:"model"),kind:.frozenModel)])
        doc.schemaVersion=21
        try WorkflowCodec.validate(doc);return doc
    }
}
