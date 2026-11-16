import Foundation

public enum TextEncodingPracticeExample {
    public static let exampleID="text-encoding"
    public static let lessonID="text-encoding"
    public static let observationsResource="text-encoding-observations"
    public static let address=WorkflowNodeAddress(domain:.data,localID:"text")
    public static func document(projectID:String=UUID().uuidString)throws->WorkflowDocument {
        let c:[String:JSONValue]=["textColumn":.string("text"),"tokenizer":.string("character"),"unicodeNormalization":.string("none"),"letterCase":.string("preserve"),"whitespace":.string("preserve"),"missingText":.string("error"),"outputColumn":.string("tokens"),"keepSourceText":true,"vocabularySource":.string("supplied"),"vocabulary":.array(["T","o"," ","b","e","a","r","n","t"].map(JSONValue.string)),"minimumFrequency":1,"vocabularyLimit":256,"idOrder":.string("frequency"),"unknownPolicy":.string("unk"),"pad":true,"bos":false,"eos":false]
        let view=WorkflowNodeAddress(domain:.data,localID:"view")
        let result=WorkflowDocument(projectID:projectID,name:"Encode text and inspect token IDs",nodes:[
            .init(address:WorkflowExamples.sourceAddress,name:"Four short texts",x:80,y:120,payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:CSVParsingConfiguration.newDefaults)),
            .init(address:address,name:"Encode characters",x:420,y:120,payload:.dataOperator(operatorID:"kwip.data.textEncode",config:c)),
            .init(address:view,name:"Explore tokens",x:760,y:120,payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))],edges:[
                .init(id:"source-text",from:.init(address:WorkflowExamples.sourceAddress,port:"table"),to:.init(address:address,port:"table"),kind:.table),
                .init(id:"text-view",from:.init(address:address,port:"table"),to:.init(address:view,port:"table"),kind:.table)])
        try WorkflowCodec.validate(result);return result
    }
    public static let trainingExampleID="text-encoding-classifier"
    public static let trainingResource="text-training-observations"
    public static func trainingDocument(projectID:String=UUID().uuidString,device:String="cpu")throws->WorkflowDocument {
        var document=try RetiredToolAuthoringUpgrade.plan(WorkflowExamples.preparedEmbeddingClassifier(projectID:projectID,device:device))
        let text=WorkflowNodeAddress(domain:.data,localID:"text"),split=WorkflowNodeAddress(domain:.data,localID:"split")
        guard let cast=document.nodes.firstIndex(where:{$0.address == .init(domain:.data,localID:"cast")}) else {throw DataFlowContractError("Columns missing")}
        document.nodes[cast].name="Text and class"
        document.nodes[cast].payload = .dataOperator(operatorID:"kwip.data.columns",config:CurrentPracticeConfiguration.columns([("text","keep"),("label","int64")]))
        guard var config=try self.document().nodes.first(where:{$0.address==address}) else {throw DataFlowContractError("Text encoding missing")}
        if case .dataOperator(_,var values)=config.payload {
            values["vocabularySource"] = .string("training");values["vocabulary"] = .array([]);config.payload = .dataOperator(operatorID:TextEncodingConfiguration.operatorID,config:values)
        }
        config.name="Learn Training vocabulary";config.x=600;config.y=40;document.nodes.append(config)
        for i in document.edges.indices where document.edges[i].from.address==split && ["training","validation"].contains(document.edges[i].from.port) {
            document.edges[i].from = .init(address:text,port:document.edges[i].from.port)
        }
        for role in ["training","validation","test"] {
            document.edges.append(.init(id:"text-"+role,from:.init(address:split,port:role),to:.init(address:text,port:role),kind:.table))
        }
        for i in document.nodes.indices {
            if document.nodes[i].address == .init(domain:.model,localID:"input") {document.nodes[i].payload = .modelPrimitive(type:"index_input",configurationVersion:2,config:["shape":.array([5])])}
            if document.nodes[i].address == .init(domain:.model,localID:"head"),case .modelPrimitive(let type,let version,var c)=document.nodes[i].payload {c["in_features"]=15;document.nodes[i].payload = .modelPrimitive(type:type,configurationVersion:version,config:c)}
        }
        let plan=NamedInputPlan(inputs:[.init(name:"tokens",nodeID:"input",representation:"tokens",dtype:"int64",sampleShape:[5],columns:["tokens"],lengthPolicy:"error",paddingValue:1,vocabularySize:4)],roles:[.init(name:"target",dtype:"int64",sampleShape:[],columns:["label"])])
        for i in document.nodes.indices where document.nodes[i].address==WorkflowExamples.trainAddress {
            if case .train(var c)=document.nodes[i].payload {c["dataMapping"] = .object(["namedInputPlan":.object(plan.values),"featureColumns":.array([.string("tokens")]),"targetColumn":.string("label")]);document.nodes[i].payload = .train(config:c)}
        }
        guard let definition=document.nodes.first(where:WorkflowModelBoundary.isDefinition) else {throw DataFlowContractError("Model definition missing")}
        document.schemaVersion=19;document.namedModelInputs=[definition.address.localID:["tokens":"input"]]
        document.name="Train a token-sequence classifier"
        try WorkflowCodec.validate(document);return document
    }

}
