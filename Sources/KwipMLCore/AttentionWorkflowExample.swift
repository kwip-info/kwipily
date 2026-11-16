import Foundation
extension ExampleFlows {
    public static func attentionClassifier(componentID:String)throws->GraphDocument {
        guard AttentionConfiguration.componentIDs.contains(componentID),componentID != "attention_mask" else {throw GraphConfigurationError("Choose a model-building attention component.")}
        var nodes:[GraphNode]=[.init(id:"input",type:"input",name:"Measurements",x:30,y:240,config:["features":2])]
        var edges:[GraphEdge]=[]
        func wire(_ source:String,_ sourcePort:String="out",_ target:String,_ targetPort:String="in") {edges.append(.init(id:"attention-wire-\(edges.count)",source:source,sourcePort:sourcePort,target:target,targetPort:targetPort))}
        let scaled=componentID == "scaled_dot_product_attention"
        func sequence(_ role:String,_ length:Int,_ y:Double)->String {
            let projection=role+"-projection",sequence=role+"-sequence"
            nodes.append(.init(id:projection,type:"linear",name:role.capitalized+" values",x:250,y:y,config:["in_features":2,"out_features":.integer(Int64(length*2)),"bias":true]))
            let dimensions=scaled ? [1,length,2]:[length,2]
            nodes.append(.init(id:sequence,type:"reshape",name:role.capitalized+" positions",x:470,y:y,config:["shape":.array(dimensions.map {.integer(Int64($0))})]))
            wire("input","out",projection);wire(projection,"out",sequence)
            return sequence
        }
        var sources:[String:String]=[:],config:[String:JSONValue]=[:]
        switch componentID {
        case "multihead_attention","scaled_dot_product_attention":
            sources["query"]=sequence("query",2,80);sources["key"]=sequence("key",3,240);sources["value"]=sequence("value",3,400)
            config=scaled ? ["mask_mode":.string("supplied")]:["embed_dim":2,"num_heads":1,"attention_mask_mode":.string("supplied"),"padding_mask_mode":.string("supplied"),"average_weights":false]
        case "transformer_encoder":
            sources["in"]=sequence("source",3,240)
            config=["d_model":2,"nhead":1,"dim_feedforward":3,"norm_first":true,"mask_mode":.string("supplied"),"padding_mask_mode":.string("supplied")]
        default:
            sources["target"]=sequence("target",2,80)
            sources[componentID == "transformer" ? "source":"memory"]=sequence("source",3,400)
            config=["d_model":2,"nhead":1,"dim_feedforward":3,"norm_first":true,"target_mask_mode":.string("causal"),"memory_mask_mode":.string("supplied"),"memory_padding_mode":.string("supplied")]
        }
        nodes.append(.init(id:"mask",type:"attention_mask",name:"Choose useful connections",x:690,y:480,config:["mode":.string("blocked"),"blocked_queries":.array([0]),"blocked_keys":.array([1])]))
        let query=sources["query"] ?? sources["target"] ?? sources["in"]!,key=sources["key"] ?? sources["source"] ?? sources["memory"] ?? sources["in"]!
        wire(query,"out","mask","query");wire(key,"out","mask","key")
        nodes.append(.init(id:"attention",type:componentID,name:"Compare useful positions",x:910,y:240,config:try AttentionConfiguration(componentID:componentID,values:config).values))
        for role in sources.keys.sorted() {wire(sources[role]!,"out","attention",role)}
        let maskPort=componentID == "multihead_attention" ? "attn_mask":componentID == "transformer_encoder" || scaled ? "mask":"memory_mask"
        wire("mask","mask","attention",maskPort)
        if !scaled {wire("mask","padding","attention",componentID == "multihead_attention" ? "key_padding_mask":componentID == "transformer_encoder" ? "src_key_padding_mask":"memory_key_padding_mask")}
        let resultWidth=componentID == "transformer_encoder" ? 6:4
        nodes += [.init(id:"flat",type:"flatten",name:"Compared features",x:1130,y:240,config:["start_dim":1,"end_dim":-1]),.init(id:"head",type:"linear",name:"Class scores",x:1350,y:240,config:["in_features":.integer(Int64(resultWidth)),"out_features":2,"bias":true]),.init(id:"output",type:"output",name:"Prediction",x:1570,y:240)]
        wire("attention","out","flat");wire("flat","out","head");wire("head","out","output")
        return .init(name:"\(componentID) attention classifier",nodes:nodes,edges:edges,execution:ExecutionSettings(device:"auto",epochs:3,batchSize:32,sampleLimit:64,seed:42,splitSeed:42,validationPercent:20,learningRate:0.01))
    }
}
extension WorkflowExamples {
    public static func preparedMultiheadAttentionClassifier(projectID:String=UUID().uuidString,device:String="auto")throws->WorkflowDocument {try preparedClassifier(projectID:projectID,device:device,model:ExampleFlows.attentionClassifier(componentID:"multihead_attention"))}
    public static func preparedTransformerClassifier(projectID:String=UUID().uuidString,device:String="auto")throws->WorkflowDocument {try preparedClassifier(projectID:projectID,device:device,model:ExampleFlows.attentionClassifier(componentID:"transformer"))}
    public static func preparedTransformerEncoderClassifier(projectID:String=UUID().uuidString,device:String="auto")throws->WorkflowDocument {try preparedClassifier(projectID:projectID,device:device,model:ExampleFlows.attentionClassifier(componentID:"transformer_encoder"))}
    public static func preparedTransformerDecoderClassifier(projectID:String=UUID().uuidString,device:String="auto")throws->WorkflowDocument {try preparedClassifier(projectID:projectID,device:device,model:ExampleFlows.attentionClassifier(componentID:"transformer_decoder"))}
    public static func preparedScaledDotProductAttentionClassifier(projectID:String=UUID().uuidString,device:String="auto")throws->WorkflowDocument {try preparedClassifier(projectID:projectID,device:device,model:ExampleFlows.attentionClassifier(componentID:"scaled_dot_product_attention"))}
    public static func preparedAttentionMaskClassifier(projectID:String=UUID().uuidString,device:String="auto")throws->WorkflowDocument {try preparedMultiheadAttentionClassifier(projectID:projectID,device:device)}
}
