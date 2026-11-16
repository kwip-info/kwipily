import Foundation

public enum SelfPlayPractice {
    public static let lessonID="self-play"
    public static let train=WorkflowNodeAddress(domain:.control,localID:"train")
    public static func document(example:String="take-away")throws->WorkflowDocument {
        let rule=try GameRuleConfiguration.example(example)
        guard case .array(let observations)=rule["observations"]else{throw DataFlowContractError("Missing observations.")}
        let n=observations.count,a=example=="take-away" ? 3:example=="placement" ? 9:7
        func address(_ d:WorkflowDomain,_ id:String)->WorkflowNodeAddress{.init(domain:d,localID:id)}
        func model(_ id:String,_ title:String,_ type:String,_ x:Double,_ y:Double,_ config:[String:JSONValue])->WorkflowNode{.init(address:address(.model,id),name:title,x:x,y:y,payload:.modelPrimitive(type:type,configurationVersion:2,config:config))}
        var c=SelfPlayConfiguration.defaults;c["iterations"]=3;c["games"]=4;c["updates"]=8;c["batch"]=16;c["evaluationPairs"]=2
        var doc=WorkflowDocument(projectID:UUID().uuidString,name:"Learn from bounded self-play",nodes:[
            .init(address:address(.data,"rules"),name:"Authored game",x:40,y:40,payload:.dataOperator(operatorID:GameRuleConfiguration.operatorID,config:["definition":.object(rule)])),
            model("observation","Game observations","input",40,340,["features":.integer(Int64(n))]),
            model("shared","Shared features","linear",320,340,["in_features":.integer(Int64(n)),"out_features":8,"bias":true]),
            model("hidden","Shared activation","tanh",600,340,[:]),
            model("policy","Action logits","linear",880,250,["in_features":8,"out_features":.integer(Int64(a)),"bias":true]),
            model("scalar","Value projection","linear",880,550,["in_features":8,"out_features":1,"bias":true]),
            model("value","Acting-player value","tanh",1160,550,[:]),
            model("output","Game policy and value","output",1440,340,["task":.string("gamePolicyValue"),"taskVersion":1]),
            .init(address:train,name:"Learn Self-play",x:1780,y:40,payload:.train(config:["selfPlay":.object(c)])),
            .init(address:address(.data,"iterations"),name:"Inspect iterations",x:2120,y:40,payload:.dataOperator(operatorID:"kwip.data.view",config:[:])),
            .init(address:address(.data,"evaluation"),name:"Inspect evaluation",x:2120,y:340,payload:.dataOperator(operatorID:"kwip.data.view",config:[:])),
            .init(address:address(.control,"session"),name:"Test learned policy",x:2120,y:640,payload:.gameSession(config:["episodes":8,"policies":.object(["A":.object(["source":.string("saved")]),"B":.object(["source":.string("random")])])])),
            .init(address:address(.control,"save"),name:"Save paired model",x:2440,y:640,payload:.saveModel(config:[:]))],edges:[])
        func edge(_ id:String,_ s:WorkflowNodeAddress,_ out:String,_ t:WorkflowNodeAddress,_ port:String,_ kind:WorkflowEdgeKind){doc.edges.append(.init(id:id,from:.init(address:s,port:out),to:.init(address:t,port:port),kind:kind))}
        for (i,e) in [("observation","shared","in"),("shared","hidden","in"),("hidden","policy","in"),("hidden","scalar","in"),("scalar","value","in"),("policy","output","policy"),("value","output","value")].enumerated(){edge("tensor-\(i)",address(.model,e.0),"out",address(.model,e.1),e.2,.tensor)}
        edge("rules",address(.data,"rules"),"definition",train,"definition",.table);edge("model",address(.model,"output"),"definition",train,"model",.modelDefinition)
        edge("iterations",train,"iterations",address(.data,"iterations"),"table",.table);edge("evaluation",train,"evaluation",address(.data,"evaluation"),"table",.table)
        edge("session-rules",address(.data,"rules"),"definition",address(.control,"session"),"definition",.table);edge("saved",train,"model",address(.control,"session"),"policyA",.frozenModel);edge("save",train,"model",address(.control,"save"),"model",.frozenModel)
        doc.schemaVersion=23;doc.modelInputs=["output":"observation"];doc.modelSourceVersions=["output":2];return doc
    }
}
