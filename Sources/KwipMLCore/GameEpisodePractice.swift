import Foundation

public enum GameEpisodePractice {
    public static let lessonID="game-episodes"
    public static let session=WorkflowNodeAddress(domain:.control,localID:"session")
    public static func document(example:String="take-away")throws->WorkflowDocument {
        let rule=try GameRuleConfiguration.example(example)
        let observations:Int;if case .array(let v)=rule["observations"] {observations=v.count}else{throw DataFlowContractError("Missing observations.")}
        let actions:Int=example=="take-away" ? 3:example=="placement" ? 9:7
        func a(_ d:WorkflowDomain,_ id:String)->WorkflowNodeAddress{.init(domain:d,localID:id)}
        var doc=WorkflowDocument(projectID:UUID().uuidString,name:"Run bounded game episodes",nodes:[
            .init(address:a(.data,"rules"),name:"Authored game",x:40,y:60,payload:.dataOperator(operatorID:GameRuleConfiguration.operatorID,config:["definition":.object(rule)])),
            .init(address:a(.model,"observation"),name:"Game observations",x:40,y:340,payload:.modelPrimitive(type:"input",configurationVersion:2,config:["features":.integer(Int64(observations))])),
            .init(address:a(.model,"policy-head"),name:"Action logits",x:320,y:340,payload:.modelPrimitive(type:"linear",configurationVersion:2,config:["in_features":.integer(Int64(observations)),"out_features":.integer(Int64(actions)),"bias":.bool(true)])),
            .init(address:a(.model,"output"),name:"Policy scores",x:600,y:340,payload:.modelPrimitive(type:"output",configurationVersion:2,config:[:])),
            .init(address:session,name:"Bounded game session",x:1180,y:80,payload:.gameSession(config:["policies":.object(["B":.object(["source":.string("random")])])])),
            .init(address:a(.data,"episodes"),name:"Inspect episodes",x:1520,y:60,payload:.dataOperator(operatorID:"kwip.data.view",config:[:])),
            .init(address:a(.data,"moves"),name:"Inspect moves",x:1520,y:340,payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))],edges:[])
        func edge(_ id:String,_ from:WorkflowNodeAddress,_ out:String,_ to:WorkflowNodeAddress,_ input:String,_ kind:WorkflowEdgeKind)->WorkflowEdge{.init(id:id,from:.init(address:from,port:out),to:.init(address:to,port:input),kind:kind)}
        doc.edges=[edge("rules",a(.data,"rules"),"definition",session,"definition",.table),edge("x",a(.model,"observation"),"out",a(.model,"policy-head"),"in",.tensor),edge("y",a(.model,"policy-head"),"out",a(.model,"output"),"in",.tensor),edge("policy",a(.model,"output"),"definition",session,"policyA",.modelDefinition),edge("episodes",session,"episodes",a(.data,"episodes"),"table",.table),edge("moves",session,"moves",a(.data,"moves"),"table",.table)]
        doc.schemaVersion=22;doc.modelInputs=["output":"observation"];doc.modelSourceVersions=["output":2]
        return doc
    }
}
