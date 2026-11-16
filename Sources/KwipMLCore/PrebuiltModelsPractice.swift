import Foundation
public enum PrebuiltModelsPractice {
    public static let lessonID="prebuilt-models"
    public static let source=WorkflowNodeAddress(domain:.data,localID:"customers")
    public static let estimator=WorkflowNodeAddress(domain:.control,localID:"segments")
    public static func document(projectID:String=UUID().uuidString)throws->WorkflowDocument {
        let typed=WorkflowNodeAddress(domain:.data,localID:"numeric"),assignments=WorkflowNodeAddress(domain:.data,localID:"assignments"),profiles=WorkflowNodeAddress(domain:.data,localID:"profiles"),save=WorkflowNodeAddress(domain:.control,localID:"save")
        var config=PrebuiltModelConfiguration.defaults("kmeans");config["featureColumns"] = .array(["usage".json,"recency".json,"breadth".json])
        let columns:[String:JSONValue]=["rows":.array(["customer_id","usage","recency","breadth"].map{.object(["source":.string($0),"included":true,"outputName":.string($0),"targetType":.string($0=="customer_id" ? "keep":"float64")])}),"castFailure":.string("error"),"trimText":false,"newColumns":.string("drop"),"missingPolicy":.string("preserve")]
        var nodes:[WorkflowNode]=[.init(address:source,name:"Synthetic customer behavior",x:30,y:160,payload:.dataOperator(operatorID:DataFlowSourceOperators.tableInput,config:CSVParsingConfiguration.newDefaults)),.init(address:estimator,name:"Find behavior segments",x:690,y:160,payload:.estimator(config:config))]
        nodes.append(.init(address:typed,name:"Declare numeric features",x:350,y:160,payload:.dataOperator(operatorID:"kwip.data.columns",config:columns)))
        nodes += [.init(address:assignments,name:"Inspect every assignment",x:1030,y:30,payload:.dataOperator(operatorID:"kwip.data.view",config:[:])),.init(address:profiles,name:"Compare segment profiles",x:1030,y:350,payload:.dataOperator(operatorID:"kwip.data.view",config:[:])),.init(address:save,name:"Save prepared-input model",x:1370,y:160,payload:.saveModel(config:[:]))]
        func table(_ id:String,_ a:WorkflowNodeAddress,_ ap:String,_ b:WorkflowNodeAddress,_ bp:String)->WorkflowEdge {.init(id:id,from:.init(address:a,port:ap),to:.init(address:b,port:bp),kind:.table)}
        var d=WorkflowDocument(projectID:projectID,name:"Explore behavior segments",nodes:nodes,edges:[table("source-numeric",source,"table",typed,"table"),table("numeric-fit",typed,"table",estimator,"training"),table("fit-assignments",estimator,"predictions",assignments,"table"),table("fit-profiles",estimator,"profiles",profiles,"table"),.init(id:"fit-save",from:.init(address:estimator,port:"model"),to:.init(address:save,port:"model"),kind:.frozenModel)])
        d.schemaVersion=23;try WorkflowCodec.validate(d);return d
    }
}
private extension String {var json:JSONValue {.string(self)}}
