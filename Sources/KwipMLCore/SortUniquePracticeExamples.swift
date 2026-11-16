import Foundation

public enum SortPracticeExample {
    public static let exampleID="sort"
    public static let sourceResources:[PreparedWorkflowExample.SourceResource]=[.init(address:.init(domain:.data,localID:"source"),resource:"sort-practice")]
    public static func resourceURL(_ name:String="sort-practice")throws->URL {try OrderingPractice.resourceURL(name)}
    public static func document(projectID:String=UUID().uuidString)throws->WorkflowDocument {
        let keys=[SortConfiguration.Key(column:"customer",comparison:"casefold"),.init(column:"priority",direction:"descending",missing:"first")]
        return try OrderingPractice.document(projectID:projectID,tool:"sort",name:"Sort support requests",config:["keys":.array(keys.map(\.value))],outputs:["table"])
    }
}
public enum UniquePracticeExample {
    public static let exampleID="unique"
    public static let sourceResources:[PreparedWorkflowExample.SourceResource]=[.init(address:.init(domain:.data,localID:"source"),resource:"unique-practice")]
    public static func resourceURL(_ name:String="unique-practice")throws->URL {try OrderingPractice.resourceURL(name)}
    public static func document(projectID:String=UUID().uuidString)throws->WorkflowDocument {
        var config=UniqueConfiguration.defaults;config["keys"] = .array([.string("customer"),.string("region")]);config["comparison"] = .string("casefold")
        return try OrderingPractice.document(projectID:projectID,tool:"unique",name:"Unique customer observations",config:config,outputs:["unique","duplicates"])
    }
}
private enum OrderingPractice {
    static func resourceURL(_ name:String)throws->URL {
        guard ["sort-practice","unique-practice"].contains(name),let url=Bundle.module.url(forResource:name,withExtension:"csv") else {throw DataFlowContractError("Ordering practice fixture is missing.")};return url
    }
    static func document(projectID:String,tool:String,name:String,config:[String:JSONValue],outputs:[String])throws->WorkflowDocument {
        let source=WorkflowNodeAddress(domain:.data,localID:"source"),operation=WorkflowNodeAddress(domain:.data,localID:tool)
        var input=CSVParsingConfiguration.newDefaults;input["emptyField"] = .string("null")
        input["schemaOverrides"] = .object(["priority":.string("Int64")])
        var nodes:[WorkflowNode]=[.init(address:source,name:"Practice records",x:40,y:160,payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:input)),.init(address:operation,name:name,x:350,y:160,payload:.dataOperator(operatorID:"kwip.data."+tool,config:config))]
        var edges:[WorkflowEdge]=[.init(from:.init(address:source,port:"table"),to:.init(address:operation,port:"table"),kind:.table)]
        for (i,port) in outputs.enumerated() {
            let view=WorkflowNodeAddress(domain:.data,localID:"view"+String(i+1))
            nodes.append(.init(address:view,name:port == "table" ? "Sorted requests":port.capitalized,x:670,y:80+Double(i)*180,payload:.dataOperator(operatorID:"kwip.data.view",config:[:])))
            edges.append(.init(from:.init(address:operation,port:port),to:.init(address:view,port:"table"),kind:.table))
        }
        let document=WorkflowDocument(projectID:projectID,name:name,nodes:nodes,edges:edges);try WorkflowCodec.validate(document);return document
    }
}
