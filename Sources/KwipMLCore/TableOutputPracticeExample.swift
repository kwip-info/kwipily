import Foundation

/// A real source-backed local write practice. No destination grant is embedded.
public enum TableOutputPracticeExample {
    public static let exampleID="table-output"
    public static let sourceResources:[PreparedWorkflowExample.SourceResource]=[.init(address:.init(domain:.data,localID:"source"),resource:"table-output-practice")]
    public static func resourceURL(_ name:String="table-output-practice")throws->URL {
        guard name == "table-output-practice",let url=Bundle.module.url(forResource:name,withExtension:"csv") else {throw DataFlowContractError("Table Output practice fixture is missing.")};return url
    }
    public static func databaseResourceURL()throws->URL {
        guard let url=Bundle.module.url(forResource:"table-output-practice",withExtension:"sqlite") else {throw DataFlowContractError("SQLite practice fixture is missing.")};return url
    }
    public static func document(projectID:String=UUID().uuidString)throws->WorkflowDocument {
        let source=WorkflowNodeAddress(domain:.data,localID:"source"),output=WorkflowNodeAddress(domain:.data,localID:"table-output")
        var input=CSVParsingConfiguration.newDefaults
        input["emptyField"] = .string("null")
        input["schemaOverrides"] = .object(["quantity":.string("Int64"),"price":.string("Float64")])
        let config:[String:JSONValue]=["filename":.string("prepared-orders.csv"),"separator":.string(";"),"columns":.array([.string("order_id"),.string("quantity"),.string("price"),.string("notes")]),"overwrite":.bool(false),"format":.string("csv"),"imagePolicy":.string("stop"),"options":.object([:])]
        let nodes:[WorkflowNode]=[.init(address:source,name:"Twelve synthetic orders",x:40,y:150,payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:input)),.init(address:output,name:"Save complete orders",x:370,y:150,payload:.dataOperator(operatorID:"kwip.data.outputDelimited",config:config))]
        let edges:[WorkflowEdge]=[.init(from:.init(address:source,port:"table"),to:.init(address:output,port:"table"),kind:.table)]
        let document=WorkflowDocument(projectID:projectID,name:"Table Output · complete local files",nodes:nodes,edges:edges)
        try WorkflowCodec.validate(document);return document
    }
}
