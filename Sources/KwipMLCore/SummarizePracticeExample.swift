import Foundation

public enum SummarizePracticeExample {
    public static let exampleID="summarize",observationsResource="summarize-practice-orders"
    public static let address=WorkflowNodeAddress(domain:.data,localID:"summary")
    public static func observationsURL()throws->URL {
        guard let url=Bundle.module.url(forResource:observationsResource,withExtension:"csv") else {throw DataFlowContractError("Summarize practice orders are missing.")};return url
    }
    public static func document(projectID:String=UUID().uuidString)throws->WorkflowDocument {
        var source=CSVParsingConfiguration.newDefaults
        source["emptyField"] = .string("null")
        source["schemaOverrides"] = .object(["quantity":.string("Int64"),"amount":.string("Float64")])
        let config:[String:JSONValue]=["groupBy":.array([.string("order_id")]),"nullKeys":.string("error"),"maxRows":.integer(100000),"aggregations":.array([
            .object(["op":.string("countRows"),"output":.string("line_count")]),
            .object(["op":.string("sum"),"column":.string("amount"),"output":.string("total_amount"),"nullPolicy":.string("error")]),
            .object(["op":.string("joinText"),"column":.string("item"),"output":.string("items"),"nullPolicy":.string("skip"),"separator":.string("\n"),"distinct":.bool(false),"skipEmpty":.bool(false)])])]
        let view=WorkflowNodeAddress(domain:.data,localID:"view")
        let nodes:[WorkflowNode]=[
            .init(address:WorkflowExamples.sourceAddress,name:"Eight order lines",x:40,y:120,payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:source)),
            .init(address:address,name:"One row per order",x:360,y:120,payload:.dataOperator(operatorID:"kwip.data.aggregate",config:config)),
            .init(address:view,name:"Explore order summaries",x:680,y:120,payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))]
        let edges:[WorkflowEdge]=[
            .init(from:.init(address:WorkflowExamples.sourceAddress,port:"table"),to:.init(address:address,port:"table"),kind:.table),
            .init(from:.init(address:address,port:"table"),to:.init(address:view,port:"table"),kind:.table)]
        let document=WorkflowDocument(projectID:projectID,name:"Summarize orders and join item text",nodes:nodes,edges:edges)
        try WorkflowCodec.validate(document);return document
    }
}
