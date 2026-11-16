import Foundation
public enum ReshapeTablePracticeExample {
    public static let exampleID="reshapeTable",wideExampleID="reshapeTableWide"
    public static let address=WorkflowNodeAddress(domain:.data,localID:"reshape")
    public static let observationsResource="reshape-long-practice",wideResource="reshape-wide-practice"
    public static func resourceURL(wide:Bool=false)throws->URL {
        guard let url=Bundle.module.url(forResource:wide ? wideResource:observationsResource,withExtension:"csv") else {throw DataFlowContractError("Reshape Table practice source is missing.")};return url
    }
    public static func document(projectID:String=UUID().uuidString,wide:Bool=false)throws->WorkflowDocument {
        var source=CSVParsingConfiguration.newDefaults
        source["emptyField"] = .string("null")
        source["schemaOverrides"] = .object(wide ? ["sales":.string("Int64"),"units":.string("Int64")]:["jan":.string("Int64"),"feb":.string("Int64")])
        var config=ReshapeTableConfiguration.defaults
        if wide {
            config["mode"] = .string("wide");config["rowKeys"] = .array([.string("customer")]);config["columnKey"] = .string("month")
            config["values"] = .array([.object(["column":.string("sales"),"aggregation":.string("sum"),"prefix":.string("sales_"),"nullPolicy":.string("skip")]),.object(["column":.string("units"),"aggregation":.string("max"),"prefix":.string("units_"),"nullPolicy":.string("skip")])])
        } else {config["idColumns"] = .array([.string("customer")]);config["valueColumns"] = .array([.string("jan"),.string("feb")])}
        let view=WorkflowNodeAddress(domain:.data,localID:"view")
        let nodes:[WorkflowNode]=[
            .init(address:WorkflowExamples.sourceAddress,name:wide ? "Monthly observations":"Monthly fields",x:40,y:120,payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:source)),
            .init(address:address,name:wide ? "One row per customer":"One row per month",x:360,y:120,payload:.dataOperator(operatorID:"kwip.data.reshapeTable",config:config)),
            .init(address:view,name:"Explore reshaped observations",x:680,y:120,payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))]
        let edges:[WorkflowEdge]=[
            .init(from:.init(address:WorkflowExamples.sourceAddress,port:"table"),to:.init(address:address,port:"table"),kind:.table),
            .init(from:.init(address:address,port:"table"),to:.init(address:view,port:"table"),kind:.table)]
        let document=WorkflowDocument(projectID:projectID,name:wide ? "Reshape monthly observations to Wide":"Reshape monthly fields to Long",nodes:nodes,edges:edges)
        try WorkflowCodec.validate(document);return document
    }
}
