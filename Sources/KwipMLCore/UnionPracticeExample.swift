import Foundation

public enum UnionPracticeExample {
    public static let exampleID="union"
    public static let address=WorkflowNodeAddress(domain:.data,localID:"union")
    public static let sources=["union-january","union-february","union-march"]
    public static func resourceURL(_ name:String)throws->URL {
        guard sources.contains(name),let url=Bundle.module.url(forResource:name,withExtension:"csv") else {throw DataFlowContractError("Union practice source is missing.")};return url
    }
    public static var sourceResources:[PreparedWorkflowExample.SourceResource] {sources.enumerated().map {.init(address:.init(domain:.data,localID:"source\($0.offset+1)"),resource:$0.element)}}
    public static func document(projectID:String=UUID().uuidString)throws->WorkflowDocument {
        var nodes:[WorkflowNode]=[];var edges:[WorkflowEdge]=[]
        for (index,source) in sourceResources.enumerated() {
            var config=CSVParsingConfiguration.newDefaults
            config["emptyField"] = .string("null")
            config["schemaOverrides"] = .object(["order_id":.string("Int64"),"amount":.string("Float64")])
            nodes.append(.init(address:source.address,name:["January orders","February orders","March orders"][index],x:40,y:40+Double(index)*160,payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:config)))
            edges.append(.init(from:.init(address:source.address,port:"table"),to:.init(address:address,port:"table\(index+1)"),kind:.table))
        }
        var config=UnionConfiguration.defaults;config["inputs"] = .array([.string("table1"),.string("table2"),.string("table3")])
        let view=WorkflowNodeAddress(domain:.data,localID:"view")
        nodes.append(.init(address:address,name:"All monthly orders",x:360,y:200,payload:.dataOperator(operatorID:"kwip.data.union",config:config)))
        nodes.append(.init(address:view,name:"Explore appended orders",x:680,y:200,payload:.dataOperator(operatorID:"kwip.data.view",config:[:])))
        edges.append(.init(from:.init(address:address,port:"table"),to:.init(address:view,port:"table"),kind:.table))
        let document=WorkflowDocument(projectID:projectID,name:"Union monthly tables in order",nodes:nodes,edges:edges)
        try WorkflowCodec.validate(document);return document
    }
}
