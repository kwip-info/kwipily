import Foundation

public enum PartitionPracticeExample {
    public static let exampleID="partition"
    public static let lessonID="partition"
    public static let observationsResource="partition-practice-observations"
    public static func observationsURL()throws->URL {
        guard let url=Bundle.module.url(forResource:observationsResource,withExtension:"csv") else {throw DataFlowContractError("Partition practice observations are missing.")}
        return url
    }
    public static let address=WorkflowNodeAddress(domain:.data,localID:"partition")
    public static var configuration:[String:JSONValue] {["strategy":.string("random"),"seed":42,"validationPercent":20,"testPercent":20]}
    public static func document(projectID:String=UUID().uuidString)throws->WorkflowDocument {
        var source=CSVParsingConfiguration.newDefaults
        source["schemaOverrides"] = .object(["time":.string("Int64"),"class":.string("Int64")])
        var nodes:[WorkflowNode]=[
            .init(address:WorkflowExamples.sourceAddress,name:"Thirty synthetic observations",x:80,y:240,payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:source)),
            .init(address:address,name:"Partition observations",x:420,y:240,payload:.dataOperator(operatorID:"kwip.data.partition",config:configuration))]
        var edges:[WorkflowEdge]=[.init(from:.init(address:WorkflowExamples.sourceAddress,port:"table"),to:.init(address:address,port:"table"),kind:.table)]
        for (index,role) in ["training","validation","test"].enumerated() {
            let view=WorkflowNodeAddress(domain:.data,localID:role+"-view")
            nodes.append(.init(address:view,name:"Explore \(role)",x:760,y:Double(80+index*220),payload:.dataOperator(operatorID:"kwip.data.view",config:[:])))
            edges.append(.init(from:.init(address:address,port:role),to:.init(address:view,port:"table"),kind:.table))
        }
        let document=WorkflowDocument(projectID:projectID,name:"Partition rows, entities and time",nodes:nodes,edges:edges)
        try WorkflowCodec.validate(document);return document
    }
}
