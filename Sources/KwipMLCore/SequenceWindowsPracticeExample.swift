import Foundation

public enum SequenceWindowsPracticeExample {
    public static let exampleID="sequence-windows"
    public static let lessonID=exampleID
    public static let observationsResource="sequence-windows-observations"
    public static let address=WorkflowNodeAddress(domain:.data,localID:"windows")
    public static func document(projectID:String=UUID().uuidString)throws->WorkflowDocument {
        var result=try TextEncodingPracticeExample.document(projectID:projectID)
        result.name="Create and inspect next-token windows"
        var c=SequenceWindowsConfiguration.defaults;c["contextLength"]=8;c["stride"]=8;c["documentColumn"] = .string("document");c["orderColumn"] = .string("fragment")
        let columns=WorkflowNodeAddress(domain:.data,localID:"order-columns")
        result.nodes.append(.init(address:columns,name:"Integer fragment order",x:380,y:120,payload:.dataOperator(operatorID:"kwip.data.columns",config:CurrentPracticeConfiguration.columns([("document","keep"),("fragment","int64"),("text","keep")]))))
        for i in result.edges.indices where result.edges[i].from.address==WorkflowExamples.sourceAddress {result.edges[i].from.address=columns}
        result.edges.append(.init(from:.init(address:WorkflowExamples.sourceAddress,port:"table"),to:.init(address:columns,port:"table"),kind:.table))
        result.nodes.append(.init(address:address,name:"Next-token examples",x:760,y:120,payload:.dataOperator(operatorID:SequenceWindowsConfiguration.operatorID,config:c)))
        for i in result.nodes.indices {
            if result.nodes[i].address.localID=="view" {result.nodes[i].x=1400;result.nodes[i].name="Explore aligned pairs"}
            if result.nodes[i].address==TextEncodingPracticeExample.address {result.nodes[i].x=720}
            if result.nodes[i].address==address {result.nodes[i].x=1060}
            if result.nodes[i].address==WorkflowExamples.sourceAddress {result.nodes[i].name="Ordered document fragments"}
        }
        for i in result.edges.indices where result.edges[i].from.address==TextEncodingPracticeExample.address {result.edges[i].from = .init(address:address,port:"table")}
        result.edges.append(.init(id:"text-windows",from:.init(address:TextEncodingPracticeExample.address,port:"table"),to:.init(address:address,port:"table"),kind:.table))
        try WorkflowCodec.validate(result);return result
    }
}
