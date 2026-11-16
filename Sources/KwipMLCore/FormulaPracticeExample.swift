import Foundation
public enum FormulaPracticeExample {
    public static let exampleID="formula",lessonID="formula",observationsResource="formula-observations"
    public static let address=WorkflowNodeAddress(domain:.data,localID:"formula")
    public static func observationsURL()throws->URL {guard let url=Bundle.module.url(forResource:observationsResource,withExtension:"csv") else {throw DataFlowContractError("Formula practice observations are missing.")};return url}
    private static func operation(_ id:String,_ expression:String,_ outputs:[(String,String)])->JSONValue {.object(["id":.string(id),"expression":.string(expression),"outputs":.array(outputs.map{.object(["name":.string($0.0),"dtype":.string($0.1),"write":.string("new")])}),"nullPolicy":.string("preserve")])}
    public static var configuration:[String:JSONValue] {["operations":.array([
        operation("total","[price] * [quantity]",[("total","float64")]),
        operation("clean","regex_replace([name], \"\\\\s+\", \" \", \"all\", \"\")",[("clean_name","string")]),
        operation("extract","regex_extract([product_label], \"^([A-Z]+)-([0-9]+)$\", \"full\", \"null\", \"\", \"1\", \"2\")",[("product_code","string"),("batch","string")]),
        operation("match","regex_match([product_label], \"^ABC\", \"search\", \"\")",[("is_valid","bool")])]) ]}
    public static func document(projectID:String=UUID().uuidString)throws->WorkflowDocument {
        var source=CSVParsingConfiguration.newDefaults
        source["emptyField"] = .string("null");source["schemaOverrides"] = .object(["price":.string("Float64"),"quantity":.string("Int64")])
        let view=WorkflowNodeAddress(domain:.data,localID:"view")
        let document=WorkflowDocument(projectID:projectID,name:"Create columns with Formula and text patterns",nodes:[
            .init(address:WorkflowExamples.sourceAddress,name:"Eight synthetic orders",x:80,y:200,payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:source)),
            .init(address:address,name:"Create useful columns",x:420,y:200,payload:.dataOperator(operatorID:"kwip.data.formula",config:configuration)),
            .init(address:view,name:"Explore all calculated rows",x:760,y:200,payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))],edges:[
                .init(from:.init(address:WorkflowExamples.sourceAddress,port:"table"),to:.init(address:address,port:"table"),kind:.table),
                .init(from:.init(address:address,port:"table"),to:.init(address:view,port:"table"),kind:.table)])
        try WorkflowCodec.validate(document);return document
    }
}
