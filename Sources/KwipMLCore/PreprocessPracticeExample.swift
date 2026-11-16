import Foundation
public enum PreprocessPracticeExample {
    public static let exampleID="preprocess",lessonID="preprocess",observationsResource="preprocess-practice-observations"
    public static let address=WorkflowNodeAddress(domain:.data,localID:"preprocess")
    public static func observationsURL()throws->URL {
        guard let url=Bundle.module.url(forResource:observationsResource,withExtension:"csv") else {throw DataFlowContractError("Preprocess practice observations are missing.")};return url
    }
    public static var configuration:PreprocessConfiguration {
        var x=PreprocessConfiguration.Column(column:"x");x.missing="mean";x.scaling="standard"
        var city=PreprocessConfiguration.Column(column:"city");city.missing="mostFrequent";city.tiePolicy="smallest";city.encoding="oneHot";city.unknownPolicy="ignore"
        return .init(columns:[x,city])
    }
    public static func document(projectID:String=UUID().uuidString)throws->WorkflowDocument {
        var source=CSVParsingConfiguration.newDefaults;source["emptyField"] = .string("null");source["schemaOverrides"] = .object(["time":.string("Int64"),"x":.string("Float64"),"target":.string("Int64")])
        let partition=WorkflowNodeAddress(domain:.data,localID:"partition")
        let config:[String:JSONValue] = ["strategy":.string("time"),"timeColumn":.string("time"),"timeEncoding":.string("epochMilliseconds"),"trainingEnd":5000,"validationEnd":8000,"embargoMilliseconds":0,"entityColumns":.array([]),"entityPolicy":.string("allowAcrossTime")]
        var nodes:[WorkflowNode]=[
            .init(address:WorkflowExamples.sourceAddress,name:"Twelve synthetic observations",x:40,y:280,payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:source)),
            .init(address:partition,name:"Reserve later observations",x:340,y:280,payload:.dataOperator(operatorID:"kwip.data.partition",config:config)),
            .init(address:address,name:"Prepare all three populations",x:660,y:280,payload:.dataOperator(operatorID:"kwip.data.preprocess",config:configuration.values))]
        var edges:[WorkflowEdge]=[.init(from:.init(address:WorkflowExamples.sourceAddress,port:"table"),to:.init(address:partition,port:"table"),kind:.table)]
        for (index,role) in ["training","validation","test"].enumerated() {
            edges.append(.init(from:.init(address:partition,port:role),to:.init(address:address,port:role),kind:.table))
            let view=WorkflowNodeAddress(domain:.data,localID:role+"-view")
            nodes.append(.init(address:view,name:"Explore prepared \(role)",x:1000,y:Double(80+index*240),payload:.dataOperator(operatorID:"kwip.data.view",config:[:])))
            edges.append(.init(from:.init(address:address,port:role),to:.init(address:view,port:"table"),kind:.table))
        }
        let document=WorkflowDocument(projectID:projectID,name:"Prepare three populations with one tool",nodes:nodes,edges:edges)
        try WorkflowCodec.validate(document);return document
    }
    public static let cleanupExampleID="preprocess-cleanup",cleanupObservationsResource="preprocess-cleanup-observations"
    public static func cleanupObservationsURL()throws->URL {
        guard let url=Bundle.module.url(forResource:cleanupObservationsResource,withExtension:"csv") else {throw DataFlowContractError("Preprocess cleanup observations are missing.")};return url
    }
    public static var cleanupConfiguration:PreprocessConfiguration {
        var text=PreprocessConfiguration.Column(column:"name")
        text.unicodeNormalization="NFKC";text.removeControls=true;text.removePunctuation=true;text.removeDigits=true
        text.whitespace="collapse";text.casePolicy="lower";text.emptyAfterCleanup="missing";text.missing="drop"
        var amount=PreprocessConfiguration.Column(column:"amount");amount.missing="constant";amount.value = .number(0)
        return .init(mode:"table",columns:[text,amount])
    }
    public static func cleanupDocument(projectID:String=UUID().uuidString)throws->WorkflowDocument {
        var source=CSVParsingConfiguration.newDefaults
        source["emptyField"] = .string("null")
        source["schemaOverrides"] = .object(["name":.string("String"),"amount":.string("Float64")])
        let view=WorkflowNodeAddress(domain:.data,localID:"cleanup-view")
        let nodes:[WorkflowNode]=[
            .init(address:WorkflowExamples.sourceAddress,name:"Text cleanup observations",x:40,y:280,payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:source)),
            .init(address:address,name:"Clean names and fixed amounts",x:380,y:280,payload:.dataOperator(operatorID:"kwip.data.preprocess",config:cleanupConfiguration.values)),
            .init(address:view,name:"Compare retained clean rows",x:720,y:280,payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))]
        let edges:[WorkflowEdge]=[
            .init(from:.init(address:WorkflowExamples.sourceAddress,port:"table"),to:.init(address:address,port:"table"),kind:.table),
            .init(from:.init(address:address,port:"table"),to:.init(address:view,port:"table"),kind:.table)]
        let document=WorkflowDocument(projectID:projectID,name:"Clean text with fixed Table rules",nodes:nodes,edges:edges)
        try WorkflowCodec.validate(document);return document
    }

}
