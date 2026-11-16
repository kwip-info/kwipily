import Foundation

public enum TextGenerationPracticeExample {
    public static let exampleID="text-generation"
    public static let address=WorkflowNodeAddress(domain:.control,localID:"generate")
    public static func document(projectID:String=UUID().uuidString,device:String="cpu")throws->WorkflowDocument {
        var document=try SequenceTrainingPracticeExample.document(projectID:projectID,device:device)
        document.name="Generate bounded next-token text"
        document.nodes.append(.init(address:address,name:"Continue a prompt",x:1600,y:200,payload:.generateText(config:["prompt":.string("aba"),"maxNewTokens":.integer(16)])))
        document.edges.append(.init(from:.init(address:WorkflowExamples.trainAddress,port:"model"),to:.init(address:address,port:"model"),kind:.frozenModel))
        try WorkflowCodec.validate(document);return document
    }
}
