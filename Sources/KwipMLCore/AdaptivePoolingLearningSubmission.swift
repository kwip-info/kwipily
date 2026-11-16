import Foundation

public struct AdaptivePoolingLearningSubmission:Encodable,Equatable {
    public let schemaVersion=1
    public let kind="adaptivePoolingGraph"
    public let graph:GraphDocument
    public init(graph:GraphDocument,task:AdaptivePoolingLearningTask)throws {self.graph=try Self.validatedGraph(graph,task:task,complete:true)}
    static func validatedGraph(_ graph:GraphDocument,task:AdaptivePoolingLearningTask,complete:Bool)throws->GraphDocument {
        let result=try TensorLearningSubmission.validatedGraph(graph,complete:complete,adaptivePoolingTask:task)
        let count=result.nodes.filter {$0.type == task.componentID}.count
        guard count<=1,!complete || count == 1 else {throw LearningContractError("Use exactly one connected requested pooling block.")}
        return result
    }
    public func encoded()throws->Data {let e=JSONEncoder();e.outputFormatting=[.sortedKeys,.withoutEscapingSlashes];let data=try e.encode(self);guard data.count<=2_000_000 else {throw LearningContractError("Submission exceeds 2 MB.")};return data}
    public static func decode(_ data:Data,task:AdaptivePoolingLearningTask)throws->Self {
        guard data.count<=2_000_000 else {throw LearningContractError("Submission exceeds 2 MB.")}
        let raw=try JSONDecoder().decode(JSONValue.self,from:data),fields=try TensorLearningSubmission.checkedObject(raw,required:["schemaVersion","kind","graph"])
        guard fields["schemaVersion"]?.integerValue == 1,fields["kind"]?.stringValue == "adaptivePoolingGraph" else {throw LearningContractError("Unsupported averaging/adaptive pooling submission.")}
        try TensorLearningSubmission.checkGraphFields(fields["graph"]!)
        struct Envelope:Decodable {let graph:GraphDocument}
        return try Self(graph:JSONDecoder().decode(Envelope.self,from:data).graph,task:task)
    }
}
