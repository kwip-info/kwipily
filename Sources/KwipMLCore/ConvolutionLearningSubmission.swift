import Foundation

public struct ConvolutionLearningSubmission: Encodable, Equatable {
    public let schemaVersion=1
    public let kind="convolutionGraph"
    public let graph:GraphDocument
    public init(graph:GraphDocument,task:ConvolutionLearningTask)throws { self.graph=try Self.validatedGraph(graph,task:task,complete:true) }
    static func validatedGraph(_ graph:GraphDocument,task:ConvolutionLearningTask,complete:Bool)throws->GraphDocument {
        let canonical=try TensorLearningSubmission.validatedGraph(graph,complete:complete,convolutionTask:task)
        let count=canonical.nodes.filter {$0.type == task.componentID}.count
        guard count<=1,!complete || count == 1 else {throw LearningContractError("Use exactly one connected \(task.componentID) block with the supplied shared kernel.")}
        return canonical
    }
    public func encoded()throws->Data {
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes]
        let data=try encoder.encode(self);guard data.count<=2_000_000 else {throw LearningContractError("Submission exceeds 2 MB.")};return data
    }
    public static func decode(_ data:Data,task:ConvolutionLearningTask)throws->Self {
        guard data.count<=2_000_000 else {throw LearningContractError("Submission exceeds 2 MB.")}
        let raw=try JSONDecoder().decode(JSONValue.self,from:data),envelope=try TensorLearningSubmission.checkedObject(raw,required:["schemaVersion","kind","graph"])
        guard envelope["schemaVersion"]?.integerValue == 1,envelope["kind"]?.stringValue == "convolutionGraph" else {throw LearningContractError("Unsupported shared-kernel submission.")}
        try TensorLearningSubmission.checkGraphFields(envelope["graph"]!)
        struct Envelope:Decodable {let graph:GraphDocument}
        return try Self(graph:JSONDecoder().decode(Envelope.self,from:data).graph,task:task)
    }
}
