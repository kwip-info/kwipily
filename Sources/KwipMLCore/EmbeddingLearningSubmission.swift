import Foundation

/// The fixture owns table values; the learner authors the lookup and layout policy.
public struct EmbeddingLearningSubmission:Encodable,Equatable {
    public let schemaVersion=1
    public let kind="embeddingGraph"
    public let graph:GraphDocument
    public init(graph:GraphDocument)throws {
        let canonical=try TensorLearningSubmission.validatedGraph(graph,complete:true,allowEmbedding:true)
        let lookups=canonical.nodes.filter {$0.type == "embedding"}
        guard lookups.count == 1 else {throw LearningContractError("Use exactly one lookup with the assessment's supplied table.")}
        let configuration=try EmbeddingConfiguration(values:lookups[0].config)
        guard !configuration.frozen,!(configuration.sparse && configuration.scaleGradientByFrequency) else {
            throw LearningContractError("This assessment requires a trainable table and compatible gradient settings.")
        }
        self.graph=canonical
    }
    public func encoded()throws->Data {
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes]
        let data=try encoder.encode(self)
        guard data.count<=2_000_000 else {throw LearningContractError("Submission exceeds 2 MB.")}
        return data
    }
    public static func decode(_ data:Data)throws->Self {
        guard data.count<=2_000_000 else {throw LearningContractError("Submission exceeds 2 MB.")}
        let raw=try JSONDecoder().decode(JSONValue.self,from:data)
        let fields=try TensorLearningSubmission.checkedObject(raw,required:["schemaVersion","kind","graph"])
        guard fields["schemaVersion"]?.integerValue == 1,fields["kind"]?.stringValue == "embeddingGraph" else {throw LearningContractError("Unsupported lookup submission.")}
        try TensorLearningSubmission.checkGraphFields(fields["graph"]!)
        struct Envelope:Decodable {let graph:GraphDocument}
        return try Self(graph:JSONDecoder().decode(Envelope.self,from:data).graph)
    }
}
