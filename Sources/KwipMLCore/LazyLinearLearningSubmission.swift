import Foundation

/// A fixture-bound path using real inferred-width parameters, never supplied answers.
public struct LazyLinearLearningSubmission: Encodable, Equatable {
    public let schemaVersion = 1
    public let kind = "lazyLinearGraph"
    public let graph: GraphDocument
    public init(graph: GraphDocument) throws { self.graph = try Self.validatedGraph(graph, complete: true) }
    static func validatedGraph(_ graph: GraphDocument, complete: Bool) throws -> GraphDocument {
        let result = try TensorLearningSubmission.validatedGraph(graph, complete: complete, allowLazyLinear: true)
        let count = result.nodes.filter { $0.type == "lazy_linear" }.count
        guard count <= 1, !complete || count == 1 else { throw LearningContractError("Use exactly one connected LazyLinear layer.") }
        return result
    }
    public func encoded() throws -> Data {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(self)
        guard data.count <= 2_000_000 else { throw LearningContractError("Submission exceeds 2 MB.") }
        return data
    }
    public static func decode(_ data: Data) throws -> Self {
        guard data.count <= 2_000_000 else { throw LearningContractError("Submission exceeds 2 MB.") }
        let raw = try JSONDecoder().decode(JSONValue.self, from: data)
        let envelope = try TensorLearningSubmission.checkedObject(raw, required: ["schemaVersion", "kind", "graph"])
        guard envelope["schemaVersion"]?.integerValue == 1, envelope["kind"]?.stringValue == "lazyLinearGraph" else { throw LearningContractError("Unsupported inferred-width submission.") }
        try TensorLearningSubmission.checkGraphFields(envelope["graph"]!)
        struct Envelope: Decodable { let graph: GraphDocument }
        return try Self(graph: JSONDecoder().decode(Envelope.self, from: data).graph)
    }
}
