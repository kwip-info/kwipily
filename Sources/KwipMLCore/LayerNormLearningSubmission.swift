import Foundation

/// Explicit learned state and optional gradient target for fixture-shaped normalization graphs.
public struct LayerNormLearningSubmission: Encodable, Equatable {
    public let graph: GraphDocument
    public let parameters: [String:[String:LearningTensor]]
    public let gradientNodeID: String?
    public init(graph: GraphDocument, parameters: [String:[String:LearningTensor]], gradientNodeID: String?) throws {
        let canonical = try TensorLearningSubmission.validatedGraph(graph,complete:true,allowLayerNorm:true)
        let active = canonical.nodes.filter { $0.type == "layer_norm" && $0.config["elementwise_affine"]?.boolValue == true }
        guard Set(parameters.keys) == Set(active.map(\.id)), gradientNodeID == nil || active.contains(where: { $0.id == gradientNodeID }) else {
            throw LearningContractError("Provide parameters for exactly the affine-enabled normalization nodes and select an active node for parameter gradients.")
        }
        for node in active {
            let config = try LayerNormConfiguration(values:node.config)
            let values = parameters[node.id]!
            let keys: Set<String> = config.bias ? ["weight","bias"] : ["weight"]
            guard Set(values.keys) == keys, values.values.allSatisfy({ $0.wellFormed && $0.shape == config.normalizedShape }) else {
                throw LearningContractError("Normalization parameters must contain exactly the active finite weight/bias tensors with matching trailing dimensions.")
            }
        }
        self.graph=canonical;self.parameters=parameters;self.gradientNodeID=gradientNodeID
    }
    private enum CodingKeys: String, CodingKey { case schemaVersion,kind,graph,parameters,gradientNodeID }
    public func encode(to encoder: Encoder) throws {
        var c=encoder.container(keyedBy:CodingKeys.self)
        try c.encode(1,forKey:.schemaVersion);try c.encode("layerNormGraph",forKey:.kind)
        try c.encode(graph,forKey:.graph);try c.encode(parameters,forKey:.parameters)
        if let gradientNodeID {try c.encode(gradientNodeID,forKey:.gradientNodeID)} else {try c.encodeNil(forKey:.gradientNodeID)}
    }
    public func encoded() throws -> Data {
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes]
        let data=try encoder.encode(self)
        guard data.count<=2_000_000 else {throw LearningContractError("Submission exceeds 2 MB.")}
        return data
    }
    public static func decode(_ data: Data) throws -> Self {
        guard data.count<=2_000_000 else {throw LearningContractError("Submission exceeds 2 MB.")}
        let raw=try JSONDecoder().decode(JSONValue.self,from:data)
        let fields=try TensorLearningSubmission.checkedObject(raw,required:["schemaVersion","kind","graph","parameters","gradientNodeID"])
        guard fields["schemaVersion"]?.integerValue == 1,fields["kind"]?.stringValue == "layerNormGraph",
              case .object(let parameters)=fields["parameters"] else {throw LearningContractError("Unsupported normalization submission.")}
        try TensorLearningSubmission.checkGraphFields(fields["graph"]!)
        for value in parameters.values {
            guard case .object(let tensors)=value else {throw LearningContractError("Malformed normalization parameter object.")}
            for tensor in tensors.values {_ = try TensorLearningSubmission.checkedObject(tensor,required:["shape","values"])}
        }
        struct Envelope: Decodable {let graph:GraphDocument;let parameters:[String:[String:LearningTensor]];let gradientNodeID:String?}
        let envelope=try JSONDecoder().decode(Envelope.self,from:data)
        return try Self(graph:envelope.graph,parameters:envelope.parameters,gradientNodeID:envelope.gradientNodeID)
    }
}
