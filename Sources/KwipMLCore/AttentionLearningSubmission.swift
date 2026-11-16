import Foundation

public struct AttentionLearningSubmission: Encodable, Equatable {
    public let schemaVersion = 1
    public let kind = "attentionGraph"
    public let graph: GraphDocument
    public init(graph: GraphDocument, task: AttentionLearningTask) throws { self.graph = try Self.validatedGraph(graph, task: task, complete: true) }
    public static func validatedGraph(_ graph: GraphDocument, task: AttentionLearningTask, complete: Bool) throws -> GraphDocument {
        guard graph.schemaVersion == 2, graph.execution == nil, graph.nodes.count <= 7, graph.edges.count <= 9, graph.name.count <= 256 else { throw LearningContractError("Use a bounded attention lesson graph without training settings.") }
        var copy = graph, nodes: [String: GraphNode] = [:], sources: [String: String] = [:], outputs: [String: String] = [:], operatorID: String?, maskID: String?, config: AttentionConfiguration?, mask: AttentionConfiguration?
        for i in copy.nodes.indices {
            var node = copy.nodes[i]
            guard !node.id.isEmpty, node.id.count <= 128, node.name.count <= 256, nodes[node.id] == nil, node.x.isFinite, node.y.isFinite else { throw LearningContractError("Use unique attention block identities and finite positions.") }
            if node.type == "input" { guard Set(node.config.keys) == ["fixture"], let role = node.config["fixture"]?.stringValue, task.sourceRoles.contains(role), sources[role] == nil else { throw LearningContractError("Use one Input for each supplied sequence role.") }; sources[role] = node.id }
            else if node.type == "output" { guard Set(node.config.keys) == ["role"], let role = node.config["role"]?.stringValue, task.outputRoles.contains(role), outputs[role] == nil else { throw LearningContractError("Use one Output for each result role.") }; outputs[role] = node.id }
            else if node.type == task.componentID, operatorID == nil { config = try .init(componentID: node.type, values: node.config); node.config = config!.values; operatorID = node.id }
            else if node.type == "attention_mask", maskID == nil { mask = try .init(componentID: node.type, values: node.config); node.config = mask!.values; maskID = node.id }
            else { throw LearningContractError("Use the requested attention block, one Attention mask and their role boundaries.") }
            nodes[node.id] = node; copy.nodes[i] = node
        }
        var expected = Set<String>()
        func address(_ source: String, _ port: String, _ target: String, _ input: String) -> String { source+"."+port+">"+target+"."+input }
        if let op = operatorID { for (role, id) in sources { expected.insert(address(id,"out",op,role)) }; for (role,id) in outputs { expected.insert(address(op,role == "result" ? "out" : role,id,"in")) } }
        if let mid = maskID { for (i, role) in task.maskReferenceRoles.enumerated() { if let id = sources[role] { expected.insert(address(id,"out",mid,i == 0 ? "query" : "key")) } }; if let op = operatorID { expected.insert(address(mid,"mask",op,task.maskInputPort)); if task == .multihead { expected.insert(address(mid,"padding",op,"key_padding_mask")) } } }
        var ids = Set<String>(), targets = Set<String>(), actual = Set<String>()
        for edge in copy.edges {
            guard !edge.id.isEmpty, edge.id.count <= 128, ids.insert(edge.id).inserted, nodes[edge.source] != nil, nodes[edge.target] != nil, targets.insert(edge.target+"."+edge.targetPort).inserted, expected.contains(address(edge.source,edge.sourcePort,edge.target,edge.targetPort)) else { throw LearningContractError("Connect each sequence, mask and result to its matching named port.") }
            actual.insert(address(edge.source,edge.sourcePort,edge.target,edge.targetPort))
        }
        if complete {
            guard Set(sources.keys) == Set(task.sourceRoles), Set(outputs.keys) == Set(task.outputRoles), let config, let mask, operatorID != nil, maskID != nil, actual == expected else { throw LearningContractError("Connect all supplied sequences, both mask references, the supplied mask and every result Output.") }
            let requested = try AttentionConfiguration(componentID:task.componentID,values:task.configuration)
            guard config.configuredInputs == requested.configuredInputs else { throw LearningContractError("Use exactly the supplied sequence and mask input ports for this exercise.") }
            let c = config.values
            guard (c["dropout"]?.doubleValue ?? 0) == 0 else { throw LearningContractError("Use no dropout for this deterministic comparison exercise.") }
            if task == .multihead { guard c["embed_dim"]?.integerValue == 2, c["num_heads"]?.integerValue == 1, (c["kdim"]?.integerValue ?? 2) == 2, (c["vdim"]?.integerValue ?? 2) == 2, c["add_bias_kv"]?.boolValue == false, c["add_zero_attn"]?.boolValue == false, c["return_weights"]?.boolValue == true, c["average_weights"]?.boolValue == false, c["attention_mask_mode"]?.stringValue == "supplied", c["padding_mask_mode"]?.stringValue == "supplied" else { throw LearningContractError("Use width2/one group, supplied masks, no extra source positions and separate group weights.") } }
            else if task == .scaled { guard c["mask_mode"]?.stringValue == "supplied" else { throw LearningContractError("Connect the supplied mask for scaled attention.") } }
            else { guard c["d_model"]?.integerValue == 2, c["nhead"]?.integerValue == 1, c["dim_feedforward"]?.integerValue == 3, (c["num_layers"]?.integerValue ?? 1) == 1, (c["num_encoder_layers"]?.integerValue ?? 1) == 1, (c["num_decoder_layers"]?.integerValue ?? 1) == 1 else { throw LearningContractError("Use width2, one comparison group, one layer and inner width3 for the supplied learned tables.") }; let key = task == .encoder ? "mask_mode" : "memory_mask_mode"; guard c[key]?.stringValue == "supplied" else { throw LearningContractError("Connect the supplied blocked-connections mask.") }; if task != .encoder { guard c["target_mask_mode"]?.stringValue == "causal" else { throw LearningContractError("Allow the requested sequence to use only current and earlier positions.") } } }
            _ = try config.resolved(layouts: config.configuredInputs.map { port in
                if let shape = task.sampleShapes[port] { return try .init(sample: shape) }
                let q = task.sampleShapes[task.maskReferenceRoles[0]]!, k = task.sampleShapes[task.maskReferenceRoles[1]]!, l = q[q.count-2], s = k[k.count-2]
                return try .init(sample: port == "key_padding_mask" ? [s] : [l,s])
            })
            _ = try mask.resolved(layouts: task.maskReferenceRoles.map { try .init(sample: task.sampleShapes[$0]!) })
        }
        return copy
    }
    public func encoded() throws -> Data { let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys,.withoutEscapingSlashes]; let data = try encoder.encode(self); guard data.count <= 2_000_000 else { throw LearningContractError("Attention submission exceeds2MB.") }; return data }
    public static func decode(_ data: Data, task: AttentionLearningTask) throws -> Self {
        guard data.count <= 2_000_000 else { throw LearningContractError("Attention submission exceeds2MB.") }
        let raw = try JSONDecoder().decode(JSONValue.self,from:data), env = try TensorLearningSubmission.checkedObject(raw,required:["schemaVersion","kind","graph"])
        guard env["schemaVersion"]?.integerValue == 1, env["kind"]?.stringValue == "attentionGraph" else { throw LearningContractError("Unsupported attention submission.") }
        let graph = try TensorLearningSubmission.checkedObject(env["graph"]!,required:["schemaVersion","name","nodes","edges"])
        guard case .array(let nodes) = graph["nodes"], case .array(let edges) = graph["edges"] else { throw LearningContractError("Use attention blocks and connections.") }
        for node in nodes { _ = try TensorLearningSubmission.checkedObject(node,required:["id","type","name","x","y","config"]) }
        for edge in edges { _ = try TensorLearningSubmission.checkedObject(edge,required:["id","source","sourcePort","target","targetPort"]) }
        struct Envelope: Decodable { let graph: GraphDocument }
        return try .init(graph: JSONDecoder().decode(Envelope.self,from:data).graph,task:task)
    }
}
