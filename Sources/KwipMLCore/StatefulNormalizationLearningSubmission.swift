import Foundation

public struct StatefulNormalizationLearningSubmission:Encodable,Equatable {
    public let schemaVersion=1
    public let kind="statefulNormalizationGraph"
    public let graph:GraphDocument
    public let parameters:[String:[String:LearningTensor]]
    public let gradientNodeID:String?
    public init(graph:GraphDocument,parameters:[String:[String:LearningTensor]],gradientNodeID:String?,task:StatefulNormalizationLearningTask)throws {
        let canonical=try Self.validatedGraph(graph,task:task,complete:true)
        try Self.validateParameters(graph:canonical,parameters:parameters,gradientNodeID:gradientNodeID,task:task,complete:true)
        self.graph=canonical;self.parameters=parameters;self.gradientNodeID=gradientNodeID
    }
    /// Park bounded draft values while controls are inactive; strict submitted state stays separate.
    public static func validateParameters(graph:GraphDocument,parameters:[String:[String:LearningTensor]],gradientNodeID:String?,task:StatefulNormalizationLearningTask,complete:Bool)throws {
        var known:[String:StatefulNormalizationConfiguration]=[:]
        for node in graph.nodes where node.type == task.componentID {known[node.id]=try StatefulNormalizationConfiguration(componentID:node.type,values:node.config)}
        let active=known.filter {$0.value.affine}
        let allowed=complete ? Set(active.keys):Set(known.keys)
        guard Set(parameters.keys).isSubset(of:allowed),!complete || Set(parameters.keys)==Set(active.keys),gradientNodeID == nil || allowed.contains(gradientNodeID!),!complete || active.isEmpty || gradientNodeID != nil else {throw LearningContractError("Keep channel values and the chosen influence target on existing normalization blocks; checking requires their active values.")}
        for (id,p) in parameters {
            let c=known[id]!,keys:Set<String>=complete ? (c.bias ? ["weight","bias"]:["weight"]):["weight","bias"]
            guard Set(p.keys).isSubset(of:keys),!complete || Set(p.keys)==keys,p.values.allSatisfy({$0.wellFormed && (complete ? $0.shape==[c.channelCount]:$0.shape.count==1 && (1...512).contains($0.shape[0]))}) else {throw LearningContractError(complete ? "Active scale and optional shift must match the current channel count. Initialize matching values after changing Channels.":"Retained channel values must be finite one-dimensional vectors with 1 to 512 entries.")}
        }
    }
    /// Project an editor bank into only active state; direct submissions still reject inactive extras.
    public static func fromDraft(graph:GraphDocument,parameters:[String:[String:LearningTensor]],gradientNodeID:String?,task:StatefulNormalizationLearningTask)throws->Self {
        let canonical=try validatedGraph(graph,task:task,complete:false)
        try validateParameters(graph:canonical,parameters:parameters,gradientNodeID:gradientNodeID,task:task,complete:false)
        var submitted:[String:[String:LearningTensor]]=[:],active=Set<String>()
        for node in canonical.nodes where node.type == task.componentID {
            let c=try StatefulNormalizationConfiguration(componentID:node.type,values:node.config)
            guard c.affine else {continue};active.insert(node.id)
            if let values=parameters[node.id] {submitted[node.id]=values.filter {$0.key == "weight" || (c.bias && $0.key == "bias")}}
        }
        return try Self(graph:canonical,parameters:submitted,gradientNodeID:gradientNodeID.flatMap {active.contains($0) ? $0:nil},task:task)
    }
    private enum CodingKeys:String,CodingKey {case schemaVersion,kind,graph,parameters,gradientNodeID}
    public func encode(to encoder:Encoder)throws {
        var c=encoder.container(keyedBy:CodingKeys.self)
        try c.encode(schemaVersion,forKey:.schemaVersion);try c.encode(kind,forKey:.kind);try c.encode(graph,forKey:.graph);try c.encode(parameters,forKey:.parameters)
        if let gradientNodeID {try c.encode(gradientNodeID,forKey:.gradientNodeID)} else {try c.encodeNil(forKey:.gradientNodeID)}
    }
    static func validatedGraph(_ graph:GraphDocument,task:StatefulNormalizationLearningTask,complete:Bool)throws->GraphDocument {
        guard graph.schemaVersion == 2,graph.execution == nil,graph.nodes.count<=32,graph.edges.count<=31 else {throw LearningContractError("Channel-normalization lessons require a bounded schema-2 graph without training settings.")}
        var copy=graph,nodes:[String:GraphNode]=[:],sources:[String]=[],outputs:[String]=[]
        for index in copy.nodes.indices {
            var node=copy.nodes[index]
            guard !node.id.isEmpty,node.id.count<=128,nodes[node.id] == nil,node.x.isFinite,node.y.isFinite else {throw LearningContractError("Invalid channel-normalization component identity or canvas position.")}
            switch node.type {
            case "input","output":
                guard node.config.isEmpty else {throw LearningContractError("Channel-normalization lesson boundaries have no settings.")}
                if node.type == "input" {sources.append(node.id)} else {outputs.append(node.id)}
            case task.componentID:node.config=try StatefulNormalizationConfiguration(componentID:node.type,values:node.config).values
            default:node.config=try TensorTransformConfiguration(operation:node.type,values:node.config).values
            }
            copy.nodes[index]=node;nodes[node.id]=node
        }
        guard sources.count<=1,outputs.count<=1,!complete || (sources.count == 1 && outputs.count == 1 && copy.nodes.filter({$0.type == task.componentID}).count == 1) else {throw LearningContractError("Connect Input to \(task.componentTitle), then connect the block to Model Output.")}
        var incoming:[String:GraphEdge]=[:],successors:[String:[String]]=[:],ids=Set<String>()
        for edge in copy.edges {
            guard !edge.id.isEmpty,ids.insert(edge.id).inserted,let source=nodes[edge.source],let target=nodes[edge.target],source.type != "output",target.type != "input",edge.sourcePort == "out",edge.targetPort == "in",incoming[target.id] == nil else {throw LearningContractError("Connect each channel-normalization input once using its out/in ports.")}
            incoming[target.id]=edge;successors[source.id,default:[]].append(target.id)
        }
        for node in copy.nodes where complete {guard (node.type == "input" || incoming[node.id] != nil),(node.type == "output" || !(successors[node.id] ?? []).isEmpty) else {throw LearningContractError("Complete the channel-normalization connections and remove unused blocks.")}}
        var pending=nodes.mapValues {incoming[$0.id] == nil ? 0:1},queue=copy.nodes.filter {pending[$0.id] == 0}.map(\.id),order:[String]=[]
        while !queue.isEmpty {let id=queue.removeFirst();order.append(id);for child in successors[id,default:[]] {pending[child,default:0]-=1;if pending[child] == 0 {queue.append(child)}}}
        guard order.count == nodes.count else {throw LearningContractError("Remove cycles from the channel-normalization graph.")}
        var layouts:[String:BatchedTensorLayout]=[:]
        for id in order where complete {
            let node=nodes[id]!
            if node.type == "input" {layouts[id]=try .init(sample:task.sampleShape)}
            else if let edge=incoming[id],let input=layouts[edge.source] {
                if node.type == task.componentID {layouts[id]=try StatefulNormalizationConfiguration(componentID:node.type,values:node.config).resolved(layout:input)}
                else if node.type == "output" {layouts[id]=input}
                else {layouts[id]=try TensorTransformConfiguration(operation:node.type,values:node.config).outputLayout(for:input)}
            }
        }
        return copy
    }
    public func encoded()throws->Data {let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes];let data=try encoder.encode(self);guard data.count<=2_000_000 else {throw LearningContractError("Submission exceeds 2 MB.")};return data}
    public static func decode(_ data:Data,task:StatefulNormalizationLearningTask)throws->Self {
        guard data.count<=2_000_000 else {throw LearningContractError("Submission exceeds 2 MB.")}
        let raw=try JSONDecoder().decode(JSONValue.self,from:data),envelope=try TensorLearningSubmission.checkedObject(raw,required:["schemaVersion","kind","graph","parameters","gradientNodeID"])
        guard envelope["schemaVersion"]?.integerValue == 1,envelope["kind"]?.stringValue == "statefulNormalizationGraph" else {throw LearningContractError("Unsupported channel-normalization submission.")}
        let graph=try TensorLearningSubmission.checkedObject(envelope["graph"]!,required:["schemaVersion","name","nodes","edges"])
        guard case .array(let nodes)=graph["nodes"],case .array(let edges)=graph["edges"],nodes.count<=32,edges.count<=31 else {throw LearningContractError("Channel-normalization graph exceeds lesson bounds.")}
        for node in nodes {_ = try TensorLearningSubmission.checkedObject(node,required:["id","type","name","x","y","config"])}
        for edge in edges {_ = try TensorLearningSubmission.checkedObject(edge,required:["id","source","sourcePort","target","targetPort"])}
        guard case .object(let parameterObjects)=envelope["parameters"] else {throw LearningContractError("Provide an explicit channel parameter map.")}
        for value in parameterObjects.values {
            guard case .object(let tensors)=value else {throw LearningContractError("Malformed parameter object.")}
            for tensor in tensors.values {_ = try TensorLearningSubmission.checkedObject(tensor,required:["shape","values"])}
        }
        struct Envelope:Decodable {let graph:GraphDocument;let parameters:[String:[String:LearningTensor]];let gradientNodeID:String?}
        let value=try JSONDecoder().decode(Envelope.self,from:data)
        return try Self(graph:value.graph,parameters:value.parameters,gradientNodeID:value.gradientNodeID,task:task)
    }
}
