import Foundation

public struct RemainingActivationLearningSubmission:Encodable,Equatable {
    public let graph:GraphDocument
    public let parameters:[String:[String:LearningTensor]]
    public let gradientNodeID:String?
    public init(graph:GraphDocument,parameters:[String:[String:LearningTensor]]=[:],gradientNodeID:String?=nil,task:RemainingActivationLearningTask)throws {
        self.graph=try Self.validatedGraph(graph,task:task,complete:true)
        try Self.validateParameters(graph:self.graph,parameters:parameters,gradientNodeID:gradientNodeID,task:task,complete:true)
        self.parameters=parameters;self.gradientNodeID=gradientNodeID
    }
    public static func validateParameters(graph:GraphDocument,parameters:[String:[String:LearningTensor]],gradientNodeID:String?,task:RemainingActivationLearningTask,complete:Bool)throws {
        let active=graph.nodes.filter {$0.type == "prelu"}
        guard task.isPReLU || (parameters.isEmpty && gradientNodeID == nil) else {throw LearningContractError("This activation task has no learned parameters.")}
        guard Set(parameters.keys).isSubset(of:Set(active.map(\.id))),gradientNodeID == nil || active.contains(where:{$0.id == gradientNodeID}),!complete || !task.isPReLU || (Set(parameters.keys) == Set(active.map(\.id)) && gradientNodeID != nil) else {throw LearningContractError("Supply slopes for each PReLU and select its slope influence.")}
        for node in active {
            guard let state=parameters[node.id] else {continue}
            let config=try RemainingActivationConfiguration(componentID:"prelu",values:node.config)
            guard Set(state.keys) == ["weight"],let weight=state["weight"],weight.wellFormed,weight.shape == [config.numParameters!] else {throw LearningContractError("PReLU state requires a finite weight array matching Slope count.")}
        }
    }
    public static func validatedGraph(_ graph:GraphDocument,task:RemainingActivationLearningTask,complete:Bool)throws->GraphDocument {
        guard graph.schemaVersion == 2,graph.execution == nil,graph.nodes.count<=32,graph.edges.count<=31 else {throw LearningContractError("Activation lesson schema, training settings or graph bounds differ.")}
        let block=task.isPReLU ? "PReLU":task == .logScores ? "LogSoftmax":"Dropout"
        if complete,graph.nodes.count<3 {throw LearningContractError("Connect Input to \(block), then \(block) to Model Output.")}
        var copy=graph,ids=Set<String>()
        for i in copy.nodes.indices {
            var node=copy.nodes[i]
            guard !node.id.isEmpty,node.id.count<=128,ids.insert(node.id).inserted,node.x.isFinite,node.y.isFinite else {throw LearningContractError("Use unique finite activation component identities and positions.")}
            if ["input","output"].contains(node.type) {guard node.config.isEmpty else {throw LearningContractError("Lesson boundaries use fixture shapes without configuration.")}}
            else if node.type == task.componentID {node.config=try RemainingActivationConfiguration(componentID:node.type,values:node.config).values}
            else {node.config=try TensorTransformConfiguration(operation:node.type,values:node.config).values}
            copy.nodes[i]=node
        }
        let inputs=copy.nodes.filter {$0.type == "input"},outputs=copy.nodes.filter {$0.type == "output"},operators=copy.nodes.filter {$0.type == task.componentID}
        guard inputs.count<=1,outputs.count<=1,operators.count<=1,!complete || (inputs.count == 1 && outputs.count == 1 && operators.count == 1) else {throw LearningContractError("Connect one Input, one \(block) and one Output.")}
        var next:[String:String]=[:],previous:[String:String]=[:],edges=Set<String>()
        for edge in copy.edges {
            guard !edge.id.isEmpty,edges.insert(edge.id).inserted,ids.contains(edge.source),ids.contains(edge.target),edge.sourcePort == "out",edge.targetPort == "in",next[edge.source] == nil,previous[edge.target] == nil,!outputs.contains(where:{$0.id == edge.source}),!inputs.contains(where:{$0.id == edge.target}) else {throw LearningContractError("Connect the activation along one input-to-output path.")}
            next[edge.source]=edge.target;previous[edge.target]=edge.source
        }
        for node in copy.nodes {
            var seen=Set<String>(),current:String?=node.id
            while let id=current {guard seen.insert(id).inserted else {throw LearningContractError("Remove activation graph cycles.")};current=next[id]}
        }
        if complete {
            var visited=Set<String>(),current:String?=inputs[0].id
            while let id=current {visited.insert(id);current=next[id]}
            guard visited.count == copy.nodes.count,previous[outputs[0].id] != nil else {throw LearningContractError("Finish the activation path and remove unused blocks.")}
        }
        return copy
    }
    private enum CodingKeys:String,CodingKey {case schemaVersion,kind,graph,parameters,gradientNodeID}
    public func encode(to encoder:Encoder)throws {
        var c=encoder.container(keyedBy:CodingKeys.self);try c.encode(1,forKey:.schemaVersion);try c.encode("remainingActivationGraph",forKey:.kind);try c.encode(graph,forKey:.graph);try c.encode(parameters,forKey:.parameters)
        if let gradientNodeID {try c.encode(gradientNodeID,forKey:.gradientNodeID)} else {try c.encodeNil(forKey:.gradientNodeID)}
    }
    public func encoded()throws->Data {
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes];let data=try encoder.encode(self)
        guard data.count<=2_000_000 else {throw LearningContractError("Activation submission exceeds 2 MB.")};return data
    }
    public static func decode(_ data:Data,task:RemainingActivationLearningTask)throws->Self {
        guard data.count<=2_000_000 else {throw LearningContractError("Activation submission exceeds 2 MB.")}
        let raw=try JSONDecoder().decode(JSONValue.self,from:data),fields=try TensorLearningSubmission.checkedObject(raw,required:["schemaVersion","kind","graph","parameters","gradientNodeID"])
        guard fields["schemaVersion"]?.integerValue == 1,fields["kind"]?.stringValue == "remainingActivationGraph",case .object(let parameters)=fields["parameters"] else {throw LearningContractError("Unsupported activation submission.")}
        try TensorLearningSubmission.checkGraphFields(fields["graph"]!)
        for node in parameters.values {guard case .object(let tensors)=node else {throw LearningContractError("Malformed parameter map.")};for value in tensors.values {_ = try TensorLearningSubmission.checkedObject(value,required:["shape","values"])}}
        struct Envelope:Decodable {let graph:GraphDocument;let parameters:[String:[String:LearningTensor]];let gradientNodeID:String?}
        let e=try JSONDecoder().decode(Envelope.self,from:data);return try Self(graph:e.graph,parameters:e.parameters,gradientNodeID:e.gradientNodeID,task:task)
    }
}
