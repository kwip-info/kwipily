import Foundation

/// Bounded fixture-owned evaluator; the loss remains a control recipe, not a tensor layer.
public struct LossLearningSubmission:Encodable,Equatable {
    public let schemaVersion=1
    public let kind="lossObjectiveGraph"
    public let graph:GraphDocument
    public let objectiveSource:WorkflowNodeAddress
    public let inputSources:[String:WorkflowNodeAddress]
    public init(graph:GraphDocument,task:LossLearningTask)throws {
        self.graph=try Self.validatedGraph(graph,task:task,complete:true)
        objectiveSource = .init(domain:.control,localID:self.graph.nodes.first {$0.type==task.componentID}!.id)
        inputSources=Dictionary(uniqueKeysWithValues:self.graph.nodes.filter {$0.type=="input"}.map {($0.config["fixture"]!.stringValue!,WorkflowNodeAddress(domain:.model,localID:$0.id))})
    }
    public static func validatedGraph(_ graph:GraphDocument,task:LossLearningTask,complete:Bool)throws->GraphDocument {
        guard graph.schemaVersion==2,graph.execution==nil,graph.nodes.count<=12,graph.edges.count<=12,graph.name.count<=256 else {throw LearningContractError("Use the bounded objective lesson graph without production training settings.")}
        var copy=graph,nodes:[String:GraphNode]=[:],inputs:[String:String]=[:],outputs:[String:String]=[:],recipe:String?,check:String?,normalization:String?,config:LossConfiguration?
        for i in copy.nodes.indices {
            var node=copy.nodes[i]
            guard !node.id.isEmpty,node.id.count<=128,node.name.count<=256,nodes[node.id]==nil,node.x.isFinite,node.y.isFinite else {throw LearningContractError("Use unique block identities and finite positions.")}
            if node.type=="input" {
                guard Set(node.config.keys)==["fixture"],let role=node.config["fixture"]?.stringValue,task.sourceRoles.contains(role),inputs[role]==nil else {throw LearningContractError("Use each supplied prediction and target role exactly once.")};inputs[role]=node.id
            } else if node.type=="output" {
                guard Set(node.config.keys)==["role"],let role=node.config["role"]?.stringValue,["raw","objective","inputInfluence"].contains(role),outputs[role]==nil else {throw LearningContractError("Use separate Individual scores, Combined score and Input influence Outputs.")};outputs[role]=node.id
            } else if node.type==task.componentID,recipe==nil {
                config=try .init(componentID:node.type,values:node.config);node.config=config!.values;recipe=node.id
            } else if node.type=="objective_check",check==nil {
                guard Set(node.config.keys)==["scalarization"],let value=node.config["scalarization"]?.stringValue,["mean","sum"].contains(value) else {throw LearningContractError("Objective check must explicitly combine kept scores with Mean or Sum.")};check=node.id
            } else if node.type=="log_softmax",task.usesLogSoftmax,normalization==nil {
                guard Set(node.config.keys)==["dim"],node.config["dim"] == -1 else {throw LearningContractError("Prepare probabilities visibly with LogSoftmax on the final class direction, −1.")};normalization=node.id
            } else {throw LearningContractError("Use the requested loss recipe and fixture-owned Objective check, not unrelated layers or hidden code.")}
            nodes[node.id]=node;copy.nodes[i]=node
        }
        func key(_ from:String,_ port:String,_ to:String,_ input:String)->String {from+"."+port+">"+to+"."+input}
        var allowed=Set<String>()
        if let check {
            for (role,id) in inputs {
                if role=="prediction",task.usesLogSoftmax {if let normalization {allowed.insert(key(id,"out",normalization,"in"));allowed.insert(key(normalization,"out",check,"prediction"))}}
                else {allowed.insert(key(id,"out",check,role))}
            }
            if let recipe {allowed.insert(key(recipe,"out",check,"objective"))}
            for (role,id) in outputs {allowed.insert(key(check,role,id,"in"))}
        }
        var ids=Set<String>(),targets=Set<String>(),actual=Set<String>()
        for edge in copy.edges {
            guard !edge.id.isEmpty,edge.id.count<=128,ids.insert(edge.id).inserted,nodes[edge.source] != nil,nodes[edge.target] != nil,targets.insert(edge.target+"."+edge.targetPort).inserted,allowed.contains(key(edge.source,edge.sourcePort,edge.target,edge.targetPort)) else {throw LearningContractError("Connect each supplied role and objective recipe to its named check port.")}
            actual.insert(key(edge.source,edge.sourcePort,edge.target,edge.targetPort))
        }
        if complete {
            guard Set(inputs.keys)==Set(task.sourceRoles),Set(outputs.keys)==["raw","objective","inputInfluence"],recipe != nil,check != nil,!task.usesLogSoftmax || normalization != nil,actual==allowed,let config else {throw LearningContractError("Connect the objective recipe, all supplied roles, visible probability preparation and all three result Outputs.")}
            if task == .bce {
                guard config.values["weight_mode"] == .string("mapped"),config.values["pos_weight_mode"] != .string("mapped") else {throw LearningContractError("This exercise supplies mapped element importance; positive-label importance comes from configured values or is disabled.")}
            }
            if task == .crossEntropy {guard config.values["target_mode"] == .string("class_indices") else {throw LearningContractError("This exercise supplies integer class choices; choose that desired class format.")}}
        }
        return copy
    }
    public static func decode(_ data:Data,task:LossLearningTask)throws->Self {
        guard data.count<=2_000_000 else {throw LearningContractError("Objective lesson submission exceeds 2MB.")}
        let raw=try JSONDecoder().decode(JSONValue.self,from:data)
        let envelope=try TensorLearningSubmission.checkedObject(raw,required:["schemaVersion","kind","graph","objectiveSource","inputSources"])
        guard envelope["schemaVersion"] == 1,envelope["kind"] == .string("lossObjectiveGraph") else {throw LearningContractError("Use the supported objective lesson submission version.")}
        let graph=try TensorLearningSubmission.checkedObject(envelope["graph"]!,required:["schemaVersion","name","nodes","edges"])
        guard case .array(let nodes)=graph["nodes"],case .array(let edges)=graph["edges"] else {throw LearningContractError("Use objective blocks and connections.")}
        for node in nodes {_ = try TensorLearningSubmission.checkedObject(node,required:["id","type","name","x","y","config"])}
        for edge in edges {_ = try TensorLearningSubmission.checkedObject(edge,required:["id","source","sourcePort","target","targetPort"])}
        _ = try TensorLearningSubmission.checkedObject(envelope["objectiveSource"]!,required:["domain","localID"])
        guard case .object(let sources)=envelope["inputSources"] else {throw LearningContractError("Use explicit supplied role source bindings.")}
        for source in sources.values {_ = try TensorLearningSubmission.checkedObject(source,required:["domain","localID"])}
        struct Envelope:Decodable {let graph:GraphDocument;let objectiveSource:WorkflowNodeAddress;let inputSources:[String:WorkflowNodeAddress]}
        let decoded=try JSONDecoder().decode(Envelope.self,from:data),submission=try Self(graph:decoded.graph,task:task)
        guard decoded.objectiveSource==submission.objectiveSource,decoded.inputSources==submission.inputSources else {throw LearningContractError("Objective and input source bindings must identify the actual connected lesson blocks.")}
        return submission
    }
    public func encoded()throws->Data {
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes]
        let data=try encoder.encode(self)
        guard data.count<=2_000_000 else {throw LearningContractError("Objective lesson submission exceeds 2MB.")};return data
    }
}
