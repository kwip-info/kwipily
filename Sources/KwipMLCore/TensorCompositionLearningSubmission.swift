import Foundation

/// Lesson inputs bind fixture roles independently of editable labels and identities.
public struct TensorCompositionLearningSubmission:Encodable,Equatable {
    public let schemaVersion=1
    public let kind="tensorCompositionGraph"
    public let graph:GraphDocument
    public init(graph:GraphDocument,task:TensorCompositionLearningTask)throws {self.graph=try Self.validatedGraph(graph,task:task,complete:true)}
    static func validatedGraph(_ graph:GraphDocument,task:TensorCompositionLearningTask,complete:Bool)throws->GraphDocument {
        guard graph.schemaVersion == 2,graph.execution == nil,(0...32).contains(graph.nodes.count),graph.edges.count<=64 else {throw LearningContractError("Tensor composition lessons require a bounded graph without training settings.")}
        if complete,graph.nodes.count < (task.isBinary ? 4:3) {
            let block=BundledTensorCompositionComponents.definition(task.componentID).title
            throw LearningContractError(task.isBinary ? "Connect two Inputs to \(block), then connect \(block) to Model Output." : "Connect Input to \(block), then connect \(block) to Model Output.")
        }
        var copy=graph,nodes:[String:GraphNode]=[:],bindings=Set<String>(),outputs:[String]=[]
        for i in copy.nodes.indices {
            var node=copy.nodes[i]
            guard !node.id.isEmpty,node.id.count<=128,nodes[node.id] == nil,node.x.isFinite,node.y.isFinite else {throw LearningContractError("Invalid tensor composition component identity or position.")}
            switch node.type {
            case "input":
                guard Set(node.config.keys) == ["fixture"],let role=node.config["fixture"]?.stringValue,(task.isBinary ? ["left","right"]:["left"]).contains(role),bindings.insert(role).inserted else {throw LearningContractError("Bind one input to each declared source fixture.")}
            case "output":
                guard node.config.isEmpty else {throw LearningContractError("Lesson output has no configuration.")}
                outputs.append(node.id)
            case task.componentID:
                node.config=try TensorCompositionConfiguration(componentID:task.componentID,values:node.config).values
            default:node.config=try TensorTransformConfiguration(operation:node.type,values:node.config).values
            }
            copy.nodes[i]=node;nodes[node.id]=node
        }
        guard outputs.count<=1,(!complete || (bindings == (task.isBinary ? ["left","right"]:["left"]) && outputs.count == 1 && copy.nodes.filter({$0.type == task.componentID}).count == 1)) else {throw LearningContractError("Connect the declared fixture inputs, one requested block and one output.")}
        var incoming:[String:Set<String>]=[:],successors:[String:[String]]=[:],edgeIDs=Set<String>()
        for edge in copy.edges {
            guard !edge.id.isEmpty,edgeIDs.insert(edge.id).inserted,let source=nodes[edge.source],let target=nodes[edge.target],edge.sourcePort == "out",source.type != "output" else {throw LearningContractError("Invalid tensor composition connection.")}
            let ports=target.type == "input" ? [] : ["concat","add","stack"].contains(target.type) ? ["a","b"] : ["in"]
            guard ports.contains(edge.targetPort),incoming[target.id,default:[]].insert(edge.targetPort).inserted else {throw LearningContractError("Connect every declared tensor composition input exactly once.")}
            successors[source.id,default:[]].append(target.id)
        }
        for node in copy.nodes where complete {
            let expected=node.type == "input" ? 0 : ["concat","add","stack"].contains(node.type) ? 2 : 1
            guard incoming[node.id,default:[]].count == expected,node.type == "output" || !(successors[node.id] ?? []).isEmpty else {throw LearningContractError("Complete the graph and remove unused components.")}
        }
        var pending=nodes.mapValues {incoming[$0.id,default:[]].count}
        var queue=copy.nodes.filter {pending[$0.id] == 0}.map(\.id),visited=0
        while !queue.isEmpty {
            let id=queue.removeFirst();visited+=1
            for child in successors[id,default:[]] {pending[child,default:0]-=1;if pending[child] == 0 {queue.append(child)}}
        }
        guard visited == nodes.count else {throw LearningContractError("Remove cycles from the tensor composition lesson graph.")}
        return copy
    }
    public func encoded()throws->Data {
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes]
        let data=try encoder.encode(self)
        guard data.count<=2_000_000 else {throw LearningContractError("Submission exceeds 2 MB.")}
        return data
    }
    public static func decode(_ data:Data,task:TensorCompositionLearningTask)throws->Self {
        guard data.count<=2_000_000 else {throw LearningContractError("Submission exceeds 2 MB.")}
        let raw=try JSONDecoder().decode(JSONValue.self,from:data)
        let envelope=try TensorLearningSubmission.checkedObject(raw,required:["schemaVersion","kind","graph"])
        guard envelope["schemaVersion"]?.integerValue == 1,envelope["kind"]?.stringValue == "tensorCompositionGraph" else {throw LearningContractError("Unsupported tensor composition submission.")}
        let graph=try TensorLearningSubmission.checkedObject(envelope["graph"]!,required:["schemaVersion","name","nodes","edges"])
        guard case .array(let nodes)=graph["nodes"],case .array(let edges)=graph["edges"],nodes.count<=32,edges.count<=64 else {throw LearningContractError("Tensor composition graph exceeds lesson bounds.")}
        for node in nodes {_ = try TensorLearningSubmission.checkedObject(node,required:["id","type","name","x","y","config"])}
        for edge in edges {_ = try TensorLearningSubmission.checkedObject(edge,required:["id","source","sourcePort","target","targetPort"])}
        struct Envelope:Decodable {let graph:GraphDocument}
        return try Self(graph:JSONDecoder().decode(Envelope.self,from:data).graph,task:task)
    }
}
