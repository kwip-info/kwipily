import Foundation

public struct SpatialPatchLearningSubmission:Encodable,Equatable {
    public let schemaVersion=1
    public let kind="spatialPatchGraph"
    public let graph:GraphDocument
    public init(graph:GraphDocument,task:SpatialPatchLearningTask)throws {self.graph=try Self.validatedGraph(graph,task:task,complete:true)}
    static func validatedGraph(_ graph:GraphDocument,task:SpatialPatchLearningTask,complete:Bool)throws->GraphDocument {
        guard graph.schemaVersion == 2,graph.execution == nil,graph.nodes.count<=32,graph.edges.count<=31 else {throw LearningContractError("Spatial patch lessons require a bounded schema-2 graph without training settings.")}
        var copy=graph,nodes:[String:GraphNode]=[:],sources:[String]=[],outputs:[String]=[]
        for index in copy.nodes.indices {
            var node=copy.nodes[index]
            guard !node.id.isEmpty,node.id.count<=128,nodes[node.id] == nil,node.x.isFinite,node.y.isFinite else {throw LearningContractError("Invalid patch component identity or position.")}
            switch node.type {
            case "input","output":
                guard node.config.isEmpty else {throw LearningContractError("Patch lesson boundaries have no settings.")}
                if node.type == "input" {sources.append(node.id)} else {outputs.append(node.id)}
            case task.componentID:node.config=try SpatialPatchConfiguration(componentID:node.type,values:node.config).values
            default:node.config=try TensorTransformConfiguration(operation:node.type,values:node.config).values
            }
            copy.nodes[index]=node;nodes[node.id]=node
        }
        guard sources.count<=1,outputs.count<=1,!complete || (sources.count == 1 && outputs.count == 1 && copy.nodes.filter({$0.type == task.componentID}).count == 1) else {throw LearningContractError("Connect Input to \(task.isUnfold ? "Unfold":"Fold"), then connect the block to Model Output.")}
        var incoming:[String:GraphEdge]=[:],successors:[String:[String]]=[:],ids=Set<String>()
        for edge in copy.edges {
            guard !edge.id.isEmpty,ids.insert(edge.id).inserted,let source=nodes[edge.source],let target=nodes[edge.target],source.type != "output",target.type != "input",edge.sourcePort == "out",edge.targetPort == "in",incoming[target.id] == nil else {throw LearningContractError("Connect each patch input once using its out/in ports.")}
            incoming[target.id]=edge;successors[source.id,default:[]].append(target.id)
        }
        for node in copy.nodes where complete {guard (node.type == "input" || incoming[node.id] != nil),(node.type == "output" || !(successors[node.id] ?? []).isEmpty) else {throw LearningContractError("Complete the patch connections and remove unused blocks.")}}
        var pending=nodes.mapValues {incoming[$0.id] == nil ? 0:1},queue=copy.nodes.filter {pending[$0.id] == 0}.map(\.id),order:[String]=[]
        while !queue.isEmpty {let id=queue.removeFirst();order.append(id);for child in successors[id,default:[]] {pending[child,default:0]-=1;if pending[child] == 0 {queue.append(child)}}}
        guard order.count == nodes.count else {throw LearningContractError("Remove cycles from the patch graph.")}
        var layouts:[String:BatchedTensorLayout]=[:]
        for id in order where complete {
            let node=nodes[id]!
            if node.type == "input" {layouts[id]=try .init(sample:task.sampleShape)}
            else if let edge=incoming[id],let input=layouts[edge.source] {
                if node.type == task.componentID {layouts[id]=try SpatialPatchConfiguration(componentID:node.type,values:node.config).resolved(layout:input)}
                else if node.type == "output" {layouts[id]=input}
                else {layouts[id]=try TensorTransformConfiguration(operation:node.type,values:node.config).outputLayout(for:input)}
            }
        }
        return copy
    }
    public func encoded()throws->Data {let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes];let data=try encoder.encode(self);guard data.count<=2_000_000 else {throw LearningContractError("Submission exceeds 2 MB.")};return data}
    public static func decode(_ data:Data,task:SpatialPatchLearningTask)throws->Self {
        guard data.count<=2_000_000 else {throw LearningContractError("Submission exceeds 2 MB.")}
        let raw=try JSONDecoder().decode(JSONValue.self,from:data),envelope=try TensorLearningSubmission.checkedObject(raw,required:["schemaVersion","kind","graph"])
        guard envelope["schemaVersion"]?.integerValue == 1,envelope["kind"]?.stringValue == "spatialPatchGraph" else {throw LearningContractError("Unsupported spatial patch submission.")}
        let graph=try TensorLearningSubmission.checkedObject(envelope["graph"]!,required:["schemaVersion","name","nodes","edges"])
        guard case .array(let nodes)=graph["nodes"],case .array(let edges)=graph["edges"],nodes.count<=32,edges.count<=31 else {throw LearningContractError("Patch graph exceeds lesson bounds.")}
        for node in nodes {_ = try TensorLearningSubmission.checkedObject(node,required:["id","type","name","x","y","config"])}
        for edge in edges {_ = try TensorLearningSubmission.checkedObject(edge,required:["id","source","sourcePort","target","targetPort"])}
        struct Envelope:Decodable {let graph:GraphDocument}
        return try Self(graph:JSONDecoder().decode(Envelope.self,from:data).graph,task:task)
    }
}
