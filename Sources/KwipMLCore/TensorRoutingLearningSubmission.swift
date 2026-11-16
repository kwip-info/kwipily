import Foundation

/// Source-port addresses remain intact: a named part is never silently rewritten to another output.
public struct TensorRoutingLearningSubmission:Encodable,Equatable {
    public let schemaVersion=1
    public let kind="tensorRoutingGraph"
    public let graph:GraphDocument
    public init(graph:GraphDocument,task:TensorRoutingLearningTask)throws {self.graph=try Self.validatedGraph(graph,task:task,complete:true)}
    static func validatedGraph(_ graph:GraphDocument,task:TensorRoutingLearningTask,complete:Bool)throws->GraphDocument {
        guard graph.schemaVersion == 2,graph.execution == nil,graph.nodes.count<=32,graph.edges.count<=64 else {throw LearningContractError("Routing lessons require a bounded schema-2 graph without training settings.")}
        var copy=graph,nodes:[String:GraphNode]=[:],bindings=Set<String>(),outputs:[String]=[],inputPorts:[String:[String]]=[:]
        for index in copy.nodes.indices {
            var node=copy.nodes[index]
            guard !node.id.isEmpty,node.id.count<=128,nodes[node.id] == nil,node.x.isFinite,node.y.isFinite else {throw LearningContractError("Invalid component identity or position.")}
            switch node.type {
            case "input":
                guard Set(node.config.keys) == ["fixture"],let role=node.config["fixture"]?.stringValue,task.sourceRoles.contains(role),bindings.insert(role).inserted else {throw LearningContractError("Bind one Input to each declared fixture source.")}
                inputPorts[node.id]=[]
            case "output":
                guard node.config.isEmpty else {throw LearningContractError("Output has no settings.")}
                outputs.append(node.id);inputPorts[node.id]=["in"]
            case task.componentID:
                let configuration=try TensorRoutingConfiguration(componentID:node.type,values:node.config)
                node.config=configuration.values;inputPorts[node.id]=configuration.configuredInputs
            case "concat":
                node.config=try TensorCompositionConfiguration(componentID:"concat",values:node.config).values;inputPorts[node.id]=["a","b"]
            default:
                node.config=try TensorTransformConfiguration(operation:node.type,values:node.config).values;inputPorts[node.id]=["in"]
            }
            copy.nodes[index]=node;nodes[node.id]=node
        }
        guard outputs.count<=1,!complete || (bindings == Set(task.sourceRoles) && outputs.count == 1 && copy.nodes.filter({$0.type == task.componentID}).count == 1) else {throw LearningContractError("Connect the declared Inputs to one \(task.componentID) block and one Output.")}
        var incoming:[String:[String:GraphEdge]]=[:],successors:[String:[String]]=[:],edgeIDs=Set<String>()
        for edge in copy.edges {
            guard !edge.id.isEmpty,edgeIDs.insert(edge.id).inserted,let source=nodes[edge.source],let target=nodes[edge.target],source.type != "output" else {throw LearningContractError("Invalid named-port connection.")}
            let ports=["split","chunk"].contains(source.type) ? (0..<8).map {"part\($0)"} : ["out"]
            guard ports.contains(edge.sourcePort),inputPorts[target.id]!.contains(edge.targetPort),incoming[target.id,default:[:]][edge.targetPort] == nil else {throw LearningContractError("Use an available source port and connect each configured input once.")}
            incoming[target.id,default:[:]][edge.targetPort]=edge;successors[source.id,default:[]].append(target.id)
        }
        for node in copy.nodes where complete {
            guard incoming[node.id,default:[:]].count == inputPorts[node.id]!.count,node.type == "output" || !(successors[node.id] ?? []).isEmpty else {throw LearningContractError("Complete the configured connections and remove unused blocks.")}
        }
        var pending=nodes.mapValues {incoming[$0.id,default:[:]].count},queue=copy.nodes.filter {pending[$0.id] == 0}.map(\.id),order:[String]=[]
        while !queue.isEmpty {let id=queue.removeFirst();order.append(id);for child in successors[id,default:[]] {pending[child,default:0]-=1;if pending[child] == 0 {queue.append(child)}}}
        guard order.count == nodes.count else {throw LearningContractError("Remove cycles from the routing lesson.")}
        // Resolve available addresses as soon as their fixture-backed shapes are known, even in a partial draft.
        var layouts:[String:[String:BatchedTensorLayout]]=[:],integerSources=Set<String>()
        for id in order {
            let node=nodes[id]!
            if node.type == "input" {let role=node.config["fixture"]!.stringValue!;layouts[id]=["out":try .init(sample:task.sampleShapes[role]!)];if role == "indices" {integerSources.insert(id)};continue}
            let edges=inputPorts[id]!.compactMap {incoming[id]?[$0]}
            if edges.count != inputPorts[id]!.count {continue}
            var inputs:[BatchedTensorLayout]=[]
            for edge in edges {
                if let source=layouts[edge.source] {guard let layout=source[edge.sourcePort] else {throw LearningContractError("\(edge.sourcePort) is unavailable for the selected input shape and settings.")};inputs.append(layout)}
            }
            if inputs.count != edges.count {continue}
            if node.type == task.componentID {
                if node.type == "gather" {guard edges[0].source != edges[1].source,!integerSources.contains(edges[0].source),integerSources.contains(edges[1].source) else {throw LearningContractError("Gather needs a number source and the provided whole-number positions.")}}
                else {guard edges.allSatisfy({!integerSources.contains($0.source)}) else {throw LearningContractError("Use floating number sources for this block.")}}
                layouts[id]=try TensorRoutingConfiguration(componentID:node.type,values:node.config).resolved(layouts:inputs)
            } else if node.type == "concat" {guard edges.allSatisfy({!integerSources.contains($0.source)}) else {throw LearningContractError("Concatenate needs number sources.")};layouts[id]=["out":try TensorCompositionConfiguration(componentID:"concat",values:node.config).resolved(layouts:inputs)]}
            else if node.type == "output" {layouts[id]=["out":inputs[0]];if integerSources.contains(edges[0].source) {integerSources.insert(id)}}
            else {layouts[id]=["out":try TensorTransformConfiguration(operation:node.type,values:node.config).outputLayout(for:inputs[0])];if integerSources.contains(edges[0].source) {integerSources.insert(id)}}
        }
        return copy
    }
    public func encoded()throws->Data {let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes];let data=try encoder.encode(self);guard data.count<=2_000_000 else {throw LearningContractError("Submission exceeds 2 MB.")};return data}
    public static func decode(_ data:Data,task:TensorRoutingLearningTask)throws->Self {
        guard data.count<=2_000_000 else {throw LearningContractError("Submission exceeds 2 MB.")}
        let raw=try JSONDecoder().decode(JSONValue.self,from:data),envelope=try TensorLearningSubmission.checkedObject(raw,required:["schemaVersion","kind","graph"])
        guard envelope["schemaVersion"]?.integerValue == 1,envelope["kind"]?.stringValue == "tensorRoutingGraph" else {throw LearningContractError("Unsupported routing submission.")}
        let graph=try TensorLearningSubmission.checkedObject(envelope["graph"]!,required:["schemaVersion","name","nodes","edges"])
        guard case .array(let nodes)=graph["nodes"],case .array(let edges)=graph["edges"],nodes.count<=32,edges.count<=64 else {throw LearningContractError("Graph exceeds lesson bounds.")}
        for node in nodes {_ = try TensorLearningSubmission.checkedObject(node,required:["id","type","name","x","y","config"])}
        for edge in edges {_ = try TensorLearningSubmission.checkedObject(edge,required:["id","source","sourcePort","target","targetPort"])}
        struct Envelope:Decodable {let graph:GraphDocument}
        return try Self(graph:JSONDecoder().decode(Envelope.self,from:data).graph,task:task)
    }
}
