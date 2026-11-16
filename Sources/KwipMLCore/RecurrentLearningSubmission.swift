import Foundation

public struct RecurrentLearningSubmission:Encodable,Equatable {
    public let schemaVersion=1
    public let kind="recurrentGraph"
    public let graph:GraphDocument
    public init(graph:GraphDocument,task:RecurrentLearningTask)throws {self.graph=try Self.validatedGraph(graph,task:task,complete:true)}
    static func validatedGraph(_ graph:GraphDocument,task:RecurrentLearningTask,complete:Bool)throws->GraphDocument {
        guard graph.schemaVersion==2,graph.execution==nil,graph.nodes.count<=7,graph.edges.count<=6,graph.name.count<=256 else {throw LearningContractError("Use a bounded recurrent lesson graph without training settings.")}
        var copy=graph,nodes:[String:GraphNode]=[:],sources:[String:String]=[:],outputs:[String:String]=[:],operatorID:String?,configuration:RecurrentConfiguration?
        for i in copy.nodes.indices {
            var node=copy.nodes[i]
            guard !node.id.isEmpty,node.id.count<=128,node.name.count<=256,nodes[node.id]==nil,node.x.isFinite,node.y.isFinite else {throw LearningContractError("Invalid recurrent block identity or position.")}
            if node.type=="input" {
                guard Set(node.config.keys)==["fixture"],let role=node.config["fixture"]?.stringValue,task.sourceRoles.contains(role),sources[role]==nil else {throw LearningContractError("Use one Input for each supplied measurement or state role.")};sources[role]=node.id
            } else if node.type=="output" {
                guard Set(node.config.keys)==["role"],let role=node.config["role"]?.stringValue,task.outputRoles.contains(role),outputs[role]==nil else {throw LearningContractError("Use a separate Output for each result and state role.")};outputs[role]=node.id
            } else if node.type==task.componentID,operatorID==nil {
                configuration=try RecurrentConfiguration(componentID:node.type,values:node.config);node.config=configuration!.values;operatorID=node.id
            } else {throw LearningContractError("Use one requested recurrent block and its role boundaries.")}
            nodes[node.id]=node;copy.nodes[i]=node
        }
        if complete {guard Set(sources.keys)==Set(task.sourceRoles),Set(outputs.keys)==Set(task.outputRoles),operatorID != nil else {throw LearningContractError("Connect every supplied role to the recurrent block and every named result to its own Output.")}}
        var ids=Set<String>(),targets=Set<String>()
        for edge in copy.edges {
            guard !edge.id.isEmpty,ids.insert(edge.id).inserted,let source=nodes[edge.source],let target=nodes[edge.target],targets.insert(edge.target+"."+edge.targetPort).inserted else {throw LearningContractError("Invalid or repeated recurrent role connection.")}
            if target.id==operatorID {
                let wanted=edge.targetPort=="in" ? "input":edge.targetPort
                guard source.type=="input",source.config["fixture"]?.stringValue==wanted,edge.sourcePort=="out",configuration!.configuredInputs.contains(edge.targetPort) else {throw LearningContractError("Connect each supplied role to its matching enabled input.")}
            } else if source.id==operatorID {
                let role=target.config["role"]?.stringValue
                guard target.type=="output",edge.targetPort=="in",edge.sourcePort==(role=="result" ? "out":role) else {throw LearningContractError("Connect out to the result Output and named state ports to matching state Outputs.")}
            } else {throw LearningContractError("Use direct role connections through the assessed recurrent block.")}
        }
        if complete {
            guard copy.edges.count==task.sourceRoles.count+task.outputRoles.count else {throw LearningContractError("Complete every measurement, starting-state and named-result connection.")}
            let c=configuration!.values
            guard c["state_mode"]?.stringValue=="supplied",c["input_size"]?.integerValue==2,c["hidden_size"]?.integerValue==2,(c["num_layers"]?.integerValue ?? 1)==1,(c["bidirectional"]?.boolValue ?? false)==false,(c["dropout"]?.doubleValue ?? 0)==0,(c["proj_size"]?.integerValue ?? 0)==0 else {throw LearningContractError("This exercise uses supplied state, width 2 and one forward layer without dropout or projection.")}
            _ = try configuration!.resolved(layouts:task.sourceRoles.map {try .init(sample:task.sampleShapes[$0]!)})
        }
        return copy
    }
    public func encoded()throws->Data {let e=JSONEncoder();e.outputFormatting=[.sortedKeys,.withoutEscapingSlashes];let d=try e.encode(self);guard d.count<=2_000_000 else {throw LearningContractError("Recurrent submission exceeds 2 MB.")};return d}
    public static func decode(_ data:Data,task:RecurrentLearningTask)throws->Self {
        guard data.count<=2_000_000 else {throw LearningContractError("Recurrent submission exceeds 2 MB.")}
        let raw=try JSONDecoder().decode(JSONValue.self,from:data),env=try TensorLearningSubmission.checkedObject(raw,required:["schemaVersion","kind","graph"])
        guard env["schemaVersion"]?.integerValue==1,env["kind"]?.stringValue=="recurrentGraph" else {throw LearningContractError("Unsupported recurrent submission.")}
        let graph=try TensorLearningSubmission.checkedObject(env["graph"]!,required:["schemaVersion","name","nodes","edges"])
        guard case .array(let nodes)=graph["nodes"],case .array(let edges)=graph["edges"] else {throw LearningContractError("Use recurrent role blocks and connections.")}
        for n in nodes {_ = try TensorLearningSubmission.checkedObject(n,required:["id","type","name","x","y","config"])}
        for e in edges {_ = try TensorLearningSubmission.checkedObject(e,required:["id","source","sourcePort","target","targetPort"])}
        struct Envelope:Decodable {let graph:GraphDocument}
        return try .init(graph:JSONDecoder().decode(Envelope.self,from:data).graph,task:task)
    }
}
