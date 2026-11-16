import Foundation

/// Only the recipe and its typed connection are authored. Train's context is fixture-owned.
public struct OptimizerLearningSubmission:Codable,Equatable {
    public let schemaVersion:Int
    public let kind:String
    public let graph:GraphDocument
    public init(graph:GraphDocument,task:OptimizerLearningTask)throws {schemaVersion=1;kind="optimizerLearningGraph";self.graph=try Self.validatedGraph(graph,task:task,complete:true)}
    public static func validatedGraph(_ graph:GraphDocument,task:OptimizerLearningTask,complete:Bool)throws->GraphDocument {
        guard graph.schemaVersion == 6,graph.execution == nil,graph.name.count<=256,graph.nodes.count<=2,graph.edges.count<=1,Set(graph.nodes.map(\.id)).count == graph.nodes.count else {throw LearningContractError("Use a bounded optimizer lesson canvas with one recipe and the supplied Train context.")}
        var copy=graph
        for i in copy.nodes.indices {
            let n=copy.nodes[i]
            guard !n.id.isEmpty,n.id.count<=128,n.name.count<=256,n.x.isFinite,n.y.isFinite,[task.componentID,"train"].contains(n.type) else {throw LearningContractError("This lesson uses only the requested optimizer and Train; recipes are not model layers.")}
            if n.type == "train" {guard n.config.isEmpty else {throw LearningContractError("The checker owns Train's examples, starting settings and update sequence.")}}
            else {copy.nodes[i].config=try OptimizerConfiguration(componentID:n.type,values:n.config).values}
        }
        guard copy.nodes.filter({$0.type == task.componentID}).count<=1,copy.nodes.filter({$0.type == "train"}).count<=1 else {throw LearningContractError("Keep one optimizer recipe and one Train context.")}
        for e in copy.edges {
            guard !e.id.isEmpty,e.id.count<=128,copy.nodes.first(where:{$0.id == e.source})?.type == task.componentID,e.sourcePort == "out",copy.nodes.first(where:{$0.id == e.target})?.type == "train",e.targetPort == "optimizer" else {throw LearningContractError("Connect optimizer out to Train optimizer. This port carries a recipe, not model numbers.")}
        }
        if complete {guard copy.nodes.count == 2,copy.edges.count == 1 else {throw LearningContractError("Add the requested optimizer and Train, then connect out to optimizer before checking.")}}
        return copy
    }
    public func encoded()throws->Data {let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes];return try encoder.encode(self)}
    public static func decode(_ data:Data,task:OptimizerLearningTask)throws->Self {
        guard data.count<=2_000_000 else {throw LearningContractError("Optimizer submission exceeds 2 MB.")}
        let raw=try JSONDecoder().decode(JSONValue.self,from:data),o=try TensorLearningSubmission.checkedObject(raw,required:["schemaVersion","kind","graph"])
        guard o["schemaVersion"]?.integerValue == 1,o["kind"]?.stringValue == "optimizerLearningGraph" else {throw LearningContractError("Unsupported optimizer lesson submission.")}
        let g=try TensorLearningSubmission.checkedObject(o["graph"]!,required:["schemaVersion","name","nodes","edges"])
        guard case .array(let nodes)=g["nodes"],case .array(let edges)=g["edges"] else {throw LearningContractError("Provide the recipe canvas.")}
        for n in nodes {_ = try TensorLearningSubmission.checkedObject(n,required:["id","type","name","x","y","config"])}
        for e in edges {_ = try TensorLearningSubmission.checkedObject(e,required:["id","source","sourcePort","target","targetPort"])}
        struct Envelope:Decodable {let graph:GraphDocument}
        return try Self(graph:JSONDecoder().decode(Envelope.self,from:data).graph,task:task)
    }
}
