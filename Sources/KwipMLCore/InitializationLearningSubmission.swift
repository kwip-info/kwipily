import Foundation

/// Initialization controls are editable; public supplied SGD and Train context are fixed.
public struct InitializationLearningSubmission:Codable,Equatable {
    public let schemaVersion:Int
    public let kind:String
    public let taskID:String
    public let graph:GraphDocument
    public init(graph:GraphDocument,task:InitializationLearningTask)throws {schemaVersion=1;kind="initializationLearningGraph";taskID=task.rawValue;self.graph=try Self.validatedGraph(graph,task:task,complete:true)}
    public static func validatedGraph(_ graph:GraphDocument,task:InitializationLearningTask,complete:Bool)throws->GraphDocument {
        guard graph.schemaVersion == 9,graph.execution == nil,graph.name.count<=256,graph.nodes.count<=3,graph.edges.count<=2,Set(graph.nodes.map(\.id)).count == graph.nodes.count,Set(graph.edges.map(\.id)).count == graph.edges.count else {throw LearningContractError("Use one initialization, the supplied SGD optimizer and Train on this lesson canvas.")}
        var copy=graph
        let supplied=try OptimizerConfiguration(componentID:"optimizer_sgd",values:InitializationLearningTask.suppliedOptimizerValues)
        for i in copy.nodes.indices {
            let n=copy.nodes[i]
            guard !n.id.isEmpty,n.id.count<=128,n.name.count<=256,n.x.isFinite,n.y.isFinite,[task.componentID,"optimizer_sgd","train"].contains(n.type),copy.nodes.filter({$0.type == n.type}).count == 1 else {throw LearningContractError("Keep one requested initialization, one supplied SGD optimizer and one Train context.")}
            if n.type == "train" {guard n.config.isEmpty else {throw LearningContractError("The checker supplies Train's examples, gradients, measurements and saved state.")}}
            else if n.type == "optimizer_sgd" {
                guard Set(n.config.keys) == Set(InitializationLearningTask.suppliedOptimizerValues.keys),try OptimizerConfiguration(componentID:n.type,values:n.config) == supplied else {throw LearningContractError("Keep the supplied SGD controls: rate 0.05, momentum 0, zero decay and dampening, with Maximize and Look ahead off.")}
                copy.nodes[i].config=supplied.values
            } else {copy.nodes[i].config=try InitializationLearningConfiguration(values:n.config).values}
        }
        var ports=Set<String>()
        for e in copy.edges {
            let source=copy.nodes.first {$0.id == e.source},target=copy.nodes.first {$0.id == e.target}
            guard !e.id.isEmpty,e.id.count<=128,e.sourcePort == "out",target?.type == "train",((source?.type == task.componentID && e.targetPort == "initialization") || (source?.type == "optimizer_sgd" && e.targetPort == "optimizer")),ports.insert(e.targetPort).inserted else {throw LearningContractError("Connect initialization out to Train initialization, and supplied SGD out to Train optimizer. Both wires carry training recipes.")}
        }
        if complete {guard copy.nodes.count == 3,copy.edges.count == 2 else {throw LearningContractError("Add the initialization, supplied SGD and Train, then connect both recipe inputs before checking.")}}
        return copy
    }
    public func encoded()throws->Data {let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes];return try encoder.encode(self)}
    public static func decode(_ data:Data,task:InitializationLearningTask)throws->Self {
        guard data.count<=2_000_000 else {throw LearningContractError("Initialization submission exceeds 2 MB.")}
        let raw=try JSONDecoder().decode(JSONValue.self,from:data),o=try TensorLearningSubmission.checkedObject(raw,required:["schemaVersion","kind","taskID","graph"])
        guard o["schemaVersion"]?.integerValue == 1,o["kind"]?.stringValue == "initializationLearningGraph",o["taskID"]?.stringValue == task.rawValue else {throw LearningContractError("Unsupported initialization lesson submission.")}
        let g=try TensorLearningSubmission.checkedObject(o["graph"]!,required:["schemaVersion","name","nodes","edges"])
        guard case .array(let nodes)=g["nodes"],case .array(let edges)=g["edges"] else {throw LearningContractError("Provide the training recipe canvas.")}
        for n in nodes {_ = try TensorLearningSubmission.checkedObject(n,required:["id","type","name","x","y","config"])}
        for e in edges {_ = try TensorLearningSubmission.checkedObject(e,required:["id","source","sourcePort","target","targetPort"])}
        struct Envelope:Decodable {let graph:GraphDocument}
        return try Self(graph:JSONDecoder().decode(Envelope.self,from:data).graph,task:task)
    }
}
