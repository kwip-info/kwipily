import Foundation

public struct GradientLearningSubmission:Codable,Equatable {
    public let schemaVersion:Int;public let kind:String;public let taskID:String;public let graph:GraphDocument
    public init(graph:GraphDocument,task:GradientLearningTask)throws {schemaVersion=1;kind="gradientLearningGraph";taskID=task.rawValue;self.graph=try Self.validatedGraph(graph,task:task,complete:true)}
    public static func validatedGraph(_ graph:GraphDocument,task:GradientLearningTask,complete:Bool)throws->GraphDocument {
        let types:Set<String>=["gradient_policy","trainability","optimizer_sgd","train"]
        guard graph.schemaVersion==10,graph.execution==nil,graph.name.count<=256,graph.nodes.count<=4,graph.edges.count<=3,Set(graph.nodes.map(\.id)).count==graph.nodes.count,Set(graph.edges.map(\.id)).count==graph.edges.count,Set(graph.nodes.map(\.type)).count==graph.nodes.count else {throw LearningContractError("Use one gradient policy, one trainability policy, supplied SGD and Train on this lesson canvas.")}
        var copy=graph
        var gradient=task.gradientValues,trainability=task.trainabilityValues
        for node in graph.nodes {
            guard types.contains(node.type),!node.id.isEmpty,node.id.count<=128,node.name.count<=256,node.x.isFinite,node.y.isFinite else {throw LearningContractError("Keep unique bounded lesson control nodes.")}
            if node.type == "gradient_policy" {gradient=node.config}
            if node.type == "trainability" {trainability=node.config}
        }
        let controls=try GradientLearningConfiguration(gradientValues:gradient,trainabilityValues:trainability,task:task)
        let optimizer=try OptimizerConfiguration(componentID:"optimizer_sgd",values:task.optimizerValues)
        for index in copy.nodes.indices {
            let node=copy.nodes[index]
            switch node.type {
            case "gradient_policy":copy.nodes[index].config=controls.gradient.values
            case "trainability":copy.nodes[index].config=controls.trainability.values
            case "optimizer_sgd":guard Set(node.config.keys)==Set(task.optimizerValues.keys),try OptimizerConfiguration(componentID:node.type,values:node.config)==optimizer else {throw LearningContractError("Keep the supplied SGD rate and memory controls.")};copy.nodes[index].config=optimizer.values
            default:guard node.config.isEmpty else {throw LearningContractError("Train's rows, model, saved parent and execution settings are supplied by the checker.")}
            }
        }
        let ports=["gradient_policy":"gradientPolicy","trainability":"trainability","optimizer_sgd":"optimizer"]
        var occupied=Set<String>()
        for edge in copy.edges {
            let source=copy.nodes.first {$0.id==edge.source},target=copy.nodes.first {$0.id==edge.target}
            guard !edge.id.isEmpty,edge.id.count<=128,edge.sourcePort=="out",target?.type=="train",let source,ports[source.type]==edge.targetPort,occupied.insert(edge.targetPort).inserted else {throw LearningContractError("Connect each recipe out to its matching Gradient policy, Trainability or Optimizer input on Train.")}
        }
        if complete {guard copy.nodes.count==4,copy.edges.count==3,occupied==Set(ports.values),controls.trainableOwnerCount>0 else {throw LearningContractError("Connect all three recipe controls and leave at least one learned array able to train.")}}
        return copy
    }
    public func encoded()throws->Data {let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes];return try encoder.encode(self)}
    public static func decode(_ data:Data,task:GradientLearningTask)throws->Self {
        guard data.count<=2_000_000 else {throw LearningContractError("Gradient lesson submission exceeds 2 MB.")}
        let raw=try JSONDecoder().decode(JSONValue.self,from:data),o=try TensorLearningSubmission.checkedObject(raw,required:["schemaVersion","kind","taskID","graph"])
        guard o["schemaVersion"]?.integerValue==1,o["kind"]?.stringValue=="gradientLearningGraph",o["taskID"]?.stringValue==task.rawValue else {throw LearningContractError("Use the requested versioned gradient lesson.")}
        let graph=try TensorLearningSubmission.checkedObject(o["graph"]!,required:["schemaVersion","name","nodes","edges"])
        guard case .array(let nodes)=graph["nodes"],case .array(let edges)=graph["edges"] else {throw LearningContractError("Use the typed recipe canvas.")}
        for node in nodes {_ = try TensorLearningSubmission.checkedObject(node,required:["id","type","name","x","y","config"])}
        for edge in edges {_ = try TensorLearningSubmission.checkedObject(edge,required:["id","source","sourcePort","target","targetPort"])}
        struct Envelope:Decodable {let graph:GraphDocument}
        return try Self(graph:JSONDecoder().decode(Envelope.self,from:data).graph,task:task)
    }
}
