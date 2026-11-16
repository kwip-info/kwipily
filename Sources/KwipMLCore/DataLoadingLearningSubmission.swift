import Foundation

public struct DataLoadingLearningSubmission:Equatable,Codable {
    public let schemaVersion:Int
    public let kind:String
    public let taskID:String
    public let graph:GraphDocument
    public init(task:DataLoadingLearningTask,graph:GraphDocument,complete:Bool=true)throws {
        guard graph.schemaVersion==2,graph.nodes.count<=3,graph.edges.count<=2,Set(graph.nodes.map(\.id)).count==graph.nodes.count,Set(graph.nodes.map(\.type)).count==graph.nodes.count,Set(graph.edges.map(\.id)).count==graph.edges.count else {throw LearningContractError("Use one Dataset, one DataLoader and the supplied Train on this lesson canvas.")}
        var copy=graph
        for index in copy.nodes.indices {
            let node=copy.nodes[index]
            guard !node.id.isEmpty,node.id.utf8.count<=128,node.name.count<=256,node.x.isFinite,node.y.isFinite else {throw LearningContractError("Keep bounded, uniquely named lesson controls.")}
            switch node.type {
            case "dataset_source":copy.nodes[index].config=try DatasetConfiguration(values:node.config).values
            case "data_loader":copy.nodes[index].config=try DataLoaderConfiguration(values:node.config).values
            case "train":guard node.config.isEmpty else {throw LearningContractError("The checker supplies Train’s model, source and saved parent.")}
            default:throw LearningContractError("Use Dataset, DataLoader and Train; ordinary tensor Input does not define a dataset recipe.")
            }
        }
        var occupied=Set<String>()
        for edge in copy.edges {
            let source=copy.nodes.first {$0.id==edge.source},target=copy.nodes.first {$0.id==edge.target}
            let port=source?.type == "dataset_source" ? "dataset":source?.type == "data_loader" ? "loader":""
            guard !edge.id.isEmpty,edge.id.utf8.count<=128,edge.sourcePort=="out",target?.type=="train",!port.isEmpty,edge.targetPort==port,occupied.insert(port).inserted else {throw LearningContractError("Connect each recipe’s out to its matching dataset or loader input on Train.")}
        }
        if complete {guard copy.nodes.count==3,copy.edges.count==2,occupied==["dataset","loader"] else {throw LearningContractError("Connect Dataset and DataLoader to the supplied Train before checking.")}}
        self.schemaVersion=1;self.kind="dataLoadingLearningGraph";self.taskID=task.rawValue;self.graph=copy
    }
    public func encoded()throws->Data {let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes];return try encoder.encode(self)}
    public static func decode(_ data:Data,task:DataLoadingLearningTask,complete:Bool=true)throws->Self {
        guard data.count<=2_000_000 else {throw LearningContractError("Data loading submission exceeds 2 MB.")}
        let raw=try JSONDecoder().decode(JSONValue.self,from:data)
        let envelope=try TensorLearningSubmission.checkedObject(raw,required:["schemaVersion","kind","taskID","graph"])
        guard envelope["schemaVersion"] == .integer(1),envelope["kind"] == .string("dataLoadingLearningGraph"),envelope["taskID"] == .string(task.rawValue) else {throw LearningContractError("Use this versioned data loading lesson.")}
        let graph=try TensorLearningSubmission.checkedObject(envelope["graph"]!,required:["schemaVersion","name","nodes","edges"])
        guard case .array(let nodes)=graph["nodes"],case .array(let edges)=graph["edges"] else {throw LearningContractError("Use the typed recipe canvas.")}
        for node in nodes {_=try TensorLearningSubmission.checkedObject(node,required:["id","type","name","x","y","config"])}
        for edge in edges {_=try TensorLearningSubmission.checkedObject(edge,required:["id","source","sourcePort","target","targetPort"])}
        struct Envelope:Decodable {let graph:GraphDocument}
        return try .init(task:task,graph:JSONDecoder().decode(Envelope.self,from:data).graph,complete:complete)
    }
}
