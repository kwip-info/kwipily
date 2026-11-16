import Foundation

/// Lesson-only fixture boundaries. Production model Input(features) is not reinterpreted.
public struct TensorLearningSubmission:Encodable,Equatable {
    public let schemaVersion=1
    public let kind="tensorGraph"
    public let graph:GraphDocument
    public init(graph:GraphDocument,allowSoftmax:Bool=false)throws {self.graph=try Self.validatedGraph(graph,complete:true,allowSoftmax:allowSoftmax)}
    static func validatedGraph(_ graph:GraphDocument,complete:Bool,allowSoftmax:Bool=false,allowLayerNorm:Bool=false,allowEmbedding:Bool=false,allowLazyLinear:Bool=false,convolutionTask:ConvolutionLearningTask?=nil,poolingTask:PoolingLearningTask?=nil,activationTask:ConfigurableActivationLearningTask?=nil,adaptivePoolingTask:AdaptivePoolingLearningTask?=nil)throws->GraphDocument {
        func invalid(_ message:String)->LearningContractError {LearningContractError(message)}
        guard graph.schemaVersion == 2,graph.execution == nil,(0...32).contains(graph.nodes.count),graph.edges.count<=31,(!complete || (graph.nodes.count>=2 && graph.edges.count == graph.nodes.count-1)) else {
            throw invalid("Tensor lessons require a schema-2 path of 2 to 32 nodes without training settings.")
        }
        var copy=graph,nodes:[String:GraphNode]=[:]
        for index in copy.nodes.indices {
            var node=copy.nodes[index]
            guard !node.id.isEmpty,node.id.count<=128,nodes[node.id] == nil,node.x.isFinite,node.y.isFinite else {throw invalid("Component identity and position must be valid and unique.")}
            if ["input","output"].contains(node.type) {
                guard node.config.isEmpty else {throw invalid("Lesson Input and Model Output use fixture shapes and have no configuration.")}
            } else if let task=adaptivePoolingTask,node.type == task.componentID {
                node.config=task == .averageTwo ? try AvgPool2dConfiguration(values:node.config).values : try AdaptivePoolingConfiguration(componentID:task.componentID,values:node.config).values
            } else if let task=activationTask,node.type == task.componentID {
                node.config=try ConfigurableActivationConfiguration(componentID:task.componentID,values:node.config).values
            } else if let task=poolingTask,node.type == task.componentID {
                switch task {
                case .maxOne:node.config=try MaxPool1dConfiguration(values:node.config).values
                case .maxTwo:node.config=try MaxPool2dConfiguration(values:node.config).values
                case .maxThree:node.config=try MaxPool3dConfiguration(values:node.config).values
                case .averageOne:node.config=try AvgPool1dConfiguration(values:node.config).values
                }
            } else if let task=convolutionTask,node.type == task.componentID {
                switch task {
                case .one:node.config=try Conv1dConfiguration(values:node.config).values
                case .two:node.config=try Conv2dConfiguration(values:node.config).values
                case .three:node.config=try Conv3dConfiguration(values:node.config).values
                case .transposedOne:node.config=try ConvTranspose1dConfiguration(values:node.config).values
                case .transposedTwo:node.config=try ConvTranspose2dConfiguration(values:node.config).values
                case .transposedThree:node.config=try ConvTranspose3dConfiguration(values:node.config).values
                }
            } else if node.type == "lazy_linear",allowLazyLinear {
                node.config=try LazyLinearConfiguration(values:node.config).values
            } else if node.type == "embedding",allowEmbedding {
                node.config=try EmbeddingConfiguration(values:node.config).values
            } else if node.type == "layer_norm",allowLayerNorm {
                node.config=try LayerNormConfiguration(values:node.config).values
            } else if node.type == "softmax",allowSoftmax {
                node.config=try SoftmaxConfiguration(values:node.config).values
            } else {node.config=try TensorTransformConfiguration(operation:node.type,values:node.config).values}
            copy.nodes[index]=node;nodes[node.id]=node
        }
        let inputs=copy.nodes.filter {$0.type == "input"},outputs=copy.nodes.filter {$0.type == "output"}
        guard inputs.count<=1,outputs.count<=1,(!complete || (inputs.count == 1 && outputs.count == 1)) else {throw invalid("Exactly one fixture Input and Model Output are required.")}
        var successors:[String:String]=[:],predecessors=Set<String>(),edgeIDs=Set<String>()
        for edge in copy.edges {
            guard !edge.id.isEmpty,edgeIDs.insert(edge.id).inserted,nodes[edge.source] != nil,nodes[edge.target] != nil,
                  edge.sourcePort == "out",edge.targetPort == "in",successors[edge.source] == nil,predecessors.insert(edge.target).inserted,
                  nodes[edge.source]?.type != "output",nodes[edge.target]?.type != "input" else {throw invalid("Connect each transform once along the Input-to-Output path.")}
            successors[edge.source]=edge.target
        }
        if !complete {return copy}
        var visited=Set<String>(),current=inputs[0].id
        while visited.insert(current).inserted {
            if current == outputs[0].id {break}
            guard let next=successors[current] else {throw invalid("The transform path is incomplete.")}
            current=next
        }
        guard current == outputs[0].id,visited.count == nodes.count else {throw invalid("Remove cycles or disconnected transform components.")}
        return copy
    }
    static func checkGraphFields(_ raw:JSONValue,maximumEdges:Int=31)throws {
        let graph=try checkedObject(raw,required:["schemaVersion","name","nodes","edges"])
        guard case .array(let nodes)=graph["nodes"],case .array(let edges)=graph["edges"],(0...32).contains(nodes.count),edges.count<=maximumEdges else {
            throw LearningContractError("Saved tensor graph exceeds its path bounds.")
        }
        for node in nodes {_ = try checkedObject(node,required:["id","type","name","x","y","config"])}
        for edge in edges {_ = try checkedObject(edge,required:["id","source","sourcePort","target","targetPort"])}
    }
    static func checkedObject(_ value:JSONValue,required:Set<String>,optional:Set<String>=[])throws->[String:JSONValue] {
            guard case .object(let fields)=value,required.isSubset(of:Set(fields.keys)),Set(fields.keys).isSubset(of:required.union(optional)) else {
                throw LearningContractError("Saved tensor submission has missing or unknown fields.")
            }
            return fields
        }
    public static func decode(_ data:Data,allowSoftmax:Bool=false)throws->Self {
        guard data.count<=2_000_000 else {throw LearningContractError("Submission exceeds 2 MB.")}
        let raw=try JSONDecoder().decode(JSONValue.self,from:data)
        let envelope=try checkedObject(raw,required:["schemaVersion","kind","graph"])
        guard envelope["schemaVersion"]?.integerValue == 1,envelope["kind"]?.stringValue == "tensorGraph" else {
            throw LearningContractError("Unsupported saved tensor submission.")
        }
        try checkGraphFields(envelope["graph"]!)
        struct Envelope:Decodable {let schemaVersion:Int;let kind:String;let graph:GraphDocument}
        let decoded=try JSONDecoder().decode(Envelope.self,from:data)
        return try Self(graph:decoded.graph,allowSoftmax:allowSoftmax)
    }
    public func encoded()throws->Data {
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes]
        let data=try encoder.encode(self)
        guard data.count<=2_000_000 else {throw LearningContractError("Submission exceeds 2 MB.")}
        return data
    }
}
