import Foundation

/// Learners supply the connected construction; private fixtures own all data.
public struct EmbeddingBagLearningSubmission:Encodable,Equatable {
    public let schemaVersion=1
    public let kind="embeddingBagGraph"
    public let graph:GraphDocument
    public init(graph:GraphDocument)throws {
        self.graph=try Self.validatedGraph(graph,complete:true)
    }
    /// Partial drafts retain independently valid controls while the learner
    /// updates source and operator one at a time. Checking requires full parity.
    public static func validatedGraph(_ graph:GraphDocument,complete:Bool)throws->GraphDocument {
        guard graph.schemaVersion == 2,graph.execution == nil,graph.compositionPlan == nil,
              graph.nodes.count<=3,graph.edges.count<=4 else {throw LearningContractError("Use one Bag Input, one EmbeddingBag and one Output without training settings.")}
        var nodes:[String:GraphNode]=[:],types=Set<String>()
        var source:GraphNode?,bag:GraphNode?,output:GraphNode?
        var sourceConfiguration:BagSourceConfiguration?,configuration:EmbeddingBagConfiguration?
        for node in graph.nodes {
            guard !node.id.isEmpty,node.id.count<=128,nodes[node.id] == nil,node.x.isFinite,node.y.isFinite,
                  types.insert(node.type).inserted else {throw LearningContractError("Component identities must be unique; use at most one of each lesson component.")}
            nodes[node.id]=node
            switch node.type {
            case "bag_input":source=node;sourceConfiguration=try BagSourceConfiguration(values:node.config)
            case "embedding_bag":bag=node;configuration=try EmbeddingBagConfiguration(values:node.config)
            case "output":
                guard node.config.isEmpty else {throw LearningContractError("Lesson Output uses the supplied group results and has no configuration.")}
                output=node
            default:throw LearningContractError("Use the real Bag Input and EmbeddingBag components, then Output.")
            }
        }
        var actual=Set<[String]>(),ids=Set<String>()
        for edge in graph.edges {
            guard !edge.id.isEmpty,edge.id.count<=128,ids.insert(edge.id).inserted,
                  let from=nodes[edge.source],let to=nodes[edge.target],
                  actual.insert([edge.source,edge.sourcePort,edge.target,edge.targetPort]).inserted else {throw LearningContractError("Connections need existing endpoints and unique identities and roles.")}
            if from.type == "bag_input",to.type == "embedding_bag" {
                guard edge.sourcePort == edge.targetPort,
                      sourceConfiguration?.outputPorts.contains(edge.sourcePort) == true,
                      configuration?.inputPorts.contains(edge.targetPort) == true else {throw LearningContractError("Connect each enabled source role to its matching EmbeddingBag input.")}
            } else {
                guard from.type == "embedding_bag",to.type == "output",edge.sourcePort == "out",edge.targetPort == "in" else {throw LearningContractError("Connect EmbeddingBag output to Model Output; bag coordinates are not ordinary tensor samples.")}
            }
        }
        if complete {
            guard let source,let bag,let output,let sourceConfiguration,let configuration,
                  sourceConfiguration.layout == configuration.layout,
                  sourceConfiguration.includeLastOffset == configuration.includeLastOffset,
                  sourceConfiguration.usesPerSampleWeights == configuration.usesPerSampleWeights else {throw LearningContractError("Complete the three components and match source and bag layout, weights and terminal-offset controls.")}
            let required=Set(configuration.inputPorts.map {[source.id,$0,bag.id,$0]}+[[bag.id,"out",output.id,"in"]])
            guard actual == required else {throw LearningContractError("Connect every enabled Bag Input role, then connect the combined result to Model Output.")}
        }
        // Match the JSON decoder's exact integral spelling locally. This does
        // not change JSONValue equality or convert Boolean/string controls.
        var canonical=graph
        for index in canonical.nodes.indices {
            canonical.nodes[index].config=canonical.nodes[index].config.mapValues(Self.canonicalNumber)
        }
        return canonical
    }
    private static func canonicalNumber(_ value:JSONValue)->JSONValue {
        switch value {
        case .number(let number):
            // Double(Int64.max) rounds up to 2^63, so the upper bound is strict.
            if number.isFinite,number.rounded(.towardZero) == number,
               number>=Double(Int64.min),number < -Double(Int64.min) {
                return .integer(Int64(number))
            }
            return value
        case .array(let values):return .array(values.map(canonicalNumber))
        case .object(let values):return .object(values.mapValues(canonicalNumber))
        default:return value
        }
    }
    public func encoded()throws->Data {
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes]
        let data=try encoder.encode(self)
        guard data.count<=2_000_000 else {throw LearningContractError("Submission exceeds 2 MB.")}
        return data
    }
    public static func decode(_ data:Data)throws->Self {
        guard data.count<=2_000_000 else {throw LearningContractError("Submission exceeds 2 MB.")}
        let raw=try JSONDecoder().decode(JSONValue.self,from:data)
        let fields=try TensorLearningSubmission.checkedObject(raw,required:["schemaVersion","kind","graph"])
        guard fields["schemaVersion"]?.integerValue == 1,fields["kind"]?.stringValue == "embeddingBagGraph" else {throw LearningContractError("Unsupported bag lesson submission.")}
        try TensorLearningSubmission.checkGraphFields(fields["graph"]!)
        struct Envelope:Decodable {let graph:GraphDocument}
        return try Self(graph:JSONDecoder().decode(Envelope.self,from:data).graph)
    }
}
