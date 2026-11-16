import Foundation

/// Typed lesson projection over caller-owned published definitions.
public enum EmbeddingBagLearningCanvas {
    public static func registry(from registry:ComponentRegistry)throws->ComponentRegistry {
        guard registry.schemaVersion == 2 else {throw LearningContractError("Refresh the component registry before opening bag lessons.")}
        let sources=registry.components.filter {$0.id == "bag_input"}
        let operators=registry.components.filter {$0.id == "embedding_bag"}
        guard sources.count == 1,operators.count == 1,
              sources[0].inputs.isEmpty,Set(sources[0].outputs) == Set(["indices","offsets","per_sample_weights"]),
              Set(operators[0].inputs) == Set(["indices","offsets","per_sample_weights"]),operators[0].outputs == ["out"],
              Set(sources[0].fields.map(\.key)).count == sources[0].fields.count,
              Set(operators[0].fields.map(\.key)).count == operators[0].fields.count else {throw LearningContractError("Bag lessons require published Bag Input and EmbeddingBag definitions. Refresh runtime support after installing the component update.")}
        _ = try BagSourceConfiguration(values:Dictionary(uniqueKeysWithValues:sources[0].fields.map {($0.key,$0.defaultValue)}))
        _ = try EmbeddingBagConfiguration(values:Dictionary(uniqueKeysWithValues:operators[0].fields.map {($0.key,$0.defaultValue)}))
        return .init(components:[sources[0],operators[0],.init(id:"output",title:"Model Output",category:"Boundaries",inputs:["in"])])
    }
    public static func inspect(_ graph:GraphDocument,registry:ComponentRegistry)->ValidationResult {
        var result=ValidationResult()
        let restricted:ComponentRegistry
        do {
            restricted=try Self.registry(from:registry)
            _ = try EmbeddingBagLearningSubmission.validatedGraph(graph,complete:false)
        } catch {result.issues=[.init(id:"bags.structure",message:error.localizedDescription)];return result}
        result=GraphValidator.validate(graph,registry:restricted,context:.explicitObjective)
        do {_ = try EmbeddingBagLearningSubmission.validatedGraph(graph,complete:true)} catch {
            if !result.issues.contains(where:{$0.message == error.localizedDescription}) {result.issues.append(.init(id:"bags.connections",message:error.localizedDescription))}
        }
        var messages=Set<String>()
        result.issues=result.issues.filter {messages.insert($0.message).inserted}
        return result
    }
    public static func portDescriptor(_ graph:GraphDocument,nodeID:String,port:String,direction:PortDirection)->CanvasPortDescriptor {
        guard let node=graph.nodes.first(where:{$0.id == nodeID}) else {return .unknown(direction:direction)}
        if node.type == "output",direction == .input,port == "in",
           let edge=graph.edges.first(where:{$0.target == nodeID && $0.targetPort == port}),
           let source=graph.nodes.first(where:{$0.id == edge.source && $0.type == "embedding_bag"}),edge.sourcePort == "out",
           let config=try? EmbeddingBagConfiguration(values:source.config) {
            return reduced(config,direction:direction)
        }
        if node.type == "bag_input",direction == .output,
           let config=try? BagSourceConfiguration(values:node.config),config.outputPorts.contains(port) {
            return coordinates(layout:config.layout,terminal:config.includeLastOffset,maximum:config.maximumItemsPerBag,port:port,direction:direction)
        }
        if node.type == "embedding_bag",let config=try? EmbeddingBagConfiguration(values:node.config) {
            if direction == .output,port == "out" {return reduced(config,direction:direction)}
            if direction == .input,config.inputPorts.contains(port) {
                return coordinates(layout:config.layout,terminal:config.includeLastOffset,maximum:nil,port:port,direction:direction)
            }
        }
        return .unknown(direction:direction,reason:"Port disabled by bag settings or not connected to a bag result")
    }
    private static func reduced(_ config:EmbeddingBagConfiguration,direction:PortDirection)->CanvasPortDescriptor {
        .init(kind:.tensor,direction:direction,role:"Combined group vectors",declared:["[bags, \(config.embedding.embeddingDimension)]","float32","One vector per observation, including empty groups","\(config.mode.capitalized) within each group; groups stay separate"])
    }
    private static func coordinates(layout:String,terminal:Bool,maximum:Int?,port:String,direction:PortDirection)->CanvasPortDescriptor {
        let shape=layout == "fixed" ? "[bags, items]":"[items]"
        let geometry=layout == "fixed" ? "Each row is one observation group":"No observation axis in the item array; offsets keep groups separate"
        var declared:[String]
        let role:String
        switch port {
        case "indices":
            role="Category IDs"
            declared=[shape,"int64",geometry,"IDs select table rows; integer gradient unavailable"]
        case "offsets":
            role="Group boundaries"
            declared=[terminal ? "[bags + 1]":"[bags]","int64","Offsets mark group starts; equal offsets preserve empty groups",terminal ? "Final offset equals the number of items":"Final group continues to the end of the item array","Integer gradient unavailable"]
        case "per_sample_weights":
            role="Item influence weights"
            declared=[shape,"float32",geometry,"Real weights stay paired with their item IDs","Weights support a gradient with sum; unused source cells contribute zero"]
        default:return .unknown(direction:direction)
        }
        if let maximum {declared.append("Maximum \(maximum) items per group")}
        return .init(kind:.tensor,direction:direction,role:role,declared:declared)
    }
}
