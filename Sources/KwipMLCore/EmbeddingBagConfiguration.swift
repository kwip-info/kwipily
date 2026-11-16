import Foundation

/// Static authoring only. Packet geometry, aggregate model allocation, optimizer
/// and device admission remain independent checks before real execution.
public struct EmbeddingBagConfiguration:Equatable,Sendable {
    public static let defaults:[String:JSONValue] = [
        "num_embeddings":16,"embedding_dim":4,"mode":.string("mean"),"padding_idx":.null,"max_norm":.null,
        "norm_type":.number(2),"scale_grad_by_freq":false,"sparse":false,
        "include_last_offset":false,"freeze":false,"layout":.string("fixed"),"per_sample_weights":false]
    public let embedding:EmbeddingConfiguration
    public let mode:String
    public let layout:String
    public let includeLastOffset:Bool
    public let usesPerSampleWeights:Bool

    public init(values:[String:JSONValue])throws {
        guard Set(values.keys)==Set(Self.defaults.keys),
              let mode=values["mode"]?.stringValue,["mean","sum","max"].contains(mode),
              let layout=values["layout"]?.stringValue,["fixed","packed"].contains(layout),
              let terminal=values["include_last_offset"]?.boolValue,
              let weighted=values["per_sample_weights"]?.boolValue else {
            throw GraphConfigurationError("EmbeddingBag requires all declared controls, a mean/sum/max reduction, fixed/packed layout and Boolean terminal-offset/weight switches.")
        }
        let extra:Set<String>=["mode","layout","include_last_offset","per_sample_weights"]
        let embedding=try EmbeddingConfiguration(values:values.filter{!extra.contains($0.key)})
        for number in [embedding.normType,embedding.maximumNorm].compactMap({$0}) {
            guard number<=Double(Float.greatestFiniteMagnitude),Float(number).isFinite,Float(number)>0 else {
                throw GraphConfigurationError("EmbeddingBag norm controls must remain positive and finite in float32; choose a representable value.")
            }
        }
        guard !(mode=="max" && (embedding.sparse || embedding.scaleGradientByFrequency)),
              !(weighted && mode != "sum"),
              !(embedding.sparse && embedding.scaleGradientByFrequency) else {
            throw GraphConfigurationError("Item weights require sum. Max cannot use sparse or frequency-scaled gradients, and sparse gradients cannot use frequency scaling.")
        }
        self.embedding=embedding;self.mode=mode;self.layout=layout
        includeLastOffset=terminal;usesPerSampleWeights=weighted
    }
    public var parameterCount:Int {embedding.parameterCount}
    public var inputPorts:[String] {
        ["indices"]+(layout=="packed" ? ["offsets"]:[])+(usesPerSampleWeights ? ["per_sample_weights"]:[])
    }
    public var values:[String:JSONValue] {
        var result=embedding.values
        result["mode"] = .string(mode);result["layout"] = .string(layout)
        result["include_last_offset"] = .bool(includeLastOffset)
        result["per_sample_weights"] = .bool(usesPerSampleWeights)
        return result
    }
}


extension EmbeddingBagConfiguration {
    public static func validatedAuthoring(_ values:[String:JSONValue],registry:ComponentRegistry)throws->[String:JSONValue] {
        if values["per_sample_weights"] == .bool(true),values["mode"] != .string("sum") {throw GraphConfigurationError("Item weights require Sum.")}
        if values["mode"] == .string("max"),(values["sparse"] == .bool(true) || values["scale_grad_by_freq"] == .bool(true)) {throw GraphConfigurationError("Maximum cannot use sparse or frequency-scaled gradients.")}
        if values["sparse"] == .bool(true),values["scale_grad_by_freq"] == .bool(true) {throw GraphConfigurationError("Sparse gradients cannot use frequency scaling.")}
        let checked=try Self(values:values)
        return try GraphMigration.effectiveConfiguration(of:.init(type:"embedding_bag",config:checked.values),schemaVersion:2,registry:registry)
    }
    public static func applying(_ values:[String:JSONValue],to node:WorkflowNode,connectedInputPorts:Set<String>=[],registry:ComponentRegistry)throws->WorkflowNode {
        guard case .modelPrimitive("embedding_bag",let version,_)=node.payload else {throw GraphConfigurationError("Choose an existing Embedding bag layer.")}
        let values=try validatedAuthoring(values,registry:registry)
        try protectPorts(values,connectedInputPorts)
        if let settings=node.parameterSettings {try LayerParameterSettings.validate(settings)}
        var next=node;next.payload = .modelPrimitive(type:"embedding_bag",configurationVersion:version,config:values);return next
    }
    public static func applying(_ values:[String:JSONValue],to node:GraphNode,connectedInputPorts:Set<String>=[],registry:ComponentRegistry)throws->GraphNode {
        guard node.type=="embedding_bag" else {throw GraphConfigurationError("Choose an existing Embedding bag layer.")}
        let values=try validatedAuthoring(values,registry:registry)
        try protectPorts(values,connectedInputPorts)
        var next=node;next.config=values;return next
    }
    private static func protectPorts(_ values:[String:JSONValue],_ connected:Set<String>)throws {
        let ports=Set(try Self(values:values).inputPorts)
        guard connected.isSubset(of:ports) else {throw GraphConfigurationError("Disconnect the offsets or item weights input before removing it from Embedding bag.")}
    }
}
