import Foundation

/// Lookup-table authoring contract. Execution support must be negotiated separately,
/// including optimizer/device compatibility for sparse gradients and renormalization.
public struct EmbeddingConfiguration:Equatable,Sendable {
    public let numberOfEmbeddings:Int
    public let embeddingDimension:Int
    public let paddingIndex:Int?
    public let maximumNorm:Double?
    public let normType:Double
    public let scaleGradientByFrequency:Bool
    public let sparse:Bool
    public let frozen:Bool
    public var parameterCount:Int {numberOfEmbeddings*embeddingDimension}

    public init(values:[String:JSONValue])throws {
        guard Set(values.keys)==["num_embeddings","embedding_dim","padding_idx","max_norm","norm_type","scale_grad_by_freq","sparse","freeze"],
              let count=values["num_embeddings"]?.integerValue,count>0,
              let width=values["embedding_dim"]?.integerValue,width>0,
              count<=TensorLayout.maximumElements/width,
              let norm=values["norm_type"]?.doubleValue,norm>0,
              let frequency=values["scale_grad_by_freq"]?.boolValue,
              let sparse=values["sparse"]?.boolValue,let frozen=values["freeze"]?.boolValue else {
            throw GraphConfigurationError("Embedding requires positive integer table dimensions within the parameter budget, a positive finite norm order, and boolean gradient/freeze options.")
        }
        let padding:Int?
        if values["padding_idx"] == .null {padding=nil}
        else {
            guard let index=values["padding_idx"]?.integerValue,index >= -count,index<count else {
                throw GraphConfigurationError("Padding index must be null or a table row in -num_embeddings..<num_embeddings.")
            }
            padding=index<0 ? count+index:index
        }
        let maximum:Double?
        if values["max_norm"] == .null {maximum=nil}
        else {
            guard let value=values["max_norm"]?.doubleValue,value>0 else {
                throw GraphConfigurationError("Maximum norm must be null or a positive finite number.")
            }
            maximum=value
        }
        numberOfEmbeddings=count;embeddingDimension=width;paddingIndex=padding
        maximumNorm=maximum;normType=norm;scaleGradientByFrequency=frequency
        self.sparse=sparse;self.frozen=frozen
    }
    public var values:[String:JSONValue] {
        ["num_embeddings":.integer(Int64(numberOfEmbeddings)),"embedding_dim":.integer(Int64(embeddingDimension)),
         "padding_idx":paddingIndex.map {.integer(Int64($0))} ?? .null,
         "max_norm":maximumNorm.map { .number($0) } ?? .null,"norm_type":.number(normType),
         "scale_grad_by_freq":.bool(scaleGradientByFrequency),"sparse":.bool(sparse),"freeze":.bool(frozen)]
    }
    /// Shape algebra only: callers must separately verify integer input dtype and index range.
    public func outputLayout(for input:BatchedTensorLayout)throws->BatchedTensorLayout {
        try BatchedTensorLayout(sample:input.sample.dimensions+[embeddingDimension])
    }
}


extension EmbeddingConfiguration {
    public static func validatedAuthoring(_ values:[String:JSONValue],registry:ComponentRegistry)throws->[String:JSONValue] {
        let result=try Self(values:values)
        guard !(result.sparse && result.scaleGradientByFrequency) else {throw GraphConfigurationError("Sparse gradients cannot use frequency scaling.")}
        return try GraphMigration.effectiveConfiguration(of:.init(type:"embedding",config:result.values),schemaVersion:2,registry:registry)
    }
    public static func applying(_ values:[String:JSONValue],to node:WorkflowNode,registry:ComponentRegistry)throws->WorkflowNode {
        guard case .modelPrimitive("embedding",_,_)=node.payload else {throw GraphConfigurationError("Choose an existing Embedding layer.")}
        let values=try validatedAuthoring(values,registry:registry)
        if let settings=node.parameterSettings {try LayerParameterSettings.validate(settings)}
        var next=node;next.payload = .modelPrimitive(type:"embedding",configurationVersion:2,config:values);return next
    }
    public static func applying(_ values:[String:JSONValue],to node:GraphNode,registry:ComponentRegistry)throws->GraphNode {
        guard node.type=="embedding" else {throw GraphConfigurationError("Choose an existing Embedding layer.")}
        var next=node;next.config=try validatedAuthoring(values,registry:registry);return next
    }
}
