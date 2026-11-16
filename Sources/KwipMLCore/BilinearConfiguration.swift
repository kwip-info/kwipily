import Foundation

/// Two independent feature widths with identical leading sample axes.
public struct BilinearConfiguration: Equatable, Sendable {
    public static let featureBounds = 1...512
    public static let maximumParameters = 2_000_000
    public let input1Features: Int
    public let input2Features: Int
    public let outputFeatures: Int
    public let bias: Bool
    public let parameterCount: Int

    public init(values: [String: JSONValue]) throws {
        guard Set(values.keys) == ["in1_features", "in2_features", "out_features", "bias"],
              let first = values["in1_features"]?.integerValue, Self.featureBounds.contains(first),
              let second = values["in2_features"]?.integerValue, Self.featureBounds.contains(second),
              let output = values["out_features"]?.integerValue, Self.featureBounds.contains(output),
              let bias = values["bias"]?.boolValue else {
            throw GraphConfigurationError("Bilinear requires first-input, second-input and output widths from 1–512, and a boolean bias choice.")
        }
        // Width bounds make multiplication safe before applying the graph parameter ceiling.
        let count = output * first * second + (bias ? output : 0)
        guard count <= Self.maximumParameters else {
            throw GraphConfigurationError("Bilinear exceeds the 2 million parameter limit. Reduce either input width or the output width.")
        }
        input1Features = first; input2Features = second; outputFeatures = output
        self.bias = bias; parameterCount = count
    }
    public var values: [String: JSONValue] {
        ["in1_features": .integer(Int64(input1Features)), "in2_features": .integer(Int64(input2Features)),
         "out_features": .integer(Int64(outputFeatures)), "bias": .bool(bias)]
    }
    public func outputLayout(first: BatchedTensorLayout, second: BatchedTensorLayout) throws -> BatchedTensorLayout {
        try first.bilinear(with: second, input1Features: input1Features,
            input2Features: input2Features, outputFeatures: outputFeatures)
    }
}


extension BilinearConfiguration {
    public static func applying(_ values:[String:JSONValue],to node:WorkflowNode,registry:ComponentRegistry)throws->WorkflowNode {
        guard case .modelPrimitive(let type,_,_)=node.payload,type=="bilinear" else {throw GraphConfigurationError("Choose an existing Bilinear layer.")}
        let validated=try Self(values:values)
        _ = try GraphMigration.effectiveConfiguration(of:.init(id:"bilinear",type:"bilinear",config:validated.values),schemaVersion:2,registry:registry)
        if let settings=node.parameterSettings {
            try LayerParameterSettings.validate(settings)
            let choices=try LayerParameterInventory.choices(type:"bilinear",config:validated.values,version:2,registry:registry)
            for (parameter,value) in settings {
                guard let choice=choices.first(where:{$0.id==parameter}) else {throw GraphConfigurationError("This change removes the authored \(parameter) initialization or trainability policy. Remove that policy explicitly before changing Bilinear.")}
                if case .object(let row)=value,case .object(let initialization)=row["initialization"],
                   let algorithm=initialization["algorithm"]?.stringValue,["xavier_uniform","kaiming_uniform"].contains(algorithm),choice.shape.count<2 {
                    throw GraphConfigurationError("Xavier and Kaiming require multidimensional weights. Choose a supported initializer for \(parameter).")
                }
            }
        }
        var next=node;next.payload = .modelPrimitive(type:"bilinear",configurationVersion:2,config:validated.values)
        return next
    }
    public static func applying(_ values:[String:JSONValue],to node:GraphNode,registry:ComponentRegistry)throws->GraphNode {
        let wrapped=WorkflowNode(address:.init(domain:.model,localID:node.id),name:node.name,x:node.x,y:node.y,payload:.modelPrimitive(type:node.type,configurationVersion:2,config:node.config),parameterSettings:node.parameterSettings)
        let changed=try applying(values,to:wrapped,registry:registry)
        guard case .modelPrimitive(_,_,let values)=changed.payload else {throw GraphConfigurationError("Bilinear settings did not resolve an existing layer.")}
        var next=node;next.config=values;return next
    }
}
