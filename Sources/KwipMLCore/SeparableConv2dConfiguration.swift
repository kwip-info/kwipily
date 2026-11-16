import Foundation

/// Construction settings for an inspectable depthwise → pointwise Block.
/// Both stages use existing convolution contracts; there is no fused execution kernel.
public struct SeparableConv2dConfiguration: Equatable, Sendable {
    public let depthwise: DepthwiseConv2dConfiguration
    public let pointwise: Conv2dConfiguration
    public var parameterCount: Int { depthwise.parameterCount + pointwise.parameterCount }

    public init(values: [String: JSONValue]) throws {
        guard Set(values.keys) == ["in_channels", "channel_multiplier", "out_channels", "kernel_size", "stride",
            "dilation", "depthwise_bias", "pointwise_bias", "padding_policy", "padding", "padding_mode"] else {
            throw GraphConfigurationError("SeparableConv2d requires exactly eleven authored settings, including independent depthwise and pointwise biases. Intermediate channels and pointwise geometry are derived.")
        }
        var spatial = values
        spatial.removeValue(forKey: "out_channels")
        spatial.removeValue(forKey: "pointwise_bias")
        spatial["bias"] = spatial.removeValue(forKey: "depthwise_bias")
        let depthwise = try DepthwiseConv2dConfiguration(values: spatial)
        let pointwise = try Conv2dConfiguration(values: [
            "in_channels": .integer(Int64(depthwise.convolution.outputChannels)),
            "out_channels": values["out_channels"]!, "kernel_size": .array([1, 1]),
            "stride": .array([1, 1]), "dilation": .array([1, 1]), "groups": 1,
            "bias": values["pointwise_bias"]!, "padding_policy": .string("valid"),
            "padding": .array([0, 0]), "padding_mode": .string("zeros")])
        guard depthwise.parameterCount + pointwise.parameterCount <= Conv2dConfiguration.maximumParameters else {
            throw GraphConfigurationError("The depthwise and pointwise stages together exceed the model's 2 million parameter limit.")
        }
        self.depthwise = depthwise; self.pointwise = pointwise
    }
    public var values: [String: JSONValue] {
        var values = depthwise.values
        values["depthwise_bias"] = values.removeValue(forKey: "bias")
        values["pointwise_bias"] = .bool(pointwise.bias)
        values["out_channels"] = .integer(Int64(pointwise.outputChannels))
        return values
    }
    public struct Resolved: Equatable, Sendable {
        public let intermediateLayout: BatchedTensorLayout
        public let outputLayout: BatchedTensorLayout
        public let multiplyAdds: Int
    }
    public func resolved(input: BatchedTensorLayout) throws -> Resolved {
        let spatial = try depthwise.resolved(input: input)
        let mixed = try pointwise.resolved(input: spatial.outputLayout)
        return Resolved(intermediateLayout: spatial.outputLayout, outputLayout: mixed.outputLayout,
                        multiplyAdds: spatial.multiplyAdds + mixed.multiplyAdds)
    }
}
