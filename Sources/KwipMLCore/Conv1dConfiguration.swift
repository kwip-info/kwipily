import Foundation

/// Bounded channels-first convolution; padding is resolved before any allocation.
public struct Conv1dConfiguration: Equatable, Sendable {
    public static let maximumParameters = 2_000_000
    public static let maximumMultiplyAdds = 128_000_000
    public let inputChannels: Int
    public let outputChannels: Int
    public let kernelSize: Int
    public let stride: Int
    public let dilation: Int
    public let groups: Int
    public let bias: Bool
    public let paddingPolicy: String
    public let padding: Int
    public let paddingMode: String
    public let parameterCount: Int

    public init(values: [String: JSONValue]) throws {
        let keys: Set<String> = ["in_channels", "out_channels", "kernel_size", "stride", "dilation", "groups", "bias", "padding_policy", "padding", "padding_mode"]
        guard Set(values.keys) == keys else { throw GraphConfigurationError("Conv1d requires exactly its channel, kernel, stride, dilation, group, bias and padding settings.") }
        func integer(_ key: String, _ bounds: ClosedRange<Int>) throws -> Int {
            guard let value = values[key]?.integerValue, bounds.contains(value) else {
                throw GraphConfigurationError("Conv1d \(key) must be an integer from \(bounds.lowerBound)–\(bounds.upperBound).")
            }
            return value
        }
        let input = try integer("in_channels", 1...512), output = try integer("out_channels", 1...512)
        let kernel = try integer("kernel_size", 1...4096), stride = try integer("stride", 1...4096)
        let dilation = try integer("dilation", 1...4096), groups = try integer("groups", 1...512)
        let padding = try integer("padding", 0...4096)
        guard let bias = values["bias"]?.boolValue,
              let policy = values["padding_policy"]?.stringValue, ["explicit", "valid", "same"].contains(policy),
              let mode = values["padding_mode"]?.stringValue, ["zeros", "reflect", "replicate", "circular"].contains(mode) else {
            throw GraphConfigurationError("Conv1d requires a Boolean bias, explicit/valid/same padding policy and zeros/reflect/replicate/circular padding mode.")
        }
        guard input % groups == 0, output % groups == 0 else { throw GraphConfigurationError("Conv1d groups must divide both input and output channels.") }
        guard policy == "explicit" || padding == 0 else { throw GraphConfigurationError("Conv1d valid/same padding requires the explicit padding amount to be zero.") }
        guard policy != "same" || stride == 1 else { throw GraphConfigurationError("Conv1d same padding requires stride 1.") }
        let count = output * (input / groups) * kernel + (bias ? output : 0)
        guard count <= Self.maximumParameters else { throw GraphConfigurationError("Conv1d exceeds the 2 million parameter limit. Reduce channels or kernel size, or increase valid groups.") }
        inputChannels = input; outputChannels = output; kernelSize = kernel
        self.stride = stride; self.dilation = dilation; self.groups = groups; self.bias = bias
        paddingPolicy = policy; self.padding = padding; paddingMode = mode; parameterCount = count
    }

    public var values: [String: JSONValue] {
        ["in_channels": .integer(Int64(inputChannels)), "out_channels": .integer(Int64(outputChannels)),
         "kernel_size": .integer(Int64(kernelSize)), "stride": .integer(Int64(stride)), "dilation": .integer(Int64(dilation)),
         "groups": .integer(Int64(groups)), "bias": .bool(bias), "padding_policy": .string(paddingPolicy),
         "padding": .integer(Int64(padding)), "padding_mode": .string(paddingMode)]
    }
    public struct Resolved: Equatable, Sendable {
        public let outputLayout: BatchedTensorLayout
        public let paddingLeft: Int
        public let paddingRight: Int
        public let multiplyAdds: Int
    }
    public func resolved(input: BatchedTensorLayout) throws -> Resolved {
        let shape = input.sample.dimensions
        guard shape.count == 2, shape[0] == inputChannels, (1...4096).contains(shape[1]) else {
            throw GraphConfigurationError("Conv1d requires sample shape [\(inputChannels), length] with length from 1–4096; channels precede length.")
        }
        let axis = try ConvolutionAxisGeometry(length: shape[1], kernel: kernelSize, stride: stride,
            dilation: dilation, padding: padding, policy: paddingPolicy, mode: paddingMode, operation: "Conv1d")
        let outputLength = axis.output
        let work = outputLength * outputChannels * (inputChannels / groups) * kernelSize
        guard work <= Self.maximumMultiplyAdds else { throw GraphConfigurationError("Conv1d exceeds 128 million multiply-adds per observation. Reduce length, channels or kernel size, or increase valid groups/stride.") }
        return try Resolved(outputLayout: BatchedTensorLayout(sample: [outputChannels, outputLength]),
            paddingLeft: axis.before, paddingRight: axis.after, multiplyAdds: work)
    }
}
