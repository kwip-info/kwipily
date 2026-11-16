import Foundation

/// Channels-first spatial convolution with explicit height/width pairs and allocation bounds.
public struct Conv2dConfiguration: Equatable, Sendable {
    public static let maximumParameters = 2_000_000
    public static let maximumMultiplyAdds = 128_000_000
    public let inputChannels: Int
    public let outputChannels: Int
    public let kernelSize: [Int]
    public let stride: [Int]
    public let dilation: [Int]
    public let groups: Int
    public let bias: Bool
    public let paddingPolicy: String
    public let padding: [Int]
    public let paddingMode: String
    public let parameterCount: Int

    public init(values: [String: JSONValue]) throws {
        guard Set(values.keys) == ["in_channels", "out_channels", "kernel_size", "stride", "dilation", "groups", "bias", "padding_policy", "padding", "padding_mode"] else {
            throw GraphConfigurationError("Conv2d requires exactly its channels, spatial pairs, groups, bias and padding settings.")
        }
        func integer(_ key: String) throws -> Int {
            guard let value = values[key]?.integerValue, (1...512).contains(value) else { throw GraphConfigurationError("Conv2d \(key) must be an integer from 1–512.") }
            return value
        }
        func pair(_ key: String, lower: Int = 1) throws -> [Int] {
            guard case .array(let items) = values[key], items.count == 2 else { throw GraphConfigurationError("Conv2d \(key) requires exactly two integers [height, width].") }
            return try items.map {
                guard let value = $0.integerValue, (lower...4096).contains(value) else { throw GraphConfigurationError("Conv2d \(key) entries must be integers from \(lower)–4096.") }
                return value
            }
        }
        let input = try integer("in_channels"), output = try integer("out_channels"), groups = try integer("groups")
        let kernel = try pair("kernel_size"), stride = try pair("stride"), dilation = try pair("dilation"), padding = try pair("padding", lower: 0)
        guard let bias = values["bias"]?.boolValue,
              let policy = values["padding_policy"]?.stringValue, ["explicit", "valid", "same"].contains(policy),
              let mode = values["padding_mode"]?.stringValue, ["zeros", "reflect", "replicate", "circular"].contains(mode) else {
            throw GraphConfigurationError("Conv2d requires a Boolean bias, explicit/valid/same policy and zeros/reflect/replicate/circular mode.")
        }
        guard input % groups == 0, output % groups == 0 else { throw GraphConfigurationError("Conv2d groups must divide both input and output channels.") }
        guard policy == "explicit" || padding == [0, 0] else { throw GraphConfigurationError("Conv2d valid/same padding requires explicit padding [0, 0].") }
        guard policy != "same" || stride == [1, 1] else { throw GraphConfigurationError("Conv2d same padding requires stride [1, 1].") }
        let count = output * (input / groups) * kernel[0] * kernel[1] + (bias ? output : 0)
        guard count <= Self.maximumParameters else { throw GraphConfigurationError("Conv2d exceeds the 2 million parameter limit. Reduce channels/kernel or increase valid groups.") }
        inputChannels = input; outputChannels = output; kernelSize = kernel; self.stride = stride
        self.dilation = dilation; self.groups = groups; self.bias = bias; paddingPolicy = policy
        self.padding = padding; paddingMode = mode; parameterCount = count
    }
    public var values: [String: JSONValue] {
        func pair(_ values: [Int]) -> JSONValue { .array(values.map { .integer(Int64($0)) }) }
        return ["in_channels": .integer(Int64(inputChannels)), "out_channels": .integer(Int64(outputChannels)),
                "kernel_size": pair(kernelSize), "stride": pair(stride), "dilation": pair(dilation),
                "groups": .integer(Int64(groups)), "bias": .bool(bias), "padding_policy": .string(paddingPolicy),
                "padding": pair(padding), "padding_mode": .string(paddingMode)]
    }
    public struct Resolved: Equatable, Sendable {
        public let outputLayout: BatchedTensorLayout
        public let paddingLeft: Int
        public let paddingRight: Int
        public let paddingTop: Int
        public let paddingBottom: Int
        public let multiplyAdds: Int
    }
    public func resolved(input: BatchedTensorLayout) throws -> Resolved {
        let shape = input.sample.dimensions
        guard shape.count == 3, shape[0] == inputChannels, (1...4096).contains(shape[1]), (1...4096).contains(shape[2]) else {
            throw GraphConfigurationError("Conv2d requires sample [\(inputChannels), height, width], with spatial dimensions from 1–4096.")
        }
        let axes = try (0..<2).map {
            try ConvolutionAxisGeometry(length: shape[$0 + 1], kernel: kernelSize[$0], stride: stride[$0], dilation: dilation[$0],
                padding: padding[$0], policy: paddingPolicy, mode: paddingMode, operation: "Conv2d")
        }
        let height = axes[0], width = axes[1]
        let padded = inputChannels * (shape[1] + height.before + height.after) * (shape[2] + width.before + width.after)
        guard padded <= TensorLayout.maximumElements else { throw GraphConfigurationError("Conv2d padded sample exceeds 100 million elements. Reduce spatial size, channels or padding.") }
        // Dimensions are bounded before multiplication; the largest product fits Int64.
        let work = height.output * width.output * outputChannels * (inputChannels / groups) * kernelSize[0] * kernelSize[1]
        guard work <= Self.maximumMultiplyAdds else { throw GraphConfigurationError("Conv2d exceeds 128 million multiply-adds per observation. Reduce spatial size/channels/kernel or increase valid groups/stride.") }
        return try Resolved(outputLayout: BatchedTensorLayout(sample: [outputChannels, height.output, width.output]),
            paddingLeft: width.before, paddingRight: width.after, paddingTop: height.before, paddingBottom: height.after, multiplyAdds: work)
    }
}
