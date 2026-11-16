import Foundation

/// Channels-first spatial convolution with explicit depth/height/width triples and allocation bounds.
public struct Conv3dConfiguration: Equatable, Sendable {
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
            throw GraphConfigurationError("Conv3d requires exactly its channels, spatial triples, groups, bias and padding settings.")
        }
        func integer(_ key: String) throws -> Int {
            guard let value = values[key]?.integerValue, (1...512).contains(value) else { throw GraphConfigurationError("Conv3d \(key) must be an integer from 1–512.") }
            return value
        }
        func triple(_ key: String, lower: Int = 1) throws -> [Int] {
            guard case .array(let items) = values[key], items.count == 3 else { throw GraphConfigurationError("Conv3d \(key) requires exactly three integers [depth, height, width].") }
            return try items.map {
                guard let value = $0.integerValue, (lower...4096).contains(value) else { throw GraphConfigurationError("Conv3d \(key) entries must be integers from \(lower)–4096.") }
                return value
            }
        }
        let input = try integer("in_channels"), output = try integer("out_channels"), groups = try integer("groups")
        let kernel = try triple("kernel_size"), stride = try triple("stride"), dilation = try triple("dilation"), padding = try triple("padding", lower: 0)
        guard let bias = values["bias"]?.boolValue,
              let policy = values["padding_policy"]?.stringValue, ["explicit", "valid", "same"].contains(policy),
              let mode = values["padding_mode"]?.stringValue, ["zeros", "reflect", "replicate", "circular"].contains(mode) else {
            throw GraphConfigurationError("Conv3d requires a Boolean bias, explicit/valid/same policy and zeros/reflect/replicate/circular mode.")
        }
        guard input % groups == 0, output % groups == 0 else { throw GraphConfigurationError("Conv3d groups must divide both input and output channels.") }
        guard policy == "explicit" || padding == [0, 0, 0] else { throw GraphConfigurationError("Conv3d valid/same padding requires explicit padding [0, 0, 0].") }
        guard policy != "same" || stride == [1, 1, 1] else { throw GraphConfigurationError("Conv3d same padding requires stride [1, 1, 1].") }
        let count = output * (input / groups) * kernel[0] * kernel[1] * kernel[2] + (bias ? output : 0)
        guard count <= Self.maximumParameters else { throw GraphConfigurationError("Conv3d exceeds the 2 million parameter limit. Reduce channels/kernel or increase valid groups.") }
        inputChannels = input; outputChannels = output; kernelSize = kernel; self.stride = stride
        self.dilation = dilation; self.groups = groups; self.bias = bias; paddingPolicy = policy
        self.padding = padding; paddingMode = mode; parameterCount = count
    }
    public var values: [String: JSONValue] {
        func triple(_ values: [Int]) -> JSONValue { .array(values.map { .integer(Int64($0)) }) }
        return ["in_channels": .integer(Int64(inputChannels)), "out_channels": .integer(Int64(outputChannels)),
                "kernel_size": triple(kernelSize), "stride": triple(stride), "dilation": triple(dilation),
                "groups": .integer(Int64(groups)), "bias": .bool(bias), "padding_policy": .string(paddingPolicy),
                "padding": triple(padding), "padding_mode": .string(paddingMode)]
    }
    public struct Resolved: Equatable, Sendable {
        public let outputLayout: BatchedTensorLayout
        public let paddingFront: Int
        public let paddingBack: Int
        public let paddingLeft: Int
        public let paddingRight: Int
        public let paddingTop: Int
        public let paddingBottom: Int
        public let multiplyAdds: Int
    }
    public func resolved(input: BatchedTensorLayout) throws -> Resolved {
        let shape = input.sample.dimensions
        guard shape.count == 4, shape[0] == inputChannels, (1...4096).contains(shape[1]), (1...4096).contains(shape[2]), (1...4096).contains(shape[3]) else {
            throw GraphConfigurationError("Conv3d requires sample [\(inputChannels), depth, height, width], with spatial dimensions from 1–4096.")
        }
        let axes = try (0..<3).map {
            try ConvolutionAxisGeometry(length: shape[$0 + 1], kernel: kernelSize[$0], stride: stride[$0], dilation: dilation[$0],
                padding: padding[$0], policy: paddingPolicy, mode: paddingMode, operation: "Conv3d")
        }
        let depth = axes[0], height = axes[1], width = axes[2]
        // Bounded per-axis lengths keep padded volume multiplication below Int64.max.
        let padded = inputChannels * (shape[1] + depth.before + depth.after)
            * (shape[2] + height.before + height.after) * (shape[3] + width.before + width.after)
        guard padded <= TensorLayout.maximumElements else { throw GraphConfigurationError("Conv3d padded sample exceeds 100 million elements. Reduce spatial size, channels or padding.") }
        let output = try BatchedTensorLayout(sample: [outputChannels, depth.output, height.output, width.output])
        // Validate output volume first; do not multiply six unrestricted spatial/feature factors.
        // Parameter validation already bounds each output channel's fan-in.
        let fanIn = (inputChannels / groups) * kernelSize[0] * kernelSize[1] * kernelSize[2]
        guard fanIn <= Self.maximumMultiplyAdds / output.sample.elementCount else {
            throw GraphConfigurationError("Conv3d exceeds 128 million multiply-adds per observation. Reduce spatial size/channels/kernel or increase valid groups/stride.")
        }
        let work = output.sample.elementCount * fanIn
        return Resolved(outputLayout: output, paddingFront: depth.before, paddingBack: depth.after,
            paddingLeft: width.before, paddingRight: width.after, paddingTop: height.before, paddingBottom: height.after, multiplyAdds: work)
    }
}
