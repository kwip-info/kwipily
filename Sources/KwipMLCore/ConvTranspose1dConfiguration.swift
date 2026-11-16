import Foundation

/// Bounded transposed convolution; output padding selects extent, not appended data.
public struct ConvTranspose1dConfiguration: Equatable, Sendable {
    public static let maximumParameters = 2_000_000
    public static let maximumMultiplyAdds = 128_000_000
    public let inputChannels: Int
    public let outputChannels: Int
    public let kernelSize: Int
    public let stride: Int
    public let padding: Int
    public let outputPadding: Int
    public let dilation: Int
    public let groups: Int
    public let bias: Bool
    public let parameterCount: Int

    public init(values: [String: JSONValue]) throws {
        guard Set(values.keys) == ["in_channels", "out_channels", "kernel_size", "stride", "padding", "output_padding", "dilation", "groups", "bias"] else {
            throw GraphConfigurationError("ConvTranspose1d requires exactly channels, kernel, stride, padding, output padding, dilation, groups and bias; only zero padding is supported.")
        }
        func integer(_ key: String, _ bounds: ClosedRange<Int>) throws -> Int {
            guard let value = values[key]?.integerValue, bounds.contains(value) else {
                throw GraphConfigurationError("ConvTranspose1d \(key) must be an integer from \(bounds.lowerBound)–\(bounds.upperBound).")
            }
            return value
        }
        let input = try integer("in_channels", 1...512), output = try integer("out_channels", 1...512)
        let kernel = try integer("kernel_size", 1...4096), stride = try integer("stride", 1...4096), dilation = try integer("dilation", 1...4096)
        let padding = try integer("padding", 0...4096), outputPadding = try integer("output_padding", 0...4096), groups = try integer("groups", 1...512)
        guard let bias = values["bias"]?.boolValue else { throw GraphConfigurationError("ConvTranspose1d bias must be a Boolean.") }
        guard input % groups == 0, output % groups == 0 else { throw GraphConfigurationError("ConvTranspose1d groups must divide both channel counts.") }
        guard outputPadding < stride || outputPadding < dilation else { throw GraphConfigurationError("ConvTranspose1d output padding must be smaller than stride or dilation.") }
        let count = input * (output / groups) * kernel + (bias ? output : 0)
        guard count <= Self.maximumParameters else { throw GraphConfigurationError("ConvTranspose1d exceeds the 2 million parameter limit.") }
        inputChannels = input; outputChannels = output; kernelSize = kernel; self.stride = stride
        self.padding = padding; self.outputPadding = outputPadding; self.dilation = dilation
        self.groups = groups; self.bias = bias; parameterCount = count
    }
    public var values: [String: JSONValue] {
        ["in_channels": .integer(Int64(inputChannels)), "out_channels": .integer(Int64(outputChannels)),
         "kernel_size": .integer(Int64(kernelSize)), "stride": .integer(Int64(stride)), "padding": .integer(Int64(padding)),
         "output_padding": .integer(Int64(outputPadding)), "dilation": .integer(Int64(dilation)),
         "groups": .integer(Int64(groups)), "bias": .bool(bias)]
    }
    public struct Resolved: Equatable, Sendable {
        public let outputLayout: BatchedTensorLayout
        public let uncroppedOutputLength: Int
        public let multiplyAdds: Int
    }
    public func resolved(input: BatchedTensorLayout) throws -> Resolved {
        let shape = input.sample.dimensions
        guard shape.count == 2, shape[0] == inputChannels, (1...4096).contains(shape[1]) else {
            throw GraphConfigurationError("ConvTranspose1d requires sample [\(inputChannels), length] with length from 1–4096.")
        }
        let uncropped = (shape[1] - 1) * stride + dilation * (kernelSize - 1) + outputPadding + 1
        let outputLength = uncropped - 2 * padding
        guard (1...4096).contains(outputLength) else { throw GraphConfigurationError("ConvTranspose1d output length must be from 1–4096. Review stride, padding, dilation and output padding.") }
        // Redundant under today's output/crop bounds, retained as explicit scratch admission.
        guard outputChannels * uncropped <= TensorLayout.maximumElements else { throw GraphConfigurationError("ConvTranspose1d uncropped output exceeds 100 million elements.") }
        let work = shape[1] * inputChannels * (outputChannels / groups) * kernelSize
        guard work <= Self.maximumMultiplyAdds else { throw GraphConfigurationError("ConvTranspose1d exceeds 128 million multiply-adds per observation before cropping.") }
        return try Resolved(outputLayout: BatchedTensorLayout(sample: [outputChannels, outputLength]),
            uncroppedOutputLength: uncropped, multiplyAdds: work)
    }
}
