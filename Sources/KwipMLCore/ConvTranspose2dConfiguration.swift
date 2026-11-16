import Foundation

/// Bounded transposed convolution; output padding selects extent, not appended data.
public struct ConvTranspose2dConfiguration: Equatable, Sendable {
    public static let maximumParameters = 2_000_000
    public static let maximumMultiplyAdds = 128_000_000
    public let inputChannels: Int
    public let outputChannels: Int
    public let kernelSize: [Int]
    public let stride: [Int]
    public let padding: [Int]
    public let outputPadding: [Int]
    public let dilation: [Int]
    public let groups: Int
    public let bias: Bool
    public let parameterCount: Int

    public init(values: [String: JSONValue]) throws {
        guard Set(values.keys) == ["in_channels", "out_channels", "kernel_size", "stride", "padding", "output_padding", "dilation", "groups", "bias"] else {
            throw GraphConfigurationError("ConvTranspose2d requires exactly channels, kernel, stride, padding, output padding, dilation, groups and bias; only zero padding is supported.")
        }
        func integer(_ key: String, _ bounds: ClosedRange<Int>) throws -> Int {
            guard let value = values[key]?.integerValue, bounds.contains(value) else {
                throw GraphConfigurationError("ConvTranspose2d \(key) must be an integer from \(bounds.lowerBound)–\(bounds.upperBound).")
            }
            return value
        }
        func pair(_ key: String, _ lower: Int = 1) throws -> [Int] {
            guard case .array(let items) = values[key], items.count == 2 else {
                throw GraphConfigurationError("ConvTranspose2d \(key) requires exactly two integers [height, width].")
            }
            return try items.map {
                guard let value = $0.integerValue, (lower...4096).contains(value) else {
                    throw GraphConfigurationError("ConvTranspose2d \(key) entries must be integers from \(lower)–4096.")
                }
                return value
            }
        }
        let input = try integer("in_channels", 1...512), output = try integer("out_channels", 1...512)
        let kernel = try pair("kernel_size"), stride = try pair("stride"), dilation = try pair("dilation")
        let padding = try pair("padding", 0), outputPadding = try pair("output_padding", 0), groups = try integer("groups", 1...512)
        guard let bias = values["bias"]?.boolValue else { throw GraphConfigurationError("ConvTranspose2d bias must be a Boolean.") }
        guard input % groups == 0, output % groups == 0 else { throw GraphConfigurationError("ConvTranspose2d groups must divide both channel counts.") }
        guard (0..<2).allSatisfy({ outputPadding[$0] < stride[$0] || outputPadding[$0] < dilation[$0] }) else { throw GraphConfigurationError("ConvTranspose2d output padding must be smaller than stride or dilation on each axis.") }
        let count = input * (output / groups) * kernel[0] * kernel[1] + (bias ? output : 0)
        guard count <= Self.maximumParameters else { throw GraphConfigurationError("ConvTranspose2d exceeds the 2 million parameter limit.") }
        inputChannels = input; outputChannels = output; kernelSize = kernel; self.stride = stride
        self.padding = padding; self.outputPadding = outputPadding; self.dilation = dilation
        self.groups = groups; self.bias = bias; parameterCount = count
    }
    public var values: [String: JSONValue] {
        func pair(_ values: [Int]) -> JSONValue { .array(values.map { .integer(Int64($0)) }) }
        return ["in_channels": .integer(Int64(inputChannels)), "out_channels": .integer(Int64(outputChannels)),
         "kernel_size": pair(kernelSize), "stride": pair(stride), "padding": pair(padding),
         "output_padding": pair(outputPadding), "dilation": pair(dilation),
         "groups": .integer(Int64(groups)), "bias": .bool(bias)]
    }
    public struct Resolved: Equatable, Sendable {
        public let outputLayout: BatchedTensorLayout
        public let uncroppedOutputShape: [Int]
        public let multiplyAdds: Int
    }
    public func resolved(input: BatchedTensorLayout) throws -> Resolved {
        let shape = input.sample.dimensions
        guard shape.count == 3, shape[0] == inputChannels,
              (1...4096).contains(shape[1]), (1...4096).contains(shape[2]) else {
            throw GraphConfigurationError("ConvTranspose2d requires sample [\(inputChannels), height, width] with spatial dimensions from 1–4096.")
        }
        let uncropped = (0..<2).map {
            (shape[$0 + 1] - 1) * stride[$0] + dilation[$0] * (kernelSize[$0] - 1) + outputPadding[$0] + 1
        }
        let spatial = (0..<2).map { uncropped[$0] - 2 * padding[$0] }
        guard spatial.allSatisfy({ (1...4096).contains($0) }) else {
            throw GraphConfigurationError("ConvTranspose2d output dimensions must be from 1–4096. Review stride, padding, dilation and output padding.")
        }
        // Check final output first; valid output and bounded crop imply uncropped axes <=12288.
        let output = try BatchedTensorLayout(sample: [outputChannels] + spatial)
        guard outputChannels * uncropped[0] * uncropped[1] <= TensorLayout.maximumElements else {
            throw GraphConfigurationError("ConvTranspose2d uncropped output exceeds 100 million elements.")
        }
        let weights = inputChannels * (outputChannels / groups) * kernelSize[0] * kernelSize[1]
        let positions = shape[1] * shape[2]
        guard weights <= Self.maximumMultiplyAdds / positions else {
            throw GraphConfigurationError("ConvTranspose2d exceeds 128 million multiply-adds per observation before cropping.")
        }
        return Resolved(outputLayout: output, uncroppedOutputShape: uncropped, multiplyAdds: positions * weights)
    }
}
