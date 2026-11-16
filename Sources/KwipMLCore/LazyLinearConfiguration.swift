import Foundation

/// Authored settings stay independent of the connected tensor's inferred width.
public struct LazyLinearConfiguration: Equatable, Sendable {
    public static let outputFeatureBounds = 1...512
    public static let inputFeatureBounds = 1...4096
    public static let maximumParameters = 2_000_000
    public let outputFeatures: Int
    public let bias: Bool

    public init(values: [String: JSONValue]) throws {
        guard Set(values.keys) == ["out_features", "bias"],
              let output = values["out_features"]?.integerValue, Self.outputFeatureBounds.contains(output),
              let bias = values["bias"]?.boolValue else {
            throw GraphConfigurationError("LazyLinear requires an output width from 1–512 and a boolean bias choice. Its input width is inferred from the connected tensor.")
        }
        outputFeatures = output
        self.bias = bias
    }

    public var values: [String: JSONValue] {
        ["out_features": .integer(Int64(outputFeatures)), "bias": .bool(bias)]
    }

    public struct Resolved: Equatable, Sendable {
        public let inputFeatures: Int
        public let outputFeatures: Int
        public let bias: Bool
        public let parameterCount: Int
        public let outputLayout: BatchedTensorLayout
    }

    /// Resolves before parameter allocation; only the final sample dimension changes.
    public func resolved(input: BatchedTensorLayout) throws -> Resolved {
        guard let width = input.sample.dimensions.last, Self.inputFeatureBounds.contains(width) else {
            throw GraphConfigurationError("LazyLinear requires an inferred final input width from 1–4096. Reshape the input or reduce its feature width.")
        }
        // Validated width bounds keep arithmetic safe before checking the parameter cap.
        let count = width * outputFeatures + (bias ? outputFeatures : 0)
        guard count <= Self.maximumParameters else {
            throw GraphConfigurationError("LazyLinear exceeds the 2 million parameter limit. Reduce the inferred input width or the output width.")
        }
        return try Resolved(inputFeatures: width, outputFeatures: outputFeatures, bias: bias,
            parameterCount: count, outputLayout: input.linear(inputFeatures: width, outputFeatures: outputFeatures))
    }
}
