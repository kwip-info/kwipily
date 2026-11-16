import Foundation

/// Explicit trailing sample normalization, mirroring the float32 worker contract.
public struct LayerNormConfiguration:Equatable,Sendable {
    public static let epsilonRange=1.1754943508222875e-38...3.4028234663852886e38
    public let normalizedShape:[Int]
    public let epsilon:Double
    public let elementwiseAffine:Bool
    public let bias:Bool
    public let parameterCount:Int

    public init(values:[String:JSONValue])throws {
        guard Set(values.keys)==["normalized_shape","eps","elementwise_affine","bias"],
              case .array(let dimensions)=values["normalized_shape"],
              (1..<TensorLayout.maximumRank).contains(dimensions.count),
              let epsilon=values["eps"]?.doubleValue,Self.epsilonRange.contains(epsilon),
              let affine=values["elementwise_affine"]?.boolValue,let bias=values["bias"]?.boolValue else {
            throw GraphConfigurationError("Layer normalization requires 1–7 trailing sample dimensions, a positive normal float32 epsilon, and boolean scale/bias options.")
        }
        let shape=dimensions.compactMap(\.integerValue)
        guard shape.count==dimensions.count else {throw GraphConfigurationError("Normalized dimensions must be positive integers.")}
        let layout=try TensorLayout(shape)
        self.normalizedShape=shape;self.epsilon=epsilon;self.elementwiseAffine=affine;self.bias=bias
        self.parameterCount=affine ? layout.elementCount*(bias ? 2:1):0
    }
    public var values:[String:JSONValue] {
        ["normalized_shape":.array(normalizedShape.map { .integer(Int64($0)) }),"eps":.number(epsilon),"elementwise_affine":.bool(elementwiseAffine),"bias":.bool(bias)]
    }
    public func outputLayout(for input:BatchedTensorLayout)throws->BatchedTensorLayout {
        guard normalizedShape.count<=input.sample.dimensions.count,
              Array(input.sample.dimensions.suffix(normalizedShape.count))==normalizedShape else {
            throw GraphConfigurationError("Normalized shape must match trailing sample dimensions exactly; the batch stays separate.")
        }
        return input
    }
}
