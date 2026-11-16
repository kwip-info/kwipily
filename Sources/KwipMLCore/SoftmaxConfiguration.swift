import Foundation

/// Explicit normalization axis; rank-dependent checks wait for an incoming tensor.
public struct SoftmaxConfiguration:Equatable,Sendable {
    public let axis:Int
    public init(values:[String:JSONValue])throws {
        guard Set(values.keys) == ["dim"],let axis=values["dim"]?.integerValue,
              axis != 0,(-TensorLayout.maximumRank..<TensorLayout.maximumRank).contains(axis) else {
            throw GraphConfigurationError("Softmax requires one integer sample axis (dim) from -8 to 7, excluding the batch axis 0.")
        }
        self.axis=axis
    }
    public var values:[String:JSONValue] {["dim":.integer(Int64(axis))]}
    public func outputLayout(for input:BatchedTensorLayout)throws->BatchedTensorLayout {
        try input.softmax(axis:axis)
    }
}
