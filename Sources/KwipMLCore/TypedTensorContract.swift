import Foundation

/// Concrete sample shape plus element type. This is inferred metadata, not measured data.
/// Operations never silently reinterpret indices as floating-point features.
public struct TypedTensorContract:Equatable {
    public let layout:BatchedTensorLayout
    public let dtype:ContractDType
    public init(sample:[Int],dtype:ContractDType)throws {
        self.layout=try BatchedTensorLayout(sample:sample);self.dtype=dtype
    }
    public init(layout:BatchedTensorLayout,dtype:ContractDType) {
        self.layout=layout;self.dtype=dtype
    }
    public func reshapedSample(_ shape:[Int])throws->Self {
        try Self(layout:layout.reshapedSample(shape),dtype:dtype)
    }
    public func transposed(_ first:Int,_ second:Int)throws->Self {
        try Self(layout:layout.transposed(first,second),dtype:dtype)
    }
    public func flattened(from first:Int,through last:Int)throws->Self {
        try Self(layout:layout.flattened(from:first,through:last),dtype:dtype)
    }
    /// Index range is a runtime value constraint and cannot be proven from shape/type alone.
    public func embedding(_ config:EmbeddingConfiguration,weightType:ContractDType)throws->Self {
        guard dtype == .int32 || dtype == .int64 else {
            throw GraphConfigurationError("Embedding requires int32 or int64 indices; received \(dtype.rawValue). Use an explicitly validated index source.")
        }
        guard weightType == .float32 || weightType == .float64 else {
            throw GraphConfigurationError("Embedding weights require a supported floating-point type.")
        }
        return try Self(layout:config.outputLayout(for:layout),dtype:weightType)
    }
    public func adding(_ other:Self)throws->Self {
        try requireMatchingType(other)
        guard dtype != .string && dtype != .bool else {
            throw GraphConfigurationError("Add requires numeric tensors.")
        }
        return try Self(layout:layout.adding(other.layout),dtype:dtype)
    }
    public func concatenated(with other:Self,axis:Int)throws->Self {
        try requireMatchingType(other)
        return try Self(layout:layout.concatenated(with:other.layout,axis:axis),dtype:dtype)
    }
    private func requireMatchingType(_ other:Self)throws {
        guard dtype == other.dtype else {
            throw GraphConfigurationError("Connected tensors have different types (\(dtype.rawValue), \(other.dtype.rawValue)). Choose an explicit conversion; implicit promotion is not enabled.")
        }
    }
    public func matrixMultiplied(by other:Self)throws->Self {
        try requireMatchingType(other)
        guard dtype == .float32 || dtype == .float64 else {
            throw GraphConfigurationError("Matrix multiplication requires float32 or float64 tensors. Choose an explicit conversion for integer inputs.")
        }
        return try Self(layout:layout.matrixMultiplied(by:other.layout),dtype:dtype)
    }
}
