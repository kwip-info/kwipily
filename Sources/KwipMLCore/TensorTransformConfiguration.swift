import Foundation

/// Validated transform settings. Canonical values serialize integer numbers for worker dispatch.
/// Rank-dependent axis and element-count checks occur when an input layout is available.
public struct TensorTransformConfiguration:Equatable,Sendable {
    private enum Operation:Equatable,Sendable {
        case reshape([Int]),transpose(Int,Int),flatten(Int,Int)
    }
    private let operation:Operation

    public init(operation:String,values:[String:JSONValue])throws {
        func fields(_ expected:Set<String>)throws {
            guard Set(values.keys) == expected else {
                throw GraphConfigurationError("\(operation) requires exactly these settings: \(expected.sorted().joined(separator:", ")).")
            }
        }
        func axis(_ key:String)throws->Int {
            guard let value=values[key]?.integerValue,value != 0,
                  (-TensorLayout.maximumRank..<TensorLayout.maximumRank).contains(value) else {
                throw GraphConfigurationError("\(key) must be an integer sample axis; axis 0 is reserved for the batch.")
            }
            return value
        }
        switch operation {
        case "reshape":
            try fields(["shape"])
            guard case .array(let raw)=values["shape"],!raw.isEmpty,raw.count<TensorLayout.maximumRank else {
                throw GraphConfigurationError("Reshape requires 1 to 7 sample dimensions, excluding the batch.")
            }
            let dimensions=try raw.map {value -> Int in
                guard let dimension=value.integerValue,dimension == -1 || dimension>0 else {
                    throw GraphConfigurationError("Reshape dimensions must be positive integers or one inferred -1.")
                }
                return dimension
            }
            guard dimensions.filter({$0 == -1}).count<=1 else {
                throw GraphConfigurationError("Only one reshape dimension can be inferred with -1.")
            }
            // Substituting one checks the minimum possible size without guessing the inferred axis.
            _ = try TensorLayout(dimensions.map {$0 == -1 ? 1:$0})
            self.operation = .reshape(dimensions)
        case "transpose":
            try fields(["dim0","dim1"])
            self.operation = .transpose(try axis("dim0"),try axis("dim1"))
        case "flatten":
            try fields(["start_dim","end_dim"])
            self.operation = .flatten(try axis("start_dim"),try axis("end_dim"))
        default:throw GraphConfigurationError("Unknown tensor transform: \(operation).")
        }
    }
    public var values:[String:JSONValue] {
        func integer(_ value:Int)->JSONValue {.integer(Int64(value))}
        switch operation {
        case .reshape(let shape):return ["shape":.array(shape.map(integer))]
        case .transpose(let first,let second):return ["dim0":integer(first),"dim1":integer(second)]
        case .flatten(let first,let last):return ["start_dim":integer(first),"end_dim":integer(last)]
        }
    }
    public func outputLayout(for input:BatchedTensorLayout)throws->BatchedTensorLayout {
        switch operation {
        case .reshape(let shape):return try input.reshapedSample(shape)
        case .transpose(let first,let second):return try input.transposed(first,second)
        case .flatten(let first,let last):return try input.flattened(from:first,through:last)
        }
    }
}
