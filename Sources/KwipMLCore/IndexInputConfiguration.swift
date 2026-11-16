import Foundation

/// Explicit integer source boundary; values must be loaded without float truncation.
public struct IndexInputConfiguration:Equatable {
    public let tensor:TypedTensorContract
    public init(values:[String:JSONValue])throws {
        guard Set(values.keys)==["shape"],case .array(let dimensions)=values["shape"] else {
            throw GraphConfigurationError("Index input requires an explicit sample shape.")
        }
        let shape=dimensions.compactMap(\.integerValue)
        guard shape.count==dimensions.count else {throw GraphConfigurationError("Index input dimensions must be positive integers.")}
        tensor=try TypedTensorContract(sample:shape,dtype:.int64)
    }
    public var values:[String:JSONValue] {["shape":.array(tensor.layout.sample.dimensions.map {.integer(Int64($0))})]}
}
