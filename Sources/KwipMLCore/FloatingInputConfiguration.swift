import Foundation

/// Alternative of the existing Model Input, never an implicit reshape.
public struct FloatingInputConfiguration {
    public let tensor:TypedTensorContract
    public init(_ values:[String:JSONValue])throws {
        guard Set(values.keys)==["shape"],case .array(let dimensions)?=values["shape"] else {throw GraphConfigurationError("Floating tensor input requires explicit sample dimensions.")}
        let shape=dimensions.compactMap(\.integerValue)
        guard dimensions.count==shape.count else {throw GraphConfigurationError("Sample dimensions must be positive integers.")}
        tensor=try TypedTensorContract(sample:shape,dtype:.float32)
    }
}
