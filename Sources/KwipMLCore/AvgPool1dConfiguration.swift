import Foundation

/// Average windows share MaxPool1d geometry at dilation one, with no index output.
public struct AvgPool1dConfiguration: Equatable, Sendable {
    private let geometry: MaxPool1dConfiguration
    public let countIncludePad: Bool
    public var kernelSize: Int { geometry.kernelSize }
    public var stride: Int? { geometry.stride }
    public var padding: Int { geometry.padding }
    public var ceilMode: Bool { geometry.ceilMode }
    public var effectiveStride: Int { geometry.effectiveStride }
    public init(values: [String: JSONValue]) throws {
        guard Set(values.keys) == ["kernel_size", "stride", "padding", "ceil_mode", "count_include_pad"],
              let include = values["count_include_pad"]?.boolValue else {
            throw GraphConfigurationError("AvgPool1d requires kernel, stride, padding, ceiling mode and Boolean count include pad.")
        }
        countIncludePad = include
        var shared = values
        shared.removeValue(forKey: "count_include_pad")
        shared["dilation"] = 1; shared["return_indices"] = false
        do { geometry = try MaxPool1dConfiguration(values: shared) }
        catch { throw GraphConfigurationError(error.localizedDescription.replacingOccurrences(of: "MaxPool1d", with: "AvgPool1d")) }
    }
    public var values: [String: JSONValue] {
        var result = geometry.values
        result.removeValue(forKey: "dilation"); result.removeValue(forKey: "return_indices")
        result["count_include_pad"] = .bool(countIncludePad)
        return result
    }
    public func resolved(input: BatchedTensorLayout) throws -> MaxPool1dConfiguration.Resolved {
        do { return try geometry.resolved(input: input) }
        catch { throw GraphConfigurationError(error.localizedDescription.replacingOccurrences(of: "MaxPool1d", with: "AvgPool1d")) }
    }
}
