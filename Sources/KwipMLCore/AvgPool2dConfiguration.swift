import Foundation

public struct AvgPool2dConfiguration: Equatable, Sendable {
    private let geometry: MaxPool2dConfiguration
    public let countIncludePad: Bool
    public let divisorOverride: Int?
    public var kernelSize: [Int] { geometry.kernelSize }
    public var stride: [Int]? { geometry.stride }
    public var padding: [Int] { geometry.padding }
    public var ceilMode: Bool { geometry.ceilMode }
    public var effectiveStride: [Int] { geometry.effectiveStride }
    public init(values: [String:JSONValue]) throws {
        guard Set(values.keys) == ["kernel_size", "stride", "padding", "ceil_mode", "count_include_pad", "divisor_override"],
              let include = values["count_include_pad"]?.boolValue else {
            throw GraphConfigurationError("AvgPool2d requires window, stride, padding, partial-window, padding-count and divisor settings.")
        }
        if values["divisor_override"] == .null { divisorOverride = nil }
        else {
            guard let divisor = values["divisor_override"]?.integerValue,
                  (-2147483647...2147483647).contains(divisor), divisor != 0 else {
                throw GraphConfigurationError("Average divisor must be a nonzero integer from −2147483647–2147483647, or empty to use the window count.")
            }
            divisorOverride = divisor
        }
        countIncludePad = include
        var shared = values
        shared.removeValue(forKey: "count_include_pad")
        shared.removeValue(forKey: "divisor_override")
        shared["dilation"] = .array([1,1])
        shared["return_indices"] = false
        do { geometry = try MaxPool2dConfiguration(values: shared) }
        catch { throw GraphConfigurationError(error.localizedDescription.replacingOccurrences(of: "MaxPool2d", with: "AvgPool2d")) }
    }
    public var values: [String:JSONValue] {
        var result = geometry.values
        result.removeValue(forKey: "dilation")
        result.removeValue(forKey: "return_indices")
        result["count_include_pad"] = .bool(countIncludePad)
        result["divisor_override"] = divisorOverride.map { .integer(Int64($0)) } ?? .null
        return result
    }
    public func resolved(input: BatchedTensorLayout) throws -> MaxPool2dConfiguration.Resolved {
        do { return try geometry.resolved(input: input) }
        catch { throw GraphConfigurationError(error.localizedDescription.replacingOccurrences(of: "MaxPool2d", with: "AvgPool2d")) }
    }
}
