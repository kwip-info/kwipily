import Foundation

/// Channels-first max pooling. Geometry is declared; argmax indices are not measurements here.
public struct MaxPool1dConfiguration: Equatable, Sendable {
    public let kernelSize: Int
    public let stride: Int?
    public let padding: Int
    public let dilation: Int
    public let ceilMode: Bool
    public let returnIndices: Bool
    public var effectiveStride: Int { stride ?? kernelSize }

    public init(values: [String: JSONValue]) throws {
        guard Set(values.keys) == ["kernel_size", "stride", "padding", "dilation", "ceil_mode", "return_indices"] else {
            throw GraphConfigurationError("MaxPool1d requires kernel, stride, padding, dilation, ceiling mode and return-indices settings.")
        }
        func integer(_ key: String, _ bounds: ClosedRange<Int>) throws -> Int {
            guard let value = values[key]?.integerValue, bounds.contains(value) else {
                throw GraphConfigurationError("MaxPool1d \(key) must be an integer from \(bounds.lowerBound)–\(bounds.upperBound).")
            }
            return value
        }
        kernelSize = try integer("kernel_size", 1...4096)
        stride = values["stride"] == .null ? nil : try integer("stride", 1...4096)
        dilation = try integer("dilation", 1...4096)
        padding = try integer("padding", 0...kernelSize / 2)
        guard let ceil = values["ceil_mode"]?.boolValue, let indices = values["return_indices"]?.boolValue else {
            throw GraphConfigurationError("MaxPool1d ceiling mode and return indices must be Boolean values.")
        }
        ceilMode = ceil; returnIndices = indices
    }
    public var values: [String: JSONValue] {
        ["kernel_size":.integer(Int64(kernelSize)), "stride":stride.map { .integer(Int64($0)) } ?? .null,
         "padding":.integer(Int64(padding)), "dilation":.integer(Int64(dilation)),
         "ceil_mode":.bool(ceilMode), "return_indices":.bool(returnIndices)]
    }
    public struct Resolved: Equatable, Sendable {
        public let outputLayout: BatchedTensorLayout
        /// Int64 locations in the input length axis; same geometry as values, no gradient.
        public let indicesLayout: BatchedTensorLayout?
    }
    public func resolved(input: BatchedTensorLayout) throws -> Resolved {
        let shape = input.sample.dimensions
        guard shape.count == 2, (1...512).contains(shape[0]), (1...4096).contains(shape[1]) else {
            throw GraphConfigurationError("MaxPool1d requires sample shape [channels, length], with channels1–512 and length1–4096.")
        }
        let step = effectiveStride
        let numerator = shape[1] + 2 * padding - dilation * (kernelSize - 1) - 1 + (ceilMode ? step - 1 : 0)
        // Swift integer division truncates toward zero; output geometry requires floor.
        var length = numerator / step - (numerator < 0 && numerator % step != 0 ? 1 : 0) + 1
        if ceilMode && (length - 1) * step >= shape[1] + padding { length -= 1 }
        guard length > 0 else {
            throw GraphConfigurationError("MaxPool1d produces no output positions. Reduce kernel/dilation or adjust padding and ceiling mode.")
        }
        let output = try BatchedTensorLayout(sample: [shape[0], length])
        return Resolved(outputLayout: output, indicesLayout: returnIndices ? output : nil)
    }
}
