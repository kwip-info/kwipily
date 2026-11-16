import Foundation

public struct AdaptivePoolingConfiguration: Equatable, Sendable {
    public static let componentIDs: Set<String> = ["adaptive_avg_pool1d", "adaptive_avg_pool2d", "adaptive_max_pool2d"]
    public let componentID: String
    public let values: [String:JSONValue]
    public let outputSizes: [Int?]
    public let returnIndices: Bool
    public init(componentID: String, values: [String:JSONValue]) throws {
        guard Self.componentIDs.contains(componentID) else {
            throw GraphConfigurationError("Unknown adaptive pooling component.")
        }
        let oneDimensional = componentID == "adaptive_avg_pool1d"
        let maximum = componentID == "adaptive_max_pool2d"
        var keys: Set<String> = oneDimensional ? ["output_size"] : ["output_height", "output_width"]
        if maximum { keys.insert("return_indices") }
        guard Set(values.keys) == keys else {
            throw GraphConfigurationError("Adaptive pooling requires exactly its declared output dimensions and optional winning-index setting.")
        }
        func size(_ key: String, nullable: Bool) throws -> Int? {
            if nullable && values[key] == .null { return nil }
            guard let value = values[key]?.integerValue, (1...4096).contains(value) else {
                throw GraphConfigurationError("\(key) must be an integer from 1–4096\(nullable ? ", or empty to keep the input axis" : "").")
            }
            return value
        }
        let sizes = try oneDimensional ? [size("output_size", nullable: false)] : [size("output_height", nullable: true), size("output_width", nullable: true)]
        if maximum && values["return_indices"]?.boolValue == nil {
            throw GraphConfigurationError("Expose winning indices must be Boolean.")
        }
        self.componentID = componentID
        self.values = values
        outputSizes = sizes
        returnIndices = maximum && values["return_indices"]?.boolValue == true
    }
    public struct Resolved: Equatable, Sendable {
        public let outputLayout: BatchedTensorLayout
        public let indicesLayout: BatchedTensorLayout?
    }
    public func resolved(input: BatchedTensorLayout) throws -> Resolved {
        let shape = input.sample.dimensions
        guard shape.count == outputSizes.count + 1, (1...512).contains(shape[0]),
              shape.dropFirst().allSatisfy({ (1...4096).contains($0) }) else {
            throw GraphConfigurationError("Adaptive pooling requires channels1–512 followed by \(outputSizes.count) spatial axes of size1–4096. The batch remains separate.")
        }
        let dimensions = [shape[0]] + outputSizes.enumerated().map { axis, size in size ?? shape[axis + 1] }
        let layout = try BatchedTensorLayout(sample: dimensions)
        return .init(outputLayout: layout, indicesLayout: returnIndices ? layout : nil)
    }
}
