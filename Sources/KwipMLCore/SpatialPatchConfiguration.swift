import Foundation

/// Extract rectangular patches or add patch columns back into an observation's grids.
public struct SpatialPatchConfiguration: Equatable, Sendable {
    public static let componentIDs: Set<String> = ["unfold", "fold"]
    public static let maximumElements = TensorLayout.maximumElements
    public let componentID: String
    public let kernelSize: [Int]
    public let stride: [Int]
    public let dilation: [Int]
    public let padding: [Int]
    public let outputSize: [Int]?

    public init(componentID: String, values: [String: JSONValue]) throws {
        guard Self.componentIDs.contains(componentID) else { throw GraphConfigurationError("Unknown spatial patch component.") }
        let allowed: Set<String> = componentID == "fold" ? ["kernel_size", "stride", "dilation", "padding", "output_size"] : ["kernel_size", "stride", "dilation", "padding"]
        guard Set(values.keys).isSubset(of: allowed) else { throw GraphConfigurationError("Use only this patch block's declared geometry settings.") }
        func pair(_ key: String, defaultValue: [Int], lower: Int = 1) throws -> [Int] {
            let value = values[key] ?? .array(defaultValue.map { .integer(Int64($0)) })
            guard case .array(let items) = value, items.count == 2 else { throw GraphConfigurationError("\(key) needs exactly two whole numbers: height, width.") }
            return try items.map {
                guard let integer = $0.integerValue, (lower...4096).contains(integer) else { throw GraphConfigurationError("\(key) entries must be whole numbers from \(lower) to 4096.") }
                return integer
            }
        }
        self.componentID = componentID
        kernelSize = try pair("kernel_size", defaultValue: [2,2])
        stride = try pair("stride", defaultValue: [1,1])
        dilation = try pair("dilation", defaultValue: [1,1])
        padding = try pair("padding", defaultValue: [0,0], lower: 0)
        outputSize = componentID == "fold" ? try pair("output_size", defaultValue: [2,2]) : nil
    }

    public var values: [String: JSONValue] {
        func pair(_ entries: [Int]) -> JSONValue { .array(entries.map { .integer(Int64($0)) }) }
        var result = ["kernel_size": pair(kernelSize), "stride": pair(stride), "dilation": pair(dilation), "padding": pair(padding)]
        if let outputSize { result["output_size"] = pair(outputSize) }
        return result
    }

    public func resolved(layout: BatchedTensorLayout) throws -> BatchedTensorLayout {
        let shape = layout.sample.dimensions
        let channels: Int, spatial: [Int]
        let area = kernelSize[0] * kernelSize[1] // Validated pair bounds fit Int.
        if componentID == "unfold" {
            guard shape.count == 3, (1...512).contains(shape[0]), shape.dropFirst().allSatisfy({ (1...4096).contains($0) }) else {
                throw GraphConfigurationError("Unfold needs [channels, height, width] per observation: 1–512 channels, height and width 1–4096.")
            }
            channels = shape[0]; spatial = Array(shape.dropFirst())
        } else {
            guard shape.count == 2, shape[0] % area == 0, (1...512).contains(shape[0] / area), let outputSize else {
                throw GraphConfigurationError("Fold needs [packed patch cells, window columns]. Packed cells must divide exactly by the window area into 1–512 channels.")
            }
            channels = shape[0] / area; spatial = outputSize
        }
        let axes = try (0..<2).map { index in
            try ConvolutionAxisGeometry(length: spatial[index], kernel: kernelSize[index], stride: stride[index], dilation: dilation[index], padding: padding[index], policy: "explicit", mode: "zeros", operation: componentID.capitalized)
        }
        let columns = axes[0].output * axes[1].output
        if componentID == "fold", shape[1] != columns { throw GraphConfigurationError("Fold expects \(columns) window columns for this output size and geometry, received \(shape[1]). Adjust output size or use matching patch columns.") }
        let packed = channels * area
        guard packed <= Self.maximumElements / columns else { throw GraphConfigurationError("Patch extraction/addition exceeds 100 million cells per observation. Reduce channels, window size or padding, or increase stride.") }
        return try BatchedTensorLayout(sample: componentID == "unfold" ? [packed,columns] : [channels] + spatial)
    }
}
