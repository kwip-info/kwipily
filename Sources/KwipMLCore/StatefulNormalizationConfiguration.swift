import Foundation

/// Channel normalization preserves geometry; actual batch statistics belong to execution.
public struct StatefulNormalizationConfiguration: Equatable, Sendable {
    public static let componentIDs: Set<String> = ["batch_norm1d", "batch_norm2d", "instance_norm1d", "instance_norm2d", "group_norm"]
    public static let maximumElements = TensorLayout.maximumElements
    public let componentID: String
    public let values: [String: JSONValue]
    public let channelCount: Int
    public let groupCount: Int?
    public let epsilon: Double
    public let affine: Bool
    public let bias: Bool
    public let tracksRunningStatistics: Bool
    public let averaging: String?
    public let momentum: Double?
    public var parameterCount: Int { affine ? channelCount * (bias ? 2 : 1) : 0 }

    public init(componentID: String, values supplied: [String: JSONValue]) throws {
        guard Self.componentIDs.contains(componentID) else { throw GraphConfigurationError("Choose a supported channel normalization component.") }
        let grouped = componentID == "group_norm", batch = componentID.hasPrefix("batch")
        var values: [String: JSONValue] = ["eps": .number(1e-5), "affine": .bool(!componentID.hasPrefix("instance")), "bias": .bool(true)]
        if grouped { values["num_channels"] = 2; values["num_groups"] = 1 }
        else {
            values["num_features"] = 2; values["momentum"] = .number(0.1); values["track_running_stats"] = .bool(batch)
            if batch { values["averaging"] = .string("moving") }
        }
        guard Set(supplied.keys).isSubset(of: Set(values.keys)) else { throw GraphConfigurationError("Normalization accepts only its declared channel, statistics and learned-adjustment controls.") }
        values.merge(supplied) { _, new in new }
        func integer(_ key: String) throws -> Int {
            guard let number = values[key]?.doubleValue, number.isFinite, (1...512).contains(number), number.rounded(.towardZero) == number else {
                throw GraphConfigurationError("Channels and groups must be whole numbers from 1 to 512.")
            }
            let result = Int(number); values[key] = .integer(Int64(result)); return result
        }
        let channels = try integer(grouped ? "num_channels" : "num_features")
        let groups: Int? = grouped ? try integer("num_groups") : nil
        guard let epsilon = values["eps"]?.doubleValue, epsilon.isFinite, LayerNormConfiguration.epsilonRange.contains(epsilon),
              let affine = values["affine"]?.boolValue, let bias = values["bias"]?.boolValue else {
            throw GraphConfigurationError("Use a positive normal float32 safety value and Boolean learned-adjustment controls.")
        }
        let tracking: Bool
        let momentum: Double?
        let averaging: String?
        if grouped {
            guard channels % groups! == 0 else { throw GraphConfigurationError("Channel groups must divide the channel count exactly.") }
            tracking = false; momentum = nil; averaging = nil
        } else {
            guard let track = values["track_running_stats"]?.boolValue, let update = values["momentum"]?.doubleValue, update.isFinite, (0...1).contains(update) else {
                throw GraphConfigurationError("Track running statistics is a Boolean; its update weight must be a finite number from 0 to 1.")
            }
            tracking = track; momentum = update
            if batch {
                guard let method = values["averaging"]?.stringValue, ["moving", "cumulative"].contains(method) else { throw GraphConfigurationError("Choose Moving or Cumulative averaging.") }
                averaging = method
            } else { averaging = nil }
            values["momentum"] = .number(update)
        }
        values["eps"] = .number(epsilon)
        self.componentID = componentID; self.values = values; self.channelCount = channels; self.groupCount = groups
        self.epsilon = epsilon; self.affine = affine; self.bias = bias; self.tracksRunningStatistics = tracking; self.momentum = momentum; self.averaging = averaging
    }

    public func resolved(layout: BatchedTensorLayout) throws -> BatchedTensorLayout {
        let shape = layout.sample.dimensions
        let ranks: [String: Set<Int>] = ["batch_norm1d": [1, 2], "batch_norm2d": [3], "instance_norm1d": [2], "instance_norm2d": [3], "group_norm": Set(1...7)]
        guard ranks[componentID]!.contains(shape.count), shape.first == channelCount, shape.dropFirst().allSatisfy({ (1...4096).contains($0) }) else {
            throw GraphConfigurationError("Connect the declared channel count and supported spatial directions; observations remain the first direction.")
        }
        if let groups = groupCount, layout.sample.elementCount / groups <= 1 {
            throw GraphConfigurationError("Each channel group needs more than one value per observation. Reduce groups or add spatial positions.")
        }
        return layout
    }
}
