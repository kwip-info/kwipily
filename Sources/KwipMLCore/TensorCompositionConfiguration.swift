import Foundation

/// Shape and configuration rules shared by canvas validation and assessed lessons.
/// Observation direction zero is preserved; repeating dimensions is always opt-in.
public struct TensorCompositionConfiguration: Equatable, Sendable {
    public static let componentIDs: Set<String> = ["concat", "add", "stack", "permute"]
    public let componentID: String
    public let dim: Int?
    public let dims: [Int]?
    public let alpha: Double
    public let sampleBroadcast: Bool
    public let values: [String: JSONValue]

    public init(componentID: String, values: [String: JSONValue]) throws {
        guard Self.componentIDs.contains(componentID) else { throw GraphConfigurationError("Unknown tensor composition component.") }
        self.componentID = componentID
        var axis: Int?, order: [Int]?, multiplier = 1.0, broadcast = false
        var canonical: [String: JSONValue]
        switch componentID {
        case "concat", "stack":
            guard Set(values.keys).isSubset(of: ["dim"]) else { throw GraphConfigurationError("Only the direction setting is supported.") }
            let selected = values.isEmpty ? -1 : values["dim"]?.integerValue
            guard let selected, selected != 0, (-8...7).contains(selected) else { throw GraphConfigurationError("Choose an integer sample direction from -8 to 7; direction 0 is reserved for observations.") }
            axis = selected; canonical = ["dim": .integer(Int64(selected))]
        case "add":
            guard Set(values.keys).isSubset(of: ["alpha", "sample_broadcast"]) else { throw GraphConfigurationError("Add supports a second-source multiplier and sample repetition only.") }
            if let value = values["alpha"] {
                switch value {
                case .integer(let n): multiplier = Double(n)
                case .number(let n): multiplier = n
                default: throw GraphConfigurationError("The second-source multiplier must be a finite number.")
                }
            }
            guard multiplier.isFinite, abs(multiplier) <= Double(Float.greatestFiniteMagnitude) else { throw GraphConfigurationError("The multiplier must fit a finite 32-bit number.") }
            if let value = values["sample_broadcast"] {
                guard case .bool(let enabled) = value else { throw GraphConfigurationError("Sample repetition must be on or off.") }
                broadcast = enabled
            }
            canonical = ["alpha": .number(multiplier), "sample_broadcast": .bool(broadcast)]
            if multiplier == 1 { canonical["alpha"] = .integer(1) }
        default:
            guard Set(values.keys).isSubset(of: ["dims"]) else { throw GraphConfigurationError("Permute supports only the direction order.") }
            let raw = values.isEmpty ? JSONValue.array([0, 2, 1]) : values["dims"]!
            guard case .array(let entries) = raw, (2...8).contains(entries.count) else { throw GraphConfigurationError("List every direction once, with observations first.") }
            let integers = entries.compactMap(\.integerValue)
            guard integers.count == entries.count else { throw GraphConfigurationError("Direction order must contain integers only.") }
            let rank = integers.count
            guard integers.allSatisfy({ (-rank..<rank).contains($0) }) else { throw GraphConfigurationError("A direction is outside the listed rank.") }
            let normalized = integers.map { $0 < 0 ? $0 + rank : $0 }
            guard normalized.first == 0, Set(normalized).count == rank else { throw GraphConfigurationError("Keep observations first and include each direction exactly once.") }
            order = integers; canonical = ["dims": .array(integers.map { .integer(Int64($0)) })]
        }
        dim = axis; dims = order; alpha = multiplier; sampleBroadcast = broadcast; self.values = canonical
    }

    public func resolved(layouts: [BatchedTensorLayout]) throws -> BatchedTensorLayout {
        let count = componentID == "permute" ? 1 : 2
        guard layouts.count == count else { throw GraphConfigurationError("Connect exactly \(count) ordered sources.") }
        let first = layouts[0]
        switch componentID {
        case "concat": return try first.concatenated(with: layouts[1], axis: dim!)
        case "add": return try first.adding(layouts[1], broadcastingSamples: sampleBroadcast)
        case "stack": return try first.stacked(with: layouts[1], axis: dim!)
        default: return try first.permuted(dims!)
        }
    }

    public func resolved(inputs: [TypedTensorContract]) throws -> TypedTensorContract {
        guard let first = inputs.first, inputs.allSatisfy({ $0.dtype == first.dtype }) else { throw GraphConfigurationError("Sources must use the same data type; convert them explicitly before joining.") }
        guard first.dtype != .string, first.dtype != .bool else { throw GraphConfigurationError("These components require numeric arrays.") }
        if componentID == "add", (alpha != 1 || sampleBroadcast), first.dtype != .float32 {
            throw GraphConfigurationError("A custom multiplier or sample repetition requires 32-bit decimal arrays.")
        }
        return TypedTensorContract(layout: try resolved(layouts: inputs.map(\.layout)), dtype: first.dtype)
    }
}
