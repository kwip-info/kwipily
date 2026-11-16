import Foundation

/// Parameter flags only: this is independent of module train/eval behavior.
public struct TrainabilityConfiguration: Equatable {
    public static let componentIDs: Set<String> = ["trainability"]
    public static let maximumRecipeBytes = 32_768
    public static let maximumOwners = 128
    public let componentID: String
    public let values: [String: JSONValue]
    public var owners: [JSONValue] { if case .array(let rows) = values["owners"] { return rows }; return [] }
    public var recipeValue: JSONValue {
        .object(["schemaVersion": 1, "configurationVersion": 1, "componentID": .string(componentID), "config": .object(values)])
    }
    public init(componentID: String = "trainability", values supplied: [String: JSONValue] = [:]) throws {
        guard Self.componentIDs.contains(componentID), Set(supplied.keys).isSubset(of: ["owners"]) else { throw GraphConfigurationError("Choose the declared trainability owner overrides only.") }
        let raw = supplied["owners"] ?? .array([])
        guard case .array(let entries) = raw, entries.count <= Self.maximumOwners else { throw GraphConfigurationError("Select no more than 128 trainability overrides.") }
        var identities = Set<String>()
        var canonical: [(String, String, JSONValue)] = []
        for entry in entries {
            let owner = try GradientPolicyWire.object(entry, keys: ["node", "parameter", "mode"])
            let node = try GradientPolicyWire.name(owner["node"]), parameter = try GradientPolicyWire.name(owner["parameter"])
            guard let mode = owner["mode"]?.stringValue, ["inherit", "trainable", "frozen"].contains(mode),
                  identities.insert(GradientPolicyWire.identity([node, parameter])).inserted else {
                throw GraphConfigurationError("Choose each native parameter once with Inherit, Trainable or Frozen behavior.")
            }
            canonical.append((node, parameter, entry))
        }
        canonical.sort {
            if $0.0 != $1.0 { return $0.0.unicodeScalars.lexicographicallyPrecedes($1.0.unicodeScalars) }
            return $0.1.unicodeScalars.lexicographicallyPrecedes($1.1.unicodeScalars)
        }
        self.componentID = componentID
        self.values = ["owners": .array(canonical.map { $0.2 })]
        try GradientPolicyWire.budget(recipeValue, maximum: Self.maximumRecipeBytes)
    }
    public static func decodeRecipe(_ value: JSONValue) throws -> Self {
        let config = try GradientPolicyWire.recipe(value, componentID: "trainability", fields: ["owners"])
        return try .init(values: config)
    }
}
