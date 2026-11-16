import Foundation

/// A versioned training control; selected native owners are resolved by the captured model.
public struct InitializationConfiguration: Equatable {
    public struct Owner: Equatable {
        public let node: String
        public let parameter: String
    }
    public struct Rule: Equatable {
        public let owner: Owner
        public let algorithm: String
        public let settings: [String: JSONValue]
    }
    public static let componentIDs: Set<String> = ["initialization"]
    public static let maximumRecipeBytes = 32_768
    public let componentID: String
    public let values: [String: JSONValue]
    public let seed: Int64
    public let rules: [Rule]
    public var recipeValue: JSONValue {
        .object(["schemaVersion": 1, "configurationVersion": 1, "componentID": .string(componentID), "config": .object(values)])
    }
    public init(componentID: String = "initialization", values supplied: [String: JSONValue]) throws {
        guard Self.componentIDs.contains(componentID), Set(supplied.keys) == ["seed", "rules"],
              case .integer(let seed) = supplied["seed"], seed >= 0,
              case .array(let rawRules) = supplied["rules"], (1...128).contains(rawRules.count) else {
            throw GraphConfigurationError("Starting settings need an exact nonnegative integer seed and 1–128 selected learned arrays.")
        }
        var rules: [Rule] = []
        var owners = Set<String>()
        for raw in rawRules {
            guard case .object(let rule) = raw, Set(rule.keys) == ["owner", "algorithm", "settings"],
                  case .object(let owner) = rule["owner"], Set(owner.keys) == ["node", "parameter"],
                  let node = owner["node"]?.stringValue, !node.isEmpty, !node.contains("\0"),
                  let parameter = owner["parameter"]?.stringValue, !parameter.isEmpty, !parameter.contains("\0"),
                  let algorithm = rule["algorithm"]?.stringValue,
                  case .object(let suppliedSettings) = rule["settings"] else {
                throw GraphConfigurationError("Select an exact model node, learned-array name, algorithm and its settings.")
            }
            // Length-qualified identity cannot confuse node/parameter separator characters.
            let identity = "\(node.utf8.count):\(node)\(parameter.utf8.count):\(parameter)"
            guard owners.insert(identity).inserted else { throw GraphConfigurationError("Select each learned array only once.") }
            var settings = try Self.defaults(algorithm)
            guard Set(suppliedSettings.keys).isSubset(of: Set(settings.keys)) else {
                throw GraphConfigurationError("Use only the selected starting-value algorithm's controls.")
            }
            settings.merge(suppliedSettings) { _, new in new }
            for key in ["gain", "a", "mean", "std"] where settings[key] != nil {
                let minimum: Double = key == "mean" ? -1_000_000 : 0
                let maximum: Double = ["gain", "a"].contains(key) ? 100 : 1_000_000
                guard let value = settings[key]?.doubleValue, value.isFinite, (minimum...maximum).contains(value),
                      Float(value).isFinite, value == 0 || Float(value) != 0 else {
                    throw GraphConfigurationError("\(key) must fit its finite bounds and remain representable as a float32 value.")
                }
                settings[key] = .number(value)
            }
            if algorithm == "kaiming_uniform" {
                guard let mode = settings["mode"]?.stringValue, ["fan_in", "fan_out"].contains(mode),
                      let nonlinearity = settings["nonlinearity"]?.stringValue, ["relu", "leaky_relu"].contains(nonlinearity) else {
                    throw GraphConfigurationError("Kaiming starting values require incoming or outgoing fan and ReLU or LeakyReLU behavior.")
                }
            }
            rules.append(.init(owner: .init(node: node, parameter: parameter), algorithm: algorithm, settings: settings))
        }
        // Compare exact Unicode scalar sequences, as Python's owner ordering does.
        rules.sort {
            if $0.owner.node != $1.owner.node { return $0.owner.node.unicodeScalars.lexicographicallyPrecedes($1.owner.node.unicodeScalars) }
            return $0.owner.parameter.unicodeScalars.lexicographicallyPrecedes($1.owner.parameter.unicodeScalars)
        }
        self.componentID = componentID
        self.seed = seed
        self.rules = rules
        self.values = ["seed": .integer(seed), "rules": .array(rules.map { rule in
            .object(["owner": .object(["node": .string(rule.owner.node), "parameter": .string(rule.owner.parameter)]),
                     "algorithm": .string(rule.algorithm), "settings": .object(rule.settings)])
        })]
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        // Python canonical JSON spells integral real controls with ".0". Reserve
        // those two bytes per control; this bounds both encodings without claiming
        // their serialized bytes are identical. Owner strings use actual UTF-8.
        let integralRealOverhead = rules.reduce(0) { total, rule in
            total + rule.settings.values.reduce(0) { count, value in
                guard case .number(let scalar) = value else { return count }
                return count + (scalar.rounded(.towardZero) == scalar ? 2 : 0)
            }
        }
        guard try encoder.encode(recipeValue).count + integralRealOverhead <= Self.maximumRecipeBytes else {
            throw GraphConfigurationError("Starting settings exceed their 32,768-byte recipe limit; reduce selected names or rules.")
        }
    }
    public static func decodeRecipe(_ value: JSONValue) throws -> InitializationConfiguration {
        guard case .object(let recipe) = value,
              Set(recipe.keys) == ["schemaVersion", "configurationVersion", "componentID", "config"],
              recipe["schemaVersion"] == .integer(1), recipe["configurationVersion"] == .integer(1),
              let componentID = recipe["componentID"]?.stringValue, case .object(let config) = recipe["config"],
              case .array(let rules) = config["rules"] else {
            throw GraphConfigurationError("Starting settings require the supported recipe version and exact declared envelope.")
        }
        // Wire data must carry all applicable controls; authoring defaults are not a repair policy.
        for rule in rules {
            guard case .object(let object) = rule, let algorithm = object["algorithm"]?.stringValue,
                  case .object(let settings) = object["settings"], Set(settings.keys) == Set(try defaults(algorithm).keys) else {
                throw GraphConfigurationError("Saved starting settings must include every applicable algorithm control.")
            }
        }
        return try .init(componentID: componentID, values: config)
    }
    public static func defaults(_ algorithm: String) throws -> [String: JSONValue] {
        switch algorithm {
        case "xavier_uniform": return ["gain": .number(1)]
        case "kaiming_uniform": return ["a": .number(0), "mode": .string("fan_in"), "nonlinearity": .string("leaky_relu")]
        case "normal": return ["mean": .number(0), "std": .number(1)]
        case "zeros": return [:]
        default: throw GraphConfigurationError("Choose Xavier uniform, Kaiming uniform, normal or zeros for starting values.")
        }
    }

}

public struct InitializationRecipe: Equatable {
    public let configuration: InitializationConfiguration
    public let source: WorkflowNodeAddress
    public init(configuration: InitializationConfiguration, source: WorkflowNodeAddress) {
        self.configuration = configuration
        self.source = source
    }
}
