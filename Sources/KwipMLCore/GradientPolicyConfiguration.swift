import Foundation

/// Standalone authoring contract. Native graph resolution belongs to execution admission.
public struct GradientPolicyConfiguration: Equatable {
    public static let componentIDs: Set<String> = ["gradient_policy"]
    public static let maximumRecipeBytes = 32_768
    public static let maximumRequests = 32
    public static let maximumUniqueTargets = 32
    public static let maximumDerivativeElements = 65_536
    public let componentID: String
    public let values: [String: JSONValue]
    public var accumulationSteps: Int { Int(values["accumulation_steps"]!.integerValue!) }
    public var gradientScale: String { values["gradient_scale"]!.stringValue! }
    public var requests: [JSONValue] { if case .array(let rows) = values["requests"] { return rows }; return [] }
    public var recipeValue: JSONValue {
        .object(["schemaVersion": 1, "configurationVersion": 1, "componentID": .string(componentID), "config": .object(values)])
    }
    public init(componentID: String = "gradient_policy", values supplied: [String: JSONValue] = [:]) throws {
        guard Self.componentIDs.contains(componentID), Set(supplied.keys).isSubset(of: ["accumulation_steps", "gradient_scale", "requests"]) else {
            throw GraphConfigurationError("Choose the declared gradient policy controls only.")
        }
        var canonical: [String: JSONValue] = ["accumulation_steps": 1, "gradient_scale": .string("mean"), "requests": .array([])]
        canonical.merge(supplied) { _, new in new }
        _ = try GradientPolicyWire.integer(canonical["accumulation_steps"], 1...32)
        guard let scale = canonical["gradient_scale"]?.stringValue, ["mean", "sum"].contains(scale),
              case .array(let rawRequests) = canonical["requests"], rawRequests.count <= Self.maximumRequests else {
            throw GraphConfigurationError("Choose Mean or Sum and no more than 32 selected derivative requests.")
        }
        var ids = Set<String>(), allTargets = Set<String>(), normalized: [JSONValue] = []
        var upstreamElements = 0
        for raw in rawRequests {
            var request = try GradientPolicyWire.object(raw, keys: ["id", "output", "partition", "row_start", "row_count", "reduction", "upstream", "targets", "unavailable"])
            let id = try GradientPolicyWire.name(request["id"])
            let allowed = CharacterSet(charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-")
            guard (1...64).contains(id.utf8.count), id.unicodeScalars.allSatisfy({ allowed.contains($0) }), ids.insert(id).inserted else {
                throw GraphConfigurationError("Give each derivative request a unique 1–64 character letter, number, underscore or dash ID.")
            }
            let output = try GradientPolicyWire.object(request["output"]!, keys: ["node", "port"])
            _ = try GradientPolicyWire.name(output["node"]); _ = try GradientPolicyWire.name(output["port"])
            guard let partition = request["partition"]?.stringValue, ["training", "validation"].contains(partition) else {
                throw GraphConfigurationError("Select the captured training or validation rows.")
            }
            let start = try GradientPolicyWire.integer(request["row_start"], 0...Int64.max)
            let rows = try GradientPolicyWire.integer(request["row_count"], 1...32)
            guard !start.addingReportingOverflow(rows).overflow else { throw GraphConfigurationError("Selected row range overflows its integer bound.") }
            guard let reduction = request["reduction"]?.stringValue, ["sum", "mean", "upstream"].contains(reduction),
                  let unavailable = request["unavailable"]?.stringValue, ["error", "reportUnavailable"].contains(unavailable),
                  case .array(let targets) = request["targets"], (1...32).contains(targets.count) else {
                throw GraphConfigurationError("Choose a result reduction, unavailable-result policy and 1–32 exact targets.")
            }
            var localTargets = Set<String>()
            for rawTarget in targets {
                guard case .object(let target) = rawTarget, let kind = target["kind"]?.stringValue else {
                    throw GraphConfigurationError("Select an actual model input or learned parameter.")
                }
                let field: String
                switch kind {
                case "input": field = "port"
                case "parameter": field = "parameter"
                default: throw GraphConfigurationError("Derivative targets are model inputs or learned parameters, not detached intermediate values.")
                }
                _ = try GradientPolicyWire.object(rawTarget, keys: ["kind", "node", field])
                let node = try GradientPolicyWire.name(target["node"]), selected = try GradientPolicyWire.name(target[field])
                let identity = GradientPolicyWire.identity([kind, node, selected])
                guard localTargets.insert(identity).inserted else { throw GraphConfigurationError("Select each derivative target only once per request.") }
                allTargets.insert(identity)
                guard allTargets.count <= Self.maximumUniqueTargets else { throw GraphConfigurationError("Select at most 32 distinct derivative targets overall.") }
            }
            if reduction == "upstream" {
                var upstream = try GradientPolicyWire.object(request["upstream"]!, keys: ["dtype", "shape", "values"])
                guard upstream["dtype"] == .string("float32"), case .array(let shape) = upstream["shape"],
                      (1...8).contains(shape.count), case .array(let numbers) = upstream["values"] else {
                    throw GraphConfigurationError("Provide a full float32 upstream array with its exact shape and row-major values.")
                }
                var count = 1
                for dimension in shape {
                    let size = Int(try GradientPolicyWire.integer(dimension, 1...4096))
                    guard count <= Self.maximumDerivativeElements / size else { throw GraphConfigurationError("Upstream arrays exceed their 65,536-element budget.") }
                    count *= size
                }
                guard shape.first == .integer(rows), numbers.count == count, upstreamElements <= Self.maximumDerivativeElements - count else {
                    throw GraphConfigurationError("Upstream shape, selected rows, value count or overall element budget differs.")
                }
                upstreamElements += count
                upstream["values"] = .array(try numbers.map { .number(try GradientPolicyWire.real($0)) })
                request["upstream"] = .object(upstream)
            } else if request["upstream"] != .null {
                throw GraphConfigurationError("Sum and Mean requests require a null upstream array.")
            }
            normalized.append(.object(request))
        }
        canonical["requests"] = .array(normalized)
        self.componentID = componentID; self.values = canonical
        try GradientPolicyWire.budget(recipeValue, maximum: Self.maximumRecipeBytes)
    }
    public static func decodeRecipe(_ value: JSONValue) throws -> Self {
        let recipe = try GradientPolicyWire.recipe(value, componentID: "gradient_policy", fields: ["accumulation_steps", "gradient_scale", "requests"])
        return try .init(values: recipe)
    }
    public func validateObjectiveCompatibility(reduction: String, scalarization: String = "mean") throws {
        guard ["mean", "sum"].contains(scalarization) else { throw GraphConfigurationError("Choose Mean or Sum objective scalarization.") }
        let effective: String
        switch reduction {
        case "mean", "batchmean": effective = "mean"
        case "sum": effective = "sum"
        case "none": effective = scalarization
        default: throw GraphConfigurationError("Choose a supported objective reduction.")
        }
        guard gradientScale == effective else { throw GraphConfigurationError("Gradient scaling must match the selected objective reduction; it never changes the objective.") }
    }
}

/// Internal wire checks shared only by the two standalone policy configurations.
enum GradientPolicyWire {
    static func object(_ value: JSONValue, keys: Set<String>) throws -> [String: JSONValue] {
        guard case .object(let object) = value, Set(object.keys) == keys else { throw GraphConfigurationError("Unknown or missing gradient/trainability fields.") }
        return object
    }
    static func integer(_ value: JSONValue?, _ range: ClosedRange<Int64>) throws -> Int64 {
        guard case .integer(let number) = value, range.contains(number) else { throw GraphConfigurationError("Choose an exact integer within the declared bounds.") }
        return number
    }
    static func name(_ value: JSONValue?) throws -> String {
        guard let text = value?.stringValue, !text.isEmpty, !text.contains("\0") else { throw GraphConfigurationError("Select a nonempty exact captured node, port or native parameter name.") }
        return text
    }
    static func identity(_ parts: [String]) -> String { parts.map { "\($0.utf8.count):\($0)" }.joined() }
    static func real(_ value: JSONValue) throws -> Double {
        guard let number = value.doubleValue, number.isFinite, Float(number).isFinite, number == 0 || Float(number) != 0 else {
            throw GraphConfigurationError("Upstream values must remain finite float32 numbers, without nonzero underflow.")
        }
        return number
    }
    static func recipe(_ value: JSONValue, componentID: String, fields: Set<String>) throws -> [String: JSONValue] {
        let recipe = try object(value, keys: ["schemaVersion", "configurationVersion", "componentID", "config"])
        guard recipe["schemaVersion"] == .integer(1), recipe["configurationVersion"] == .integer(1), recipe["componentID"] == .string(componentID) else {
            throw GraphConfigurationError("Unsupported gradient/trainability recipe identity or version.")
        }
        return try object(recipe["config"]!, keys: fields)
    }
    static func budget(_ value: JSONValue, maximum: Int) throws {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        func overhead(_ value: JSONValue) -> Int {
            switch value {
            case .number(let number): return number.rounded(.towardZero) == number ? 2 : 0
            case .array(let values): return values.reduce(0) { $0 + overhead($1) }
            case .object(let values): return values.values.reduce(0) { $0 + overhead($1) }
            default: return 0
            }
        }
        guard try encoder.encode(value).count + overhead(value) <= maximum else {
            throw GraphConfigurationError("Gradient/trainability recipe exceeds its 32,768-byte portable limit.")
        }
    }
}
