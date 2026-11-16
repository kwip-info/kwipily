import KwipMLCore

/// Keeps conditional policy keys aligned with the worker's strict JSON schema.
enum FittedDataConfiguration {
    static func choosing(_ value: String, for key: String, in existing: [String: JSONValue]) -> [String: JSONValue] {
        var config = existing; config[key] = .string(value)
        if key == "strategy" { config["tiePolicy"] = value == "mostFrequent" ? config["tiePolicy"] ?? .string("error") : nil }
        if key == "emptyPolicy" { config["emptyValue"] = value == "constant" ? config["emptyValue"] ?? .number(0) : nil }
        return config
    }
}
