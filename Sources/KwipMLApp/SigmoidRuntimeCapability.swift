import Foundation
import KwipMLCore

/// Exact optional worker admission. Sigmoid has no configurable mathematical
/// parameters; nonempty configuration and broader dtype/rank claims are refused.
struct SigmoidRuntimeCapability: Decodable, Equatable {
    let version: ContractVersion
    let adapterID: String
    let graphSchemaVersions: [Int]
    let graphDTypes: [String]
    let configuration: [String: JSONValue]
    let outputDTypes: [String: String]
    let maximumRank: Int
    let preservesBatch: Bool
    let preservesShape: Bool
    let parameterless: Bool
    let deterministic: Bool

    private struct Key: CodingKey {
        let stringValue: String
        var intValue: Int? { nil }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { return nil }
        init(_ name: String) { stringValue = name }
    }
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: Key.self)
        let expected: Set<String> = ["version", "adapterID", "graphSchemaVersions", "graphDTypes", "configuration", "outputDTypes", "maximumRank", "preservesBatch", "preservesShape", "parameterless", "deterministic"]
        guard Set(c.allKeys.map(\.stringValue)) == expected else {
            throw GraphConfigurationError("Sigmoid capability has unknown or missing fields.")
        }
        let revision = try c.nestedContainer(keyedBy: Key.self, forKey: Key("version"))
        guard Set(revision.allKeys.map(\.stringValue)) == ["major", "minor", "patch"] else {
            throw GraphConfigurationError("Sigmoid capability version has unknown or missing fields.")
        }
        version = ContractVersion(try revision.decode(Int.self, forKey: Key("major")), try revision.decode(Int.self, forKey: Key("minor")), try revision.decode(Int.self, forKey: Key("patch")))
        adapterID = try c.decode(String.self, forKey: Key("adapterID"))
        graphSchemaVersions = try c.decode([Int].self, forKey: Key("graphSchemaVersions"))
        graphDTypes = try c.decode([String].self, forKey: Key("graphDTypes"))
        configuration = try c.decode([String: JSONValue].self, forKey: Key("configuration"))
        outputDTypes = try c.decode([String: String].self, forKey: Key("outputDTypes"))
        maximumRank = try c.decode(Int.self, forKey: Key("maximumRank"))
        preservesBatch = try c.decode(Bool.self, forKey: Key("preservesBatch"))
        preservesShape = try c.decode(Bool.self, forKey: Key("preservesShape"))
        parameterless = try c.decode(Bool.self, forKey: Key("parameterless"))
        deterministic = try c.decode(Bool.self, forKey: Key("deterministic"))
        guard version == ContractVersion(), adapterID == "kwip.runner.graph-v2.sigmoid", graphSchemaVersions == [2],
              graphDTypes == ["float32"], configuration.isEmpty, outputDTypes == ["out": "float32"],
              maximumRank == 8, preservesBatch, preservesShape, parameterless, deterministic else {
            throw GraphConfigurationError("Sigmoid capability differs from the reviewed schema, empty configuration, dense float32 layout or parameterless behavior.")
        }
    }
    static func decode(_ data: Data) throws -> Self {
        guard data.count <= 4096 else { throw GraphConfigurationError("Sigmoid capability exceeds 4096 bytes.") }
        return try JSONDecoder().decode(Self.self, from: data)
    }
}
