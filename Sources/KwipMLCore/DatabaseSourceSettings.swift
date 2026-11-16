import Foundation
import CryptoKit

/// Portable, exact-table database settings. Usernames, passwords and local file grants are absent.
public enum DatabaseSourceSettings {
    public static let defaults: [String: JSONValue] = [
        "backend": .string("sqlite"), "host": .string("localhost"), "port": .integer(5432),
        "database": .string(""), "schema": .string("main"), "table": .string(""),
        "columns": .array([]), "orderBy": .array([]), "rowLimit": .integer(0), "rowOffset": .integer(0),
        "fetchBatchRows": .integer(2048), "timeoutSeconds": .integer(30), "tlsMode": .string("verifyFull"),
        "decimalMode": .string("text"), "dateTimeMode": .string("iso8601"), "binaryMode": .string("error")
    ]
    public static func connectionIdentity(_ options: [String: JSONValue]) throws -> String {
        let config = try normalized(options, requireTable: false)
        let keys: Set<String> = ["backend", "host", "port", "database", "tlsMode"]
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        return SHA256.hash(data: try encoder.encode(config.filter { keys.contains($0.key) })).map { String(format: "%02x", $0) }.joined()
    }
    public static func normalized(_ options: [String: JSONValue], requireTable: Bool = true) throws -> [String: JSONValue] {
        guard Set(options.keys).isSubset(of: Set(defaults.keys)) else { throw DataFlowContractError("Unknown database source setting. Credentials belong in private connection storage.") }
        let result = defaults.merging(options) { _, saved in saved }
        func text(_ name: String) throws -> String {
            guard case .string(let value) = result[name], !value.contains("\0"), value.utf8.count <= 256 else { throw DataFlowContractError("Database names must be bounded text.") }
            return value
        }
        let backend = try text("backend"), host = try text("host"), database = try text("database"), schema = try text("schema"), table = try text("table")
        guard ["sqlite", "postgresql", "mysql"].contains(backend), !requireTable || !table.isEmpty else { throw DataFlowContractError("Choose SQLite, PostgreSQL, or MySQL and one exact table.") }
        for (key, bounds) in [("port", 1...65_535), ("rowLimit", 0...5_000_000), ("rowOffset", 0...2_147_483_647), ("fetchBatchRows", 1...8192), ("timeoutSeconds", 1...300)] {
            guard case .integer(let value) = result[key], bounds.contains(Int(value)) else { throw DataFlowContractError("Database source numeric setting is outside its supported range.") }
        }
        for key in ["columns", "orderBy"] {
            guard case .array(let values) = result[key], values.count <= 128 else { throw DataFlowContractError("Choose at most 128 ordered columns.") }
            var seen = Set<String>()
            for value in values {
                guard case .string(let name) = value, !name.isEmpty, !name.contains("\0"), name.utf8.count <= 256, seen.insert(name).inserted else { throw DataFlowContractError("Column selections must be unique bounded names.") }
            }
        }
        for (key, values) in [("tlsMode", ["verifyFull", "disabledLoopback"]), ("decimalMode", ["text", "error"]), ("dateTimeMode", ["iso8601", "error"]), ("binaryMode", ["error", "base64", "hex"])] {
            guard case .string(let value) = result[key], values.contains(value) else { throw DataFlowContractError("Unsupported database TLS or decoding setting.") }
        }
        if backend == "sqlite" {
            guard schema.isEmpty || schema == "main" else { throw DataFlowContractError("SQLite reads only the selected file’s main schema.") }
        } else {
            let allowed = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-")
            guard !host.isEmpty, !requireTable || !database.isEmpty, host.unicodeScalars.allSatisfy({ allowed.contains($0) }) else { throw DataFlowContractError("Enter a hostname or IP address and database, without a connection URI.") }
            if result["tlsMode"] == .string("disabledLoopback") {
                let octets = host.split(separator: ".").compactMap { Int($0) }
                let loopback = host == "::1" || (octets.count == 4 && octets[0] == 127 && octets.allSatisfy { (0...255).contains($0) })
                guard loopback else { throw DataFlowContractError("Unencrypted database access requires a literal loopback IP address.") }
            }
            if backend == "mysql" { guard schema.isEmpty || schema == "main" || schema == database else { throw DataFlowContractError("MySQL schema must match its database.") } }
        }
        return result
    }
}
