import Foundation

/// Selected derivative values have their own permission, independent of other captures.
public struct DerivativeCapturePolicy: Codable, Equatable {
    public static let maximumWireBytes = 256
    public static let factory = Self(values: false)
    public let schemaVersion: Int
    public let values: Bool

    public init(values: Bool = false) { schemaVersion = 1; self.values = values }
    public var jsonValue: JSONValue { .object(["schemaVersion": 1, "values": .bool(values)]) }

    public init(jsonValue: JSONValue) throws {
        guard case .object(let fields) = jsonValue,
              Set(fields.keys) == ["schemaVersion", "values"],
              case .integer(1) = fields["schemaVersion"],
              case .bool(let permission) = fields["values"] else {
            throw GraphConfigurationError("Use derivative capture schemaVersion 1 and an explicit Boolean values permission only.")
        }
        self.init(values: permission)
    }

    private enum CodingKeys: String, CodingKey { case schemaVersion, values }
    private struct Field: CodingKey {
        let stringValue: String
        var intValue: Int? { nil }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { return nil }
    }
    public init(from decoder: Decoder) throws {
        guard Set(try decoder.container(keyedBy: Field.self).allKeys.map(\.stringValue)) == ["schemaVersion", "values"] else {
            throw GraphConfigurationError("Unknown or missing derivative capture permission fields.")
        }
        let fields = try decoder.container(keyedBy: CodingKeys.self)
        guard try fields.decode(Int.self, forKey: .schemaVersion) == 1 else {
            throw GraphConfigurationError("Unsupported derivative capture permission version.")
        }
        self.init(values: try fields.decode(Bool.self, forKey: .values))
    }

    /// Use this bounded entry point for external JSON. Decoder alone cannot retain a
    /// JSON number's lexical spelling or detect keys discarded by its container.
    public static func decode(_ data: Data) throws -> Self {
        guard data.count <= maximumWireBytes, let text = String(data: data, encoding: .utf8) else {
            throw GraphConfigurationError("Derivative capture permission must be valid UTF-8 JSON within 256 bytes.")
        }
        // This tiny policy has exactly two scalar members. Match their original
        // tokens before decoding keys, preserving integer spelling and duplicates.
        let key = #""(?:[^"\\\x00-\x1F]|\\(?:["\\/bfnrt]|u[0-9A-Fa-f]{4}))*""#
        let scalar = #"(?:true|false|-?(?:0|[1-9][0-9]*))"#
        let whitespace = #"[ \t\r\n]*"#
        let member = "(" + key + ")" + whitespace + ":" + whitespace + "(" + scalar + ")"
        let expression = try NSRegularExpression(pattern: "\\A" + whitespace + "\\{" + whitespace + member + whitespace + "," + whitespace + member + whitespace + "\\}" + whitespace + "\\z")
        guard let match = expression.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) else {
            throw GraphConfigurationError("Use exactly schemaVersion 1 and a Boolean values permission; duplicate fields and numeric versions are refused.")
        }
        var tokens: [String: String] = [:]
        for pair in [(1, 2), (3, 4)] {
            guard let keyRange = Range(match.range(at: pair.0), in: text),
                  let valueRange = Range(match.range(at: pair.1), in: text) else {
                throw GraphConfigurationError("Malformed derivative capture permission.")
            }
            let name = try JSONDecoder().decode(String.self, from: Data(text[keyRange].utf8))
            guard tokens[name] == nil else { throw GraphConfigurationError("Duplicate derivative capture permission field.") }
            tokens[name] = String(text[valueRange])
        }
        guard Set(tokens.keys) == ["schemaVersion", "values"], tokens["schemaVersion"] == "1",
              let permission = tokens["values"], ["true", "false"].contains(permission) else {
            throw GraphConfigurationError("Use derivative capture schemaVersion 1 and an explicit Boolean values permission only.")
        }
        return Self(values: permission == "true")
    }

    public func wireData() throws -> Data {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(self)
        guard data.count <= Self.maximumWireBytes else { throw GraphConfigurationError("Derivative capture permission exceeds 256 bytes.") }
        return data
    }

    public func validate(gradientPolicy: GradientPolicyConfiguration) throws {
        guard gradientPolicy.requests.isEmpty || values else {
            throw GraphConfigurationError("Enable derivative values capture before running selected derivative requests.")
        }
    }
}
