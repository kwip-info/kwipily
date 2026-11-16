import Foundation

/// Global defaults applied independently to each flow folder; never an execution-storage quota.
public struct FlowAuthoringRetentionPreferences: Codable, Equatable, Sendable {
    public let maximumFiles: Int
    public let maximumBytes: Int
    public static let factory = try! Self(maximumFiles: 4096, maximumBytes: 256 * 1024 * 1024)

    public init(maximumFiles: Int, maximumBytes: Int) throws {
        guard (1...100_000).contains(maximumFiles), (1...1_099_511_627_776).contains(maximumBytes) else {
            throw AppPreferencesError.invalid("authoring evidence requires 1–100,000 records and 1 byte–1 TiB per flow folder.")
        }
        self.maximumFiles = maximumFiles; self.maximumBytes = maximumBytes
    }
    private struct Key: CodingKey {
        var stringValue: String
        var intValue: Int? { nil }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { return nil }
    }
    private enum CodingKeys: String, CodingKey { case maximumFiles, maximumBytes }
    public init(from decoder: Decoder) throws {
        let fields = try decoder.container(keyedBy: Key.self)
        guard Set(fields.allKeys.map(\.stringValue)) == ["maximumFiles", "maximumBytes"] else {
            throw AppPreferencesError.invalid("unexpected or missing authoring-evidence fields.")
        }
        let values = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(maximumFiles: values.decode(Int.self, forKey: .maximumFiles), maximumBytes: values.decode(Int.self, forKey: .maximumBytes))
    }
}
