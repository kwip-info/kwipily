import Foundation

public struct FileInputOptionChoice: Codable, Equatable {
    public let value: String
    public let title: String
}
public struct FileInputOption: Codable, Equatable, Identifiable {
    public let key: String
    public let title: String
    public let kind: String
    public let defaultValue: JSONValue
    public let choices: [FileInputOptionChoice]?
    public let min: Double?
    public let max: Double?
    public var id: String { key }
}
public struct FileInputReaderDefinition: Codable, Equatable {
    public let title: String
    public let fields: [FileInputOption]
}
public struct FileInputOptionsSchema: Codable, Equatable {
    public let schemaVersion: Int
    public let extensions: [String: [String]]
    public let metadataFields: [String]
    public let readers: [String: FileInputReaderDefinition]
}

/// Portable parsing and collection policy. Access grants and source paths are private state.
public enum FileInputConfiguration {
    private static let catalog: Result<FileInputOptionsSchema, Error> = Result {
        guard let url = Bundle.module.url(forResource: "file-input-options", withExtension: "json") else { throw DataFlowContractError("File Input reader options are unavailable.") }
        let data = try Data(contentsOf: url)
        guard data.count <= 131_072 else { throw DataFlowContractError("File Input reader options exceed their catalog bound.") }
        let schema = try JSONDecoder().decode(FileInputOptionsSchema.self, from: data)
        guard schema.schemaVersion == 1, Set(schema.readers.keys) == ["text", "pdf", "docx", "rtf", "odt", "image", "audio"], Set(schema.extensions.keys) == Set(schema.readers.keys) else { throw DataFlowContractError("Unsupported File Input reader catalog.") }
        for reader in schema.readers.values {
            guard !reader.title.isEmpty, (1...64).contains(reader.fields.count), Set(reader.fields.map(\.key)).count == reader.fields.count else { throw DataFlowContractError("Invalid File Input reader field catalog.") }
            for field in reader.fields {
                guard !field.key.isEmpty, !field.title.isEmpty, ["choice", "boolean", "integer", "number", "string", "integerList"].contains(field.kind) else { throw DataFlowContractError("Unsupported File Input reader field.") }
                if field.kind == "choice" { guard let choices = field.choices, !choices.isEmpty, Set(choices.map(\.value)).count == choices.count else { throw DataFlowContractError("Invalid File Input reader choices.") } }
                if ["integer", "number"].contains(field.kind) { guard let low = field.min, let high = field.max, low.isFinite, high.isFinite, low <= high else { throw DataFlowContractError("Invalid File Input reader bounds.") } }
            }
        }
        return schema
    }
    public static func loadSchema() throws -> FileInputOptionsSchema { try catalog.get() }
    public static var readerDefinitions: [String: FileInputReaderDefinition] { (try? loadSchema().readers) ?? [:] }
    public static var extensionGroups: [String: [String]] { (try? loadSchema().extensions) ?? [:] }
    public static var supportedExtensions: Set<String> { Set(allExtensions) }
    public static var metadataFields: [String] { (try? loadSchema().metadataFields) ?? [] }
    public static var readerOrder: [String] { ["text", "pdf", "docx", "rtf", "odt", "image", "audio"] }
    public static var allExtensions: [String] { readerOrder.flatMap { extensionGroups[$0] ?? [] } }
    public static var newDefaults: [String: JSONValue] {
        ["sourceMode": .string("file"), "extensions": .array(allExtensions.filter { !(extensionGroups["audio"] ?? []).contains($0) }.map(JSONValue.string)),
         "includeSubfolders": .bool(false), "includeHidden": .bool(false),
         "includePatterns": .array([.string("**/*")]), "excludePatterns": .array([]),
         "symlinks": .string("skip"), "readMode": .string("content"), "errorPolicy": .string("fail"), "unknownFiles": .string("metadata"),
         "maxFiles": .integer(1000), "maxDepth": .integer(32), "maxFileBytes": .integer(67_108_864), "maxTotalBytes": .integer(268_435_456),
         "metadataFields": .array(["file_id", "name", "relative_path", "extension", "detected_type", "byte_size"].map(JSONValue.string)),
         "readers": .object(readerDefinitions.mapValues { .object(Dictionary(uniqueKeysWithValues: $0.fields.map { ($0.key, $0.defaultValue) })) })]
    }
    public static func validate(_ config: [String: JSONValue]) throws -> [String: JSONValue] {
        let schema = try loadSchema(), defaults = newDefaults
        guard Set(config.keys).isSubset(of: Set(defaults.keys)) else { throw DataFlowContractError("Unknown File Input setting. Source paths belong in private bindings.") }
        var result = defaults.merging(config) { _, value in value }
        let choices = ["sourceMode": ["file", "directory"], "symlinks": ["skip", "error"], "readMode": ["content", "metadata"], "errorPolicy": ["fail", "record", "skip"], "unknownFiles": ["metadata", "error", "skip"]]
        for (key, values) in choices { guard case .string(let value) = result[key], values.contains(value) else { throw DataFlowContractError("Choose a supported File Input \(key).") } }
        for key in ["includeSubfolders", "includeHidden"] { guard case .bool = result[key] else { throw DataFlowContractError("\(key) requires a Boolean.") } }
        for (key, upper) in ["maxFiles": Int64(1000), "maxDepth": 32, "maxFileBytes": 67_108_864, "maxTotalBytes": 268_435_456] {
            guard case .integer(let value) = result[key], (1...upper).contains(value) else { throw DataFlowContractError("\(key) must be positive and within its File Input resource bound.") }
        }
        if case .integer(let fileBytes) = result["maxFileBytes"], case .integer(let totalBytes) = result["maxTotalBytes"], fileBytes > totalBytes { throw DataFlowContractError("Per-file byte limit must not exceed the total byte limit.") }
        let extensions = try strings(result["extensions"], maximum: 128)
        guard Set(extensions).count == extensions.count, extensions.allSatisfy({ $0.range(of: "^[a-z0-9][a-z0-9_-]{0,31}$", options: .regularExpression) != nil }) else { throw DataFlowContractError("Extensions must be distinct lowercase names without dots or paths. Empty selects all files.") }
        let metadata = try strings(result["metadataFields"], maximum: 128)
        guard Set(metadata).count == metadata.count, metadata.allSatisfy(schema.metadataFields.contains), metadata.contains("file_id") else { throw DataFlowContractError("Keep file_id and choose distinct supported metadata fields.") }
        if result["readMode"] == .string("metadata"), result["metadataFields"] == .array([]) { throw DataFlowContractError("Metadata-only reading needs at least one metadata field.") }
        for key in ["includePatterns", "excludePatterns"] {
            let patterns = try strings(result[key], maximum: 128)
            guard Set(patterns).count == patterns.count else { throw DataFlowContractError("File filter patterns must be distinct.") }
            for pattern in patterns {
                let drive = pattern.range(of: "^[A-Za-z]:", options: .regularExpression) != nil
                guard !pattern.isEmpty, pattern.unicodeScalars.count <= 512, !pattern.contains("\0"), !pattern.hasPrefix("/"), !pattern.contains("\\"), !pattern.contains("://"), !drive, !pattern.split(separator: "/", omittingEmptySubsequences: false).contains("..") else { throw DataFlowContractError("File filters must be relative glob patterns without parent traversal.") }
            }
        }
        guard case .object(let readers) = result["readers"], Set(readers.keys).isSubset(of: Set(schema.readers.keys)) else { throw DataFlowContractError("Unknown File Input reader.") }
        var normalizedReaders: [String: JSONValue] = [:]
        for name in schema.readers.keys {
            guard case .object(let options) = readers[name] ?? .object([:]) else { throw DataFlowContractError("Reader options must be an object.") }
            normalizedReaders[name] = .object(try validateReaderOptions(name, options))
        }
        result["readers"] = .object(normalizedReaders)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        guard try encoder.encode(result).count <= 32_768 else { throw DataFlowContractError("File Input settings exceed their 32 KiB transport bound.") }
        return result
    }
    public static func validateReaderOptions(_ name: String, _ options: [String: JSONValue]) throws -> [String: JSONValue] {
        guard let reader = try loadSchema().readers[name], Set(options.keys).isSubset(of: Set(reader.fields.map(\.key))) else { throw DataFlowContractError("Unknown File Input reader option.") }
        var result = Dictionary(uniqueKeysWithValues: reader.fields.map { ($0.key, $0.defaultValue) })
        result.merge(options) { _, value in value }
        for field in reader.fields {
            let value = result[field.key]!
            func invalid() throws -> Never { throw DataFlowContractError("Choose a valid \(reader.title) \(field.title.lowercased()).") }
            switch field.kind {
            case "choice": guard case .string(let text) = value, field.choices?.contains(where: { $0.value == text }) == true else { try invalid() }
            case "boolean": guard case .bool = value else { try invalid() }
            case "integer": guard case .integer(let integer) = value, let low = field.min, let high = field.max, Double(integer) >= low, Double(integer) <= high else { try invalid() }
            case "number":
                let number: Double
                switch value { case .integer(let integer): number = Double(integer); case .number(let floating): number = floating; default: try invalid() }
                guard number.isFinite, let low = field.min, let high = field.max, (low...high).contains(number) else { try invalid() }
            case "string": guard case .string(let text) = value, text.unicodeScalars.count <= 4096, !text.contains("\0") else { try invalid() }
            case "integerList":
                guard case .array(let values) = value, values.count <= 1000, Set(values).count == values.count else { try invalid() }
                for value in values { guard case .integer(let number) = value, (field.key == "pages" ? (1...1_000_000).contains(number) : [Int64(0), 90, 180, 270].contains(number)) else { try invalid() } }
                if field.key == "orientations", values.isEmpty { try invalid() }
            default: try invalid()
            }
        }
        if name == "image", result["frames"] == .string("range"), case .integer(let start) = result["frameStart"], case .integer(let end) = result["frameEnd"], start > end { throw DataFlowContractError("The last image frame must not precede the first.") }
        return result
    }
    private static func strings(_ value: JSONValue?, maximum: Int) throws -> [String] {
        guard case .array(let values) = value, values.count <= maximum else { throw DataFlowContractError("File Input list exceeds its supported bound.") }
        return try values.map { guard case .string(let text) = $0 else { throw DataFlowContractError("File Input selections require text.") }; return text }
    }
}
