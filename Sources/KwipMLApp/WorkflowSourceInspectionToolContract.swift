import Foundation
import KwipMLCore

/// Source inspection accepts one captured source and never reopens a caller-owned path.
enum WorkflowSourceInspectionToolContract {
    static func configurationJSON(_ configuration: [String: JSONValue]) throws -> String {
        try TableInputConfiguration.transportJSON(configuration)
    }
    static func fileLimit(path: String) throws -> Int {
        if path.hasPrefix("input/file-collections/") {return try WorkflowPreparationToolContract.fileLimit(path:path)}
        let parts = path.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 3, parts[0] == "input", parts[1] == "data",
              !parts[2].isEmpty, parts[2] != ".", parts[2] != "..",
              !parts[2].contains("\0"), parts[2].utf8.count <= 255 else { throw WorkflowFileLeaseError.unsafePath }
        return 10_000_000
    }
    static func validate(arguments: [String], files: [WorkflowArchiveFile], directory: URL?) throws {
        if arguments.count==4,arguments[0]=="--inspect-data",arguments[2]=="--file-options" {
            guard (1...9).contains(files.count),arguments[3].utf8.count<=32768,arguments[1].utf8.count<=4096,
                  arguments.allSatisfy({!$0.isEmpty && !$0.contains("\0")}),
                  let config=try? JSONDecoder().decode([String:JSONValue].self,from:Data(arguments[3].utf8)),
                  (try? FileInputConfiguration.validate(config)) != nil else {throw WorkflowFileLeaseError.unsafePath}
            let manifests=files.filter {$0.path.hasPrefix("input/file-collections/") && $0.path.hasSuffix("/manifest.json")}
            guard manifests.count==1 else {throw WorkflowFileLeaseError.unsafePath}
            let manifest=manifests[0],parts=try WorkflowArchiveCodec.pathParts(manifest.path)
            guard parts.count==4,parts[2]==manifest.sha256,arguments[1].hasPrefix("/"),!arguments[1].contains("//"),!arguments[1].split(separator:"/").contains(".."),!arguments[1].split(separator:"/").contains(".") else {throw WorkflowFileLeaseError.unsafePath}
            if let directory {guard arguments[1]==directory.appendingPathComponent(manifest.path).path else {throw WorkflowFileLeaseError.unsafePath}}
            else {guard arguments[1].hasSuffix("/"+manifest.path) else {throw WorkflowFileLeaseError.unsafePath}}
            guard Set(files.map(\.path)).count==files.count else {throw WorkflowFileLeaseError.unsafePath}
            for file in files {guard file.path.hasPrefix("input/file-collections/"+manifest.sha256+"/"),file.bytes>=0,file.bytes <= (try fileLimit(path:file.path)),WorkflowArchiveCodec.isHash(file.sha256) else {throw WorkflowFileLeaseError.unsafePath}}
            return
        }
        guard files.count == 1, arguments.count == 4, arguments[0] == "--inspect-data",
              ["--csv-separator", "--csv-options", "--table-options"].contains(arguments[2]),
              arguments.enumerated().allSatisfy({ index, value in
                  !value.isEmpty && !value.contains("\0")
                    && value.utf8.count <= (index == 3 && arguments[2] == "--table-options" ? 32_768 : 4096)
              }) else { throw WorkflowFileLeaseError.unsafePath }
        if arguments[2] == "--csv-separator" {
            guard CSVParsingConfiguration.isSeparator(arguments[3]) else { throw WorkflowFileLeaseError.unsafePath }
        } else {
            guard let data = arguments[3].data(using: .utf8),
                  let config = try? JSONDecoder().decode([String: JSONValue].self, from: data),
                  (arguments[2] == "--table-options"
                    ? (try? TableInputConfiguration.validate(config)) != nil
                    : (try? CSVParsingConfiguration.validate(config)) != nil) else { throw WorkflowFileLeaseError.unsafePath }
            if arguments[2] == "--table-options" {
                guard (try? TableInputConfiguration.format(of: config)) != .csv || CSVParsingConfiguration.isPolars(config) else { throw WorkflowFileLeaseError.unsafePath }
            }
        }
        let file = files[0], limit = try fileLimit(path: files[0].path)
        guard (0...limit).contains(file.bytes), WorkflowArchiveCodec.isHash(file.sha256),
              arguments[1].hasPrefix("/"), !arguments[1].contains("//"),
              !arguments[1].split(separator: "/").contains("."), !arguments[1].split(separator: "/").contains("..") else { throw WorkflowFileLeaseError.unsafePath }
        if let directory {
            guard directory.isFileURL, arguments[1] == directory.appendingPathComponent(file.path).path else { throw WorkflowFileLeaseError.unsafePath }
        } else {
            guard arguments[1].hasSuffix("/" + file.path) else { throw WorkflowFileLeaseError.unsafePath }
        }
    }
    static func verifyInputs(arguments:[String]=[],files: [WorkflowArchiveFile], directory: URL) throws {
        if let manifest=files.first(where:{$0.path.hasPrefix("input/file-collections/") && $0.path.hasSuffix("/manifest.json")}) {
            guard (1...9).contains(files.count),Set(files.map(\.path)).count==files.count else {throw WorkflowFileLeaseError.unsafePath}
            for file in files {try WorkflowPreviewToolContract.verifyFile(file:file,directory:directory,limit:fileLimit(path:file.path))}
            let parts=try WorkflowArchiveCodec.pathParts(manifest.path)
            let bytes=try WorkflowEvaluationOwnership.readProtocol(directory.appendingPathComponent(parts.dropLast().joined(separator:"/")),name:"manifest.json",limit:2_000_000)
            let collection=try FileInputCollectionManifest.decode(bytes)
            if !arguments.isEmpty {
                guard arguments.count==4,arguments[2]=="--file-options" else {throw WorkflowFileLeaseError.unsafePath}
                let requested=try JSONDecoder().decode([String:JSONValue].self,from:Data(arguments[3].utf8)),selected=try FileInputConfiguration.validate(requested)
                guard collection.selection==selected.filter({FileInputCollectionManifest.selectionKeys.contains($0.key)}) else {throw WorkflowFileLeaseError.unsafePath}
            }
            guard collection.members.count<=8,WorkflowArchiveCodec.hash(bytes)==manifest.sha256,
                  try collection.ownedPaths(manifestPath:manifest.path,inventory:Dictionary(uniqueKeysWithValues:files.map {($0.path,$0)})) == Set(files.map(\.path)) else {throw WorkflowFileLeaseError.unsafePath}
        } else {
            guard files.count == 1 else { throw WorkflowFileLeaseError.unsafePath }
            try WorkflowPreviewToolContract.verifyFile(file: files[0], directory: directory, limit: fileLimit(path: files[0].path))
        }
    }
}

/// A verified table sample; callers publish it only after the supervised operation exits successfully.
struct SourceInspectionResult: Decodable, Equatable {
    struct ColumnSchema: Decodable, Equatable {
        let name: String
        let dtype: String
        let nullable: Bool?
    }
    struct Parsing: Decodable, Equatable {
        let validatedRows: Int
        let previewRows: Int
        let sampleOnly: Bool
        let options: [String: JSONValue]
    }
    struct Collection:Decodable,Equatable {
        let complete:Bool
        let selectedFiles:Int
        let observedFiles:Int
        let sourceMode:String
        let readMode:String
        let imagePixelsIncluded:Bool
    }
    let columns: [String]
    let rows: [[String]]
    let rowCount: Int
    let sourceSHA256: String
    var apiFetch:[String:JSONValue]? = nil
    var collection:Collection? = nil
    var parsing: Parsing? = nil
    var nullCells: [[Int]]? = nil
    var schema: [ColumnSchema]? = nil
    var columnSchema: [DataColumnSchema]? { schema?.map { .init(name: $0.name, dtype: $0.dtype, nullable: $0.nullable) } }
    private enum CodingKeys: String, CodingKey {
        case columns, rows, rowCount, sourceSHA256, parsing, nullCells,collection,apiFetch
        case schema = "columnSchema"
    }

    static func decode(_ data: Data, expectedSourceSHA256: String) throws -> Self {
        guard data.count <= 2_000_000,
              let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              Set(object.keys).isSubset(of: ["event", "columns", "rows", "rowCount", "sourceSHA256", "parsing", "nullCells", "columnSchema","collection","apiFetch"]),
              Set(["event", "columns", "rows", "rowCount", "sourceSHA256"]).isSubset(of: Set(object.keys)),
              object["event"] as? String == "dataPreview" else { throw WorkflowFileLeaseError.unsafePath }
        let result = try JSONDecoder().decode(Self.self, from: data)
        guard (1...128).contains(result.columns.count), Set(result.columns).count == result.columns.count,
              result.columns.allSatisfy({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && $0.utf8.count <= 4096 }),
              (0...100_000).contains(result.rowCount), result.rows.count <= 8, result.rows.count <= result.rowCount,
              result.rows.allSatisfy({ $0.count == result.columns.count && $0.allSatisfy({ $0.unicodeScalars.count <= 200 }) }),
              WorkflowArchiveCodec.isHash(result.sourceSHA256), result.sourceSHA256 == expectedSourceSHA256 else { throw WorkflowFileLeaseError.unsafePath }
        if let fetch=result.apiFetch {guard Set(fetch.keys)==["complete","pages","rows","responseBytes","capturedAt"],fetch["complete"] == .bool(true),case .integer(let pages)=fetch["pages"],(1...1000).contains(pages),case .integer(let rows)=fetch["rows"],(0...500_000).contains(rows),case .integer(let bytes)=fetch["responseBytes"],(0...50_000_000).contains(bytes),let date=fetch["capturedAt"]?.stringValue,date.utf8.count<=64,result.parsing?.options["format"] == .string("api") else {throw WorkflowFileLeaseError.unsafePath}}
        if let collection=result.collection {
            guard let metadata=object["collection"] as? [String:Any],Set(metadata.keys)==["complete","selectedFiles","observedFiles","sourceMode","readMode","imagePixelsIncluded"],
                  (0...8).contains(collection.selectedFiles),(collection.selectedFiles...1000).contains(collection.observedFiles),["file","directory"].contains(collection.sourceMode),["metadata","content"].contains(collection.readMode),!collection.imagePixelsIncluded,
                  let parsing=result.parsing,parsing.sampleOnly,(try? FileInputConfiguration.validate(parsing.options)) != nil,
                  parsing.options["sourceMode"]?.stringValue==collection.sourceMode,parsing.options["readMode"]?.stringValue==collection.readMode else {throw WorkflowFileLeaseError.unsafePath}
        }
        if let parsing = result.parsing {
            guard let metadata = object["parsing"] as? [String: Any],
                  Set(metadata.keys) == ["validatedRows", "previewRows", "sampleOnly", "options"],
                  parsing.validatedRows == result.rowCount, parsing.previewRows == result.rows.count,
                  (!parsing.sampleOnly || result.rowCount == result.rows.count),
                  (result.collection != nil ? (try? FileInputConfiguration.validate(parsing.options)) != nil : (try? TableInputConfiguration.validate(parsing.options)) != nil) else { throw WorkflowFileLeaseError.unsafePath }
        }
        if result.rowCount == 0 {
            guard let parsing = result.parsing, parsing.sampleOnly,
                  result.collection != nil || (try? TableInputConfiguration.format(of: parsing.options)) != .csv || CSVParsingConfiguration.isPolars(parsing.options) else { throw WorkflowFileLeaseError.unsafePath }
        }
        if let schema = result.schema {
            guard let metadata = object["columnSchema"] as? [[String: Any]], metadata.count == result.columns.count,
                  schema.map(\.name) == result.columns,
                  zip(schema, metadata).allSatisfy({ column, fields in
                      Set(fields.keys) == ["name", "dtype", "nullable"]
                        && (["string", "bool", "int64", "float64"].contains(column.dtype) || result.parsing?.options["format"] == .string("api") && column.dtype == "datetime")
                        && (fields["nullable"] is NSNull || fields["nullable"] is Bool)
                  }) else { throw WorkflowFileLeaseError.unsafePath }
        }
        if let cells = result.nullCells {
            guard cells.count <= result.rows.count * result.columns.count,
                  Set(cells).count == cells.count,
                  cells.allSatisfy({ $0.count == 2 && result.rows.indices.contains($0[0])
                      && result.columns.indices.contains($0[1]) && result.rows[$0[0]][$0[1]].isEmpty }) else { throw WorkflowFileLeaseError.unsafePath }
        }
        return result
    }
}
