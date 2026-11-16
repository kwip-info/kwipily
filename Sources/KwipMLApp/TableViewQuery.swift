import Foundation
import KwipMLCore

struct TableViewQuery: Codable, Equatable {
    static let maximumCellScalars = 1_048_576
    var schemaVersion = 2
    var offset: Int
    var limit: Int
    var search: String
    let sourceSHA256: String
    // Legacy name: schema 3 pins the captured table here until its lazy index is admitted.
    let indexSHA256: String
    var sortColumn: Int? = nil
    var sortDescending = false
    var inputKind: String? = nil
    var formula:[String:JSONValue]? = nil
    var fullCellColumn:Int? = nil
    var fullCellStart:Int? = nil
    func validate(workload: ExecutionWorkloadPolicy? = nil) throws {
        let capacity = TableTransferCapacity(workload: workload)
        if schemaVersion == 4 {guard let formula,offset == 0,limit<=8,search.isEmpty,sortColumn == nil,!sortDescending else {throw WorkflowFileLeaseError.unsafePath};_ = try FormulaDraft(configuration:formula)} else if formula != nil {throw WorkflowFileLeaseError.unsafePath}
        if schemaVersion==5 {
            guard (0...Self.maximumCellScalars).contains(fullCellStart ?? 0),let column=fullCellColumn,(0..<capacity.columns).contains(column),limit==1,search.isEmpty,sortColumn==nil,!sortDescending,formula==nil else {throw WorkflowFileLeaseError.unsafePath}
        } else if fullCellColumn != nil || fullCellStart != nil {throw WorkflowFileLeaseError.unsafePath}
        guard (1...5).contains(schemaVersion), (schemaVersion == 3 ? inputKind == "table" && sourceSHA256 == indexSHA256 : (schemaVersion == 4 || schemaVersion == 5) ? (inputKind == nil || inputKind == "index" || inputKind == "table" && sourceSHA256 == indexSHA256):inputKind == nil), sortColumn.map({ (0..<capacity.columns).contains($0) }) ?? true, (0...capacity.rows).contains(offset), (1...TableTransferLimits.pageRows).contains(limit),
              search.unicodeScalars.count <= 256, !search.contains("\0"),
              WorkflowArchiveCodec.isHash(sourceSHA256), WorkflowArchiveCodec.isHash(indexSHA256) else { throw WorkflowFileLeaseError.unsafePath }
    }
    static func decode(_ bytes: Data, workload: ExecutionWorkloadPolicy? = nil) throws -> Self {
        guard bytes.count <= 256_000,
              let object = try JSONSerialization.jsonObject(with: bytes) as? [String: Any],
              Set(object.keys).isSuperset(of: ["schemaVersion","offset","limit","search","sourceSHA256","indexSHA256"]),
              Set(object.keys).isSubset(of: ["schemaVersion","offset","limit","search","sourceSHA256","indexSHA256","sortColumn","sortDescending","inputKind","formula","fullCellColumn","fullCellStart"]) else { throw WorkflowFileLeaseError.unsafePath }
        if (object["schemaVersion"] as? Int ?? 0) < 3, object["inputKind"] != nil {throw WorkflowFileLeaseError.unsafePath}
        var normalized = object
        if normalized["sortDescending"] == nil { normalized["sortDescending"] = false }
        if normalized["schemaVersion"] as? Int == 1, object["sortColumn"] != nil || object["sortDescending"] != nil { throw WorkflowFileLeaseError.unsafePath }
        let result = try JSONDecoder().decode(Self.self, from: JSONSerialization.data(withJSONObject: normalized))
        try result.validate(workload: workload); return result
    }
}

struct TableViewPage: Decodable, Equatable {
    let schemaVersion: Int
    let columns: [[String: JSONValue]]
    let rows: [[JSONValue]]
    let positions: [Int]
    let offset: Int
    let nextOffset: Int?
    let totalRows: Int
    let matchedRows: Int
    let cellTruncated: Bool
    let sourceSHA256: String
    var verification: String? = nil
    var cellOffset:Int? = nil
    var cellTotalCharacters:Int? = nil
    struct MaterializedIndex: Decodable, Equatable {
        let path: String
        let bytes: Int
        let sha256: String
    }
    let materializedIndex: MaterializedIndex?
    static func decode(_ bytes: Data, query: TableViewQuery, workload: ExecutionWorkloadPolicy? = nil) throws -> Self {
        let capacity = TableTransferCapacity(workload: workload)
        try query.validate(workload: workload)
        guard bytes.count <= TableTransferLimits.pageBytes + 100_000,
              let object = try JSONSerialization.jsonObject(with: bytes) as? [String: Any],
              Set(object.keys).subtracting(query.fullCellColumn == nil ? ["materializedIndex","verification"]:["materializedIndex","verification","cellOffset","cellTotalCharacters"]) == ["event","schemaVersion","columns","rows","positions","offset","nextOffset","totalRows","matchedRows","cellTruncated","sourceSHA256"],
              object["event"] as? String == "tableViewPage" else { throw WorkflowFileLeaseError.unsafePath }
        let page = try JSONDecoder().decode(Self.self, from: bytes)
        if page.verification != nil {
            guard page.verification == "pinnedRanges", query.inputKind == "table",
                  query.search.isEmpty, query.sortColumn == nil, !query.sortDescending,
                  query.formula == nil, page.materializedIndex == nil,
                  object["materializedIndex"] == nil,
                  page.matchedRows == page.totalRows,
                  page.positions == Array(query.offset..<(query.offset + page.rows.count)) else { throw WorkflowFileLeaseError.unsafePath }
        } else if object["verification"] != nil { throw WorkflowFileLeaseError.unsafePath }
        else if query.inputKind == "table" {
            guard let rawIndex=object["materializedIndex"] as? [String:Any],Set(rawIndex.keys) == ["path","bytes","sha256"],let index=page.materializedIndex,index.path == "artifacts/view.sqlite3",(1...capacity.artifactBytes).contains(index.bytes),WorkflowArchiveCodec.isHash(index.sha256) else {throw WorkflowFileLeaseError.unsafePath}
        } else if page.materializedIndex != nil { throw WorkflowFileLeaseError.unsafePath }
        guard page.schemaVersion == 1, page.offset == query.offset, page.sourceSHA256 == query.sourceSHA256,
              (1...capacity.columns).contains(page.columns.count), (0...min(capacity.rows, capacity.cells / page.columns.count)).contains(page.totalRows),
              (0...page.totalRows).contains(page.matchedRows), page.rows.count <= query.limit,
              page.positions.count == page.rows.count,
              page.positions.allSatisfy({ (0..<page.totalRows).contains($0) }),
              Set(page.positions).count == page.positions.count,
              query.sortColumn.map({ page.columns.indices.contains($0) }) ?? true,
              query.sortColumn != nil || zip(page.positions, page.positions.dropFirst()).allSatisfy({ query.sortDescending ? $0.0 > $0.1 : $0.0 < $0.1 }),
              page.nextOffset == (query.fullCellColumn == nil && page.offset + page.rows.count < page.matchedRows ? page.offset + page.rows.count : nil),
              page.rows.allSatisfy({ row in row.count == page.columns.count && row.enumerated().allSatisfy { index, value in
                  if page.columns[index]["dtype"] == .string("int64Sequence") {
                      guard case .array(let values)=value,values.count<=65536 else {return false}
                      return values.allSatisfy {if case .integer(let i)=$0 {return i>=0};return false}
                  }
                  if [JSONValue.string("audio"),.string("tensor")].contains(page.columns[index]["dtype"] ?? .null) {
                      if value == .null {return page.columns[index]["nullable"] == .bool(true)}
                      guard let bytes=try? JSONEncoder().encode(value),let ref=try? MediaArtifactReference.decode(bytes) else {return false}
                      return ref.kind == (page.columns[index]["dtype"] == .string("audio") ? "audioSamplesRef":"preparedTensorRef")
                  }
                  if page.columns[index]["dtype"] == .string("image") {
                      if value == .null { return true }
                      guard case .object = value, let data = try? JSONEncoder().encode(value) else { return false }
                      return (try? FileImageReference.decode(data)) != nil
                  }
                  let column = page.columns[index]
                  if let dtype = column["dtype"]?.stringValue, TemporalColumnContract.types.contains(dtype) {
                      if value == .null { return column["nullable"] == .bool(true) }
                      guard case .integer(let integer) = value else { return false }
                      return TemporalColumnContract.validValue(integer, dtype: dtype)
                  }
                  switch value { case .string(let text): return text.unicodeScalars.count <= (query.fullCellColumn == nil ? 2000:16_384)
                  case .integer, .number, .bool, .null: return true
                  default: return false }
              } }) else { throw WorkflowFileLeaseError.unsafePath }
        if query.fullCellColumn != nil {
            guard page.columns.count==1,!page.cellTruncated,page.positions==[query.offset],page.cellOffset==(query.fullCellStart ?? 0),let total=page.cellTotalCharacters,(0...TableViewQuery.maximumCellScalars).contains(total) else {throw WorkflowFileLeaseError.unsafePath}
            if case .string(let text)=page.rows.first?.first {guard total >= (page.cellOffset ?? 0),text.unicodeScalars.count==min(16_384,total-(page.cellOffset ?? 0)) else {throw WorkflowFileLeaseError.unsafePath}}
        } else if page.cellOffset != nil || page.cellTotalCharacters != nil {throw WorkflowFileLeaseError.unsafePath}
        if let column = query.sortColumn, [JSONValue.string("image"),.string("audio"),.string("tensor"),.string("int64Sequence")].contains(page.columns[column]["dtype"] ?? .null) { throw WorkflowFileLeaseError.unsafePath }
        var names = Set<String>()
        for column in page.columns {
            guard case .string(let name) = column["name"], !name.isEmpty, name.utf8.count <= 4096,
                  names.insert(name).inserted, case .string(let dtype) = column["dtype"], case .bool = column["nullable"],
                  (["string", "int64", "float64", "bool", "image", "audio", "tensor", "int64Sequence"] + TemporalColumnContract.types).contains(dtype),
                  Set(column.keys).subtracting(["displayLabel"]) == (TemporalColumnContract.types.contains(dtype) ? ["name", "dtype", "nullable", "temporal"] : ["name", "dtype", "nullable"]),
                  column["displayLabel"].map({ value in
                      guard case .string(let label) = value else { return false }
                      return label.unicodeScalars.count <= 400
                  }) ?? true,
                  TemporalColumnContract.validMetadata(column["temporal"], dtype: dtype) else { throw WorkflowFileLeaseError.unsafePath }
        }
        return page
    }
}
