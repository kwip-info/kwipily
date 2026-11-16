import Foundation
import CoreFoundation
import KwipMLCore

struct TableGridSort: Equatable {
    var column: Int? = nil
    var descending = false
    func toggled(_ selected: Int?) -> Self { .init(column: selected, descending: column == selected ? !descending : false) }
}

/// Pure presentation of an existing bounded worker preview; no source reads.
struct CapturedDataGrid {
    struct Column: Identifiable {
        let id: Int
        let name: String
        let label: String
        let dtype: String
        let nullability: String
    }
    struct Cell {
        let text: String
        let isAbsent: Bool
        var sequence:[Int64]? = nil
        var image: FileImageReference? = nil
        var media: MediaArtifactReference? = nil
        var temporalSortValue: Int64? = nil
    }
    struct Row: Identifiable {
        let index: Int
        let cells: [Cell]
        var id: Int { index }
    }
    let columns: [Column]
    let rows: [Row]
    let totalRows: Int?
    let partition: String
    let truncated: Bool
    let hasUnmappedValues: Bool

    init(preview: [String: Any], rowLimit:Int=8) {
        let capturedColumns = preview["columns"] as? [[String: Any]] ?? []
        columns = capturedColumns.enumerated().map { index, column in
            let name = column["name"] as? String ?? "Column \(index + 1)"
            return Column(id: index, name: name, label: column["displayLabel"] as? String ?? name,
                          dtype: column["dtype"] as? String ?? "not recorded",
                          nullability: (column["nullable"] as? Bool).map { $0 ? "nullable" : "required" } ?? "not recorded")
        }
        let capturedRows = Array((preview["rows"] as? [[Any]] ?? []).prefix(max(0,min(256,rowLimit))))
        rows = capturedRows.enumerated().map { index, values in
            Row(index: index, cells: capturedColumns.indices.map { column in
                column < values.count ? Self.cell(values[column], column: capturedColumns[column]) : Cell(text: "not captured", isAbsent: true)
            })
        }
        totalRows = preview["rowCount"] as? Int
        partition = preview["partition"] as? String ?? "partition not recorded"
        truncated = preview["previewTruncated"] as? Bool == true
        hasUnmappedValues = capturedRows.contains { $0.count > capturedColumns.count }
    }

    var selectionIdentity: [String] {
        [partition, totalRows.map(String.init) ?? "unknown"] + columns.flatMap { [$0.name, $0.label, $0.dtype, $0.nullability] } + rows.flatMap { $0.cells.flatMap { [$0.text, $0.image?.sha256 ?? $0.media?.sha256 ?? ""] } }
    }

    func filteredRows(_ query: String) -> [Row] {
        let text = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return rows }
        return rows.filter { row in row.cells.contains { $0.text.localizedCaseInsensitiveContains(text) } }
    }

    func displayRows(query: String, sort: TableGridSort) -> [Row] {
        let effective = sort.column.flatMap { columns.indices.contains($0) && ["image","audio","tensor","int64Sequence"].contains(columns[$0].dtype) ? TableGridSort() : nil } ?? sort
        return filteredRows(query).sorted { left, right in
            guard let index = effective.column, columns.indices.contains(index),
                  left.cells.indices.contains(index), right.cells.indices.contains(index) else {
                return effective.descending ? left.index > right.index : left.index < right.index
            }
            let a = left.cells[index], b = right.cells[index]
            if a.isAbsent != b.isAbsent { return !a.isAbsent }
            if a.isAbsent { return left.index < right.index }
            var comparison: ComparisonResult
            if let x = a.temporalSortValue, let y = b.temporalSortValue {
                comparison = x < y ? .orderedAscending : (x > y ? .orderedDescending : .orderedSame)
            } else if ["int64", "float64"].contains(columns[index].dtype),
               let x = Decimal(string: a.text, locale: Locale(identifier: "en_US_POSIX")),
               let y = Decimal(string: b.text, locale: Locale(identifier: "en_US_POSIX")) {
                comparison = x < y ? .orderedAscending : (x > y ? .orderedDescending : .orderedSame)
            } else { comparison = a.text.compare(b.text, options: .literal) }
            if comparison == .orderedSame { return left.index < right.index }
            return effective.descending ? comparison == .orderedDescending : comparison == .orderedAscending
        }
    }

    static func cell(_ value: Any, column: [String: Any]) -> Cell {
        if column["dtype"] as? String == "int64Sequence" {
            guard let raw=value as? [NSNumber],raw.count<=65536,raw.allSatisfy({CFGetTypeID($0) != CFBooleanGetTypeID() && $0.doubleValue>=0 && $0.doubleValue==Double($0.int64Value)}) else {return Cell(text:"Invalid token sequence",isAbsent:true)}
            let ids=raw.map(\.int64Value)
            let summary="["+ids.prefix(12).map(String.init).joined(separator:", ")+(ids.count>12 ? ", …]":"]")
            return Cell(text:summary+" · \(ids.count) tokens",isAbsent:false,sequence:ids)
        }
        guard let dtype = column["dtype"] as? String, TemporalColumnContract.types.contains(dtype), !(value is NSNull) else { return cell(value) }
        guard JSONSerialization.isValidJSONObject([value]),
              let bytes = try? JSONSerialization.data(withJSONObject: [value]),
              let values = try? JSONDecoder().decode([JSONValue].self, from: bytes), case .integer(let integer) = values.first,
              let fields = column["temporal"] as? [String: Any],
              let metadataBytes = try? JSONSerialization.data(withJSONObject: fields),
              let metadata = try? JSONDecoder().decode(JSONValue.self, from: metadataBytes),
              let formatted = TemporalTablePresentation.format(integer, dtype: dtype, metadata: metadata) else {
            return Cell(text: "Invalid temporal value", isAbsent: true)
        }
        return Cell(text: formatted, isAbsent: false, temporalSortValue: integer)
    }

    static func cell(_ value: Any) -> Cell {
        if let object=value as? [String:Any],let kind=object["kind"] as? String,["audioSamplesRef","preparedTensorRef"].contains(kind) {
            guard let bytes=try? JSONSerialization.data(withJSONObject:object),let ref=try? MediaArtifactReference.decode(bytes) else {return Cell(text:"Invalid media reference",isAbsent:true)}
            return Cell(text:ref.shape.map(String.init).joined(separator:" × ")+" · "+ref.dtype+" · "+ref.axes.joined(separator:" × "),isAbsent:false,media:ref)
        }
        if let object = value as? [String: Any], object["kind"] as? String == "imagePixelsRef" {
            guard JSONSerialization.isValidJSONObject(object),
                  let data = try? JSONSerialization.data(withJSONObject: object),
                  let reference = try? FileImageReference.decode(data) else {
                return Cell(text: "Invalid image reference", isAbsent: true)
            }
            return Cell(text: "\(reference.width) × \(reference.height) · \(reference.channels) channels · \(reference.dtype) · \(reference.layout)", isAbsent: false, image: reference)
        }
        if value is NSNull { return Cell(text: "null", isAbsent: true) }
        if let string = value as? String { return Cell(text: string.isEmpty ? "\"\"" : string, isAbsent: false) }
        if let number = value as? NSNumber {
            if CFGetTypeID(number) == CFBooleanGetTypeID() { return Cell(text: number.boolValue ? "true" : "false", isAbsent: false) }
            if number.doubleValue.isNaN { return Cell(text: "NaN", isAbsent: false) }
            if number.doubleValue.isInfinite { return Cell(text: number.doubleValue > 0 ? "∞" : "−∞", isAbsent: false) }
            return Cell(text: number.stringValue, isAbsent: false)
        }
        if JSONSerialization.isValidJSONObject(value),
           let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys, .withoutEscapingSlashes]),
           let text = String(data: data, encoding: .utf8) { return Cell(text: text, isAbsent: false) }
        return Cell(text: String(describing: value), isAbsent: false)
    }
}
