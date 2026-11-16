import Foundation
import KwipMLCore

/// Display bounds for the existing worker preview protocol. Full table/state
/// numerical and lineage validation belongs to the worker's archived reader.
enum WorkflowPreparationPreviewValidation {
    static func validate(_ value: JSONValue, kind: String, operatorID: String) throws {
        guard case .object(let output) = value, case .object(let lineage) = output["lineage"],
              case .string(let role) = lineage["role"], ["unassigned", "training", "validation", "test"].contains(role),
              hash(lineage["sourceDigest"]), hash(lineage["membershipDigest"]) else { throw invalid() }
        if kind == "state" {
            guard output["kind"] == .string("state"), (role == "training" || operatorID == TextEncodingConfiguration.operatorID),
                  case .object(let summary) = output["summary"], summary["operator"] == .string(operatorID),
                  hash(summary["contentDigest"]), hash(summary["fitDataVersion"]),
                  case .array(let columns) = summary["columns"], (!columns.isEmpty || operatorID == "kwip.data.preprocess"), columns.count <= 128,
                  columns.allSatisfy({ if case .string(let name) = $0 { return !name.isEmpty && name.unicodeScalars.count <= 200 }; return false }) else { throw invalid() }
            return
        }
        guard ["table", "dataset"].contains(kind), output["kind"] == .string("table"),
              case .object(let preview) = output["preview"], preview["partition"] == .string(role),
              case .integer(let count) = preview["rowCount"], count >= 0,
              case .array(let columns) = preview["columns"], !columns.isEmpty, columns.count <= 128,
              count <= 5_000_000 / Int64(columns.count),
              case .array(let rows) = preview["rows"], rows.count <= 8, rows.count <= count,
              case .array(let diagnostics) = preview["diagnostics"], diagnostics.count <= 16,
              diagnostics.allSatisfy({ if case .object = $0 { return true }; return false }),
              case .integer(let diagnosticCount) = preview["diagnosticCount"], diagnosticCount >= diagnostics.count,
              case .array(let fitted) = lineage["fittedStateDigests"], fitted.count <= 128, fitted.allSatisfy({hash($0)}),
              case .string(let partition) = lineage["partitionDigest"], partition.isEmpty || WorkflowArchiveCodec.isHash(partition) else { throw invalid() }
        // Version2 is a live-only observation with explicitly deferred content
        // identity. Durable table readers retain their canonical hash contract.
        if output["schemaVersion"] == .integer(2) {
            guard preview["dataVersion"] == .null,lineage["dataVersion"] == .null else {throw invalid()}
        } else {
            guard output["schemaVersion"] == nil,hash(preview["dataVersion"]),preview["dataVersion"] == lineage["dataVersion"] else {throw invalid()}
        }
        if let truncated = preview["previewTruncated"], case .bool = truncated {} else if preview["previewTruncated"] != nil { throw invalid() }
        var names = Set<String>(), schema: [(String, Bool)] = []
        for column in columns {
            guard case .object(let fields) = column, case .string(let name) = fields["name"],
                  !name.isEmpty, name.unicodeScalars.count <= 200, names.insert(name).inserted,
                  case .string(let dtype) = fields["dtype"], (["float64", "int64", "string", "bool", "image", "audio", "tensor", "int64Sequence"] + TemporalColumnContract.types).contains(dtype),
                  case .bool(let nullable) = fields["nullable"] else { throw invalid() }
            if let label = fields["displayLabel"] {
                guard case .string(let text) = label, text.unicodeScalars.count <= 400 else { throw invalid() }
            }
            guard TemporalColumnContract.validMetadata(fields["temporal"],dtype:dtype) else {throw invalid()}
            schema.append((dtype, nullable))
        }
        for row in rows {
            guard case .array(let cells) = row, cells.count == schema.count else { throw invalid() }
            for (cell, column) in zip(cells, schema) {
                switch cell {
                case .null: guard column.1 else { throw invalid() }
                case .string(let text): guard column.0 == "string", text.unicodeScalars.count <= 200 else { throw invalid() }
                case .bool: guard column.0 == "bool" else { throw invalid() }
                case .integer(let integer): guard column.0 == "int64" || column.0 == "float64" || TemporalColumnContract.validValue(integer,dtype:column.0) else { throw invalid() }
                case .number: guard column.0 == "float64" else { throw invalid() }
                case .array(let values):
                    guard column.0 == "int64Sequence",values.count<=65536,values.allSatisfy({if case .integer(let id)=$0 {return id>=0};return false}) else {throw invalid()}
                case .object:
                    if column.0 == "image" {_=try FileImageReference.decode(JSONEncoder().encode(cell))}
                    else {guard ["audio","tensor"].contains(column.0) else {throw invalid()};let ref=try MediaArtifactReference.decode(JSONEncoder().encode(cell));guard ref.kind == (column.0 == "audio" ? "audioSamplesRef":"preparedTensorRef") else {throw invalid()}}
                default: throw invalid()
                }
            }
        }
    }
    private static func hash(_ value: JSONValue?) -> Bool {
        guard case .string(let text) = value else { return false }
        return WorkflowArchiveCodec.isHash(text)
    }
    private static func invalid() -> WorkflowArchiveError { .init("Inspection preview has an invalid table/state shape, lineage or display bound.") }
}
