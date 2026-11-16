import Foundation
import KwipMLCore

/// Bounded presentation of an actual row-major flattened scalar capture.
/// Coordinates require a valid whole shape; absence never becomes an invented axis.
struct TensorSampleInspection {
    struct Row: Identifiable {
        let id: Int
        let coordinates: [Int]?
        let value: String
        let state: String
        /// Finite magnitude for display shading only; exact integer text is never derived from this value.
        let numericValue: Double?
    }
    struct Matrix {
        let rowAxis: Int
        let columnAxis: Int
        let fixedIndices: [Int: Int]
        let rowIndices: [Int]
        let columnIndices: [Int]
        let cells: [[Row]]
        let isPartialWindow: Bool
    }
    static let maximumDisplayedScalars = 256

    let heatmap:TensorHeatmapSummary?
    let heatmapPlayback:TensorHeatmapPlayback?
    let shape: [Int]?
    let dtype: String
    let retainedCount: Int
    let totalCount: Int?
    let axes: [String]
    let rows: [Row]
    let availability: String
    let isSampled: Bool

    init(_ preview: [String: Any]) {
        let capturedType = preview["dtype"] as? String ?? "Unknown type"
        dtype = capturedType
        let candidate = Self.validShape(preview["shape"])
        if let raw=preview["heatmap"],let data=try? JSONSerialization.data(withJSONObject:raw),data.count<=1_000_000,let value=try? JSONDecoder().decode(JSONValue.self,from:data) {heatmap=try? TensorHeatmapSummary(value:value,expectedShape:candidate)} else {heatmap=nil}
        if let raw=preview["heatmapPlayback"],let data=try? JSONSerialization.data(withJSONObject:raw),data.count<=1_800_000,let value=try? JSONDecoder().decode(JSONValue.self,from:data) {heatmapPlayback=try? TensorHeatmapPlayback(value:value)} else {heatmapPlayback=nil}
        var count: Int? = 1
        if let candidate, candidate.count <= 8, candidate.allSatisfy({ $0 >= 0 }) {
            for dimension in candidate {
                guard let previous = count else { break }
                let product = previous.multipliedReportingOverflow(by: dimension)
                count = product.overflow ? nil : product.partialValue
            }
        } else { count = nil }
        let validShape = count == nil ? nil : candidate
        shape = validShape
        totalCount = count
        axes = (validShape ?? []).indices.map { "Axis \($0)" }
        let raw = preview["values"] as? [Any]
        retainedCount = raw?.count ?? 0
        if let status = preview["captureStatus"] {
            guard let status = status as? String, ["captured", "omittedByPolicy"].contains(status) else {
                rows = []; availability = "Invalid capture status"; isSampled = false; return
            }
            if status == "omittedByPolicy" {
                rows = []; isSampled = false
                availability = raw?.isEmpty == true ? "Not captured by this run’s policy" : "Invalid capture: values contradict policy"
                return
            }
        }
        guard let raw else {
            rows = []; availability = "Values unavailable"; isSampled = false; return
        }
        let coordinatesKnown = count.map { raw.count <= $0 } ?? false
        let projectedRows = raw.prefix(Self.maximumDisplayedScalars).enumerated().map { offset, value in
            let formatted = Self.format(value, dtype: capturedType)
            var coordinate: [Int]? = nil
            if coordinatesKnown, let shape = validShape, count != 0 {
                var remainder = offset
                var indices = Array(repeating: 0, count: shape.count)
                for axis in shape.indices.reversed() {
                    indices[axis] = remainder % shape[axis]
                    remainder /= shape[axis]
                }
                coordinate = indices
            }
            return Row(id: offset, coordinates: coordinate, value: formatted.0, state: formatted.1, numericValue: formatted.2)
        }
        rows = projectedRows
        let sampled = count.map { raw.count < $0 } ?? !raw.isEmpty
        isSampled = sampled
        if raw.isEmpty { availability = count == 0 ? "Empty tensor" : "No retained scalar values" }
        else if let count, raw.count > count { availability = "Captured values exceed shape; coordinates unavailable" }
        else { availability = sampled ? "Retained scalar sample" : "All scalar values retained" }
    }

    /// Matrix is possible only when every actual scalar is retained and bounded for display.
    var canShowMatrix: Bool { shape?.count == 2 && totalCount == retainedCount && retainedCount <= 256 && rows.count == retainedCount && rows.allSatisfy { $0.coordinates != nil && $0.state == "Captured" } }

    /// The default plane uses the final two axes and zero on every other axis.
    var defaultMatrix: Matrix? {
        guard let shape, shape.count >= 2 else { return nil }
        let row = shape.count - 2, column = shape.count - 1
        let fixed = Dictionary(uniqueKeysWithValues: shape.indices.filter { $0 != row && $0 != column }.map { ($0, 0) })
        return try? matrix(rowAxis: row, columnAxis: column, fixedIndices: fixed)
    }

    /// Returns the largest complete prefix rectangle of the requested plane inside
    /// the bounded capture. Every cell is an existing retained row; unavailable cells
    /// are refused rather than filled, reread or inferred from a model/source.
    func matrix(rowAxis: Int, columnAxis: Int, fixedIndices: [Int: Int],
                maximumRows: Int = 16, maximumColumns: Int = 16) throws -> Matrix {
        func refuse(_ message: String) -> DataFlowContractError { .init(message) }
        guard let shape, shape.count >= 2, let totalCount, totalCount > 0,
              shape.allSatisfy({ $0 > 0 }), retainedCount <= totalCount,
              !rows.isEmpty, rows.count <= Self.maximumDisplayedScalars,
              rows.allSatisfy({ $0.coordinates != nil && $0.state == "Captured" }) else {
            throw refuse("Matrix unavailable: this capture has no valid finite numeric plane. Inspect Coordinates for retained values and their states.")
        }
        guard shape.indices.contains(rowAxis), shape.indices.contains(columnAxis), rowAxis != columnAxis else {
            throw refuse("Choose two distinct captured axes for rows and columns.")
        }
        let remaining = Set(shape.indices).subtracting([rowAxis, columnAxis])
        guard Set(fixedIndices.keys) == remaining,
              fixedIndices.allSatisfy({ shape.indices.contains($0.key) && (0..<shape[$0.key]).contains($0.value) }) else {
            throw refuse("Choose one in-range fixed index for every other axis, and none for the row or column axis.")
        }
        guard (1...Self.maximumDisplayedScalars).contains(maximumRows),
              (1...Self.maximumDisplayedScalars).contains(maximumColumns),
              maximumRows <= Self.maximumDisplayedScalars / maximumColumns else {
            throw refuse("A matrix window may display at most 256 retained scalar values.")
        }
        var lookup: [Int: Row] = [:]
        for row in rows {
            guard lookup.updateValue(row, forKey: row.id) == nil else { throw refuse("The capture repeats a scalar position.") }
        }
        let rowLimit = min(shape[rowAxis], maximumRows), columnLimit = min(shape[columnAxis], maximumColumns)
        var candidates: [[Row]] = [], bestRows = 0, bestColumns = 0, minimumColumns = columnLimit
        for rowIndex in 0..<rowLimit {
            var values: [Row] = []
            for columnIndex in 0..<columnLimit {
                var coordinates = Array(repeating: 0, count: shape.count)
                for (axis, index) in fixedIndices { coordinates[axis] = index }
                coordinates[rowAxis] = rowIndex; coordinates[columnAxis] = columnIndex
                var offset = 0
                for axis in shape.indices { offset = offset * shape[axis] + coordinates[axis] }
                guard let captured = lookup[offset], captured.coordinates == coordinates else { break }
                values.append(captured)
            }
            minimumColumns = min(minimumColumns, values.count)
            guard minimumColumns > 0 else { break }
            candidates.append(values)
            let count = candidates.count
            if count * minimumColumns > bestRows * bestColumns { bestRows = count; bestColumns = minimumColumns }
        }
        guard bestRows > 0, bestColumns > 0 else {
            throw refuse("This slice is not present in the retained capture. Choose another fixed index or inspect Coordinates; no source reread is performed.")
        }
        let cells = candidates.prefix(bestRows).map { Array($0.prefix(bestColumns)) }
        return .init(rowAxis: rowAxis, columnAxis: columnAxis, fixedIndices: fixedIndices,
            rowIndices: Array(0..<bestRows), columnIndices: Array(0..<bestColumns), cells: cells,
            isPartialWindow: bestRows < shape[rowAxis] || bestColumns < shape[columnAxis])
    }

    private static func validShape(_ raw: Any?) -> [Int]? {
        guard let dimensions = raw as? [Any], dimensions.count <= 8 else { return nil }
        var result: [Int] = []
        for dimension in dimensions {
            guard dimension is NSNumber, JSONSerialization.isValidJSONObject([dimension]),
                  let data = try? JSONSerialization.data(withJSONObject: [dimension]),
                  let decoded = try? JSONDecoder().decode([JSONValue].self, from: data),
                  case .integer(let integer) = decoded.first, let value = Int(exactly: integer), value >= 0 else { return nil }
            result.append(value)
        }
        return result
    }
    private static func format(_ raw: Any, dtype: String) -> (String, String, Double?) {
        if raw is NSNull { return ("null", "Null", nil) }
        guard ["int32", "int64", "float32", "float64", "bool"].contains(dtype) else { return ("—", "Unsupported type", nil) }
        if let number = raw as? NSNumber, !number.doubleValue.isFinite {
            return (number.doubleValue.isNaN ? "NaN" : (number.doubleValue > 0 ? "+∞" : "−∞"), "Nonfinite", nil)
        }
        guard raw is NSNumber, JSONSerialization.isValidJSONObject([raw]),
              let data = try? JSONSerialization.data(withJSONObject: [raw]),
              let values = try? JSONDecoder().decode([JSONValue].self, from: data), let value = values.first else {
            return ("—", "Invalid", nil)
        }
        if dtype == "bool" {
            guard case .bool(let boolean) = value else { return ("—", "Invalid", nil) }
            return (boolean ? "true" : "false", "Captured", nil)
        }
        if dtype == "int32" || dtype == "int64" {
            guard case .integer(let integer) = value,
                  dtype != "int32" || (Int64(Int32.min)...Int64(Int32.max)).contains(integer) else { return ("—", "Invalid", nil) }
            return (String(integer), "Captured", Double(integer))
        }
        guard let number = value.doubleValue, number.isFinite,
              dtype != "float32" || Float(number).isFinite else { return ("—", "Invalid", nil) }
        return (String(number), "Captured", number)
    }
}
