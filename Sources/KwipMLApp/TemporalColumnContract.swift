import Foundation
import KwipMLCore

/// Native admission for the worker-owned canonical temporal v1 table contract.
enum TemporalColumnContract {
    static let types = ["date", "datetime", "time"]
    static func validMetadata(_ value: JSONValue?, dtype: String) -> Bool {
        guard types.contains(dtype) else { return value == nil }
        guard case .object(let fields) = value,
              fields["version"] == .integer(1),
              fields["unit"] == .string(dtype == "date" ? "day" : "microsecond") else { return false }
        if dtype == "datetime" {
            guard Set(fields.keys) == ["version", "unit", "timezone"], case .string(let zone) = fields["timezone"],
                  !zone.isEmpty, zone.count <= 128, TimeZone(identifier: zone) != nil else { return false }
            return true
        }
        return Set(fields.keys) == ["version", "unit"]
    }
    static func validValue(_ value: Int64, dtype: String) -> Bool {
        switch dtype {
        case "date": return (-719162...2932896).contains(value)
        case "datetime": return (-62135596800000000...253402300799999999).contains(value)
        case "time": return value >= 0 && value < 86_400_000_000
        default: return false
        }
    }
}
