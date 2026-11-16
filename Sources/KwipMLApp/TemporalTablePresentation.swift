import Foundation
import KwipMLCore

/// Integer civil arithmetic retains every microsecond, including negative epoch values.
enum TemporalTablePresentation {
    static func format(_ value: Int64, dtype: String, metadata: JSONValue?) -> String? {
        guard TemporalColumnContract.validMetadata(metadata, dtype: dtype), TemporalColumnContract.validValue(value, dtype: dtype) else { return nil }
        if dtype == "date" { return date(value) }
        if dtype == "time" { return clock(value / 1_000_000) + String(format: ".%06lld", value % 1_000_000) }
        guard case .object(let fields) = metadata, case .string(let name) = fields["timezone"], let zone = TimeZone(identifier: name) else { return nil }
        var seconds = value / 1_000_000, micros = value % 1_000_000
        if micros < 0 { seconds -= 1; micros += 1_000_000 }
        // Integral seconds are exactly representable across the supported years; fractions never enter Date.
        let offset = zone.secondsFromGMT(for: Date(timeIntervalSince1970: Double(seconds)))
        let local = seconds + Int64(offset)
        let day = floorDivide(local, 86400), time = local - day * 86400
        guard let dayText = date(day) else { return nil }
        let absolute = abs(offset)
        var suffix = String(format: "%@%02d:%02d", offset < 0 ? "-" : "+", absolute / 3600, absolute % 3600 / 60)
        if absolute % 60 != 0 { suffix += String(format: ":%02d", absolute % 60) }
        return dayText + "T" + clock(time) + String(format: ".%06lld", micros) + suffix
    }
    private static func floorDivide(_ value: Int64, _ divisor: Int64) -> Int64 { value / divisor - (value % divisor < 0 ? 1 : 0) }
    private static func clock(_ seconds: Int64) -> String { String(format: "%02lld:%02lld:%02lld", seconds / 3600, seconds % 3600 / 60, seconds % 60) }
    private static func date(_ epochDay: Int64) -> String? {
        let z = epochDay + 719468
        let era = floorDivide(z, 146097), doe = z - era * 146097
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365
        var year = yoe + era * 400
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100), mp = (5 * doy + 2) / 153
        let day = doy - (153 * mp + 2) / 5 + 1, month = mp + (mp < 10 ? 3 : -9)
        if month <= 2 { year += 1 }
        guard (1...9999).contains(year) else { return nil }
        return String(format: "%04lld-%02lld-%02lld", year, month, day)
    }
}
