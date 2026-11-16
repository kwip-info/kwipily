import SwiftUI
import KwipMLCore

/// Compact editor embedded in Formula's existing atomic Apply/Revert draft.
struct FormulaTemporalEditor: View {
    @Binding var value: FormulaGuidedTemporal
    let columns: [DataColumnSchema]
    let accent: Color
    private var sourceColumns: [DataColumnSchema] {
        columns.filter { value.kind == "temporal_parse" ? $0.dtype == "string" : value.kind == "temporal_timezone" ? $0.dtype == "datetime" : ["date", "datetime", "time"].contains($0.dtype) }
    }
    private var sourceType: String { columns.first { $0.name == value.column }?.dtype ?? "datetime" }
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if columns.isEmpty { field("Source column", text: $value.column) }
            else { choice("Source column", selection: $value.column, options: sourceColumns.map { ($0.name, $0.name) }) }
            if value.kind == "temporal_parse" {
                choice("Result type", selection: $value.dtype, options: [("date","Date"),("datetime","DateTime"),("time","Time")])
                pattern
                if value.dtype == "datetime" { field("Source timezone for naive values", text: $value.sourceTimezone); field("Result timezone", text: $value.timezone) }
                choice("Invalid value", selection: $value.invalid, options: policies)
                if value.dtype == "datetime" {
                    choice("Ambiguous local time", selection: $value.ambiguous, options: [("stop","Stop run"),("earliest","Earliest instant"),("latest","Latest instant"),("missing","Missing")])
                    choice("Nonexistent local time", selection: $value.nonexistent, options: policies)
                }
            } else if value.kind == "temporal_format" {
                pattern
                if sourceType == "datetime" { field("Display timezone", text: $value.timezone) }
            } else if value.kind == "temporal_timezone" {
                field("Result timezone", text: $value.timezone)
            } else if value.kind == "temporal_part" {
                choice("Part", selection: $value.part, options: parts.map { ($0.0,$0.1) })
                if sourceType == "datetime" { field("Extract in timezone", text: $value.timezone) }
            } else if value.kind == "temporal_difference" {
                choice("Subtract column", selection: $value.otherColumn, options: columns.filter { $0.dtype == sourceType }.map { ($0.name,$0.name) })
                choice("Units", selection: $value.units, options: [("days","Days"),("hours","Hours"),("minutes","Minutes"),("seconds","Seconds"),("milliseconds","Milliseconds"),("microseconds","Microseconds")])
                choice("Output", selection: $value.representation, options: [("decimal","Decimal"),("integer","Exact integer")])
            }
        }
    }
    private var policies: [(String,String)] { [("stop","Stop run"),("missing","Missing")] }
    private var parts: [(String,String)] {
        let date = [("year","Year"),("month","Month"),("day","Day"),("weekday","Weekday · Monday = 1"),("dayOfYear","Day of year")]
        let time = [("hour","Hour"),("minute","Minute"),("second","Second"),("microsecond","Microsecond")]
        return sourceType == "date" ? date : sourceType == "time" ? time : date + time + [("utcOffsetSeconds","UTC offset in seconds")]
    }
    private var patternType: String { value.kind == "temporal_parse" ? value.dtype : sourceType }
    private var presets: [(String,String)] {
        if patternType == "date" { return [("%Y-%m-%d","Year-month-day"),("%m/%d/%Y","Month/day/year"),("%d/%m/%Y","Day/month/year")] }
        if patternType == "time" { return [("%H:%M:%S","Hours:minutes:seconds"),("%H:%M:%S.%f","Clock with microseconds")] }
        return [("%Y-%m-%d %H:%M:%S","Date and clock"),("%Y-%m-%dT%H:%M:%S%z","Date and clock with offset")]
    }
    private var pattern: some View {
        VStack(alignment: .leading, spacing: 5) {
            choice("Pattern", selection: Binding(get: {
                value.pattern == "iso" ? "iso" : presets.contains(where: { $0.0 == value.pattern }) ? value.pattern : "custom"
            }, set: { next in
                value.pattern = next == "custom" ? (patternType == "date" ? "%Y.%m.%d" : patternType == "time" ? "%H.%M.%S" : "%Y-%m-%dT%H:%M:%S.%f%z") : next
            }), options: [("iso","ISO 8601")] + presets + [("custom","Custom numeric pattern")])
            if value.pattern != "iso" && !presets.contains(where: { $0.0 == value.pattern }) { field("Numeric pattern", text: $value.pattern) }
        }
    }
    private func field(_ title: String, text: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 4) { Text(title); TextField(title,text: text).flatField().accessibilityLabel(title) }
    }
    private func choice(_ title: String, selection: Binding<String>, options: [(String,String)]) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
            FlatChoiceControl(title: title, selection: selection.wrappedValue, options: options.map { .init(id:$0.0,title:$0.1) }, accent: accent) { selection.wrappedValue = $0 }
        }
    }
}
