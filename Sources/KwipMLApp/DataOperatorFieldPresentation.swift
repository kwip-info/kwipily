import Foundation

/// Shared UI copy and exact insertion for the bounded data expression grammar.
enum DataOperatorFieldPresentation {
    static func expressionReference(_ name: String) -> String? {
        guard !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              name.unicodeScalars.count <= 200, !name.contains("]"), !name.contains("\n"), !name.contains("\r") else { return nil }
        return "[" + name + "]"
    }
    static func orderedKeys(_ keys: [String]) -> [String] {
        let order = ["expression", "column", "pattern", "flags", "replacement", "groups", "outputColumn", "dtype", "matchMode", "replaceMode", "noMatchPolicy", "nullPolicy", "replaceExisting", "filename", "separator", "columns", "overwrite"]
        return order.filter(keys.contains) + keys.filter { !order.contains($0) }.sorted()
    }
    static func title(for key: String) -> String {
        ["expression":"Rule using column values", "column":"Source text column", "pattern":"Text pattern", "flags":"Pattern options", "replacement":"Replacement text", "groups":"Capture groups and output names", "outputColumn":"Output column name", "dtype":"Output value type", "matchMode":"Where the pattern must match", "replaceMode":"Matches to replace", "noMatchPolicy":"When no match is found", "nullPolicy":"When a value is missing", "replaceExisting":"Allow replacing existing columns", "filename":"Filename", "separator":"Separator", "columns":"Ordered output columns", "overwrite":"Allow replacing an existing file"][key] ?? key
    }
    static func optionTitle(_ value: String, key: String) -> String {
        if key == "separator" { return [",":"Comma", "\t":"Tab", ";":"Semicolon", "|":"Pipe"][value] ?? value }
        if key == "dtype" { return ["float64":"Decimal number · float64", "int64":"Whole number · int64", "string":"Text · string", "bool":"True or false · bool"][value] ?? value }
        if key == "flags" { return ["ignoreCase":"Ignore uppercase/lowercase differences", "multiline":"Match line starts and ends within a value", "dotAll":"Let a dot match line breaks"][value] ?? value }
        if key == "matchMode" { return ["search":"Find a matching part", "full":"Match the complete value"][value] ?? value }
        if key == "replaceMode" { return ["first":"Replace the first match", "all":"Replace all matches"][value] ?? value }
        if key == "noMatchPolicy" { return ["null":"Keep the output missing", "error":"Stop when no match is found"][value] ?? value }
        if key == "nullPolicy" { return ["error":"Stop with an error", "preserve":"Keep missing values missing", "nonmatching":"Send missing values to nonmatching output"][value] ?? value }
        return value
    }
}
