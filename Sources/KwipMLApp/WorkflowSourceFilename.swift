import Foundation

/// Retain only a format suffix, never the user's original filename, in captures.
enum WorkflowSourceFilename {
    static func suffix(_ url: URL) -> String {
        let value = url.pathExtension.lowercased()
        return !value.isEmpty && value.count <= 10 && value.utf8.allSatisfy {
            (97...122).contains($0) || (48...57).contains($0)
        } ? value : "data"
    }
    static func archive(index: Int, source: URL) -> String { "source-\(index).\(suffix(source))" }
    static func validArchive(_ name: String, index: Int) -> Bool {
        let prefix = "source-\(index)."
        guard name.hasPrefix(prefix) else { return false }
        let ext = String(name.dropFirst(prefix.count))
        return !ext.isEmpty && ext.count <= 10 && ext.utf8.allSatisfy {
            (97...122).contains($0) || (48...57).contains($0)
        }
    }
}
