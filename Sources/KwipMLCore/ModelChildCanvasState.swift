import Foundation

/// Presentation identity follows instance paths, not definition pins or transient editor sessions.
/// The enclosing workspace sidecar supplies project identity; named arrangements retain the same paths.
public enum ModelChildCanvasKey {
    public static func make(path: [String]) throws -> String {
        guard (1...8).contains(path.count), path.allSatisfy({ !$0.isEmpty && $0.unicodeScalars.count <= 128 }) else {
            throw WorkspaceArrangementError("Block canvas paths require 1–8 bounded instance identities.")
        }
        return "block:" + (try JSONEncoder().encode(path)).base64EncodedString()
    }
    public static func isValid(_ key: String) -> Bool {
        guard key.hasPrefix("block:"), key.utf8.count <= 8192,
              let bytes = Data(base64Encoded: String(key.dropFirst(6))),
              let path = try? JSONDecoder().decode([String].self, from: bytes),
              let canonical = try? make(path: path) else { return false }
        return key == canonical
    }
}
