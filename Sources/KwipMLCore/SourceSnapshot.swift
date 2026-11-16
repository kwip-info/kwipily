import Foundation
import CryptoKit

/// Copies authorized bytes into app-owned storage before a sandboxed worker opens them.
/// Content identity is independent of the user's label or original filesystem location.
public struct SourceSnapshot {
    public let url: URL
    public let sha256: String
    public let byteCount: Int

    public static func prepare(source: URL, directory: URL, maximumBytes: Int = 10_000_000) throws -> SourceSnapshot {
        let properties = try source.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        guard properties.isRegularFile == true, let size = properties.fileSize, size <= maximumBytes else {
            throw NSError(domain: "KwipML.Source", code: 1, userInfo: [NSLocalizedDescriptionKey: "Choose a regular source file no larger than \(maximumBytes) bytes."])
        }
        let handle = try FileHandle(forReadingFrom: source)
        defer { try? handle.close() }
        let bytes = try handle.read(upToCount: maximumBytes + 1) ?? Data()
        guard bytes.count <= maximumBytes else { throw NSError(domain: "KwipML.Source", code: 2, userInfo: [NSLocalizedDescriptionKey: "Source grew beyond the size limit while being read. No snapshot was published."]) }
        let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
        let folder = directory.appendingPathComponent(digest, isDirectory: true)
        let target = folder.appendingPathComponent(source.lastPathComponent)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        // Never trust a cached filename alone: repair changed bytes before returning it.
        if (try? Data(contentsOf: target)) != bytes { try bytes.write(to: target, options: .atomic) }
        return SourceSnapshot(url: target, sha256: digest, byteCount: bytes.count)
    }
}
