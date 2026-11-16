import Foundation
import KwipMLCore

/// Bounded regular-file reads shared by private editor sidecars and immutable source snapshots.
enum WorkspaceDocumentReader {
    static func read(_ url: URL) throws -> Data {
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey])
        guard values.isRegularFile == true, values.isSymbolicLink != true, (values.fileSize ?? Int.max) <= 2_000_000 else { throw CompositeContractError(.resourceLimit, "Project must be a regular file of at most 2 MB.") }
        let handle = try FileHandle(forReadingFrom: url); defer { try? handle.close() }
        let data = try handle.read(upToCount: 2_000_001) ?? Data()
        guard data.count <= 2_000_000 else { throw CompositeContractError(.resourceLimit, "Project exceeds 2 MB.") }
        return data
    }
}
