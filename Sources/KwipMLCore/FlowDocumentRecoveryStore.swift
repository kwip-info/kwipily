import Foundation
import Darwin

public struct FlowDocumentRecoveryRecord: Codable, Equatable {
    public let schemaVersion: Int
    public let sessionID: UUID
    public let base: FlowDocumentRevision
    public let content: Data
    public let supersedes: [UUID]
    public func document() throws -> FlowDocument { try FlowDocumentCodec.decode(content) }
}

/// Per-session records preserve unsaved branches without changing a document's saved revision.
public enum FlowDocumentRecoveryStore {
    private static let maximumRecordBytes = 6_000_000
    public static func write(_ document: FlowDocument, base: FlowDocument, beside url: URL,
                             sessionID: UUID, supersedes: [UUID], replacing: FlowDocumentRecoveryRecord? = nil) throws -> FlowDocumentRecoveryRecord {
        guard document.id == base.id, document.revision == base.revision,
              supersedes.count <= 128, Set(supersedes).count == supersedes.count,
              !supersedes.contains(sessionID) else { throw FlowDocumentError.invalidDocument }
        let record = FlowDocumentRecoveryRecord(schemaVersion: 1, sessionID: sessionID,
            base: try FlowDocumentCodec.reference(base), content: try FlowDocumentCodec.encode(document), supersedes: supersedes)
        let folder = try directory(beside: url, documentID: document.id, create: true)
        let target = folder.appendingPathComponent(sessionID.uuidString + ".json")
        return try locked(target) {
        if FileManager.default.fileExists(atPath: target.path) {
            let previous = try read(target)
            guard previous.sessionID == sessionID, previous == replacing else { throw FlowDocumentError.revisionMismatch }
        } else if replacing != nil { throw FlowDocumentError.revisionMismatch }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(record)
        guard data.count <= maximumRecordBytes else { throw FlowDocumentError.resourceLimit }
        try data.write(to: target, options: .atomic)
        let handle = try FileHandle(forWritingTo: target)
        try handle.synchronize(); try handle.close()
        return record
        }
    }
    public static func records(beside url: URL, documentID: UUID, includeSuperseded: Bool = false) throws -> [FlowDocumentRecoveryRecord] {
        let folder = try directory(beside: url, documentID: documentID, create: false)
        guard FileManager.default.fileExists(atPath: folder.path) else { return [] }
        let paths = try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)
        guard paths.count <= 4096 else { throw FlowDocumentError.resourceLimit }
        var records: [FlowDocumentRecoveryRecord] = [], bytes = 0
        var identities = Set<UUID>()
        for path in paths.filter({ $0.pathExtension != "lock" }).sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            guard path.pathExtension == "json", let id = UUID(uuidString: path.deletingPathExtension().lastPathComponent) else { throw FlowDocumentError.invalidDocument }
            let record = try read(path)
            guard record.sessionID == id, identities.insert(id).inserted, record.base.documentID == documentID else { throw FlowDocumentError.invalidDocument }
            bytes += record.content.count
            guard bytes <= 32_000_000 else { throw FlowDocumentError.resourceLimit }
            records.append(record)
        }
        guard records.count <= 128 else { throw FlowDocumentError.resourceLimit }
        // Cyclic ancestry is invalid evidence, not permission to hide every pending branch.
        let byID = Dictionary(uniqueKeysWithValues: records.map { ($0.sessionID, $0) })
        var completed = Set<UUID>()
        func visit(_ record: FlowDocumentRecoveryRecord, active: Set<UUID>) throws {
            guard !active.contains(record.sessionID) else { throw FlowDocumentError.cycle }
            if completed.contains(record.sessionID) { return }
            var next = active; next.insert(record.sessionID)
            for id in record.supersedes {
                if let ancestor = byID[id], ancestor.base == record.base { try visit(ancestor, active: next) }
            }
            completed.insert(record.sessionID)
        }
        for record in records { try visit(record, active: []) }
        if includeSuperseded { return records }
        // Only a record against the same saved base may supersede another session's recovery branch.
        return records.filter { candidate in
            !records.contains { $0.base == candidate.base && $0.supersedes.contains(candidate.sessionID) }
        }
    }
    public static func remove(_ record: FlowDocumentRecoveryRecord, beside url: URL) throws {
        let folder = try directory(beside: url, documentID: record.base.documentID, create: false)
        let target = folder.appendingPathComponent(record.sessionID.uuidString + ".json")
        guard FileManager.default.fileExists(atPath: target.path) else { return }
        try locked(target) {
            guard FileManager.default.fileExists(atPath: target.path) else { return }
            guard try read(target) == record else { throw FlowDocumentError.revisionMismatch }
            try FileManager.default.removeItem(at: target)
        }
    }
    private static func locked<T>(_ target: URL, operation: () throws -> T) throws -> T {
        let lock = target.deletingPathExtension().appendingPathExtension("lock")
        let fd = open(lock.path, O_CREAT | O_RDWR | O_NOFOLLOW | O_CLOEXEC, mode_t(0o600))
        guard fd >= 0 else { throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno)) }
        defer { close(fd) }
        guard flock(fd, LOCK_EX | LOCK_NB) == 0 else { throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno)) }
        defer { flock(fd, LOCK_UN) }
        return try operation()
    }
    private static func directory(beside url: URL, documentID: UUID, create: Bool) throws -> URL {
        let parent = url.deletingLastPathComponent().resolvingSymlinksInPath()
        let base = parent.appendingPathComponent(".kwipflow-recovery", isDirectory: true)
        let folder = base.appendingPathComponent(documentID.uuidString, isDirectory: true)
        for item in [base, folder] {
            if FileManager.default.fileExists(atPath: item.path) {
                let values = try item.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
                guard values.isDirectory == true, values.isSymbolicLink != true else { throw FlowDocumentError.unsafeReference }
            } else if create { try FileManager.default.createDirectory(at: item, withIntermediateDirectories: false) }
        }
        return folder
    }
    private static func read(_ url: URL) throws -> FlowDocumentRecoveryRecord {
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey])
        guard values.isRegularFile == true, values.isSymbolicLink != true else { throw FlowDocumentError.unsafeReference }
        guard let size = values.fileSize, size <= maximumRecordBytes else { throw FlowDocumentError.resourceLimit }
        let data = try Data(contentsOf: url)
        guard let raw = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              Set(raw.keys) == ["schemaVersion", "sessionID", "base", "content", "supersedes"] else { throw FlowDocumentError.invalidDocument }
        let record = try JSONDecoder().decode(FlowDocumentRecoveryRecord.self, from: data)
        guard record.schemaVersion == 1, record.supersedes.count <= 128,
              Set(record.supersedes).count == record.supersedes.count,
              !record.supersedes.contains(record.sessionID), record.base.sha256.count == 64,
              record.base.sha256.allSatisfy({ "0123456789abcdef".contains($0) }) else { throw FlowDocumentError.invalidDocument }
        let document = try record.document()
        guard document.id == record.base.documentID, document.revision == record.base.revision else { throw FlowDocumentError.invalidDocument }
        return record
    }
}
