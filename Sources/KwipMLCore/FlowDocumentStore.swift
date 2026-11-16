import Foundation
import Darwin

/// File-backed flow closure storage. Migration publishes a new directory; it never replaces a source project.
public enum FlowDocumentStore {
    public static func commitMigration(_ plan: FlowDocumentMigrationPlan, to destination: URL) throws -> URL {
        try commitMigration(plan, to: destination, afterWrite: { _ in })
    }

    static func commitMigration(_ plan: FlowDocumentMigrationPlan, to destination: URL,
                                afterWrite: (URL) throws -> Void) throws -> URL {
        _ = try plan.snapshot().workflow()
        let fm = FileManager.default
        let parent = destination.deletingLastPathComponent().resolvingSymlinksInPath()
        let target = parent.appendingPathComponent(destination.lastPathComponent, isDirectory: true)
        let staging = parent.appendingPathComponent(".kwipflow-stage-" + UUID().uuidString, isDirectory: true)
        try fm.createDirectory(at: staging, withIntermediateDirectories: false)
        defer { try? fm.removeItem(at: staging) }
        func write(_ document: FlowDocument, path: String) throws {
            let url = staging.appendingPathComponent(path)
            guard url.standardizedFileURL.path.hasPrefix(staging.path + "/"),
                  !path.split(separator: "/").contains("..") else { throw FlowDocumentError.unsafeReference }
            try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            let bytes = try FlowDocumentCodec.encode(document)
            try bytes.write(to: url, options: .withoutOverwriting)
            let handle = try FileHandle(forWritingTo: url)
            try handle.synchronize(); try handle.close()
            try afterWrite(url)
        }
        for path in plan.subflows.keys.sorted() { try write(plan.subflows[path]!, path: path) }
        try write(plan.root, path: "Main.kwipflow")
        guard try snapshot(at: staging.appendingPathComponent("Main.kwipflow")) == plan.snapshot() else {
            throw FlowDocumentError.invalidDocument
        }
        // Same-filesystem exclusive rename publishes all documents together, without replacing a racing writer.
        guard renamex_np(staging.path, target.path, UInt32(RENAME_EXCL)) == 0 else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
        return target.appendingPathComponent("Main.kwipflow")
    }

    /// Optimistic save under a cooperating-writer lock. Old revisions remain available to pinned callers.
    public static func save(_ document: FlowDocument, at url: URL, expected: FlowDocumentRevision) throws {
        try save(document, at: url, expected: expected, beforeCommit: { _ in })
    }
    static func save(_ document: FlowDocument, at url: URL, expected: FlowDocumentRevision,
                     beforeCommit: (Int32) throws -> Void) throws {
        try FlowDocumentSaveTransaction.save(document, at: url, expected: expected, beforeCommit: beforeCommit)
    }

    private static func archiveURL(for reference: FlowDocumentRevision, beside url: URL) -> URL {
        url.deletingLastPathComponent().appendingPathComponent(".revisions")
            .appendingPathComponent(reference.documentID.uuidString)
            .appendingPathComponent(reference.sha256 + ".kwipflow")
    }

    /// Bounded current-head read for explicit revision review; does not substitute an archived revision.
    public static func readHead(at url: URL) throws -> FlowDocument { try decodeFile(url) }

    private static func decodeFile(_ url: URL) throws -> FlowDocument {
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey])
        guard values.isRegularFile == true, values.isSymbolicLink != true else { throw FlowDocumentError.unsafeReference }
        guard let size = values.fileSize, size <= FlowDocumentCodec.maximumBytes else { throw FlowDocumentError.resourceLimit }
        return try FlowDocumentCodec.decode(Data(contentsOf: url))
    }

    /// Every declared location proves its pin, whether the revision is the current head or archived.
    private static func decodePinnedFile(_ url: URL, expected: FlowDocumentRevision) throws -> FlowDocument {
        if let current = try? decodeFile(url), try FlowDocumentCodec.reference(current) == expected {
            return current
        }
        let archive = archiveURL(for: expected, beside: url)
        guard archive.resolvingSymlinksInPath() == archive else { throw FlowDocumentError.unsafeReference }
        let document = try decodeFile(archive)
        guard try FlowDocumentCodec.reference(document) == expected else { throw FlowDocumentError.revisionMismatch }
        return document
    }

    public static func snapshot(at rootURL: URL) throws -> FlowDocumentSnapshot {
        try snapshot(of: decodeFile(rootURL), at: rootURL)
    }
    /// Resolve an uncommitted root against the same pinned-file rules before a save or explicit adoption.
    public static func snapshot(of root: FlowDocument, at rootURL: URL) throws -> FlowDocumentSnapshot {
        var locations: [FlowDocumentRevision: URL] = [:]
        func read(_ url: URL, expected: FlowDocumentRevision? = nil) throws -> FlowDocument {
            let document = try expected.map { try decodePinnedFile(url, expected: $0) } ?? decodeFile(url)
            let parent = url.deletingLastPathComponent().resolvingSymlinksInPath()
            for dependency in document.fileReferences {
                let child = parent.appendingPathComponent(dependency.relativePath).standardizedFileURL
                guard child.resolvingSymlinksInPath() == child else { throw FlowDocumentError.unsafeReference }
                if let existing = locations[dependency.revision], existing != child {
                    // Multiple paths may name the same pinned contents; verify every declared path.
                    _ = try decodePinnedFile(child, expected: dependency.revision)
                } else { locations[dependency.revision] = child }
            }
            return document
        }
        let parent = rootURL.deletingLastPathComponent().resolvingSymlinksInPath()
        for dependency in root.fileReferences {
            let child = parent.appendingPathComponent(dependency.relativePath).standardizedFileURL
            guard child.resolvingSymlinksInPath() == child else { throw FlowDocumentError.unsafeReference }
            locations[dependency.revision] = child
        }
        return try FlowDocumentResolver.resolve(root) { reference in
            guard let url = locations[reference] else { throw FlowDocumentError.missingDependency }
            return try read(url, expected: reference)
        }
    }
}
