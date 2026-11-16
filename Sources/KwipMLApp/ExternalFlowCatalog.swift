import Foundation
import KwipMLCore

/// Private location identities and grants; never modifies or embeds paths in portable flow files.
@MainActor final class ExternalFlowCatalog {
    struct Record: Codable, Equatable, Identifiable {
        let id: UUID
        let projectID: String
        let title: String
        let folderPath: String
        let rootFilename: String
        let bookmarkData: Data
        var rootURL: URL { URL(fileURLWithPath: folderPath, isDirectory: true).appendingPathComponent(rootFilename) }
    }
    private struct Envelope: Codable { let schemaVersion: Int; let record: Record }
    private let directory: URL
    private let access: FlowFolderAccess
    private static let maximumBytes = 1_400_000
    init(directory: URL, access: FlowFolderAccess? = nil) {
        self.directory = directory; self.access = access ?? FlowFolderAccess()
    }
    /// Cached metadata remains listable even when external files or grants are unavailable.
    func list() throws -> [Record] {
        guard FileManager.default.fileExists(atPath: directory.path) else { return [] }
        let urls = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
        guard urls.count <= 1_000 else { throw FlowFolderAccess.Failure("The external project catalog exceeds its entry limit.") }
        return try urls.filter { $0.pathExtension == "json" }.map { url in
            guard let id = UUID(uuidString: url.deletingPathExtension().lastPathComponent) else { throw FlowFolderAccess.Failure("External project catalog contains an invalid record name.") }
            return try read(id)
        }.sorted { $0.id.uuidString < $1.id.uuidString }
    }
    /// Called only after the user explicitly selects the containing folder with read/write access.
    func attach(folder: URL, rootURL: URL) throws -> Record {
        guard rootURL.isFileURL, FlowFolderAccess.validRootFilename(rootURL.lastPathComponent) else { throw FlowFolderAccess.Failure("Choose a workflow file directly inside its flow folder.") }
        let existing = try list()
        let preliminaryID = UUID()
        let grant = try access.grant(folder: folder, rootFilename: rootURL.lastPathComponent, locationID: preliminaryID)
        guard rootURL.deletingLastPathComponent().resolvingSymlinksInPath().standardizedFileURL == grant.lease.folderURL else {
            throw FlowFolderAccess.Failure("The selected folder must directly contain this workflow file.")
        }
        let snapshot = try FlowDocumentStore.snapshot(at: grant.lease.rootURL)
        let workflow = try snapshot.workflow()
        let previous = existing.first { $0.rootURL == grant.lease.rootURL }
        guard previous != nil || existing.count < 1_000 else { throw FlowFolderAccess.Failure("The external project catalog is full. Remove an unused location before adding another.") }
        guard previous == nil || previous?.projectID == workflow.projectID else { throw FlowFolderAccess.Failure("This location now contains a different project. Its existing local history has been retained.") }
        let record = Record(id: previous?.id ?? preliminaryID, projectID: workflow.projectID, title: workflow.name,
                            folderPath: grant.lease.folderURL.path, rootFilename: rootURL.lastPathComponent, bookmarkData: grant.bookmarkData)
        try validate(record)
        let data = try JSONEncoder().encode(Envelope(schemaVersion: 1, record: record))
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try DiagnosticFileIO.writeAtomic(data, root: directory, parts: [record.id.uuidString + ".json"])
        return record
    }
    func open(_ record: Record) throws -> FlowFolderAccess.Lease {
        let lease = try openFolder(record)
        let workflow = try FlowDocumentStore.snapshot(at: lease.rootURL).workflow()
        guard workflow.projectID == record.projectID else { throw FlowFolderAccess.Failure("The selected file no longer contains this project. Its saved local history has been retained.") }
        return lease
    }
    /// Explicitly selected replacement location preserves local history identity.
    /// Callers must first close all editors retaining this location's old grant.
    func relink(_ record: Record, folder: URL, rootURL: URL) throws -> Record {
        guard try read(record.id) == record else { throw FlowFolderAccess.Failure("This saved location changed. Refresh Projects before locating it again.") }
        guard rootURL.isFileURL, FlowFolderAccess.validRootFilename(rootURL.lastPathComponent) else {
            throw FlowFolderAccess.Failure("Choose the workflow file directly inside its flow folder.")
        }
        let grant = try access.grant(folder: folder, rootFilename: rootURL.lastPathComponent, locationID: record.id)
        guard rootURL.deletingLastPathComponent().resolvingSymlinksInPath().standardizedFileURL == grant.lease.folderURL else {
            throw FlowFolderAccess.Failure("The selected folder must directly contain this workflow file.")
        }
        let workflow = try FlowDocumentStore.snapshot(at: grant.lease.rootURL).workflow()
        guard workflow.projectID == record.projectID else {
            throw FlowFolderAccess.Failure("This file belongs to another project. The saved location and its local history were retained.")
        }
        guard try !list().contains(where: { $0.id != record.id && $0.rootURL == grant.lease.rootURL }) else {
            throw FlowFolderAccess.Failure("This file already has its own linked location and history. Open that entry instead.")
        }
        let replacement = Record(id: record.id, projectID: record.projectID, title: workflow.name,
            folderPath: grant.lease.folderURL.path, rootFilename: rootURL.lastPathComponent, bookmarkData: grant.bookmarkData)
        try validate(replacement)
        guard try read(record.id) == record else { throw FlowFolderAccess.Failure("This saved location changed. Refresh Projects before locating it again.") }
        try DiagnosticFileIO.writeAtomic(JSONEncoder().encode(Envelope(schemaVersion: 1, record: replacement)),
            root: directory, parts: [record.id.uuidString + ".json"])
        return replacement
    }
    /// Child editors can retain/reopen the granted folder without creating or
    /// validating a root editor. Individual child identity is checked by its opener.
    func openFolder(_ record: Record) throws -> FlowFolderAccess.Lease {
        guard try read(record.id) == record else { throw FlowFolderAccess.Failure("The external project permission changed. Refresh the project library before reopening it.") }
        return try access.open(locationID: record.id, folderPath: record.folderPath,
                               rootFilename: record.rootFilename, bookmarkData: record.bookmarkData)
    }
    private func read(_ id: UUID) throws -> Record {
        let data = try DiagnosticFileIO.read(root: directory, parts: [id.uuidString + ".json"], limit: Self.maximumBytes)
        let envelope = try JSONDecoder().decode(Envelope.self, from: data)
        guard envelope.schemaVersion == 1, envelope.record.id == id else { throw FlowFolderAccess.Failure("External project catalog identity is invalid.") }
        try validate(envelope.record)
        return envelope.record
    }
    private func validate(_ record: Record) throws {
        guard UUID(uuidString: record.projectID)?.uuidString == record.projectID,
              record.title.utf8.count <= 4_096,
              record.folderPath.hasPrefix("/"), !record.folderPath.contains("\0"), record.folderPath.utf8.count <= 4_096,
              URL(fileURLWithPath: record.folderPath).standardizedFileURL.path == record.folderPath,
              FlowFolderAccess.validRootFilename(record.rootFilename), !record.bookmarkData.isEmpty, record.bookmarkData.count <= 1_000_000 else {
            throw FlowFolderAccess.Failure("External project catalog metadata is invalid.")
        }
    }
}
