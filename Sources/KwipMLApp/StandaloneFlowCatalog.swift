import Foundation
import KwipMLCore

/// Private grants for independently opened model-block documents. Never imports them.
@MainActor final class StandaloneFlowCatalog {
    struct Record: Codable, Equatable, Identifiable {
        let id: UUID
        let documentID: UUID
        let title: String
        let folderPath: String
        let filename: String
        let bookmarkData: Data
        var url: URL { URL(fileURLWithPath: folderPath, isDirectory: true).appendingPathComponent(filename) }
    }
    struct Opened { let record: Record; let lease: FlowFolderAccess.Lease }
    private struct Envelope: Codable { let schemaVersion: Int; let record: Record }
    private let directory: URL
    private let access: FlowFolderAccess
    init(directory: URL, access: FlowFolderAccess? = nil) { self.directory = directory; self.access = access ?? FlowFolderAccess() }
    func list() throws -> [Record] {
        guard FileManager.default.fileExists(atPath: directory.path) else { return [] }
        let entries = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
        guard entries.count <= 1000 else { throw FlowFolderAccess.Failure("The independent subflow catalog exceeds its entry limit.") }
        return try entries.filter { $0.pathExtension == "json" }.map { url in
            guard let id = UUID(uuidString: url.deletingPathExtension().lastPathComponent) else { throw FlowFolderAccess.Failure("The independent subflow catalog contains an invalid identity.") }
            return try read(id)
        }.sorted { $0.id.uuidString < $1.id.uuidString }
    }
    func attach(folder: URL, file: URL) throws -> Record {
        let entries = try list(), proposedID = UUID()
        let grant = try access.grant(folder: folder, rootFilename: file.lastPathComponent, locationID: proposedID)
        guard file.resolvingSymlinksInPath().standardizedFileURL.path == grant.lease.rootURL.path else {
            throw FlowFolderAccess.Failure("Choose the folder directly containing this subflow file.")
        }
        let document = try FlowDocumentStore.readHead(at: grant.lease.rootURL)
        guard case .modelBlock(let block) = document.payload else { throw FlowFolderAccess.Failure("Open a model-block document here. Workflow roots belong in Open project.") }
        let previous = entries.first { $0.url.path == grant.lease.rootURL.path }
        guard previous == nil || previous?.documentID == document.id else { throw FlowFolderAccess.Failure("This path now contains a different subflow. Its saved session has been retained.") }
        guard previous != nil || entries.count < 1000 else { throw FlowFolderAccess.Failure("The independent subflow catalog is full.") }
        let record = Record(id: previous?.id ?? proposedID, documentID: document.id, title: block.title,
                            folderPath: grant.lease.folderURL.path, filename: file.lastPathComponent, bookmarkData: grant.bookmarkData)
        try validate(record)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try DiagnosticFileIO.writeAtomic(JSONEncoder().encode(Envelope(schemaVersion: 1, record: record)), root: directory, parts: [record.id.uuidString + ".json"])
        return record
    }
    func open(_ id: UUID) throws -> Opened {
        let record = try read(id)
        let lease = try access.open(locationID: record.id, folderPath: record.folderPath, rootFilename: record.filename, bookmarkData: record.bookmarkData)
        let document = try FlowDocumentStore.readHead(at: lease.rootURL)
        guard document.id == record.documentID, case .modelBlock = document.payload else { throw FlowFolderAccess.Failure("The saved subflow identity changed. Its session has been retained.") }
        return Opened(record: record, lease: lease)
    }
    private func read(_ id: UUID) throws -> Record {
        let bytes = try DiagnosticFileIO.read(root: directory, parts: [id.uuidString + ".json"], limit: 1_400_000)
        let envelope = try JSONDecoder().decode(Envelope.self, from: bytes)
        guard envelope.schemaVersion == 1, envelope.record.id == id else { throw FlowFolderAccess.Failure("The saved subflow catalog version or identity is unsupported.") }
        try validate(envelope.record); return envelope.record
    }
    private func validate(_ record: Record) throws {
        guard record.title.utf8.count <= 4096, record.folderPath.hasPrefix("/"), !record.folderPath.contains("\0"),
              record.folderPath.utf8.count <= 4096, URL(fileURLWithPath: record.folderPath).standardizedFileURL.path == record.folderPath,
              FlowFolderAccess.validRootFilename(record.filename), !record.bookmarkData.isEmpty, record.bookmarkData.count <= 1_000_000 else {
            throw FlowFolderAccess.Failure("The saved subflow catalog metadata is invalid.")
        }
    }
}
