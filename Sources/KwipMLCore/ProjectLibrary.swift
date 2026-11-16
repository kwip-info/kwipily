import Foundation

public struct ProjectRecord: Codable, Identifiable, Equatable {
    public var id: String
    public var name: String
    public var path: String
    public var guidedCourseID:String?
    public var updatedAt: Date
    public init(id: String, name: String, path: String, updatedAt: Date) { self.id=id; self.name=name; self.path=path; self.updatedAt=updatedAt }
}

public enum ProjectLibraryError: LocalizedError {
    case invalidIdentity, unsupportedSchema, resourceLimit, unsafeFile, documentKindMismatch
    public var errorDescription: String? {
        switch self {
        case .documentKindMismatch: return "This project uses a different authoring format. Open and save its source with the matching editor."
        case .invalidIdentity: return "Project identity must be a UUID."
        case .unsupportedSchema: return "This project uses an unsupported schema version."
        case .resourceLimit: return "Project exceeds the POC document limits or contains non-finite values."
        case .unsafeFile: return "Project documents must be regular files, not symbolic links."
        }
    }
}

/// Single-writer project store. Each UUID-named document is the source of truth;
/// the recent-project index is derived so a failed index write cannot orphan a save.
public final class ProjectLibrary {
    private let directory: URL
    private let manager = FileManager.default
    private var lastSave = Date.distantPast
    private let maximumBytes = 2_000_000
    private var externalFlow: (url: URL, projectID: String)?
    public init(directory: URL) { self.directory = directory.standardizedFileURL }

    /// The app owns the directory access lease and a separate private workspace for this location.
    /// External source bytes remain authoritative; this is not an import or a legacy fallback.
    public init(directory: URL, externalFlowRoot: URL, projectID: String) {
        self.directory = directory.standardizedFileURL
        self.externalFlow = (externalFlowRoot.standardizedFileURL, projectID)
    }

    public func documentURL(id: String) throws -> URL {
        guard let uuid = UUID(uuidString:id), uuid.uuidString.lowercased() == id.lowercased() else { throw ProjectLibraryError.invalidIdentity }
        return directory.appendingPathComponent(uuid.uuidString).appendingPathExtension("kwipml")
    }

    public func list() throws -> [ProjectRecord] {
        guard manager.fileExists(atPath:directory.path) else { return [] }
        let urls = try manager.contentsOfDirectory(at:directory,includingPropertiesForKeys:[.contentModificationDateKey,.isRegularFileKey,.isSymbolicLinkKey],options:[.skipsHiddenFiles])
        // A corrupt document remains on disk for recovery and does not hide healthy projects.
        return urls.compactMap { url -> ProjectRecord? in
            guard url.pathExtension == "kwipml", let uuid = UUID(uuidString:url.deletingPathExtension().lastPathComponent), let graph = try? read(url), let values = try? url.resourceValues(forKeys:[.contentModificationDateKey]) else { return nil }
            return ProjectRecord(id:uuid.uuidString,name:graph.name,path:url.path,updatedAt:values.contentModificationDate ?? .distantPast)
        }.sorted { $0.updatedAt == $1.updatedAt ? $0.id < $1.id : $0.updatedAt > $1.updatedAt }
    }

    public func save(_ graph: GraphDocument, id: String? = nil) throws -> ProjectRecord {
        try validateDocument(graph)
        let identity = id ?? UUID().uuidString
        let url = try documentURL(id:identity)
        if manager.fileExists(atPath:url.path) {
            let existing = try readData(url)
            if let object = try? JSONSerialization.jsonObject(with:existing) as? [String:Any], object["kind"] != nil {
                throw ProjectLibraryError.documentKindMismatch
            }
        }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted,.sortedKeys]
        return try write(encoder.encode(graph), id: identity, name: graph.name)
    }

    public func load(id: String) throws -> GraphDocument { try read(documentURL(id:id)) }
    public func importDocument(from: URL) throws -> ProjectRecord { try save(read(from)) }

    /// Imports a new managed parent, preserving flat input bytes or rebasing only Block parent identities.
    /// External portable filenames do not need to match the embedded parent UUID.
    public func importModelDocument(from: URL) throws -> ProjectRecord {
        let original = try readData(from)
        let document = try ModelProjectCodec.decode(original)
        var identity: String
        var destination: URL
        repeat {
            identity = UUID().uuidString
            destination = try documentURL(id: identity)
        } while manager.fileExists(atPath: destination.path)
            || (try? destination.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) == true

        let name: String
        let data: Data
        switch document {
        case .legacyFlat(let graph):
            name = graph.name
            // This is a validated copy, not an editing migration or canonical reserialization.
            data = original
        case .blockModel(var model):
            name = model.source.name
            model.projectID = identity
            for index in model.drafts.indices { model.drafts[index].projectID = identity }
            data = try ModelProjectCodec.encode(.blockModel(model))
        }
        return try write(data, id: identity, name: name)
    }

    /// Document-aware entry points are explicit while legacy callers remain flat-only.
    public func loadDocument(id: String) throws -> ModelProjectDocument {
        let document = try ModelProjectCodec.decode(readData(documentURL(id:id)))
        if case .blockModel(let model) = document,
           UUID(uuidString:model.projectID) != UUID(uuidString:id) { throw ProjectLibraryError.invalidIdentity }
        return document
    }
    public func listDocuments() throws -> [ProjectRecord] {
        guard manager.fileExists(atPath:directory.path) else { return [] }
        let urls = try manager.contentsOfDirectory(at:directory,includingPropertiesForKeys:[.contentModificationDateKey],options:[.skipsHiddenFiles])
        return urls.compactMap { url -> ProjectRecord? in
            guard url.pathExtension == "kwipml", let id = UUID(uuidString:url.deletingPathExtension().lastPathComponent),
                  let document = try? loadDocument(id:id.uuidString),
                  let values = try? url.resourceValues(forKeys:[.contentModificationDateKey]) else { return nil }
            let name: String
            switch document { case .legacyFlat(let graph): name = graph.name; case .blockModel(let model): name = model.source.name }
            return ProjectRecord(id:id.uuidString,name:name,path:url.path,updatedAt:values.contentModificationDate ?? .distantPast)
        }.sorted { $0.updatedAt == $1.updatedAt ? $0.id < $1.id : $0.updatedAt > $1.updatedAt }
    }
    public func saveDocument(_ document: ModelProjectDocument, id: String? = nil) throws -> ProjectRecord {
        let identity: String
        let name: String
        switch document {
        case .legacyFlat(let graph):
            // Explicit document history may restore a pre-Block flat snapshot.
            // Legacy save(_:id:) remains protected against accidental flattening.
            identity = id ?? UUID().uuidString; name = graph.name
        case .blockModel(let model):
            identity = id ?? model.projectID; name = model.source.name
            guard UUID(uuidString:identity) == UUID(uuidString:model.projectID) else { throw ProjectLibraryError.invalidIdentity }
        }
        let data = try ModelProjectCodec.encode(document)
        let url = try documentURL(id: identity)
        if existsIncludingSymbolicLink(url) {
            // Model-only callers may restore flat/Block history, but cannot silently
            // replace a workflow or use malformed tagged source as a flat fallback.
            let existing = try loadSource(id: identity)
            if case .workflow = existing { throw ProjectLibraryError.documentKindMismatch }
        }
        return try write(data, id: identity, name: name)
    }

    /// The independent marker survives a missing/moved project directory; no legacy fallback follows cutover.
    public func fileBackedProjectURL(id: String) throws -> URL? {
        guard let identity = UUID(uuidString: id) else { throw ProjectLibraryError.invalidIdentity }
        if let externalFlow {
            guard UUID(uuidString: externalFlow.projectID) == identity else { throw ProjectLibraryError.invalidIdentity }
            let head = try FlowDocumentStore.readHead(at: externalFlow.url)
            guard head.id == identity, case .workflow = head.payload else { throw ProjectLibraryError.documentKindMismatch }
            return externalFlow.url
        }
        let marked = try readFlowMarker(id: identity)
        let folder = directory.appendingPathComponent("FlowProjects", isDirectory: true)
        guard existsIncludingSymbolicLink(folder) else {
            if marked { throw ProjectLibraryError.unsafeFile }; return nil
        }
        try requireDirectory(folder)
        let project = folder.appendingPathComponent(identity.uuidString, isDirectory: true)
        guard existsIncludingSymbolicLink(project) else {
            if marked { throw ProjectLibraryError.unsafeFile }; return nil
        }
        try requireDirectory(project)
        return project.resolvingSymlinksInPath().appendingPathComponent("Main.kwipflow")
    }
    public func activateFileBackedWorkflow(id: String) throws -> ProjectRecord {
        if let url = try fileBackedProjectURL(id: id) {
            guard case .workflow(let source) = try loadSource(id: id) else { throw ProjectLibraryError.documentKindMismatch }
            if externalFlow == nil { try writeFlowMarker(id: UUID(uuidString: id)!) }
            return ProjectRecord(id: source.projectID, name: source.name, path: url.resolvingSymlinksInPath().path, updatedAt: Date())
        }
        guard case .workflow(let source) = try loadSource(id: id) else { throw ProjectLibraryError.documentKindMismatch }
        // Export migration preserves drafts; active editing awaits a dedicated independent draft editor.
        guard source.drafts.isEmpty else { throw FlowDocumentError.pendingDrafts }
        let folder = directory.appendingPathComponent("FlowProjects", isDirectory: true)
        if existsIncludingSymbolicLink(folder) { try requireDirectory(folder) }
        else { try manager.createDirectory(at: folder, withIntermediateDirectories: false) }
        let plan = try FlowDocumentMigration.plan(source)
        let url = try FlowDocumentStore.commitMigration(plan, to: folder.appendingPathComponent(UUID(uuidString: id)!.uuidString))
        try writeFlowMarker(id: UUID(uuidString: id)!)
        return ProjectRecord(id: source.projectID, name: source.name, path: url.resolvingSymlinksInPath().path, updatedAt: Date())
    }
    private func flowMarkerURL(id: UUID) -> URL {
        directory.appendingPathComponent(id.uuidString + ".flow-project")
    }
    private func flowMarkerBytes(id: UUID) -> Data { Data(("kwipml-flow-project-v1\n" + id.uuidString + "\n").utf8) }
    private func readFlowMarker(id: UUID) throws -> Bool {
        let url = flowMarkerURL(id: id)
        guard existsIncludingSymbolicLink(url) else { return false }
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey])
        guard values.isRegularFile == true, values.isSymbolicLink != true,
              let size = values.fileSize, size <= 128,
              try Data(contentsOf: url) == flowMarkerBytes(id: id) else { throw ProjectLibraryError.unsafeFile }
        return true
    }
    private func writeFlowMarker(id: UUID) throws {
        if try readFlowMarker(id: id) { return }
        let url = flowMarkerURL(id: id)
        do { try flowMarkerBytes(id: id).write(to: url, options: .withoutOverwriting) }
        catch {
            if (try? readFlowMarker(id: id)) == true { return }
            throw error
        }
        let handle = try FileHandle(forWritingTo: url)
        try handle.synchronize(); try handle.close()
    }
    private func requireDirectory(_ url: URL) throws {
        let value = try url.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        guard value.isDirectory == true, value.isSymbolicLink != true else { throw ProjectLibraryError.unsafeFile }
    }

    /// All supported portable source kinds; no automatic conversion to another kind.
    public func loadSource(id: String) throws -> ProjectSourceDocument {
        if let url = try fileBackedProjectURL(id: id) {
            let workflow = try FlowDocumentStore.snapshot(at: url).workflow()
            guard UUID(uuidString: workflow.projectID) == UUID(uuidString: id) else { throw ProjectLibraryError.invalidIdentity }
            return .workflow(workflow) // Derived read projection; saveSource cannot write it back.
        }
        let document = try ProjectSourceCodec.decode(readData(documentURL(id: id)))
        if let embedded = document.projectID,
           UUID(uuidString: embedded) != UUID(uuidString: id) { throw ProjectLibraryError.invalidIdentity }
        return document
    }

    public func listSources() throws -> [ProjectRecord] {
        if let externalFlow {
            let source = try loadSource(id: externalFlow.projectID)
            let date = try externalFlow.url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate ?? .distantPast
            var record = ProjectRecord(id: externalFlow.projectID, name: source.name, path: externalFlow.url.path, updatedAt: date)
            if case .workflow(let workflow) = source { record.guidedCourseID = workflow.guidedBuild?.courseID }
            return [record]
        }
        guard manager.fileExists(atPath: directory.path) else { return [] }
        let urls = try manager.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.contentModificationDateKey], options: [.skipsHiddenFiles])
        var records: [String: ProjectRecord] = [:]
        for url in urls where url.pathExtension == "kwipml" {
            guard let id = UUID(uuidString: url.deletingPathExtension().lastPathComponent),
                  (try? fileBackedProjectURL(id: id.uuidString)) == nil,
                  let document = try? loadSource(id: id.uuidString),
                  let values = try? url.resourceValues(forKeys: [.contentModificationDateKey]) else { continue }
            var record = ProjectRecord(id: id.uuidString, name: document.name, path: url.path, updatedAt: values.contentModificationDate ?? .distantPast)
            if case .workflow(let workflow) = document { record.guidedCourseID = workflow.guidedBuild?.courseID }
            records[id.uuidString] = record
        }
        let flows = directory.appendingPathComponent("FlowProjects", isDirectory: true)
        var flowIDs = Set(urls.filter { $0.pathExtension == "flow-project" }.compactMap { UUID(uuidString: $0.deletingPathExtension().lastPathComponent) })
        if existsIncludingSymbolicLink(flows) {
            try requireDirectory(flows)
            for folder in try manager.contentsOfDirectory(at: flows, includingPropertiesForKeys: [.contentModificationDateKey]) {
                if let id = UUID(uuidString: folder.lastPathComponent) { flowIDs.insert(id) }
            }
        }
        for id in flowIDs {
            let url = flows.appendingPathComponent(id.uuidString).resolvingSymlinksInPath().appendingPathComponent("Main.kwipflow")
            let source = try? loadSource(id: id.uuidString)
            let date = (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            var record = ProjectRecord(id: id.uuidString, name: source?.name ?? "Unavailable workflow", path: url.path, updatedAt: date)
            if case .workflow(let workflow) = source { record.guidedCourseID = workflow.guidedBuild?.courseID }
            records[id.uuidString] = record
        }
        return records.values.sorted { $0.updatedAt == $1.updatedAt ? $0.id < $1.id : $0.updatedAt > $1.updatedAt }
    }

    public func saveSource(_ document: ProjectSourceDocument, id: String? = nil) throws -> ProjectRecord {
        let identity = id ?? document.projectID ?? UUID().uuidString
        guard try fileBackedProjectURL(id: identity) == nil else { throw ProjectLibraryError.documentKindMismatch }
        let url = try documentURL(id: identity)
        if let embedded = document.projectID,
           UUID(uuidString: embedded) != UUID(uuidString: identity) { throw ProjectLibraryError.invalidIdentity }
        let data = try ProjectSourceCodec.encode(document)
        if existsIncludingSymbolicLink(url) {
            let existing = try loadSource(id: identity)
            // Model history may restore a pre-Block flat snapshot. Workflow conversion
            // requires a separate, explicit migration transaction, not an ordinary save.
            let oldWorkflow: Bool
            let newWorkflow: Bool
            if case .workflow = existing { oldWorkflow = true } else { oldWorkflow = false }
            if case .workflow = document { newWorkflow = true } else { newWorkflow = false }
            guard oldWorkflow == newWorkflow else { throw ProjectLibraryError.documentKindMismatch }
        }
        var record=try write(data, id: identity, name: document.name)
        if case .workflow(let workflow)=document {record.guidedCourseID=workflow.guidedBuild?.courseID}
        return record
    }

    public func importSource(from url: URL) throws -> ProjectRecord {
        let original = try readData(url)
        let source = try ProjectSourceCodec.decode(original)
        let identity = try freshIdentity()
        let data: Data
        switch source {
        case .legacyFlat:
            // Preserve historical formatting and unknown optional execution fields.
            data = original
        case .blockModel(var model):
            model.projectID = identity
            for index in model.drafts.indices { model.drafts[index].projectID = identity }
            data = try ProjectSourceCodec.encode(.blockModel(model))
        case .workflow(var workflow):
            workflow.projectID = identity
            for index in workflow.drafts.indices { workflow.drafts[index].projectID = identity }
            data = try ProjectSourceCodec.encode(.workflow(workflow))
        }
        return try write(data, id: identity, name: source.name)
    }

    private func freshIdentity() throws -> String {
        var identity: String
        repeat { identity = UUID().uuidString }
        while try existsIncludingSymbolicLink(documentURL(id: identity))
        return identity
    }

    private func existsIncludingSymbolicLink(_ url: URL) -> Bool {
        manager.fileExists(atPath: url.path)
            || (try? url.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) == true
    }

    /// Single commit point for all formats; metadata failure cannot report a committed
    /// source write as failed and leave the in-memory owner behind the file.
    private func write(_ data: Data, id: String, name: String) throws -> ProjectRecord {
        guard externalFlow == nil else { throw ProjectLibraryError.documentKindMismatch }
        guard try fileBackedProjectURL(id: id) == nil else { throw ProjectLibraryError.documentKindMismatch }
        guard data.count <= maximumBytes else { throw ProjectLibraryError.resourceLimit }
        let url = try documentURL(id: id)
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        if existsIncludingSymbolicLink(url) { try regularFile(url) }
        try data.write(to: url, options: .atomic)
        let stamp = max(Date(), lastSave.addingTimeInterval(0.001)); lastSave = stamp
        try? manager.setAttributes([.modificationDate: stamp], ofItemAtPath: url.path)
        return ProjectRecord(id: UUID(uuidString: id)!.uuidString, name: name, path: url.path, updatedAt: stamp)
    }

    private func regularFile(_ url: URL) throws {
        let values = try url.resourceValues(forKeys:[.isRegularFileKey,.isSymbolicLinkKey])
        guard values.isRegularFile == true, values.isSymbolicLink != true else { throw ProjectLibraryError.unsafeFile }
    }
    private func readData(_ url: URL) throws -> Data {
        try regularFile(url)
        let attributes = try manager.attributesOfItem(atPath:url.path)
        guard let size = attributes[.size] as? NSNumber, size.intValue <= maximumBytes else { throw ProjectLibraryError.resourceLimit }
        let data = try Data(contentsOf:url)
        guard data.count <= maximumBytes else { throw ProjectLibraryError.resourceLimit }
        return data
    }
    private func read(_ url: URL) throws -> GraphDocument {
        let data = try readData(url)
        if let object = try JSONSerialization.jsonObject(with:data) as? [String:Any], object["kind"] != nil { throw ProjectLibraryError.documentKindMismatch }
        let graph = try JSONDecoder().decode(GraphDocument.self,from:data)
        try validateDocument(graph)
        return graph
    }
    private func validateDocument(_ graph: GraphDocument) throws {
        guard [1,2].contains(graph.schemaVersion) else { throw ProjectLibraryError.unsupportedSchema }
        guard graph.nodes.count <= 128, graph.edges.count <= 256,
              graph.nodes.allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.config.values.allSatisfy(\.isFiniteJSON) }) else { throw ProjectLibraryError.resourceLimit }
    }
}
