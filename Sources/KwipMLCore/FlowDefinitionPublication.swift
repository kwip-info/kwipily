import Foundation
import Darwin

/// A completed definition fork publishes new immutable files; it never edits a pinned child.
enum FlowDefinitionPublication {
    struct Receipt: Codable {
        let schemaVersion: Int
        let rootFilename: String
        let expected: FlowDocumentRevision
        let candidate: FlowDocument
        let children: [String: FlowDocument]
    }
    struct Plan {
        let root: FlowDocument
        let children: [String: FlowDocument]
    }
    static func plan(current: FlowDocument, snapshot: FlowDocumentSnapshot, next: WorkflowDocument,
                     newID: () -> UUID = { UUID() }) throws -> Plan {
        try WorkflowCodec.validate(next)
        let previous = try snapshot.workflow()
        guard previous.projectID == next.projectID else { throw FlowDocumentError.pendingDrafts }
        let old = Dictionary(uniqueKeysWithValues: previous.definitions.map { ($0.reference, $0) })
        for definition in next.definitions {
            if let saved = old[definition.reference], saved != definition { throw FlowDocumentError.conflictingDefinition }
        }
        var documents: [FlowDocumentRevision: FlowDocument] = [:]
        for document in snapshot.documents { documents[try FlowDocumentCodec.reference(document)] = document }
        var locations = Dictionary(uniqueKeysWithValues: current.dependencies.map { ($0.definition, $0) })
        var visited = Set<FlowDocumentRevision>()
        func locate(_ document: FlowDocument, base: String) throws {
            guard visited.insert(try FlowDocumentCodec.reference(document)).inserted else { return }
            for dependency in document.dependencies {
                let path = base + dependency.relativePath
                guard let child = documents[dependency.revision] else { throw FlowDocumentError.missingDependency }
                if locations[dependency.definition] == nil {
                    locations[dependency.definition] = .init(definition: dependency.definition, revision: dependency.revision, relativePath: path)
                }
                let pieces = path.split(separator: "/").dropLast()
                try locate(child, base: pieces.isEmpty ? "" : pieces.joined(separator: "/") + "/")
            }
        }
        try locate(current, base: "")
        let definitions = Dictionary(uniqueKeysWithValues: next.definitions.map { ($0.reference, $0) })
        var usedIDs = Set(snapshot.documents.map(\.id)), active = Set<CompositeDefinitionReference>()
        var children: [String: FlowDocument] = [:]
        func build(_ reference: CompositeDefinitionReference, depth: Int) throws -> FlowDocumentDependency {
            guard definitions[reference] != nil else { throw FlowDocumentError.missingDependency }
            if let existing = locations[reference] { return existing }
            guard depth <= 8 else { throw FlowDocumentError.resourceLimit }
            guard active.insert(reference).inserted else { throw FlowDocumentError.cycle }
            defer { active.remove(reference) }
            let definition = definitions[reference]!
            var dependencies: [FlowDocumentDependency] = [], seen = Set<CompositeDefinitionReference>()
            for instance in definition.body.instances where seen.insert(instance.definition).inserted {
                dependencies.append(try build(instance.definition, depth: depth + 1))
            }
            let id = newID()
            guard usedIDs.insert(id).inserted else { throw FlowDocumentError.invalidDocument }
            let child = FlowDocument(id: id, payload: .modelBlock(definition), dependencies: dependencies)
            let pin = try FlowDocumentCodec.reference(child), path = id.uuidString + ".kwipflow"
            children[path] = child; documents[pin] = child
            let dependency = FlowDocumentDependency(definition: reference, revision: pin, relativePath: path)
            locations[reference] = dependency
            return dependency
        }
        var root = current
        root.dependencies = try next.definitions.map { try build($0.reference, depth: 1) }
        root.draftDependencies = []
        for draft in next.drafts {
            let id = newID()
            guard usedIDs.insert(id).inserted else { throw FlowDocumentError.invalidDocument }
            let child = FlowDocument(id: id, payload: .modelBlockDraft(draft))
            let pin = try FlowDocumentCodec.reference(child)
            let path = ".kwipflow-draft-" + id.uuidString + ".kwipflow"
            children[path] = child; documents[pin] = child
            root.draftDependencies.append(.init(revision: pin, relativePath: path))
        }
        if !root.draftDependencies.isEmpty { root.schemaVersion = 2 }
        root.revision = next.revision
        var portable = next; portable.definitions = []; portable.drafts = []
        root.payload = .workflow(portable)
        let projected = try FlowDocumentResolver.resolve(root) { pin in
            guard let document = documents[pin] else { throw FlowDocumentError.missingDependency }
            return document
        }
        guard try projected.workflow() == next else { throw FlowDocumentError.invalidDocument }
        return .init(root: root, children: children)
    }

    /// Called only while the root CAS lock is held after its expected head passed.
    /// Every interrupted write remains accompanied by a complete bounded recovery plan.
    static func publish(_ plan: Plan, beside rootURL: URL, expected: FlowDocumentRevision,
                        directoryDescriptor: Int32? = nil, retention: FlowPublicationRetention = .init(), retentionPreferences: (() throws -> FlowAuthoringRetentionPreferences)? = nil, afterWrite: ((URL) throws -> Void)? = nil) throws {
        guard !plan.children.isEmpty else { return }
        let parent = rootURL.deletingLastPathComponent()
        guard let canonical = realpath(parent.path, nil) else { throw FlowDocumentError.unsafeReference }
        let directory = URL(fileURLWithPath: String(cString: canonical)); free(canonical)
        let fd = directoryDescriptor.map { dup($0) } ?? open(directory.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW_ANY | O_CLOEXEC)
        guard fd >= 0 else { throw FlowDocumentError.unsafeReference }
        defer { close(fd) }
        // Different root documents can share a folder; serialize its evidence admission and writes.
        let publicationLock = openat(fd, ".kwipflow-publication.lock", O_CREAT | O_RDWR | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC, mode_t(0o600))
        guard publicationLock >= 0 else { throw FlowDocumentError.unsafeReference }
        defer { close(publicationLock) }
        var lockInfo = stat()
        guard fstat(publicationLock, &lockInfo) == 0, lockInfo.st_mode & S_IFMT == S_IFREG else { throw FlowDocumentError.unsafeReference }
        guard flock(publicationLock, LOCK_EX | LOCK_NB) == 0 else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
        defer { flock(publicationLock, LOCK_UN) }
        // Capture the current validated global policy once, under both root and folder locks.
        let effectiveRetention: FlowPublicationRetention
        if let retentionPreferences {
            let policy = try retentionPreferences()
            effectiveRetention = .init(maximumFiles: policy.maximumFiles, maximumBytes: policy.maximumBytes)
        } else { effectiveRetention = retention }
        let receipt = Receipt(schemaVersion: 1, rootFilename: rootURL.lastPathComponent, expected: expected,
                              candidate: plan.root, children: plan.children)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let recovery = try encoder.encode(receipt)
        guard recovery.count <= 32_000_000 else { throw FlowDocumentError.resourceLimit }
        let draftBytes = try plan.children.filter { $0.key.hasPrefix(".kwipflow-draft-") }.map { try FlowDocumentCodec.encode($0.value) }
        try effectiveRetention.check(directory: fd, incoming: [recovery] + draftBytes)
        func write(_ bytes: Data, name: String) throws {
            guard !name.isEmpty, !name.contains("/"), name != ".", name != ".." else { throw FlowDocumentError.unsafeReference }
            let output = openat(fd, name, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, mode_t(0o600))
            guard output >= 0 else { throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno)) }
            defer { close(output) }
            try bytes.withUnsafeBytes { raw in
                var offset = 0
                while offset < raw.count {
                    let count = Darwin.write(output, raw.baseAddress!.advanced(by: offset), raw.count - offset)
                    if count < 0, errno == EINTR { continue }
                    guard count > 0 else { throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno)) }
                    offset += count
                }
            }
            guard fsync(output) == 0, fsync(fd) == 0 else { throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno)) }
            try afterWrite?(directory.appendingPathComponent(name))
        }
        try write(recovery, name: ".kwipflow-publication-" + UUID().uuidString + ".json")
        for path in plan.children.keys.sorted() { try write(FlowDocumentCodec.encode(plan.children[path]!), name: path) }
    }
}
