import Foundation

/// One root authority selects a storage backend; resolved definitions are a read projection, never saved copies.
public final class WorkflowSourcePersistence {
    public struct State: Equatable {
        public let document: WorkflowDocument
        public let flowDocument: FlowDocument?
    }
    public struct Adoption: Equatable {
        public let before: State
        public let after: State
        public let definition: CompositeDefinitionReference
        public let expectedHead: FlowDocumentRevision
        fileprivate init(before: State, after: State, definition: CompositeDefinitionReference, expectedHead: FlowDocumentRevision) {
            self.before = before; self.after = after; self.definition = definition; self.expectedHead = expectedHead
        }
    }
    public private(set) var state: State
    public let fileURL: URL?
    var publicationRetention = FlowPublicationRetention()
    /// App injects the global policy; Core never knows where preferences are stored.
    public var authoringRetentionPreferences: (() throws -> FlowAuthoringRetentionPreferences)?
    var publicationDidWrite: ((URL) throws -> Void)?
    private let library: ProjectLibrary
    private let projectID: String
    public init(library: ProjectLibrary, projectID: String) throws {
        self.library = library; self.projectID = projectID
        fileURL = try library.fileBackedProjectURL(id: projectID)
        if let fileURL {
            let snapshot = try FlowDocumentStore.snapshot(at: fileURL)
            guard let root = snapshot.documents.first, try FlowDocumentCodec.reference(root) == snapshot.root,
                  UUID(uuidString: projectID) == root.id else { throw FlowDocumentError.invalidDocument }
            state = .init(document: try snapshot.workflow(), flowDocument: root)
        } else {
            guard case .workflow(let source) = try library.loadSource(id: projectID) else { throw ProjectLibraryError.documentKindMismatch }
            state = .init(document: source, flowDocument: nil)
        }
    }
    public func currentRecord() throws -> ProjectRecord {
        var result = ProjectRecord(id: projectID, name: state.document.name,
            path: try fileURL?.path ?? library.documentURL(id: projectID).path, updatedAt: Date())
        result.guidedCourseID = state.document.guidedBuild?.courseID
        return result
    }
    @discardableResult public func save(_ next: WorkflowDocument, restoring envelope: FlowDocument? = nil) throws -> ProjectRecord {
        guard next.projectID == state.document.projectID else { throw ProjectLibraryError.invalidIdentity }
        if let current = state.flowDocument, let fileURL {
            var publication: FlowDefinitionPublication.Plan?
            if envelope == nil, (next.definitions != state.document.definitions || next.drafts != state.document.drafts) {
                publication = try FlowDefinitionPublication.plan(current: current,
                    snapshot: FlowDocumentStore.snapshot(of: current, at: fileURL), next: next)
            }
            var candidate = envelope ?? publication?.root ?? current
            guard candidate.id == current.id,
                  case .workflow = candidate.payload else { throw FlowDocumentError.invalidDocument }
            var portable = next; portable.definitions = []; portable.drafts = []
            candidate.revision = next.revision; candidate.payload = .workflow(portable)
            let expected = try FlowDocumentCodec.reference(current)
            try FlowDocumentStore.save(candidate, at: fileURL, expected: expected) { directoryDescriptor in
                if let publication {
                    try FlowDefinitionPublication.publish(publication, beside: fileURL, expected: expected,
                        directoryDescriptor: directoryDescriptor, retention: publicationRetention,
                        retentionPreferences: authoringRetentionPreferences, afterWrite: publicationDidWrite)
                }
                guard try FlowDocumentStore.snapshot(of: candidate, at: fileURL).workflow() == next else { throw FlowDocumentError.invalidDocument }
            }
            state = .init(document: next, flowDocument: candidate)
            return try currentRecord()
        }
        guard envelope == nil else { throw FlowDocumentError.invalidDocument }
        let record = try library.saveSource(.workflow(next), id: projectID)
        state = .init(document: next, flowDocument: nil)
        return record
    }
    /// Discovery is separate from review/commit and never advances a caller's saved pin.
    public func adoptionHead(definition: CompositeDefinitionReference) throws -> FlowDocumentRevision {
        try FlowDocumentCodec.reference(adoptionContext(definition: definition).child)
    }
    public func prepareAdoption(definition: CompositeDefinitionReference, expectedHead: FlowDocumentRevision) throws -> Adoption {
        let context = try adoptionContext(definition: definition)
        guard try FlowDocumentCodec.reference(context.child) == expectedHead,
              expectedHead != context.root.dependencies[context.index].revision else { throw FlowDocumentError.revisionMismatch }
        guard context.root.revision < UInt64.max else { throw FlowDocumentError.resourceLimit }
        var candidate = context.root
        candidate.dependencies[context.index].revision = expectedHead
        candidate.revision += 1
        guard case .workflow(var portable) = candidate.payload else { throw FlowDocumentError.invalidDocument }
        portable.revision = candidate.revision
        if case .modelBlock(let selected)=context.child.payload,selected.body.nodes.contains(where:{$0.parameterSettings != nil}) {
            WorkflowModelInput.migrateOwnershipMetadata(&portable);portable.schemaVersion=max(15,portable.schemaVersion)
        }
        candidate.payload = .workflow(portable)
        var snapshot = try FlowDocumentStore.snapshot(of:candidate,at:context.url)
        if portable.schemaVersion<15,snapshot.documents.contains(where:{item in
            if case .modelBlock(let block)=item.payload {return block.body.nodes.contains {$0.parameterSettings != nil}};return false
        }) {
            WorkflowModelInput.migrateOwnershipMetadata(&portable);portable.schemaVersion=15
            candidate.payload = .workflow(portable)
            snapshot = try FlowDocumentStore.snapshot(of:candidate,at:context.url)
        }
        let document = try snapshot.workflow()
        guard document.drafts.isEmpty else { throw FlowDocumentError.pendingDrafts }
        return Adoption(before: state, after: .init(document: document, flowDocument: candidate),
                        definition: definition, expectedHead: expectedHead)
    }
    @discardableResult public func commitAdoption(_ proposal: Adoption) throws -> ProjectRecord {
        guard state == proposal.before else { throw FlowDocumentError.revisionMismatch }
        // Re-read both saved heads and the exact closure. UI review never licenses a different revision.
        let checked = try prepareAdoption(definition: proposal.definition, expectedHead: proposal.expectedHead)
        guard checked == proposal, let envelope = checked.after.flowDocument else { throw FlowDocumentError.revisionMismatch }
        return try save(checked.after.document, restoring: envelope)
    }
    private func adoptionContext(definition: CompositeDefinitionReference) throws -> (root: FlowDocument, index: Int, child: FlowDocument, url: URL) {
        guard let url = fileURL, let root = state.flowDocument else { throw FlowDocumentError.invalidDocument }
        guard root.draftDependencies.isEmpty, state.document.drafts.isEmpty else { throw FlowDocumentError.pendingDrafts }
        guard try library.fileBackedProjectURL(id: projectID) == url,
              try FlowDocumentCodec.reference(FlowDocumentStore.readHead(at: url)) == FlowDocumentCodec.reference(root) else { throw FlowDocumentError.revisionMismatch }
        guard let index = root.dependencies.firstIndex(where: { $0.definition == definition }) else { throw FlowDocumentError.missingDependency }
        let dependency = root.dependencies[index]
        let childURL = url.deletingLastPathComponent().appendingPathComponent(dependency.relativePath).standardizedFileURL
        guard childURL.resolvingSymlinksInPath() == childURL else { throw FlowDocumentError.unsafeReference }
        let child = try FlowDocumentStore.readHead(at: childURL)
        guard child.id == dependency.revision.documentID, case .modelBlock(let block) = child.payload,
              block.reference == definition else { throw FlowDocumentError.revisionMismatch }
        return (root, index, child, url)
    }

}
