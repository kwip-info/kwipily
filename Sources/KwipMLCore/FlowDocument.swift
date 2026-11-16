import Foundation
import CryptoKit

public enum FlowDocumentError: String, Error, Equatable, LocalizedError {
    case pendingDrafts, invalidDocument, unsafeReference, missingDependency, revisionMismatch, cycle, resourceLimit, conflictingDefinition
    public var errorDescription: String? {
        switch self {
        case .pendingDrafts: return "This workflow has an unapplied child draft. Preserve or finish that draft before converting its files."
        case .invalidDocument: return "The flow document has an unsupported format or inconsistent identity."
        case .unsafeReference: return "Subflow locations must be normalized relative file paths."
        case .missingDependency: return "A pinned subflow revision is missing. Locate its file or restore the saved revision."
        case .revisionMismatch: return "The subflow differs from the pinned revision. Restore that revision or explicitly adopt the change."
        case .cycle: return "Subflow dependencies contain a document cycle."
        case .resourceLimit: return "The flow exceeds its document, dependency, depth or byte limit."
        case .conflictingDefinition: return "Different subflows claim the same block definition revision."
        }
    }
}

public struct FlowDocumentRevision: Codable, Equatable, Hashable {
    public var documentID: UUID
    public var revision: UInt64
    public var sha256: String
    public init(documentID: UUID, revision: UInt64, sha256: String) {
        self.documentID = documentID; self.revision = revision; self.sha256 = sha256
    }
}
public struct FlowDocumentDependency: Codable, Equatable {
    public var definition: CompositeDefinitionReference
    public var revision: FlowDocumentRevision
    public var relativePath: String
    public init(definition: CompositeDefinitionReference, revision: FlowDocumentRevision, relativePath: String) {
        self.definition = definition; self.revision = revision; self.relativePath = relativePath
    }
}
public struct FlowDocumentFileReference: Codable, Equatable {
    public var revision: FlowDocumentRevision
    public var relativePath: String
    public init(revision: FlowDocumentRevision, relativePath: String) {
        self.revision = revision; self.relativePath = relativePath
    }
}
public enum FlowDocumentPayload: Equatable, Codable {
    case workflow(WorkflowDocument)
    case modelBlock(CompositeDefinition)
    case modelBlockDraft(ModelBlockDraft)
    private enum Keys: String, CodingKey { case kind, content }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: Keys.self)
        switch try c.decode(String.self, forKey: .kind) {
        case "workflow": self = .workflow(try c.decode(WorkflowDocument.self, forKey: .content))
        case "modelBlock": self = .modelBlock(try c.decode(CompositeDefinition.self, forKey: .content))
        case "modelBlockDraft": self = .modelBlockDraft(try c.decode(ModelBlockDraft.self, forKey: .content))
        default: throw FlowDocumentError.invalidDocument
        }
    }
    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: Keys.self)
        switch self {
        case .workflow(let value): try c.encode("workflow", forKey: .kind); try c.encode(value, forKey: .content)
        case .modelBlock(let value): try c.encode("modelBlock", forKey: .kind); try c.encode(value, forKey: .content)
        case .modelBlockDraft(let value): try c.encode("modelBlockDraft", forKey: .kind); try c.encode(value, forKey: .content)
        }
    }
}
/// A file's authored canvas. Dependencies are references, never mutable copies owned by its parent.
public struct FlowDocument: Codable, Equatable {
    public var schemaVersion = 1
    public var kind = "flowDocument"
    public var id: UUID
    public var revision: UInt64
    public var payload: FlowDocumentPayload
    public var dependencies: [FlowDocumentDependency]
    public var draftDependencies: [FlowDocumentFileReference]
    public var fileReferences: [FlowDocumentFileReference] {
        dependencies.map { .init(revision: $0.revision, relativePath: $0.relativePath) } + draftDependencies
    }
    public init(id: UUID = UUID(), revision: UInt64 = 0, payload: FlowDocumentPayload,
                dependencies: [FlowDocumentDependency] = [], draftDependencies: [FlowDocumentFileReference] = []) {
        self.id = id; self.revision = revision; self.payload = payload; self.dependencies = dependencies
        self.draftDependencies = draftDependencies
        if case .modelBlockDraft = payload { schemaVersion = 2 }
        if !draftDependencies.isEmpty { schemaVersion = 2 }
    }
    private enum CodingKeys: String, CodingKey { case schemaVersion, kind, id, revision, payload, dependencies, draftDependencies }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion = try c.decode(Int.self, forKey: .schemaVersion)
        kind = try c.decode(String.self, forKey: .kind)
        id = try c.decode(UUID.self, forKey: .id); revision = try c.decode(UInt64.self, forKey: .revision)
        payload = try c.decode(FlowDocumentPayload.self, forKey: .payload)
        dependencies = try c.decode([FlowDocumentDependency].self, forKey: .dependencies)
        draftDependencies = try c.decodeIfPresent([FlowDocumentFileReference].self, forKey: .draftDependencies) ?? []
    }
    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(schemaVersion, forKey: .schemaVersion); try c.encode(kind, forKey: .kind)
        try c.encode(id, forKey: .id); try c.encode(revision, forKey: .revision)
        try c.encode(payload, forKey: .payload); try c.encode(dependencies, forKey: .dependencies)
        if schemaVersion >= 2 { try c.encode(draftDependencies, forKey: .draftDependencies) }
    }
}
public enum FlowDocumentCodec {
    public static let maximumBytes = 4_000_000
    public static func encode(_ document: FlowDocument) throws -> Data {
        try validate(document)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(document)
        guard data.count <= maximumBytes else { throw FlowDocumentError.resourceLimit }
        return data
    }
    public static func decode(_ data: Data) throws -> FlowDocument {
        guard data.count <= maximumBytes else { throw FlowDocumentError.resourceLimit }
        guard let raw = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              Set(raw.keys) == Set(["schemaVersion", "kind", "id", "revision", "payload", "dependencies"] + ((raw["schemaVersion"] as? Int) == 2 ? ["draftDependencies"] : [])),
              let payload = raw["payload"] as? [String: Any], Set(payload.keys) == ["kind", "content"],
              let dependencies = raw["dependencies"] as? [[String: Any]] else { throw FlowDocumentError.invalidDocument }
        for dependency in dependencies {
            guard Set(dependency.keys) == ["definition", "revision", "relativePath"],
                  let reference = dependency["revision"] as? [String: Any], Set(reference.keys) == ["documentID", "revision", "sha256"],
                  let definition = dependency["definition"] as? [String: Any], Set(definition.keys) == ["id", "version"] else { throw FlowDocumentError.invalidDocument }
        }
        if raw["schemaVersion"] as? Int == 2 {
            guard let drafts = raw["draftDependencies"] as? [[String: Any]] else { throw FlowDocumentError.invalidDocument }
            for draft in drafts {
                guard Set(draft.keys) == ["revision", "relativePath"],
                      let reference = draft["revision"] as? [String: Any], Set(reference.keys) == ["documentID", "revision", "sha256"] else { throw FlowDocumentError.invalidDocument }
            }
        }
        if payload["kind"] as? String == "modelBlockDraft" {
            guard let draft = payload["content"] as? [String: Any],
                  Set(draft.keys) == ["projectID", "instancePath", "baseReference", "baseRevision", "proposed"] else { throw FlowDocumentError.invalidDocument }
        }
        if payload["kind"] as? String == "workflow", let content = payload["content"] {
            _ = try WorkflowCodec.decode(JSONSerialization.data(withJSONObject: content))
        }
        let document = try JSONDecoder().decode(FlowDocument.self, from: data)
        try validate(document)
        return document
    }
    public static func reference(_ document: FlowDocument) throws -> FlowDocumentRevision {
        .init(documentID: document.id, revision: document.revision,
              sha256: SHA256.hash(data: try encode(document)).map { String(format: "%02x", $0) }.joined())
    }
    public static func validate(_ document: FlowDocument) throws {
        guard [1, 2].contains(document.schemaVersion), document.kind == "flowDocument",
              document.schemaVersion == 2 || document.draftDependencies.isEmpty else { throw FlowDocumentError.invalidDocument }
        guard document.dependencies.count <= 64, document.draftDependencies.count <= 1 else { throw FlowDocumentError.resourceLimit }
        guard Set(document.dependencies.map(\.definition)).count == document.dependencies.count else { throw FlowDocumentError.conflictingDefinition }
        for dependency in document.fileReferences {
            let path = dependency.relativePath
            guard !path.isEmpty, path.utf8.count <= 1024, !path.contains(":"), !path.contains("\\"),
                  !path.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) }),
                  path.split(separator: "/", omittingEmptySubsequences: false).allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else { throw FlowDocumentError.unsafeReference }
            guard dependency.revision.sha256.utf8.count == 64,
                  dependency.revision.sha256.allSatisfy({ "0123456789abcdef".contains($0) }) else { throw FlowDocumentError.invalidDocument }
        }
        guard document.dependencies.allSatisfy({ !$0.definition.id.isEmpty && $0.definition.id.count <= 128 }) else { throw FlowDocumentError.invalidDocument }
        switch document.payload {
        case .workflow(let workflow):
            guard UUID(uuidString: workflow.projectID) == document.id, workflow.revision == document.revision,
                  workflow.definitions.isEmpty, workflow.drafts.isEmpty else { throw FlowDocumentError.invalidDocument }
            try WorkflowCodec.validate(workflow)
        case .modelBlock(let definition):
            guard document.draftDependencies.isEmpty else { throw FlowDocumentError.invalidDocument }
            try ModelProjectCodec.validate(.blockModel(.init(projectID: document.id.uuidString,
                source: .init(root: .init(), definitions: [definition]))))
        case .modelBlockDraft(let draft):
            guard document.schemaVersion == 2, document.draftDependencies.isEmpty,
                  UUID(uuidString: draft.projectID) != nil, (1...8).contains(draft.instancePath.count),
                  draft.instancePath.allSatisfy({ !$0.isEmpty && $0.unicodeScalars.count <= 128 }),
                  draft.proposed.reference == draft.baseReference else { throw FlowDocumentError.invalidDocument }
            // Parent/path consistency is checked against the complete restored workflow; the draft's
            // proposed graph remains editable, including incomplete and unresolved connections.
            try ModelProjectCodec.validate(.blockModel(.init(projectID: draft.projectID,
                source: .init(root: .init(), definitions: [draft.proposed]))))
        }
    }
}

/// Portable immutable closure. Revalidation never consults mutable files or the network.
public struct FlowDocumentSnapshot: Codable, Equatable {
    public var root: FlowDocumentRevision
    public var documents: [FlowDocument]
    public func workflow() throws -> WorkflowDocument {
        guard documents.count <= 128 else { throw FlowDocumentError.resourceLimit }
        var lookup: [FlowDocumentRevision: FlowDocument] = [:]
        for document in documents {
            let reference = try FlowDocumentCodec.reference(document)
            guard lookup[reference] == nil else { throw FlowDocumentError.invalidDocument }
            lookup[reference] = document
        }
        guard let rootDocument = lookup[root] else { throw FlowDocumentError.missingDependency }
        let checked = try FlowDocumentResolver.resolve(rootDocument) { reference in
            guard let document = lookup[reference] else { throw FlowDocumentError.missingDependency }
            return document
        }
        guard checked.documents.count == documents.count, case .workflow(var workflow) = rootDocument.payload else { throw FlowDocumentError.invalidDocument }
        let definitions = checked.documents.compactMap { if case .modelBlock(let definition) = $0.payload { return definition }; return nil }
        let direct = rootDocument.dependencies.map(\.definition)
        let byReference = Dictionary(uniqueKeysWithValues: definitions.map { ($0.reference, $0) })
        workflow.definitions = direct.compactMap { byReference[$0] } + definitions.filter { !direct.contains($0.reference) }
        workflow.drafts = try rootDocument.draftDependencies.map { dependency in
            guard let document = lookup[dependency.revision], case .modelBlockDraft(let draft) = document.payload else { throw FlowDocumentError.invalidDocument }
            return draft
        }
        try WorkflowCodec.validate(workflow)
        return workflow
    }
}
public enum FlowDocumentResolver {
    public static func resolve(_ root: FlowDocument, maximumDocuments: Int = 128, maximumDepth: Int = 8,
                               load: (FlowDocumentRevision) throws -> FlowDocument) throws -> FlowDocumentSnapshot {
        guard (1...128).contains(maximumDocuments), (0...8).contains(maximumDepth) else { throw FlowDocumentError.resourceLimit }
        var result: [FlowDocument] = [], visited = Set<FlowDocumentRevision>(), active = Set<UUID>()
        var definitions: [CompositeDefinitionReference: FlowDocumentRevision] = [:]
        var bytes = 0
        func visit(_ document: FlowDocument, depth: Int) throws {
            guard !active.contains(document.id) else { throw FlowDocumentError.cycle }
            guard depth <= maximumDepth else { throw FlowDocumentError.resourceLimit }
            let data = try FlowDocumentCodec.encode(document)
            let reference = try FlowDocumentCodec.reference(document)
            if visited.contains(reference) { return }
            guard result.count < maximumDocuments, data.count <= 32_000_000 - bytes else { throw FlowDocumentError.resourceLimit }
            bytes += data.count; result.append(document); visited.insert(reference); active.insert(document.id)
            defer { active.remove(document.id) }
            if case .modelBlock(let definition) = document.payload {
                if let existing = definitions[definition.reference], existing != reference { throw FlowDocumentError.conflictingDefinition }
                definitions[definition.reference] = reference
            }
            for dependency in document.dependencies {
                guard !active.contains(dependency.revision.documentID) else { throw FlowDocumentError.cycle }
                let child = try load(dependency.revision)
                guard try FlowDocumentCodec.reference(child) == dependency.revision else { throw FlowDocumentError.revisionMismatch }
                try visit(child, depth: depth + 1)
                guard case .modelBlock(let definition) = child.payload, definition.reference == dependency.definition else { throw FlowDocumentError.revisionMismatch }
            }
            for dependency in document.draftDependencies {
                guard !active.contains(dependency.revision.documentID) else { throw FlowDocumentError.cycle }
                let child = try load(dependency.revision)
                guard try FlowDocumentCodec.reference(child) == dependency.revision,
                      case .modelBlockDraft = child.payload else { throw FlowDocumentError.revisionMismatch }
                try visit(child, depth: depth + 1)
            }
            let required: Set<CompositeDefinitionReference>
            switch document.payload {
            case .workflow(let workflow):
                required = Set(workflow.nodes.compactMap { if case .modelBlock(let reference, _) = $0.payload { return reference }; return nil })
            case .modelBlock(let definition): required = Set(definition.body.instances.map(\.definition))
            case .modelBlockDraft: required = [] // A proposed graph is not an executable definition.
            }
            guard required.isSubset(of: Set(document.dependencies.map(\.definition))) else { throw FlowDocumentError.missingDependency }
        }
        try visit(root, depth: 0)
        return .init(root: try FlowDocumentCodec.reference(root), documents: result)
    }
}
