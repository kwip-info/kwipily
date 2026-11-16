import Foundation

/// Pure planning: nothing is written or removed until a separate transactional writer commits the plan.
public struct FlowDocumentMigrationPlan: Equatable {
    public let root: FlowDocument
    public let subflows: [String: FlowDocument]
    public func snapshot() throws -> FlowDocumentSnapshot {
        var lookup: [FlowDocumentRevision: FlowDocument] = [:]
        for child in subflows.values { lookup[try FlowDocumentCodec.reference(child)] = child }
        return try FlowDocumentResolver.resolve(root) { reference in
            guard let child = lookup[reference] else { throw FlowDocumentError.missingDependency }
            return child
        }
    }
}
public enum FlowDocumentMigration {
    public static func plan(_ source: WorkflowDocument, newID: () -> UUID = { UUID() }) throws -> FlowDocumentMigrationPlan {
        try WorkflowCodec.validate(source)
        guard let rootID = UUID(uuidString: source.projectID) else { throw FlowDocumentError.invalidDocument }
        var ids: [CompositeDefinitionReference: UUID] = [:], used = Set([rootID])
        let definitions = Dictionary(uniqueKeysWithValues: source.definitions.map { ($0.reference, $0) })
        for definition in source.definitions {
            let id = newID()
            guard used.insert(id).inserted else { throw FlowDocumentError.invalidDocument }
            ids[definition.reference] = id
        }
        var built: [CompositeDefinitionReference: FlowDocument] = [:], active = Set<CompositeDefinitionReference>()
        func build(_ reference: CompositeDefinitionReference, depth: Int) throws -> FlowDocument {
            guard !active.contains(reference) else { throw FlowDocumentError.cycle }
            guard depth <= 8 else { throw FlowDocumentError.resourceLimit }
            if let document = built[reference] { return document }
            guard let definition = definitions[reference], let id = ids[reference] else { throw FlowDocumentError.missingDependency }
            active.insert(reference); defer { active.remove(reference) }
            var seen = Set<CompositeDefinitionReference>(), dependencies: [FlowDocumentDependency] = []
            for instance in definition.body.instances where seen.insert(instance.definition).inserted {
                let child = try build(instance.definition, depth: depth + 1)
                dependencies.append(.init(definition: instance.definition, revision: try FlowDocumentCodec.reference(child),
                                          relativePath: child.id.uuidString + ".kwipflow"))
            }
            let document = FlowDocument(id: id, payload: .modelBlock(definition), dependencies: dependencies)
            _ = try FlowDocumentCodec.encode(document)
            built[reference] = document
            return document
        }
        var children: [String: FlowDocument] = [:], dependencies: [FlowDocumentDependency] = []
        for definition in source.definitions {
            let child = try build(definition.reference, depth: 1)
            let path = "Subflows/" + child.id.uuidString + ".kwipflow"
            children[path] = child
            dependencies.append(.init(definition: definition.reference, revision: try FlowDocumentCodec.reference(child), relativePath: path))
        }
        var draftDependencies: [FlowDocumentFileReference] = []
        for draft in source.drafts {
            let id = newID()
            guard used.insert(id).inserted else { throw FlowDocumentError.invalidDocument }
            var seen = Set<CompositeDefinitionReference>(), links: [FlowDocumentDependency] = []
            for reference in [draft.baseReference] + draft.proposed.body.instances.map(\.definition)
                where seen.insert(reference).inserted {
                // Missing proposed references remain unresolved authoring state. Existing definitions
                // still travel with the draft; the proposed body is never published by migration.
                guard let child = built[reference] else { continue }
                links.append(.init(definition: reference, revision: try FlowDocumentCodec.reference(child),
                                   relativePath: child.id.uuidString + ".kwipflow"))
            }
            let child = FlowDocument(id: id, payload: .modelBlockDraft(draft), dependencies: links)
            let path = "Subflows/" + id.uuidString + ".draft.kwipflow"
            children[path] = child
            draftDependencies.append(.init(revision: try FlowDocumentCodec.reference(child), relativePath: path))
        }
        var rootSource = source; rootSource.definitions = []; rootSource.drafts = []
        let root = FlowDocument(id: rootID, revision: source.revision, payload: .workflow(rootSource),
                                dependencies: dependencies, draftDependencies: draftDependencies)
        let plan = FlowDocumentMigrationPlan(root: root, subflows: children)
        // Resolve and reconstruct before offering any file writes. This catches unbound roots too.
        guard try plan.snapshot().workflow() == source else { throw FlowDocumentError.invalidDocument }
        return plan
    }
}
