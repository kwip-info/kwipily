import Foundation

public struct ModelRootEditingResult {
    public let document: ModelProjectDocument
    public let command: GraphCommandResult
}

/// Pure root authoring transactions. Projections are command inputs, never a second stored graph.
public enum ModelRootEditing {
    public static func apply(_ command: GraphCommand, to document: ModelProjectDocument,
                             selection: GraphSelection = .init(), registry: ComponentRegistry,
                             makeID: () -> String = { UUID().uuidString }) throws -> ModelRootEditingResult {
        try ModelProjectCodec.validate(document)
        switch document {
        case .legacyFlat(let graph):
            let result = try GraphCommandEngine.apply(command, to: .init(document: graph, selection: selection), registry: registry, makeID: makeID)
            let next = ModelProjectDocument.legacyFlat(result.after.document)
            _ = try ModelProjectCodec.encode(next)
            return .init(document: next, command: result)
        case .blockModel(var model):
            try requireNoDraft(model)
            let result = try ModelScopeEditing.apply(command, to: model.source.root,
                                                     definitions: model.source.definitions, name: model.source.name,
                                                     execution: model.source.execution, selection: selection,
                                                     registry: registry, makeID: makeID)
            guard result.command.changesDocument else { return .init(document: document, command: result.command) }
            model.source.root = result.scope
            try advance(&model)
            let updated = ModelProjectDocument.blockModel(model)
            _ = try ModelProjectCodec.encode(updated)
            return .init(document: updated, command: result.command)
        }
    }

    /// First insertion alone promotes a flat source. Existing pins are immutable and may be reused exactly.
    public static func insertBlock(_ document: ModelProjectDocument, definition: CompositeDefinition,
                                   instance: CompositeInstance, projectID: String) throws -> ModelProjectDocument {
        try ModelProjectCodec.validate(document)
        var model: ModelAuthoringDocument
        switch document {
        case .legacyFlat(let graph): model = try ModelProjectCodec.promote(graph, projectID: projectID)
        case .blockModel(let existing): model = existing
        }
        guard model.projectID == projectID else { throw error("Block insertion belongs to a different parent project.") }
        try requireNoDraft(model)
        guard instance.definition == definition.reference else { throw error("The Block instance must use the supplied exact definition pin.") }
        guard !model.source.root.nodes.contains(where: { $0.id == instance.id }),
              !model.source.root.instances.contains(where: { $0.id == instance.id }),
              !model.source.root.edges.contains(where: { $0.id == instance.id }) else {
            throw error("The inserted Block needs a new identity.")
        }
        if let existing = model.source.definitions.first(where: { $0.reference == definition.reference }) {
            guard existing == definition else { throw error("A different Block definition already uses this immutable pin.") }
        } else { model.source.definitions.append(definition) }
        model.source.root.instances.append(instance)
        try advance(&model)
        let result = ModelProjectDocument.blockModel(model)
        _ = try ModelProjectCodec.encode(result)
        return result
    }

    private static func requireNoDraft(_ model: ModelAuthoringDocument) throws {
        guard model.drafts.isEmpty else { throw error("Apply or discard the open Block draft before changing the parent model.") }
    }
    private static func advance(_ model: inout ModelAuthoringDocument) throws {
        guard model.revision < UInt64.max else { throw error("The model revision limit has been reached.") }
        model.revision += 1
    }
    private static func error(_ text: String) -> CompositeContractError { .init(.invalidBoundary, text) }
}
