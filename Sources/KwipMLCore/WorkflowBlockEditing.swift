import Foundation

/// Reuses the reviewed Block transactions without replacing the canonical mixed source.
/// The transient model is an operation input only; its primitive projection is never written back.
public enum WorkflowBlockEditing {
    public static func beginDraft(_ document: WorkflowDocument, root: WorkflowNodeAddress,
                                  nestedPath: [String] = []) throws -> WorkflowDocument {
        let model = try transient(document, root: root)
        let edited = try ModelBlockEditing.beginDraft(model, path: [root.localID] + nestedPath)
        return try patchDraft(document, model: edited)
    }

    public static func updateDraft(_ document: WorkflowDocument, proposed: CompositeDefinition,
                                   expectedBaseRevision: UInt64, root: WorkflowNodeAddress,
                                   nestedPath: [String] = []) throws -> WorkflowDocument {
        let model = try transient(document, root: root)
        let edited = try ModelBlockEditing.updateDraft(model, proposed: proposed,
            expectedBaseRevision: expectedBaseRevision, path: [root.localID] + nestedPath)
        return try patchDraft(document, model: edited)
    }

    public static func discardDraft(_ document: WorkflowDocument, root: WorkflowNodeAddress,
                                    nestedPath: [String] = [], expectedBaseRevision: UInt64) throws -> WorkflowDocument {
        let model = try transient(document, root: root)
        let edited = try ModelBlockEditing.discardDraft(model, path: [root.localID] + nestedPath,
                                                       expectedBaseRevision: expectedBaseRevision)
        return try patchDraft(document, model: edited)
    }

    public static func applyDraft(_ document: WorkflowDocument, root: WorkflowNodeAddress,
                                  nestedPath: [String] = [], expectedBaseRevision: UInt64,
                                  registry: ComponentRegistry? = nil,
                                  idProvider: () -> String = { UUID().uuidString }) throws -> WorkflowDocument {
        let model = try transient(document, root: root)
        let edited = try ModelBlockEditing.applyDraft(model, path: [root.localID] + nestedPath,
            expectedBaseRevision: expectedBaseRevision, registry: registry, idProvider: idProvider)
        guard let replacement = edited.source.root.instances.first(where: { $0.id == root.localID }),
              let definition = edited.source.definitions.first(where: { $0.reference == replacement.definition }),
              let nodeIndex = document.nodes.firstIndex(where: { $0.address == root }),
              case .modelBlock(_, let overrides) = document.nodes[nodeIndex].payload else {
            throw failure("The selected workflow Block no longer resolves.")
        }
        // Edge kinds are authored claims, not proof of compatibility. Even unfinished
        // cross-domain or incorrectly typed wires must not lose their named boundary.
        for edge in document.edges {
            if edge.from.address == root && !definition.outputs.contains(where: { $0.id == edge.from.port }) {
                throw failure("An existing workflow wire targets a removed Block output.")
            }
            if edge.to.address == root && !definition.inputs.contains(where: { $0.id == edge.to.port }) {
                throw failure("An existing workflow wire targets a removed Block input.")
            }
        }
        var result = document
        result.nodes[nodeIndex].payload = .modelBlock(reference: replacement.definition, overrides: overrides)
        result.definitions = edited.source.definitions
        if result.definitions.contains(where:{$0.body.nodes.contains(where:{$0.parameterSettings != nil})}) {
            WorkflowModelInput.migrateOwnershipMetadata(&result);result.schemaVersion=max(15,result.schemaVersion)
        }
        result.drafts = edited.drafts
        result.revision = edited.revision
        try WorkflowCodec.validate(result)
        return result
    }

    private static func patchDraft(_ document: WorkflowDocument, model: ModelAuthoringDocument) throws -> WorkflowDocument {
        var result = document
        // Migration and its origin are one draft transaction; pins/definitions/revision stay unchanged.
        result.drafts = model.drafts
        if result.drafts.contains(where:{$0.proposed.body.nodes.contains(where:{$0.parameterSettings != nil})}) {
            WorkflowModelInput.migrateOwnershipMetadata(&result);result.schemaVersion=max(15,result.schemaVersion)
        }
        if model.schemaVersion == 2 && document.schemaVersion < 11 { result.schemaVersion = 11 }
        for index in result.drafts.indices {
            if case .model = result.drafts[index].migrationOrigin {
                if let retained = document.drafts.first?.migrationOrigin {
                    try retained.validate(workflowSchemaVersion: document.schemaVersion)
                    result.drafts[index].migrationOrigin = retained
                } else {
                    result.drafts[index].migrationOrigin = .workflow(schemaVersion: document.schemaVersion)
                }
            }
        }
        if result.drafts.isEmpty, case .workflow(let originalVersion) = document.drafts.first?.migrationOrigin {
            result.schemaVersion = originalVersion
        }
        try WorkflowCodec.validate(result)
        return result
    }

    private static func transient(_ document: WorkflowDocument, root: WorkflowNodeAddress) throws -> ModelAuthoringDocument {
        try WorkflowCodec.validate(document)
        guard root.domain == .model,
              let selected = document.nodes.first(where: { $0.address == root }),
              case .modelBlock = selected.payload else {
            throw failure("Select a model-domain Block instance to edit its definition.")
        }
        let nodes: [GraphNode] = document.nodes.compactMap { node in
            guard case .modelPrimitive(let type, _, let config) = node.payload else { return nil }
            return GraphNode(id: node.address.localID, type: type, name: node.name, x: node.x, y: node.y, config: config, parameterSettings:node.parameterSettings)
        }
        let instances: [CompositeInstance] = document.nodes.compactMap { node in
            guard case .modelBlock(let reference, let overrides) = node.payload else { return nil }
            return CompositeInstance(id: node.address.localID, definition: reference, name: node.name,
                                     x: node.x, y: node.y, overrides: overrides)
        }
        let edges = document.edges.filter {
            $0.kind == .tensor && $0.from.address.domain == .model && $0.to.address.domain == .model
        }.map {
            GraphEdge(id: $0.id, source: $0.from.address.localID, sourcePort: $0.from.port,
                      target: $0.to.address.localID, targetPort: $0.to.port)
        }
        // Mixed per-primitive versions remain exclusively in WorkflowDocument. These
        // operations validate edited definitions, not the parent's executable tensor shapes.
        let scope = CompositeGraph(nodes: nodes, instances: instances, edges: edges)
        let composed = document.schemaVersion >= 11
        let source = CompositeProjectDocument(schemaVersion: composed ? 3 : 2, name: document.name, root: scope, definitions: document.definitions)
        var drafts = document.drafts
        for index in drafts.indices {
            if let origin = drafts[index].migrationOrigin {
                try origin.validate(workflowSchemaVersion: document.schemaVersion)
                drafts[index].migrationOrigin = .model(modelSchemaVersion: 1, sourceSchemaVersion: 2)
            }
        }
        return ModelAuthoringDocument(projectID: document.projectID, revision: document.revision,
                                      source: source, drafts: drafts, schemaVersion: composed ? 2 : 1)
    }

    private static func failure(_ message: String) -> CompositeContractError {
        CompositeContractError(.immutableRevision, message)
    }
}
