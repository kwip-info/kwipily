import Foundation

public struct ModelScopeProjection {
    public let graph: GraphDocument
    public let registry: ComponentRegistry
}

public struct ModelScopeEditingResult {
    public let scope: CompositeGraph
    public let command: GraphCommandResult
}

/// Shared structural authoring projection for root and child scopes. Declared Block ports are
/// available to commands; unresolved boundary widths are not a claim of executable compatibility.
public enum ModelScopeEditing {
    public static func project(_ scope: CompositeGraph, definitions: [CompositeDefinition], name: String = "Block",
                               execution: ExecutionSettings? = nil, registry: ComponentRegistry) throws -> ModelScopeProjection {
        try context(scope, definitions: definitions, name: name, execution: execution, registry: registry).projection
    }

    public static func apply(_ command: GraphCommand, to scope: CompositeGraph, definitions: [CompositeDefinition],
                             name: String = "Block", execution: ExecutionSettings? = nil,
                             selection: GraphSelection = .init(), registry: ComponentRegistry,
                             makeID: () -> String = { UUID().uuidString }) throws -> ModelScopeEditingResult {
        let context = try context(scope, definitions: definitions, name: name, execution: execution, registry: registry)
        let result = try GraphCommandEngine.apply(command, to: .init(document: context.projection.graph, selection: selection),
                                                 registry: context.projection.registry, makeID: makeID)
        guard result.changesDocument else { return .init(scope: scope, command: result) }
        let originals = Dictionary(uniqueKeysWithValues: scope.nodes.map { ($0.id, $0) })
        let instances = Dictionary(uniqueKeysWithValues: scope.instances.map { ($0.id, $0) })
        let copyOrigins = Dictionary(uniqueKeysWithValues: result.copies.nodes.map { ($0.value, $0.key) })
        var next = scope
        let changesPrimitiveConfiguration:Bool
        switch command {
        case .setActivation,.setPooling,.setConvolution,.setNormalization,.setLinear,.setRecurrent,.setBilinear,.setEmbedding,.setEmbeddingBag,.setAttention,.setReshape,.setTensorOperation:changesPrimitiveConfiguration=true;next.configurationVersion=result.after.document.schemaVersion
        default:changesPrimitiveConfiguration=false
        }
        next.nodes = []; next.instances = []; next.edges = result.after.document.edges
        for node in result.after.document.nodes {
            let originalID = copyOrigins[node.id] ?? node.id
            if let definition = context.definitionsByType[node.type] {
                guard var instance = instances[originalID], instance.definition == definition.reference else {
                    throw error("Insert Blocks explicitly; a projected component cannot create an authoring instance.")
                }
                instance.id = node.id; instance.name = node.name; instance.x = node.x; instance.y = node.y
                next.instances.append(instance)
            } else {
                var stored = node
                // Preserve authored values, including omitted legacy bias and defaults, for copies too.
                if !changesPrimitiveConfiguration,let original = originals[originalID] { stored.config = original.config }
                next.nodes.append(stored)
            }
        }
        try validate(next, definitions: definitions, name: name, execution: execution)
        return .init(scope: next, command: result)
    }

    private struct Context {
        let projection: ModelScopeProjection
        let definitionsByType: [String: CompositeDefinition]
    }

    private static func context(_ scope: CompositeGraph, definitions: [CompositeDefinition], name: String,
                                execution: ExecutionSettings?, registry: ComponentRegistry) throws -> Context {
        try validate(scope, definitions: definitions, name: name, execution: execution)
        var definitionsByType: [String: CompositeDefinition] = [:]
        var typesByReference: [CompositeDefinitionReference: String] = [:]
        var components = registry.components
        var reservedTypes = Set(components.map(\.id) + scope.nodes.map(\.type))
        for (index, definition) in definitions.enumerated() {
            var type = "kwip.authoring.block.\(index)"
            while reservedTypes.contains(type) { type += "_" }
            reservedTypes.insert(type)
            definitionsByType[type] = definition
            typesByReference[definition.reference] = type
            components.append(.init(id: type, title: definition.title, category: "Blocks",
                                    inputs: definition.inputs.map(\.id), outputs: definition.outputs.map(\.id),
                                    fields: definition.parameters.map(\.field)))
        }
        let instanceNodes = try scope.instances.map { instance -> GraphNode in
            guard let type = typesByReference[instance.definition], let definition = definitionsByType[type] else {
                throw error("Resolve the missing Block definition before editing this scope.")
            }
            var config = Dictionary(uniqueKeysWithValues: definition.parameters.map { ($0.field.key, $0.field.defaultValue) })
            config.merge(instance.overrides) { _, value in value }
            return GraphNode(id: instance.id, type: type, name: instance.name, x: instance.x, y: instance.y, config: config)
        }
        let graph = GraphDocument(schemaVersion: scope.configurationVersion ?? 1, name: name,
                                  nodes: scope.nodes + instanceNodes, edges: scope.edges, execution: execution)
        return .init(projection: .init(graph: graph, registry: .init(components: components)), definitionsByType: definitionsByType)
    }

    private static func validate(_ scope: CompositeGraph, definitions: [CompositeDefinition], name: String,
                                 execution: ExecutionSettings?) throws {
        // Reuse the document codec's bounds, identity and finite-value validation for an ephemeral
        // scope envelope. This identity is never returned, persisted or used for computational names.
        let composed = definitions.contains { $0.composition != nil }
        let source = CompositeProjectDocument(schemaVersion: composed ? 3 : 2, name: name, root: scope, definitions: definitions, execution: execution)
        let envelope = ModelAuthoringDocument(projectID: "00000000-0000-0000-0000-000000000000", source: source, schemaVersion: composed ? 2 : 1)
        _ = try ModelProjectCodec.encode(.blockModel(envelope))
    }

    private static func error(_ text: String) -> CompositeContractError { .init(.invalidBoundary, text) }
}
