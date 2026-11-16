import Foundation
import Combine
import KwipMLCore

/// Borrowed inspector commands; the parent remains the only persisted authority.
@MainActor final class ModuleCompositionInspectorContext: ObservableObject {
    enum Command {
        case select(CompositionPolicy.Kind)
        case move(String, Int)
        case rename(String, String)
        case refresh
    }
    let adapter: ModelCanvasAdapter
    let origin: UUID
    @Published private(set) var rejection: String?
    init(adapter: ModelCanvasAdapter) { self.adapter = adapter; origin = adapter.boundaryRevision }
    @Published private(set) var validationNote = ""
    @discardableResult func perform(_ command: Command) -> Bool {
        do {
            guard adapter.boundaryContextIsActive, origin == adapter.boundaryRevision,
                  let definition = adapter.activeDefinition else {
                throw GraphConfigurationError("This Block editor changed or closed. Use its current Container controls.")
            }
            let baseline = try? snapshot(definition)
            let ports = try childPorts(definition)
            let validate: ModuleCompositionEditing.Validation = { candidate in
                try CompositeDefinitionValidation.validate(candidate, definitions: self.adapter.definitions,
                    registry: self.adapter.primitiveRegistry)
                if baseline != nil { _ = try self.snapshot(candidate) }
            }
            let proposed: CompositeDefinition
            switch command {
            case .select(let kind):
                proposed = try ModuleCompositionEditing.selecting(kind: kind, in: definition, childPorts: ports, validate: validate)
            case .move(let id, let position):
                proposed = try ModuleCompositionEditing.movingChild(localID: id, to: position, in: definition, childPorts: ports, validate: validate)
            case .rename(let id, let key):
                proposed = try ModuleCompositionEditing.renamingKey(localID: id, key: key, in: definition, childPorts: ports, validate: validate)
            case .refresh:
                proposed = try ModuleCompositionEditing.refreshingMembership(in: definition, childPorts: ports, validate: validate)
            }
            guard adapter.owner.updateDraft(proposed, lease: adapter.owner.lease(path: adapter.path)) else {
                throw GraphConfigurationError(adapter.owner.error ?? "The parent could not save this Block draft.")
            }
            rejection = nil
            validationNote = baseline == nil ? "Block connections checked. Complete the parent flow to check its input dimensions before running." : "Block connections and parent dimensions checked. Apply the Block draft to use this change."
            return true
        } catch { rejection = error.localizedDescription; return false }
    }
    private func childPorts(_ definition: CompositeDefinition) throws -> [String: CompositionPolicy.ChildPorts] {
        let resolved = try CompositeConfigurationResolver.resolve(definition, definitions: adapter.definitions, registry: adapter.primitiveRegistry)
        var result: [String: CompositionPolicy.ChildPorts] = [:]
        let dynamic = resolved.nodes.contains { ["split", "chunk"].contains($0.type) }
        var withoutPolicy = definition; withoutPolicy.composition = nil
        let expanded = dynamic ? try snapshot(withoutPolicy) : nil
        let validation = expanded.map { GraphValidator.validate($0.graph, registry: adapter.primitiveRegistry, context: .explicitObjective) }
        for node in resolved.nodes {
            guard let declaration = adapter.primitiveRegistry.component(type: node.type) else { throw GraphConfigurationError("Resolve the missing Block component before configuring its container.") }
            var layout: BatchedTensorLayout?
            if ["split", "chunk"].contains(node.type) {
                guard let expanded, let validation,
                      let id = expanded.origins.first(where: { $0.value.instancePath == adapter.path && $0.value.localNodeID == node.id })?.key,
                      let input = expanded.graph.edges.first(where: { $0.target == id && $0.targetPort == "in" }),
                      let observed = validation.outputTensorContracts[.init(input.source, input.sourcePort)]?.layout else {
                    throw GraphConfigurationError("Connect the parent flow before editing this container: its Split or Chunk output ports depend on the incoming dimensions.")
                }
                layout = observed
            }
            result[node.id] = .init(inputs: declaration.configuredInputs(node.config), outputs: declaration.configuredOutputs(node.config, inputLayout: layout))
        }
        for instance in resolved.instances {
            guard let child = adapter.definitions.first(where: { $0.reference == instance.definition }) else { throw GraphConfigurationError("Resolve the missing nested Block revision before editing this container.") }
            result[instance.id] = .init(inputs: child.inputs.map(\.id), outputs: child.outputs.map(\.id))
        }
        return result
    }
    /// Pure, unpublished previews use the existing authoring/compiler transactions.
    /// No preview gets a file, history entry, runtime or independent editing authority.
    private func snapshot(_ proposed: CompositeDefinition) throws -> ModelExecutionSnapshot {
        if let authority = adapter.owner as? WorkflowBlockCanvasAuthority {
            let owner = authority.owner
            let updated = try WorkflowBlockEditing.updateDraft(owner.document, proposed: proposed,
                expectedBaseRevision: owner.document.revision, root: authority.root, nestedPath: Array(adapter.path.dropFirst()))
            let applied = try WorkflowBlockEditing.applyDraft(updated, root: authority.root,
                nestedPath: Array(adapter.path.dropFirst()), expectedBaseRevision: updated.revision, registry: adapter.primitiveRegistry)
            var failure: Error?
            for node in applied.nodes {
                guard case .train = node.payload else { continue }
                do {
                    let plan = try WorkflowCompiler.compile(applied, train: node.address, modelRegistry: adapter.primitiveRegistry,
                        dataRegistry: owner.dataRegistry, contractVersion: WorkflowArchiveCodec.trainingCompilerContractVersion(forWorkflowSchema: applied.schemaVersion))
                    if plan.model.origins.values.contains(where: { $0.instancePath.starts(with: adapter.path) }) { return plan.model }
                } catch { failure = error }
            }
            throw failure ?? GraphConfigurationError("Complete the connected Train flow to resolve this Block's incoming dimensions.")
        }
        guard case .blockModel(let model) = adapter.owner.canvasDocument else { throw GraphConfigurationError("Open a Block draft in its parent model.") }
        let updated = try ModelBlockEditing.updateDraft(model, proposed: proposed, expectedBaseRevision: model.revision, path: adapter.path)
        let applied = try ModelBlockEditing.applyDraft(updated, path: adapter.path, expectedBaseRevision: updated.revision, registry: adapter.primitiveRegistry)
        return try ModelExecutionSnapshotBuilder.build(.blockModel(applied), registry: adapter.primitiveRegistry, context: .explicitObjective)
    }
}
