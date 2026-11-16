import Foundation
import KwipMLCore

extension WorkspaceStore {
    struct CodeGenerationSnapshot {
        let inputName: String
        let bytes: Data
        let signature: Data
        let identity: String
        let model: GraphDocument?
    }
    func codeGenerationSnapshot() throws -> CodeGenerationSnapshot {
        if let host = workflowSession, workflowCodeScope != .model {
            let owner = host.owner
            guard !host.isRetired, activeProjectID == owner.projectID, owner.isCurrent(owner.lease()), !owner.hasPendingInspectorEdits else {
                throw DataFlowContractError("Open the current workflow and resolve pending component edits before generating source.")
            }
            let plan = try WorkflowCodePlan.compile(owner.document, scope: workflowCodeScope,
                modelRegistry: owner.modelRegistry, dataRegistry: owner.dataRegistry, sampling: preparationSampling)
            let bytes = try plan.encoded()
            let identity = try plan.contentIdentity
            return .init(inputName: "input/workflow-code.json", bytes: bytes, signature: Data(identity.utf8),
                identity: identity, model: nil)
        }
        let model = try codeGenerationGraph()
        guard GraphValidator.validate(model, registry: registry, context: workflowSession == nil ? .bundledDataset : .preparedWorkflow).isValid,
              let signature = semanticSignature(model) else {
            throw DataFlowContractError("Repair the graph’s connections and dimensions to generate executable source.")
        }
        return .init(inputName: "input/graph.json", bytes: try JSONEncoder().encode(model), signature: signature,
            identity: try ModelExecutionIdentity.exportContentIdentity(model), model: model)
    }
    /// A derived snapshot only. Never install this graph into the authoring workspace.
    func codeGenerationGraph() throws -> GraphDocument {
        guard let host = workflowSession else { return graph }
        let owner = host.owner
        guard !host.isRetired, activeProjectID == owner.projectID, owner.isCurrent(owner.lease()) else {
            throw DataFlowContractError("Open the current workflow before generating source.")
        }
        guard !owner.hasPendingInspectorEdits else {
            throw DataFlowContractError("Resolve pending component edits before generating source.")
        }
        let definitions = owner.document.nodes.filter { WorkflowModelBoundary.isVisibleDefinition($0,in:owner.document) }.map(\.address)
        let selected = owner.selection.nodeAddresses.intersection(Set(definitions))
        let definition: WorkflowNodeAddress
        if selected.count == 1, let first = selected.first { definition = first }
        else {
            let trains = owner.document.nodes.filter { node in
                guard owner.selection.nodeAddresses.contains(node.address) else { return false }
                if case .train = node.payload { return true }; return false
            }
            let connected = trains.count == 1 ? owner.document.edges.filter { $0.to.address == trains[0].address && $0.to.port == "model" }.map(\.from.address) : []
            if connected.count == 1, definitions.contains(connected[0]) { definition = connected[0] }
            else if definitions.count == 1, let first = definitions.first { definition = first }
            else { throw DataFlowContractError(definitions.isEmpty ? "Add a Model definition and connect its model Output to generate source." : "Select one Model definition or its Train component to generate source.") }
        }
        return try WorkflowCompiler.modelSnapshot(owner.document, definition: definition,
            modelRegistry: owner.modelRegistry, dataRegistry: owner.dataRegistry).graph
    }

    var codeGenerationContextID: String { workflowSession?.canvas.historyContextID ?? historyContextID }
    var codeValidation: ValidationResult {
        do { _ = try codeGenerationSnapshot(); return .init(issues: []) }
        catch { return .init(issues: [.init(id: "code-source", message: error.localizedDescription)]) }
    }
}
