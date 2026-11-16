import Foundation
import Combine
import KwipMLCore

struct ModelEditorLease: Equatable {
    let projectID: String
    let sessionID: UUID
    let editRevision: UInt64
    let instancePath: [String]
}

/// One transactional source owner. Child windows borrow scoped leases, never their own project copy.
@MainActor final class ModelAuthoringStore: ObservableObject {
    let projectID: String
    let sessionID = UUID()
    private let library: ProjectLibrary
    @Published private(set) var document: ModelProjectDocument
    @Published private(set) var error: String?
    @Published private(set) var isBusy = false
    private var editRevision: UInt64 = 0
    private var undoStates: [ModelProjectDocument] = []
    private var redoStates: [ModelProjectDocument] = []
    var canUndo: Bool { !isBusy && !hasDraft && !undoStates.isEmpty }
    var canRedo: Bool { !isBusy && !hasDraft && !redoStates.isEmpty }
    private var hasDraft: Bool {
        if case .blockModel(let model) = document { return !model.drafts.isEmpty }; return false
    }
    init(library: ProjectLibrary, projectID: String) throws {
        self.library = library; self.projectID = projectID
        document = try library.loadDocument(id: projectID)
    }
    func lease(path: [String] = []) -> ModelEditorLease {
        .init(projectID: projectID, sessionID: sessionID, editRevision: editRevision, instancePath: path)
    }
    func setExecutionBusy(_ value: Bool) { isBusy = value }
    func executionSnapshot() throws -> ModelExecutionSnapshot {
        try ModelExecutionSnapshotBuilder.build(document, registry: .builtIn())
    }
    @discardableResult func editRootScope(_ mutation: (inout CompositeGraph) -> Void, lease token: ModelEditorLease) -> Bool {
        perform {
            try check(token, root: true)
            guard !hasDraft else { throw failure("Apply or discard the open Block draft before changing the parent model.") }
            switch document {
            case .legacyFlat(var graph):
                let original = CompositeGraph(nodes: graph.nodes, edges: graph.edges, configurationVersion: graph.schemaVersion)
                var scope = original; mutation(&scope)
                try checkScopeMutation(original, scope)
                graph.nodes = scope.nodes; graph.edges = scope.edges
                return try persist(.legacyFlat(graph), history: document)
            case .blockModel(var model):
                let original = model.source.root
                mutation(&model.source.root)
                try checkScopeMutation(original, model.source.root)
                guard model.source.root != original else { error = nil; return true }
                try advance(&model)
                return try persist(.blockModel(model), history: document)
            }
        }
    }
    @discardableResult func editMetadata(_ mutation: (inout String, inout ExecutionSettings?) -> Void, lease token: ModelEditorLease) -> Bool {
        perform {
            try check(token, root: true)
            guard !hasDraft else { throw failure("Apply or discard the open Block draft before changing the parent model.") }
            switch document {
            case .legacyFlat(var graph):
                mutation(&graph.name, &graph.execution)
                return try persist(.legacyFlat(graph), history: document)
            case .blockModel(var model):
                var name = model.source.name, execution = model.source.execution
                mutation(&name, &execution)
                guard name != model.source.name || execution != model.source.execution else { error = nil; return true }
                model.source.name = name; model.source.execution = execution
                try advance(&model)
                return try persist(.blockModel(model), history: document)
            }
        }
    }
    @discardableResult func editDraftScope(_ mutation: (inout CompositeGraph) -> Void, lease token: ModelEditorLease) -> Bool {
        perform {
            try check(token)
            let model = try authoring()
            var proposed = try draft(in: model, token: token).proposed
            let original = proposed.body; mutation(&proposed.body)
            try checkScopeMutation(original, proposed.body)
            let next = try ModelBlockEditing.updateDraft(model, proposed: proposed, expectedBaseRevision: model.revision, path: token.instancePath)
            return try persist(.blockModel(next))
        }
    }
    @discardableResult func applyDraftCommand(_ command: GraphCommand, selection: GraphSelection = .init(), lease token: ModelEditorLease) -> GraphCommandResult? {
        do {
            try check(token)
            let model = try authoring()
            var proposed = try draft(in: model, token: token).proposed
            var boundFields=Set<String>()
            var primitiveOriginal:GraphNode?
            let replacement:(id:String,type:String,values:[String:JSONValue]?)?
            switch command {
            case .setActivation(let id,let function,let values):replacement=(id,function.rawValue,values)
            case .setPooling(let id,let variant,let values):replacement=(id,variant.rawValue,values)
            case .setConvolution(let id,let variant,let values):replacement=(id,variant.rawValue,values)
            case .setBilinear(let id,let values):replacement=(id,"bilinear",values)
            case .setEmbedding(let id,let values):replacement=(id,"embedding",values)
            case .setEmbeddingBag(let id,let values):replacement=(id,"embedding_bag",values)
            case .setRecurrent(let id,let method,let values):replacement=(id,RecurrentAuthoringConfiguration.primitiveID(method,isCell:proposed.body.nodes.first(where:{$0.id==id})?.type.hasSuffix("_cell") ?? false),values)
            case .setTensorOperation(let id,let type,let values):replacement=(id,type,values)
            case .setReshape(let id,let variant,let values):replacement=(id,variant.rawValue,values)
            case .setLinear(let id,let variant,let values):replacement=(id,variant.rawValue,values)
            case .setNormalization(let id,let variant,let values):replacement=(id,variant.rawValue,values)
            default:replacement=nil
            }
            if let replacement {
                guard let original=proposed.body.nodes.first(where:{$0.id==replacement.id}) else {throw failure("Choose an authored component in the current Block draft.")}
                primitiveOriginal=original
                if case .setRecurrent(_,let method,let configuration)=command {
                    let inputs=Set(proposed.inputs.flatMap(\.targets).filter {$0.componentID==original.id}.map(\.portID))
                    let outputs=Set(proposed.outputs.filter {$0.source.componentID==original.id}.map {$0.source.portID})
                    _=try RecurrentAuthoringConfiguration.switching(original,configurationVersion:proposed.body.configurationVersion ?? 1,to:method,configuration:configuration,connectedInputPorts:inputs,connectedOutputPorts:outputs,registry:.builtIn())
                }
                if case .setEmbeddingBag(_,let configuration)=command {
                    let inputs=Set(proposed.inputs.flatMap(\.targets).filter {$0.componentID==original.id}.map(\.portID))
                    _=try EmbeddingBagConfiguration.applying(configuration,to:original,connectedInputPorts:inputs,registry:.builtIn())
                }
                if case .setPooling(_,let variant,let configuration)=command {
                    let exposed=Set(proposed.outputs.filter {$0.source.componentID==original.id}.map { $0.source.portID })
                    _=try PoolingConfiguration.switching(original,configurationVersion:proposed.body.configurationVersion ?? 1,to:variant,configuration:configuration,connectedOutputPorts:exposed,registry:.builtIn())
                }
                boundFields=Set((proposed.parameters.flatMap(\.targets)+(proposed.integerProducts ?? []).flatMap(\.targets))
                    .filter {$0.componentID==replacement.id}.map(\.field))
                guard boundFields.isEmpty || original.type==replacement.type else {throw failure("Remove exposed or derived bindings before replacing this component variant.")}
                if let configuration=replacement.values,!boundFields.isEmpty {
                    let resolved=try CompositeConfigurationResolver.resolve(proposed,definitions:model.source.definitions,registry:.builtIn())
                    guard let displayed=resolved.nodes.first(where:{$0.id==replacement.id}),
                          boundFields.allSatisfy({configuration[$0]==displayed.config[$0]}) else {throw failure("Edit exposed values through the Block boundary.")}
                }
            }
            let result = try ModelScopeEditing.apply(command, to: proposed.body, definitions: model.source.definitions,
                                                    name: proposed.title, selection: selection, registry: .builtIn())
            var changedScope=result.scope
            if let original=primitiveOriginal,let index=changedScope.nodes.firstIndex(where:{$0.id==original.id}) {
                for field in boundFields {changedScope.nodes[index].config[field]=original.config[field]}
            }
            proposed.body = changedScope
            let next = try ModelBlockEditing.updateDraft(model, proposed: proposed, expectedBaseRevision: model.revision, path: token.instancePath)
            _ = try persist(.blockModel(next))
            error = nil
            return result.command
        } catch { self.error = error.localizedDescription; return nil }
    }
    private func draft(in model: ModelAuthoringDocument, token: ModelEditorLease) throws -> ModelBlockDraft {
        guard let draft = model.drafts.first, draft.instancePath == token.instancePath,
              draft.projectID == projectID, draft.baseRevision == model.revision else {
            throw failure("This child editor does not own the current Block draft.")
        }
        return draft
    }
    private func checkScopeMutation(_ original: CompositeGraph, _ changed: CompositeGraph) throws {
        guard original.configurationVersion == changed.configurationVersion,
              original.instances.count == changed.instances.count,
              Set(changed.instances.map(\.id)).count == changed.instances.count,
              original.instances.allSatisfy({ old in changed.instances.contains { $0.id == old.id && $0.definition == old.definition } }) else {
            throw failure("Use explicit Block insertion or graph commands for structural changes; configuration edits must preserve Block pins and schema.")
        }
    }
    @discardableResult func applyRootCommand(_ command: GraphCommand, selection: GraphSelection = .init(), lease token: ModelEditorLease) -> GraphCommandResult? {
        do {
            try check(token, root: true)
            let result = try ModelRootEditing.apply(command, to: document, selection: selection, registry: .builtIn())
            if result.document != document { _ = try persist(result.document, history: document) }
            error = nil
            return result.command
        } catch { self.error = error.localizedDescription; return nil }
    }
    @discardableResult func insertBlock(definition: CompositeDefinition, instance: CompositeInstance, lease token: ModelEditorLease) -> Bool {
        perform {
            try check(token, root: true)
            let next = try ModelRootEditing.insertBlock(document, definition: definition, instance: instance, projectID: projectID)
            return try persist(next, history: document)
        }
    }
    @discardableResult func replaceSource(_ source: CompositeProjectDocument, lease token: ModelEditorLease) -> Bool {
        perform {
            try check(token, root: true)
            guard !hasDraft else { throw failure("Apply or discard the open Block draft before changing the parent model.") }
            var model = try authoring()
            guard model.source != source else { error = nil; return true }
            model.source = source; try advance(&model)
            return try persist(.blockModel(model), history: document)
        }
    }
    @discardableResult func beginDraft(path: [String], lease token: ModelEditorLease) -> Bool {
        perform {
            try check(token, root: true)
            let next = try ModelBlockEditing.beginDraft(authoring(), path: path)
            return try persist(.blockModel(next))
        }
    }
    @discardableResult func updateDraft(_ proposed: CompositeDefinition, lease token: ModelEditorLease) -> Bool {
        perform {
            try check(token)
            let model = try authoring()
            let next = try ModelBlockEditing.updateDraft(model, proposed: proposed, expectedBaseRevision: model.revision, path: token.instancePath)
            return try persist(.blockModel(next))
        }
    }
    @discardableResult func discardDraft(lease token: ModelEditorLease) -> Bool {
        perform {
            try check(token)
            let model = try authoring()
            return try persist(.blockModel(ModelBlockEditing.discardDraft(model, path: token.instancePath, expectedBaseRevision: model.revision)))
        }
    }
    @discardableResult func applyDraft(lease token: ModelEditorLease) -> Bool {
        perform {
            try check(token)
            let draft = try authoring()
            let next = try ModelBlockEditing.applyDraft(draft, path: token.instancePath, expectedBaseRevision: draft.revision)
            let baseline = try ModelBlockEditing.discardDraft(draft, path: token.instancePath, expectedBaseRevision: draft.revision)
            return try persist(.blockModel(next), history: .blockModel(baseline))
        }
    }
    @discardableResult func undo(lease token: ModelEditorLease) -> Bool { restoreHistory(token, undo: true) }
    @discardableResult func redo(lease token: ModelEditorLease) -> Bool { restoreHistory(token, undo: false) }
    private func restoreHistory(_ token: ModelEditorLease, undo: Bool) -> Bool {
        perform {
            try check(token, root: true)
            guard !hasDraft, let previous = (undo ? undoStates : redoStates).last else { throw failure("No available history action; resolve any open Block draft first.") }
            var next = previous
            if case .blockModel(var model) = next {
                model.revision = try authoring().revision
                try advance(&model); next = .blockModel(model)
            }
            let current = document
            _ = try persist(next)
            if undo { undoStates.removeLast(); redoStates.append(current) }
            else { redoStates.removeLast(); undoStates.append(current) }
            return true
        }
    }
    private func authoring() throws -> ModelAuthoringDocument {
        switch document {
        case .legacyFlat(let graph): return try ModelProjectCodec.promote(graph, projectID: projectID)
        case .blockModel(let model): return model
        }
    }
    private func check(_ token: ModelEditorLease, root: Bool = false) throws {
        guard !isBusy else { throw failure("Stop the active execution before editing the model.") }
        guard token.projectID == projectID, token.sessionID == sessionID, token.editRevision == editRevision,
              !root || token.instancePath.isEmpty else { throw failure("This editor refers to an earlier model state. Reopen the Block before applying changes.") }
    }
    private func advance(_ model: inout ModelAuthoringDocument) throws {
        guard model.revision < UInt64.max else { throw failure("The model revision limit has been reached.") }
        model.revision += 1
    }
    private func persist(_ next: ModelProjectDocument, history: ModelProjectDocument? = nil) throws -> Bool {
        guard next != document else { error = nil; return true }
        guard editRevision < UInt64.max else { throw failure("The editor revision limit has been reached.") }
        _ = try library.saveDocument(next, id: projectID)
        if let history {
            undoStates.append(history)
            if undoStates.count > 100 { undoStates.removeFirst(undoStates.count - 100) }
            redoStates.removeAll()
        }
        document = next; editRevision += 1; error = nil
        return true
    }
    private func perform(_ action: () throws -> Bool) -> Bool {
        do { return try action() }
        catch { self.error = error.localizedDescription; return false }
    }
    private func failure(_ text: String) -> CompositeContractError { .init(.invalidBoundary, text) }
}
