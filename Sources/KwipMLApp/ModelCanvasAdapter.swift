import SwiftUI
import Combine
import KwipMLCore

/// Window-local selection and presentation over the parent's sole persisted model source.
@MainActor final class ModelCanvasAdapter: ObservableObject, CanvasWorkspace, BlockBoundaryEditing {
    let owner: any ModelCanvasAuthority
    let path: [String]
    let primitiveRegistry: ComponentRegistry
    private var observation: AnyCancellable?
    private var documentObservation: AnyCancellable?
    private var boundaryIdentity: Data?
    @Published private var active = true
    @Published private var selection = GraphSelection() {didSet {if oldValue != selection {selectionGeneration &+= 1}}}
    @Published private(set) var selectionGeneration:UInt64=0
    @Published var boundaryRevision = UUID()
    private var restoringCanvasView = false
    @Published var canvasViewState: CanvasViewState { didSet { if active && !restoringCanvasView { onCanvasViewChange?(canvasViewState) } } }
    @Published var canvasViewRevision = UUID()
    @Published var pendingSourceID: String?
    @Published var statusMessage = "Model source saved with its parent project."
    var capturedOutput: (([String], String) -> WorkflowCapturedDataOutput?)?

    var onOpenBlock: (([String]) -> Void)?
    var onCanvasViewChange: ((CanvasViewState) -> Void)?

    init(owner: any ModelCanvasAuthority, path: [String] = [], presentation: CanvasPresentationSettings = .init()) throws {
        guard !path.isEmpty || owner.allowsRootCanvas else {
            throw CompositeContractError(.invalidBoundary, "This authority edits a child Block only. Use the workflow canvas for its parent source.")
        }
        self.owner = owner; self.path = path
        if owner.isFileBacked { statusMessage = "" }
        primitiveRegistry = try owner.modelCanvasRegistry()
        canvasViewState = CanvasViewState(presentation: presentation)
        boundaryIdentity = Self.boundaryIdentity(in: owner.canvasDocument, path: path)
        observation = owner.canvasChanges.sink { [weak self] _ in self?.objectWillChange.send() }
        // Published values arrive before owner.canvasDocument changes; inspect the delivered value.
        documentObservation = owner.canvasDocuments.sink { [weak self] document in
            guard let self else { return }
            let identity = Self.boundaryIdentity(in: document, path: self.path)
            guard identity != self.boundaryIdentity else { return }
            self.boundaryIdentity = identity
            self.boundaryRevision = UUID()
        }
    }
    func invalidate() { active = false; capturedOutput = nil; onOpenBlock = nil; observation = nil; documentObservation = nil }
    func capturedCanvasViewState() -> CanvasViewState {
        var value = canvasViewState
        value.selectedNodeIDs = selectedNodeIDs
        value.selectedEdgeIDs = selection.edgeIDs.intersection(Set(graph.edges.map(\.id)))
        return value
    }
    func restoreCanvasViewState(_ value: CanvasViewState) {
        guard active else { return }
        restoringCanvasView = true
        defer { restoringCanvasView = false }
        selection = .init(nodeIDs: value.selectedNodeIDs.intersection(Set(graph.nodes.map(\.id))),
                          edgeIDs: value.selectedEdgeIDs.intersection(Set(graph.edges.map(\.id))))
        var filtered = value
        filtered.selectedNodeIDs = selection.nodeIDs; filtered.selectedEdgeIDs = selection.edgeIDs
        canvasViewState = filtered
        pendingSourceID = nil
        canvasViewRevision = UUID()
    }
    var contextKey: String { owner.projectID + ":" + owner.sessionID.uuidString + ":" + ((try? JSONEncoder().encode(path).base64EncodedString()) ?? "") }
    var boundaryContextIsActive: Bool { active && editingDraft && !isBusy }
    var breadcrumbTitles: [String] {
        let rootTitle: String
        switch owner.canvasDocument { case .legacyFlat(let graph): rootTitle = graph.name; case .blockModel(let model): rootTitle = model.source.name }
        var result = [rootTitle], body = model?.source.root
        for id in path {
            guard let instance = body?.instances.first(where: { $0.id == id }),
                  let definition = definitions.first(where: { $0.reference == instance.definition }) else { result.append("Unavailable Block"); break }
            result.append(instance.name.isEmpty ? definition.title : instance.name); body = definition.body
        }
        return result
    }
    func updateBoundary(context: UUID, _ mutation: (inout CompositeDefinition) -> Void) {
        guard boundaryContextIsActive, context == boundaryRevision, var proposed = activeDefinition else { return }
        let token = owner.lease(path: path)
        mutation(&proposed)
        report(owner.updateDraft(proposed, lease: token))
    }
    @discardableResult func setBoundaryValue(key: String, value: JSONValue, context: UUID) -> Bool {
        guard boundaryContextIsActive, context == boundaryRevision, var proposed = activeDefinition,
              let index = proposed.parameters.firstIndex(where: { $0.field.key == key }),
              let accepted = RegistryFieldCompatibility.normalized(value, for: proposed.parameters[index].field) else { return false }
        proposed.parameters[index].field.defaultValue = accepted
        let result = owner.updateDraft(proposed, lease: owner.lease(path: path)); report(result); return result
    }
    @discardableResult func releaseIntegerProduct(at index: Int, context: UUID) -> Bool {
        guard boundaryContextIsActive, context == boundaryRevision, let definition = activeDefinition else { return false }
        do {
            let proposed = try CompositeBindingEditing.releasingIntegerProduct(at: index, from: definition, definitions: definitions, registry: primitiveRegistry)
            let accepted = owner.updateDraft(proposed, lease: owner.lease(path: path)); report(accepted); return accepted
        } catch { statusMessage = error.localizedDescription; return false }
    }
    var historyContextID: String { "model-scope:" + contextKey }
    var historyContextIsActive: Bool { active && !isBusy }
    var canUndo: Bool { historyContextIsActive && owner.scopeCanUndo(path: path) }
    var canRedo: Bool { historyContextIsActive && owner.scopeCanRedo(path: path) }
    func undo() { guard canUndo else { return }; report(owner.scopeUndo(lease: owner.lease(path: path))); clearSelection() }
    func redo() { guard canRedo else { return }; report(owner.scopeRedo(lease: owner.lease(path: path))); clearSelection() }
    // Parent model operations remain explicitly owned by the legacy model store;
    // a workflow child authority can never mutate a projected parent graph.
    private var rootOwner: ModelAuthoringStore? { owner as? ModelAuthoringStore }
    private var model: ModelAuthoringDocument? {
        if case .blockModel(let model) = owner.canvasDocument { return model }; return nil
    }
    var definitions: [CompositeDefinition] { model?.source.definitions ?? [] }
    func canvasIconType(nodeID: String) -> String {
        if let instance = scope?.instances.first(where: { $0.id == nodeID }),
           let definition = definitions.first(where: { $0.reference == instance.definition }) {
            return ModuleCompositionPresentation.iconType(definition.composition?.kind)
        }
        return graph.nodes.first { $0.id == nodeID }?.type ?? "unknown"
    }
    var activeDefinition: CompositeDefinition? { Self.definition(in: owner.canvasDocument, path: path) }
    private static func definition(in document: ModelProjectDocument, path: [String]) -> CompositeDefinition? {
        guard !path.isEmpty, case .blockModel(let model) = document else { return nil }
        if let draft = model.drafts.first, draft.instancePath == path { return draft.proposed }
        var body = model.source.root
        var found: CompositeDefinition?
        for id in path {
            guard let instance = body.instances.first(where: { $0.id == id }),
                  let definition = model.source.definitions.first(where: { $0.reference == instance.definition }) else { return nil }
            found = definition; body = definition.body
        }
        return found
    }
    private struct BoundaryIdentity: Encodable {
        let projectID: String
        let path: [String]
        let draftBase: CompositeDefinitionReference?
        let draftRevision: UInt64?
        let definition: CompositeDefinition
    }
    private static func boundaryIdentity(in document: ModelProjectDocument, path: [String]) -> Data? {
        guard case .blockModel(let model) = document, var definition = definition(in: document, path: path) else { return nil }
        // Keep editable values out of view identity so typing never recreates its field.
        definition.title = ""; definition.body.edges = []
        for index in definition.body.nodes.indices {
            definition.body.nodes[index].name = ""; definition.body.nodes[index].x = 0; definition.body.nodes[index].y = 0
            definition.body.nodes[index].config = [:]
        }
        for index in definition.body.instances.indices {
            definition.body.instances[index].name = ""; definition.body.instances[index].x = 0; definition.body.instances[index].y = 0
            definition.body.instances[index].overrides = [:]
        }
        for index in definition.parameters.indices { definition.parameters[index].field.defaultValue = .integer(0) }
        let draft = model.drafts.first(where: { $0.instancePath == path })
        let value = BoundaryIdentity(projectID: model.projectID, path: path, draftBase: draft?.baseReference,
                                     draftRevision: draft?.baseRevision, definition: definition)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        return try? encoder.encode(value)
    }
    var scope: CompositeGraph? {
        if !path.isEmpty { return activeDefinition?.body }
        switch owner.canvasDocument {
        case .legacyFlat(let graph): return CompositeGraph(nodes: graph.nodes, edges: graph.edges, configurationVersion: graph.schemaVersion)
        case .blockModel(let model): return model.source.root
        }
    }
    var title: String {
        if !path.isEmpty { return activeDefinition?.title ?? "Unavailable Block" }
        switch owner.canvasDocument { case .legacyFlat(let graph): return graph.name; case .blockModel(let model): return model.source.name }
    }
    var editingDraft: Bool { !path.isEmpty && model?.drafts.first?.instancePath == path }
    var isBusy: Bool { !active || owner.isBusy || scope == nil || (path.isEmpty ? !(model?.drafts.isEmpty ?? true) : !editingDraft) }
    private func project() throws -> ModelScopeProjection {
        guard let scope else { throw CompositeContractError(.missingDefinition, "This Block path no longer exists in its parent project.") }
        let visible = try activeDefinition.map { try CompositeConfigurationResolver.resolve($0, definitions: definitions, registry: primitiveRegistry) } ?? scope
        return try ModelScopeEditing.project(visible, definitions: definitions, name: title, registry: primitiveRegistry)
    }
    var projectionError: String? { do { _ = try project(); return nil } catch { return error.localizedDescription } }
    var graph: GraphDocument {
        if let projected = try? project().graph { return projected }
        guard let scope else { return GraphDocument(name: title) }
        return (try? ModelScopeEditing.project(scope, definitions: definitions, name: title, registry: primitiveRegistry).graph) ?? GraphDocument(name: title)
    }
    var registry: ComponentRegistry {
        if let projected = try? project().registry { return projected }
        guard let scope else { return primitiveRegistry }
        return (try? ModelScopeEditing.project(scope, definitions: definitions, name: title, registry: primitiveRegistry).registry) ?? primitiveRegistry
    }
    var validation: ValidationResult { ValidationResult() }
    var previewWeightLabel: String { "Authored scope · execution validates the complete model" }
    func capturedSnapshot(for id:String) -> WorkflowCapturedDataOutput? {
        guard active,graph.nodes.contains(where:{$0.id == id}),let capture=capturedOutput?(path,id),
              capture.projectID == owner.projectID,capture.sessionID == owner.sessionID,
              capture.address == .init(domain:.model,localID:id) else { return nil }
        return capture
    }
    func previewStatus(for id: String) -> String {
        guard let capture=capturedSnapshot(for:id) else {return "Not evaluated"}
        if editingDraft || !capture.matchesCurrentConstruction || !capture.matchesCurrentBindings {return "Stale"}
        return capture.phase == .completed ? "Captured" : "Partial capture"
    }
    func canvasEvidenceLabel(nodeID:String) -> String {
        switch previewStatus(for:nodeID) {
        case "Captured":return "Captured output · current sources not revalidated."
        case "Stale":return "Captured from the applied model; current draft or bindings differ."
        case "Partial capture":return "Partial captured output · inspect run status in Results."
        default:return "No captured output for this component."
        }
    }
    func canvasPortDescription(nodeID: String, port: String) -> String { "Declared tensor port · enclosing shape is checked before execution" }
    var selectedNodeIDs: Set<String> { selection.nodeIDs.intersection(Set(graph.nodes.map(\.id))) }
    var selectedNodeID: String? { selectedNodeIDs.sorted().first }
    var capturedSelectionEdges: Set<String> { selection.edgeIDs.intersection(Set(graph.edges.map(\.id))) }
    var selectedEdgeID: String? { selection.edgeIDs.intersection(Set(graph.edges.map(\.id))).sorted().first }
    func selectNode(_ id: String, extending: Bool) {
        guard active, graph.nodes.contains(where: { $0.id == id }) else { return }
        var ids = extending ? selectedNodeIDs : []
        if extending && ids.contains(id) { ids.remove(id) } else { ids.insert(id) }
        selection = GraphSelection(nodeIDs: ids)
    }
    func selectEdge(_ id: String) { guard active, graph.edges.contains(where: { $0.id == id }) else { return }; selection = GraphSelection(edgeIDs: [id]) }
    func clearSelection() { selection = .init() }
    func canOpenBlock(_ id: String) -> Bool { active && !owner.isBusy && scope?.instances.contains(where: { $0.id == id }) == true }
    func openBlock(_ id: String) { guard canOpenBlock(id) else { return }; onOpenBlock?(path + [id]) }
    func canvasSummary(_ node: GraphNode) -> String {
        if let instance = scope?.instances.first(where: { $0.id == node.id }) {
            let v = instance.definition.version
            return "Block · \(v.major).\(v.minor).\(v.patch)"
        }
        if node.type == "linear" { return "\(CanvasDisplayValue.dimension(node.config["in_features"]?.doubleValue)) → \(CanvasDisplayValue.dimension(node.config["out_features"]?.doubleValue)) features" }
        return primitiveRegistry.component(type: node.type)?.title ?? node.type
    }
    private func report(_ accepted: Bool) { statusMessage = accepted ? (owner.isFileBacked ? "" : owner.editStatus) : owner.error ?? "Change was not applied." }
    private func apply(_ command: GraphCommand) {
        guard !isBusy else { return }
        let token = owner.lease(path: path)
        let result: GraphCommandResult?
        if path.isEmpty {
            guard owner.allowsRootCanvas else { statusMessage = "Use the workflow canvas for parent edits."; return }
            result = owner.applyRootCommand(command, selection: selection, lease: token)
        } else { result = owner.applyDraftCommand(command, selection: selection, lease: token) }
        if let result { selection = result.after.selection; report(true) }
        else { statusMessage = owner.error ?? "Change was not applied." }
    }
    private func editScope(lease token: ModelEditorLease? = nil, _ mutation: (inout CompositeGraph) -> Void) -> Bool {
        guard !isBusy else { return false }
        let token = token ?? owner.lease(path: path)
        let accepted: Bool
        if path.isEmpty {
            guard owner.allowsRootCanvas else { statusMessage = "Use the workflow canvas for parent edits."; return false }
            accepted = owner.editRootScope(mutation, lease: token)
        } else { accepted = owner.editDraftScope(mutation, lease: token) }
        report(accepted); return accepted
    }
    func moveNodes(ids: Set<String>, dx: Double, dy: Double) { apply(.move(nodeIDs: ids, offset: .init(x: dx, y: dy))) }
    func duplicateSelection() { apply(.duplicate(nodeIDs: selectedNodeIDs, offset: .init(), semantics: .independent)) }
    func deleteSelection() { apply(.delete(.init(nodeIDs: selectedNodeIDs, edgeIDs: selectedEdgeID.map { [$0] } ?? []))) }
    func connect(source: String, sourcePort: String, target: String, targetPort: String) {
        apply(.connect(source: .init(source, sourcePort), target: .init(target, targetPort))); pendingSourceID = nil
    }
    func reconnect(edgeID: String, source: String, sourcePort: String, target: String, targetPort: String) {
        apply(.reconnect(edgeID: edgeID, source: .init(source, sourcePort), target: .init(target, targetPort)))
    }
    func canConnect(source: String, sourcePort: String, target: GraphNode, port: String) -> Bool {
        guard !isBusy, let projection = try? project() else { return false }
        return (try? GraphCommandEngine.apply(.connect(source: .init(source, sourcePort), target: .init(target.id, port)), to: .init(document: projection.graph), registry: projection.registry)) != nil
    }
    var canInsertRootBlock:Bool {path.isEmpty && rootOwner != nil}
    func canAddType(_ type: String) -> Bool { (canInsertRootBlock && type == ModelBlockPalette.type) || (primitiveRegistry.component(type: type) != nil && (path.isEmpty || !["input", "output"].contains(type))) }
    func addNode(type: String, x: Double, y: Double) {
        if type == ModelBlockPalette.type {
            guard path.isEmpty, !isBusy, let rootOwner else { statusMessage = "Insert parent Blocks from the owning model or workflow canvas."; return }
            let block = ModelBlockPalette.make(x: x, y: y)
            let accepted = rootOwner.insertBlock(definition: block.definition, instance: block.instance, lease: owner.lease())
            report(accepted); if accepted { selectNode(block.instance.id, extending: false) }; return
        }
        guard let component = primitiveRegistry.component(type: type), path.isEmpty || !["input", "output"].contains(type) else { return }
        let node = GraphNode(type: type, name: component.title, x: x, y: y, config: Dictionary(uniqueKeysWithValues: component.fields.map { ($0.key, $0.defaultValue) }))
        if editScope({ $0.nodes.append(node) }) { selectNode(node.id, extending: false) }
    }
    func insertNode(type: String, edgeID: String, x: Double, y: Double) {
        guard let component = primitiveRegistry.component(type: type), component.inputs.count == 1, component.outputs.count == 1 else { return }
        let node = GraphNode(type: type, name: component.title, x: x, y: y, config: Dictionary(uniqueKeysWithValues: component.fields.map { ($0.key, $0.defaultValue) }))
        apply(.insertOnWire(edgeID: edgeID, node: node, inputPort: component.inputs[0], outputPort: component.outputs[0]))
    }
    func renameSelected(_ value: String, lease token: ModelEditorLease) {
        guard let id = selectedNodeID else { return }
        _ = editScope(lease: token) { scope in
            if let i = scope.nodes.firstIndex(where: { $0.id == id }) { scope.nodes[i].name = String(value.prefix(80)) }
            if let i = scope.instances.firstIndex(where: { $0.id == id }) { scope.instances[i].name = String(value.prefix(80)) }
        }
    }
    func controlledParameter(nodeID: String, field: String) -> CompositeParameter? {
        activeDefinition?.parameters.first { $0.targets.contains(.init(nodeID, field)) }
    }
    func derivedParameter(nodeID: String, field: String) -> CompositeIntegerProductBinding? {
        activeDefinition?.integerProducts?.first { $0.targets.contains(.init(nodeID, field)) }
    }
    @discardableResult func setParameterSettings(_ settings:[String:JSONValue],nodeID:String,expected:[String:JSONValue],lease token:ModelEditorLease)->Bool {
        guard selectedNodeIDs == [nodeID],let node=graph.nodes.first(where:{$0.id==nodeID}),node.parameterSettings ?? [:] == expected,(try? LayerParameterSettings.validate(settings)) != nil else{return false}
        return editScope(lease:token) {scope in
            if let index=scope.nodes.firstIndex(where:{$0.id==nodeID}) {scope.nodes[index].parameterSettings=settings.isEmpty ? nil:settings}
        }
    }
    @discardableResult func setConfigValue(_ key: String, value: JSONValue, nodeID: String, lease token: ModelEditorLease) -> Bool {
        guard selectedNodeID == nodeID, controlledParameter(nodeID: nodeID, field: key) == nil, derivedParameter(nodeID: nodeID, field: key) == nil,
              let node = graph.nodes.first(where: { $0.id == nodeID }),
              let field = registry.component(type: node.type)?.fields.first(where: { $0.key == key }),
              let accepted = RegistryFieldCompatibility.normalized(value, for: field) else { return false }
        return editScope(lease: token) { scope in
            if let i = scope.nodes.firstIndex(where: { $0.id == nodeID }) { scope.nodes[i].config[key] = accepted }
            if let i = scope.instances.firstIndex(where: { $0.id == nodeID }) { scope.instances[i].overrides[key] = accepted }
        }
    }
}
