import SwiftUI
import Combine
import KwipMLCore

struct CompositeEditorState: Codable, Equatable {
    var project: CompositeProjectDocument
    var activeReference: CompositeDefinitionReference?
    var draft: CompositeDefinition?
    var draftBase: CompositeDefinitionReference?
}

@MainActor final class CompositeWorkspaceStore: ObservableObject, CanvasWorkspace {
    let workspace: WorkspaceStore
    let projectID: String?
    let primitiveRegistry: ComponentRegistry
    @Published var boundaryRevision = UUID()
    @Published var state = CompositeEditorState(project: CompositeExamples.parallelClassifier) {
        didSet {
            if state.activeReference != oldValue.activeReference { canvasViewRevision = UUID() }
            if state.activeReference != oldValue.activeReference || state.draftBase != oldValue.draftBase || Self.boundaryIdentity(state.draft) != Self.boundaryIdentity(oldValue.draft) { boundaryRevision = UUID() }
        }
    }
    private static func boundaryIdentity(_ definition: CompositeDefinition?) -> Data? {
        guard var value = definition else { return nil }
        value.title = ""; value.body.edges = []
        for index in value.body.nodes.indices { value.body.nodes[index].name="";value.body.nodes[index].x=0;value.body.nodes[index].y=0;value.body.nodes[index].config=[:] }
        for index in value.body.instances.indices { value.body.instances[index].name="";value.body.instances[index].x=0;value.body.instances[index].y=0;value.body.instances[index].overrides=[:] }
        for index in value.parameters.indices { value.parameters[index].field.defaultValue=0 }
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        return try? encoder.encode(value)
    }
    @Published var selectedNodeIDs = Set<String>()
    @Published var selectedNodeID: String?
    @Published var selectedEdgeID: String?
    @Published var pendingSourceID: String?
    @Published var statusMessage = "Root instances retain exact definition revisions."
    var canvasViewState: CanvasViewState {
        get { projectID == workspace.activeProjectID ? workspace.compositeCanvasStates[contextKey] ?? CanvasViewState(presentation: workspace.appPreferences.canvasDefaults) : CanvasViewState() }
        set { recordCanvasViewState(newValue,contextKey:contextKey,revision:canvasViewRevision) }
    }
    func recordCanvasViewState(_ value:CanvasViewState,contextKey:String,revision:UUID) {
        guard let projectID,projectID == workspace.activeProjectID,contextKey == self.contextKey,revision == canvasViewRevision,!workspace.isRestoringArrangement else { return }
        var proposed=workspace.captureWorkspaceView();proposed.compositeCanvasStates[contextKey]=value
        guard (try? proposed.validate()) != nil else {return}
        if workspace.compositeCanvasStates[contextKey] != value { workspace.compositeCanvasStates[contextKey]=value }
    }
    @Published var canvasViewRevision = UUID()
    @Published var historySessionRevision = UUID()

    var history = EditHistory<CompositeEditorState>()
    private var workspaceObservation: AnyCancellable?
    private var arrangementRevisionObservation: AnyCancellable?
    var project: CompositeProjectDocument { state.project }
    var activeDefinition: CompositeDefinition? {
        guard let reference = state.activeReference else { return nil }
        if state.draftBase == reference, let draft = state.draft { return draft }
        return project.definitions.first { $0.reference == reference }
    }
    var editingDraft: Bool { state.activeReference != nil && state.activeReference == state.draftBase && state.draft != nil }
    var isBusy: Bool { workspace.isBusy || (state.activeReference != nil && !editingDraft) }
    var previewWeightLabel: String { "Composite authoring · no execution preview" }
    var saveURL: URL { workspace.workspaceDirectory.appendingPathComponent("composite/projects/\(projectID ?? "unassigned")/editor.json") }
    var scope: CompositeGraph { activeDefinition?.body ?? project.root }
    var contextKey: String { state.activeReference.map(Self.typeKey) ?? "root" }
    init(workspace: WorkspaceStore) {
        self.workspace = workspace; projectID = workspace.activeProjectID; primitiveRegistry = workspace.registry
        workspaceObservation = workspace.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }
        arrangementRevisionObservation = workspace.$canvasViewRevision.dropFirst().sink { [weak self] _ in
            guard let self,self.projectID == self.workspace.activeProjectID else {return}
            self.canvasViewRevision = UUID()
        }
        if FileManager.default.fileExists(atPath: saveURL.path), projectID != nil { reopen() }
    }
    static func versionText(_ version: ContractVersion) -> String { "\(version.major).\(version.minor).\(version.patch)" }
    static func typeKey(_ reference: CompositeDefinitionReference) -> String { "composite:\(reference.id)@\(versionText(reference.version))" }
    var registry: ComponentRegistry {
        var components = primitiveRegistry.components.filter { state.activeReference == nil || !["input", "output"].contains($0.id) }
        if state.activeReference == nil {
            components += project.definitions.map { ComponentDefinition(id: Self.typeKey($0.reference), title: $0.title + " " + Self.versionText($0.reference.version), category: "Reusable", inputs: $0.inputs.map(\.id), outputs: $0.outputs.map(\.id), fields: $0.parameters.map(\.field)) }
        } else {
            // Existing nested instances remain inspectable; adding recursive blocks is not offered.
            components += scope.instances.compactMap { instance in project.definitions.first(where: { $0.reference == instance.definition }).map { ComponentDefinition(id: Self.typeKey($0.reference), title: $0.title, category: "Reusable", inputs: $0.inputs.map(\.id), outputs: $0.outputs.map(\.id), fields: $0.parameters.map(\.field)) } }
            components = components.reduce(into: []) { values, item in if !values.contains(where: { $0.id == item.id }) { values.append(item) } }
        }
        return ComponentRegistry(components: components)
    }
    func configuredBody() throws -> CompositeGraph {
        if let definition = activeDefinition {
            return try CompositeConfigurationResolver.resolve(definition, definitions: project.definitions, registry: primitiveRegistry)
        }
        return try CompositeMigration.toCurrentBody(scope, registry: primitiveRegistry)
    }
    var projectionError: String? { do { _ = try configuredBody(); return nil } catch { return error.localizedDescription } }
    var graph: GraphDocument {
        let body = (try? configuredBody()) ?? scope
        return GraphDocument(name: activeDefinition?.title ?? project.name, nodes: body.nodes + body.instances.map { instance in
            var config = project.definitions.first(where: { $0.reference == instance.definition })?.parameters.reduce(into: [String: JSONValue]()) { $0[$1.field.key] = $1.field.defaultValue } ?? [:]
            config.merge(instance.overrides) { _, override in override }
            return GraphNode(id: instance.id, type: Self.typeKey(instance.definition), name: instance.name, x: instance.x, y: instance.y, config: config)
        }, edges: body.edges)
    }
    var validation: ValidationResult { ValidationResult() } // Full tensor validity is checked by expansion, not fabricated for an open boundary.
    func previewStatus(for id: String) -> String { "Not evaluated" }
    func canvasSummary(_ node: GraphNode) -> String {
        if let instance = scope.instances.first(where: { $0.id == node.id }) { return "Pinned " + Self.versionText(instance.definition.version) }
        if node.type == "softmax" { return "Softmax · axis \(node.config["dim"]?.integerValue ?? -1)" }
        if node.type == "gelu" { return node.config["approximate_tanh"]?.boolValue == true ? "GELU · tanh approximation" : "GELU · exact" }
        if node.type == "linear" { return "\(CanvasDisplayValue.dimension(node.config["in_features"]?.doubleValue)) → \(CanvasDisplayValue.dimension(node.config["out_features"]?.doubleValue))" + (node.config["bias"]?.boolValue == false ? " · no bias" : "") }
        return editingDraft ? "Definition draft" : "Primitive"
    }
    func canvasPortDescription(nodeID: String, port: String) -> String { "Tensor boundary · validate expanded architecture" }
    func selectNode(_ id: String, extending: Bool = false) {
        if extending { if selectedNodeIDs.contains(id) { selectedNodeIDs.remove(id) } else { selectedNodeIDs.insert(id) } } else { selectedNodeIDs = [id] }
        selectedNodeID = selectedNodeIDs.contains(id) ? id : selectedNodeIDs.sorted().first; selectedEdgeID = nil
    }
    func selectEdge(_ id: String) { clearSelection(); selectedEdgeID = id }
    func clearSelection() { selectedNodeIDs = []; selectedNodeID = nil; selectedEdgeID = nil }
    private func edit(_ action: (inout CompositeEditorState) -> Void) {
        guard !isBusy else { statusMessage = "Create a draft revision before editing this pinned definition."; return }
        let before = state; action(&state); history.record(before: before, after: state)
    }
    private func replaceScope(_ value: CompositeGraph, in state: inout CompositeEditorState) {
        if state.activeReference != nil { state.draft?.body = value } else { state.project.root = value }
    }
    func apply(_ command: GraphCommand) {
        guard !isBusy else { statusMessage = "Pinned definitions are immutable. Create a draft revision."; return }
        do {
            let result = try GraphCommandEngine.apply(command, to: GraphEditorState(document: graph, selection: GraphSelection(nodeIDs: selectedNodeIDs, edgeIDs: selectedEdgeID.map { [$0] } ?? [])), registry: registry)
            let oldInstances = Dictionary(uniqueKeysWithValues: scope.instances.map { ($0.id, $0) })
            var next = CompositeGraph(edges: result.after.document.edges)
            for node in result.after.document.nodes {
                if let definition = project.definitions.first(where: { Self.typeKey($0.reference) == node.type }) {
                    let original = oldInstances[node.id] ?? result.copies.nodes.first(where: { $0.value == node.id }).flatMap { oldInstances[$0.key] }
                    next.instances.append(CompositeInstance(id: node.id, definition: definition.reference, name: node.name, x: node.x, y: node.y, overrides: original?.overrides ?? [:]))
                } else {
                    var stored = node
                    if let original = scope.nodes.first(where: { $0.id == node.id }) { stored.config = original.config }
                    next.nodes.append(stored)
                }
            }
            edit { self.replaceScope(next, in: &$0) }
            selectedNodeIDs = result.after.selection.nodeIDs; selectedNodeID = selectedNodeIDs.sorted().first; selectedEdgeID = nil
        } catch { statusMessage = error.localizedDescription }
    }
    func moveNodes(ids: Set<String>, dx: Double, dy: Double) { apply(.move(nodeIDs: ids, offset: GraphOffset(x: dx, y: dy))) }
    func duplicateSelection() { apply(.duplicate(nodeIDs: selectedNodeIDs, offset: GraphOffset(), semantics: .independent)) }
    func deleteSelection() { apply(.delete(GraphSelection(nodeIDs: selectedNodeIDs, edgeIDs: selectedEdgeID.map { [$0] } ?? []))) }
    func addNode(type: String, x: Double, y: Double) {
        guard let definition = registry.component(type: type), scope.nodes.count + scope.instances.count < 128 else { return }
        var next = scope; let id = UUID().uuidString
        if let composite = project.definitions.first(where: { Self.typeKey($0.reference) == type }) {
            guard state.activeReference == nil else { statusMessage = "Add reusable instances in the root workspace."; return }
            next.instances.append(CompositeInstance(id: id, definition: composite.reference, name: composite.title, x: x, y: y))
        } else { next.nodes.append(GraphNode(id: id, type: type, name: definition.title, x: x, y: y, config: Dictionary(uniqueKeysWithValues: definition.fields.map { ($0.key, $0.defaultValue) }))) }
        edit { self.replaceScope(next, in: &$0) }; selectNode(id)
    }
    func connect(source: String, sourcePort: String, target: String, targetPort: String) { apply(.connect(source: GraphPortAddress(source, sourcePort), target: GraphPortAddress(target, targetPort))); pendingSourceID = nil }
    func reconnect(edgeID: String, source: String, sourcePort: String, target: String, targetPort: String) { apply(.reconnect(edgeID: edgeID, source: GraphPortAddress(source, sourcePort), target: GraphPortAddress(target, targetPort))) }
    func insertNode(type: String, edgeID: String, x: Double, y: Double) {
        guard let definition = registry.component(type: type), definition.inputs.count == 1, definition.outputs.count == 1 else { return }
        apply(.insertOnWire(edgeID: edgeID, node: GraphNode(type: type, name: definition.title, x: x, y: y, config: Dictionary(uniqueKeysWithValues: definition.fields.map { ($0.key, $0.defaultValue) })), inputPort: definition.inputs[0], outputPort: definition.outputs[0]))
    }
    func canConnect(source: String, sourcePort: String, target: GraphNode, port: String) -> Bool {
        (try? GraphCommandEngine.apply(.connect(source: GraphPortAddress(source, sourcePort), target: GraphPortAddress(target.id, port)), to: GraphEditorState(document: graph), registry: registry)) != nil
    }
    func renameSelected(_ name: String) {
        guard let id = selectedNodeID else { return }; var next = scope
        if let index = next.nodes.firstIndex(where: { $0.id == id }) { next.nodes[index].name = String(name.prefix(80)) }
        if let index = next.instances.firstIndex(where: { $0.id == id }) { next.instances[index].name = String(name.prefix(80)) }
        edit { self.replaceScope(next, in: &$0) }
    }
    func controlledParameter(nodeID: String, field: String) -> CompositeParameter? {
        activeDefinition?.parameters.first { $0.targets.contains(CompositeParameterTarget(nodeID, field)) }
    }
    func derivedParameter(nodeID: String, field: String) -> CompositeIntegerProductBinding? {
        activeDefinition?.integerProducts?.first { $0.targets.contains(.init(nodeID, field)) }
    }
    func setNumeric(_ key: String, value: Double) {
        guard let id = selectedNodeID else { return }
        _ = setConfigValue(key, value: .number(value), nodeID: id)
    }
    @discardableResult func setConfigValue(_ key: String, value: JSONValue, nodeID: String) -> Bool {
        guard !isBusy, projectID == workspace.activeProjectID, selectedNodeID == nodeID,
              let node = graph.nodes.first(where: { $0.id == nodeID }), let field = registry.component(type: node.type)?.fields(configuration:node.config).first(where: { $0.key == key }), let accepted = RegistryFieldCompatibility.normalized(value, for: field) else {
            statusMessage = "Value not applied. Use the field's declared type and bounds."; return false
        }
        guard controlledParameter(nodeID: nodeID, field: key) == nil, derivedParameter(nodeID: nodeID, field: key) == nil else { statusMessage = "Edit this exposed parameter's default in the boundary panel."; return false }
        var next = scope
        if let index = next.nodes.firstIndex(where: { $0.id == nodeID }) { next.nodes[index].config[key] = accepted }
        if let index = next.instances.firstIndex(where: { $0.id == nodeID }) { next.instances[index].overrides[key] = accepted }
        edit { self.replaceScope(next, in: &$0) }
        return graph.nodes.first(where: { $0.id == nodeID })?.config[key] == accepted
    }
    func openDefinition(instanceID: String) {
        guard !workspace.isBusy, let instance = scope.instances.first(where: { $0.id == instanceID }) else { return }
        state.activeReference = instance.definition
        clearSelection()
    }
    func showRoot() {
        guard !workspace.isBusy else { return }; state.activeReference = nil
        clearSelection()
    }
    func beginRevision() {
        guard !workspace.isBusy, let reference = state.activeReference, var definition = activeDefinition else { return }
        guard state.draft == nil || state.draftBase == reference else { statusMessage = "Publish or discard the existing draft first."; return }
        do { definition.body = try CompositeMigration.toCurrentBody(definition.body, registry: primitiveRegistry) }
        catch { statusMessage = error.localizedDescription; return }
        let before = state; state.draft = definition; state.draftBase = reference; history.record(before: before, after: state)
        statusMessage = "Editing a new revision draft. Existing instances remain pinned."
    }
    func discardDraft() { guard !workspace.isBusy else { return }; let before = state; state.draft = nil; state.draftBase = nil; history.record(before: before, after: state) }
    func updateBoundary(_ action: (inout CompositeDefinition) -> Void) { guard editingDraft else { return }; edit { if var draft = $0.draft { action(&draft); $0.draft = draft } } }
    func updateBoundary(context: UUID, _ action: (inout CompositeDefinition) -> Void) {
        guard context == boundaryRevision else { return }
        updateBoundary(action)
    }
    func setBoundaryDefault(key: String, value: Double, context: UUID) {
        _ = setBoundaryValue(key: key, value: .number(value), context: context)
    }
    @discardableResult func setBoundaryValue(key: String, value: JSONValue, context: UUID) -> Bool {
        guard context == boundaryRevision, editingDraft, !workspace.isBusy, projectID == workspace.activeProjectID,
              let field = activeDefinition?.parameters.first(where: { $0.field.key == key })?.field, let accepted = RegistryFieldCompatibility.normalized(value, for: field) else { return false }
        updateBoundary(context: context) { definition in
            guard let index = definition.parameters.firstIndex(where: { $0.field.key == key }) else { return }
            definition.parameters[index].field.defaultValue = accepted
        }
        return activeDefinition?.parameters.first(where: { $0.field.key == key })?.field.defaultValue == accepted
    }
    @discardableResult func releaseIntegerProduct(at index: Int, context: UUID) -> Bool {
        guard boundaryContextIsActive, context == boundaryRevision, let definition = activeDefinition else { return false }
        do {
            let proposed = try CompositeBindingEditing.releasingIntegerProduct(at: index, from: definition, definitions: project.definitions, registry: primitiveRegistry)
            updateBoundary(context: context) { $0 = proposed }
            return activeDefinition == proposed
        } catch { statusMessage = error.localizedDescription; return false }
    }
    func publishRevision() throws {
        guard !workspace.isBusy, var draft = state.draft, let base = state.draftBase else { throw CompositeContractError(.immutableRevision, "Create a revision draft first.") }
        try CompositeDraftValidation.validate(draft, registry: primitiveRegistry)
        let latest = project.definitions.filter { $0.reference.id == base.id && $0.reference.version.major == base.version.major && $0.reference.version.minor == base.version.minor }.map { $0.reference.version.patch }.max() ?? base.version.patch
        guard latest < Int.max else { throw CompositeContractError(.resourceLimit, "Choose a new version series; patch version is exhausted.") }
        draft.reference.version.patch = latest + 1
        let next = try CompositeProjectEdits.addingRevision(draft, to: project)
        let before = state; state.project = next; state.activeReference = draft.reference; state.draft = nil; state.draftBase = nil
        history.record(before: before, after: state); clearSelection()
        statusMessage = "Published \(Self.versionText(draft.reference.version)). Upgrade instances explicitly; expansion validates the complete model."
    }
    func upgradeInstance(_ id: String, to reference: CompositeDefinitionReference) throws {
        guard !workspace.isBusy, state.activeReference == nil else { throw CompositeContractError(.immutableRevision, "Upgrade root instances explicitly from the root workspace.") }
        let next = try CompositeProjectEdits.pinningInstance(id, to: reference, in: project)
        edit { $0.project = next }; statusMessage = "Instance pin changed. Validate the expanded architecture before use."
    }
    func undo() { guard !workspace.isBusy, let value = history.undo(current: state) else { return }; state = value; boundaryRevision = UUID(); clearSelection(); canvasViewRevision = UUID() }
    func redo() { guard !workspace.isBusy, let value = history.redo(current: state) else { return }; state = value; boundaryRevision = UUID(); clearSelection(); canvasViewRevision = UUID() }
}
