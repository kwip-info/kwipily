import Foundation
import KwipMLCore

extension WorkspaceStore {
    var editorState: GraphEditorState {
        var nodes = selectedNodeIDs
        if let selectedNodeID { nodes.insert(selectedNodeID) }
        var edges = selectedEdgeIDs
        if let selectedEdgeID { edges.insert(selectedEdgeID) }
        nodes.formIntersection(Set(graph.nodes.map(\.id)))
        edges.formIntersection(Set(graph.edges.map(\.id)))
        return GraphEditorState(document: graph, selection: GraphSelection(nodeIDs: nodes, edgeIDs: edges))
    }
    func installEditorState(_ state: GraphEditorState) {
        graph = state.document
        selectedNodeIDs = state.selection.nodeIDs
        selectedEdgeIDs = state.selection.edgeIDs
        if selectedNodeID.map({ !selectedNodeIDs.contains($0) }) ?? true { selectedNodeID = graph.nodes.first { selectedNodeIDs.contains($0.id) }?.id }
        selectedEdgeID = selectedNodeIDs.isEmpty ? graph.edges.first { selectedEdgeIDs.contains($0.id) }?.id : nil
        pendingSourceID = nil
    }
    @discardableResult func applyCommand(_ command: GraphCommand) -> Bool {
        guard modelAuthoring == nil else { statusMessage = "Use the canonical model canvas for graph commands."; return false }
        guard !isBusy else { statusMessage = "Stop the current run before changing the graph."; return false }
        do {
            let result = try GraphCommandEngine.apply(command, to: editorState, registry: registry)
            if result.changesDocument { history.record(before: result.before, after: result.after) }
            installEditorState(result.after)
            if result.changesDocument { hasUnsavedChanges = true; saveManagedProject() }
            return true
        } catch { statusMessage = error.localizedDescription; return false }
    }
    func clearSelection() { selectedNodeIDs = []; selectedEdgeIDs = []; selectedNodeID = nil; selectedEdgeID = nil }
    func selectNode(_ id: String, extending: Bool = false) {
        guard graph.nodes.contains(where: { $0.id == id }) else { return }
        if extending {
            if selectedNodeIDs.contains(id) { selectedNodeIDs.remove(id) } else { selectedNodeIDs.insert(id) }
        } else { selectedNodeIDs = [id] }
        selectedNodeID = selectedNodeIDs.contains(id) ? id : graph.nodes.first { selectedNodeIDs.contains($0.id) }?.id
        selectedEdgeID = nil; selectedEdgeIDs = []
    }
    func selectEdge(_ id: String) {
        guard graph.edges.contains(where: { $0.id == id }) else { return }
        clearSelection(); selectedEdgeID = id; selectedEdgeIDs = [id]
    }
    func moveNodes(ids: Set<String>, dx: Double, dy: Double) { applyCommand(.move(nodeIDs: ids, offset: GraphOffset(x: dx, y: dy))) }
    func duplicateSelection() {
        guard !editorState.selection.nodeIDs.isEmpty else { return }
        if applyCommand(.duplicate(nodeIDs: editorState.selection.nodeIDs, offset: GraphOffset(), semantics: .independent)) { statusMessage = "Duplicated selected blocks with independent parameters. External connections were not copied." }
    }
    func deleteSelection() { applyCommand(.delete(editorState.selection)) }
    func reconnect(edgeID: String, source: String, sourcePort: String, target: String, targetPort: String) {
        if applyCommand(.reconnect(edgeID: edgeID, source: GraphPortAddress(source, sourcePort), target: GraphPortAddress(target, targetPort))) { statusMessage = "Connection updated. Undo restores the previous endpoints." }
    }
    func insertNode(type: String, edgeID: String, x: Double, y: Double) {
        guard let definition = registry.component(type: type), definition.inputs.count == 1, definition.outputs.count == 1 else { statusMessage = "Wire insertion requires one explicit input and output. Add multi-input blocks normally, then connect them."; return }
        let node = GraphNode(type: type, name: definition.title, x: x, y: y, config: Dictionary(uniqueKeysWithValues: definition.fields.map { ($0.key, $0.defaultValue) }))
        if applyCommand(.insertOnWire(edgeID: edgeID, node: node, inputPort: definition.inputs[0], outputPort: definition.outputs[0])) { statusMessage = "Inserted \(definition.title) on the connection. Undo restores the original wire." }
    }
}
