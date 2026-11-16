import Foundation
import Combine
import KwipMLCore

/// Retained for one selection generation. Values may change without recreating the field,
/// but returning to the same node after selection/reload/undo never revives this authority.
@MainActor final class PreparationDataNodeContext: ObservableObject, DataNodeEditing {
    let store: DataFlowWorkspaceStore
    let nodeID: String
    let contextIdentity = UUID().uuidString
    private let generation: String
    private var active = true
    private var observation: AnyCancellable?
    private let sourceAction: ((DataNodeSourceOperation) -> Void)?
    private var pendingKeys = Set<String>()

    init(store: DataFlowWorkspaceStore, nodeID: String, sourceAction: ((DataNodeSourceOperation) -> Void)? = nil) {
        self.store = store; self.nodeID = nodeID; self.sourceAction = sourceAction
        generation = store.numericEditOrigin(nodeID:nodeID,key:"context")
        observation = store.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }
    }
    var isCurrent: Bool {
        active && store.projectID == store.workspace.activeProjectID && store.selectedNodeID == nodeID
            && generation == store.numericEditOrigin(nodeID:nodeID,key:"context")
            && store.project.graph.nodes.contains { $0.id == nodeID }
    }
    var isBusy: Bool { !isCurrent || store.isBusy || store.workspace.isBusy }
    var node: DataFlowNode? { isCurrent ? store.project.graph.nodes.first { $0.id == nodeID } : nil }
    var authoredName: String { isCurrent ? store.project.editor[nodeID]?.name ?? "" : "" }
    var operatorTitle: String { node.map { store.operators?.operatorDefinition(id:$0.operatorID)?.title ?? $0.operatorID } ?? "Unavailable node" }
    var operatorDefinition: DataFlowOperatorDefinition? { node.flatMap { store.operators?.operatorDefinition(id: $0.operatorID) } }
    var statusMessage: String { store.statusMessage }
    var usesFixtures: Bool { store.policy == .normalizationLesson }
    var fixtureColumns: [DataColumnSchema] { store.policy.fixtureColumns }
    var upstreamSchema: [DataColumnSchema] { isCurrent ? store.availableColumnSchema(for:nodeID,verifySources:false) : [] }
    func upstreamSchema(forInput port:String)->[DataColumnSchema] { isCurrent ? store.availableColumnSchema(for:nodeID,verifySources:false,inputPort:port):[] }
    func inputSourceName(forInput port:String)->String? {
        guard isCurrent,let edge=store.project.graph.edges.first(where:{$0.to.node == nodeID && $0.to.port == port}),let node=store.project.graph.nodes.first(where:{$0.id == edge.from.node}) else {return nil}
        let name=store.project.editor[node.id]?.name ?? ""
        return name.isEmpty ? store.operators?.operatorDefinition(id:node.operatorID)?.title:name
    }
    var schemaDisclosure: String { usesFixtures ? "Declared lesson fixture columns; execution validates the construction." : "Declared columns from recorded source headers and configuration; sources not revalidated here." }
    var sourceDisplayName: String? { isCurrent ? store.project.sourceBindings[nodeID].map { URL(fileURLWithPath:$0).lastPathComponent } : nil }
    var sourceIsLinked: Bool { isCurrent && store.project.sourceBindings[nodeID] != nil }
    var sourceInspectionPreview: SourceInspectionResult? { isCurrent && !usesFixtures ? store.sourceInspectionPreview(for: nodeID) : nil }
    var sourcePermissionSummary: String {
        guard sourceIsLinked else { return "No local source permission selected." }
        return sourceInspectionPreview == nil
            ? "Local source linked. Refresh to verify access and inspect its contents."
            : "Read access succeeded for this recorded snapshot. External changes or revoked access require Refresh; no source reads occur while editing."
    }
    var sourceSeparatorValues: [String] { store.sourceSeparatorValues }
    var outputs: [String:Any]? { isCurrent && !usesFixtures ? store.previews[nodeID] : nil }
    private var captureMatches: Bool { store.resultGraph == store.project.graph && store.resultBindings == store.project.sourceBindings }
    var outputStatus: String {
        guard outputs != nil else { return "Not evaluated" }
        guard captureMatches else { return "Stale captured flow" }
        return store.restoredResult ? "Historical · sources unchecked" : "Captured · sources not revalidated"
    }
    var retainedRows: String? {
        guard isCurrent, !usesFixtures, captureMatches, !store.restoredResult,
              let input=store.project.graph.edges.first(where:{$0.to.node == nodeID && $0.to.port == "table"}),
              let incoming=store.previews[input.from.node]?[input.from.port] as? [String:Any],
              let before=(incoming["preview"] as? [String:Any])?["rowCount"] as? Int,
              let outgoing=store.previews[nodeID]?["table"] as? [String:Any],
              let after=(outgoing["preview"] as? [String:Any])?["rowCount"] as? Int else {return nil}
        return "Captured run: \(before) input → \(after) retained · \(max(0,before-after)) removed. Sources not revalidated."
    }
    private func accepts(_ expected:[String:JSONValue]) -> Bool {
        guard !isBusy, let node else { return false }
        guard node.config == expected else { store.statusMessage="Configuration changed. Reload this editor before applying its draft.";return false }
        return true
    }
    @discardableResult func configure(_ config:[String:JSONValue],expected:[String:JSONValue])->Bool {
        guard accepts(expected),let position=store.project.graph.nodes.firstIndex(where:{$0.id == nodeID}) else {return false}
        guard !RetiredDataOperatorMigration.retired.contains(store.project.graph.nodes[position].operatorID) else {store.statusMessage="Retired tool settings require exact project upgrade or explicit repair.";return false}
        do {
            var candidate=store.project.graph;candidate.nodes[position].config=config
            _ = try DataFlowCodec.encodeGraph(candidate)
            store.configure(config,nodeID:nodeID)
            return node?.config == config
        } catch {store.statusMessage=error.localizedDescription;return false}
    }
    @discardableResult func applyTableConfiguration(_ config:[String:JSONValue],expected:[String:JSONValue])->Bool {
        guard accepts(expected) else {return false}
        guard let node,!RetiredDataOperatorMigration.retired.contains(node.operatorID) else {store.statusMessage="Retired tool settings require exact project upgrade or explicit repair.";return false}
        return store.applyTableConfiguration(config,nodeID:nodeID,expected:expected)
    }
    @discardableResult func rename(_ name:String)->Bool {
        guard !isBusy else {return false}
        store.rename(name)
        return authoredName == String(name.prefix(80))
    }
    func numericOrigin(key:String)->String { generation + ":" + contextIdentity + ":" + key }
    func currentNumericOrigin(key:String)->String? { !isBusy ? numericOrigin(key:key) : nil }
    func setNumericPending(origin:String,key:String,pending:Bool) {
        guard isCurrent,origin == numericOrigin(key:key) else {return}
        let storeOrigin=store.numericEditOrigin(nodeID:nodeID,key:key)
        store.setNumericEditPending(origin:storeOrigin,nodeID:nodeID,key:key,pending:pending)
        if pending {pendingKeys.insert(key)} else {pendingKeys.remove(key)}
    }
    func retire() {
        for key in pendingKeys {setNumericPending(origin:numericOrigin(key:key),key:key,pending:false)}
        active=false;observation=nil
    }
    private var permitsSourceOperation:Bool { !isBusy && !usesFixtures && store.allowsPreparationOperations && node?.operatorID == "kwip.data.sourceCSV" }
    @discardableResult func chooseSource()->Bool {
        guard permitsSourceOperation else {return false}
        if let sourceAction {sourceAction(.choose)} else {store.chooseSource(ifStillCurrent:{[weak self] in self?.permitsSourceOperation == true})}
        return true
    }
    @discardableResult func refreshSource()->Bool {
        guard permitsSourceOperation else {return false}
        if let sourceAction {sourceAction(.refresh)} else {store.inspectSource(id:nodeID)}
        return true
    }
    @discardableResult func setSeparator(_ value:String,expected:[String:JSONValue])->Bool {
        guard permitsSourceOperation,accepts(expected),sourceSeparatorValues.contains(value) else {return false}
        if let sourceAction {sourceAction(.separator(value))} else {store.configureSourceSeparator(value,nodeID:nodeID)}
        return true
    }
}
