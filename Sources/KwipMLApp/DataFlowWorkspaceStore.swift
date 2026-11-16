import AppKit
import Combine
import SwiftUI
import KwipMLCore

@MainActor final class DataFlowWorkspaceStore: ObservableObject, CanvasWorkspace {
    let workspace: WorkspaceStore
    let policy: DataFlowEditorPolicy
    @Published var historySessionRevision = UUID() { didSet { invalidateNumericEdits() } }
    @Published private(set) var pendingNumericEdits = Set<String>()
    private var numericEditRevision = UUID()
    var hasPendingNumericEdits: Bool { !pendingNumericEdits.isEmpty }
    func numericEditOrigin(nodeID: String, key: String) -> String { "\(numericEditRevision):\(nodeID):\(key)" }
    func setNumericEditPending(origin: String, nodeID: String, key: String, pending: Bool) {
        guard projectID == workspace.activeProjectID, selectedNodeID == nodeID,
              origin == numericEditOrigin(nodeID:nodeID,key:key),
              !key.isEmpty, key.count <= 128, project.graph.nodes.contains(where: { $0.id == nodeID }) else { return }
        if pending { pendingNumericEdits.insert(origin) } else { pendingNumericEdits.remove(origin) }
    }
    private func invalidateNumericEdits() { numericEditRevision = UUID(); pendingNumericEdits.removeAll() }
    private var workspaceObservation: AnyCancellable?
    var canvasViewRevision: UUID { workspace.canvasViewRevision }
    var canvasViewState: CanvasViewState { get { workspace.dataCanvasViewState } set { if workspace.activeProjectID == projectID { workspace.dataCanvasViewState = newValue } } }
    let operators: DataFlowOperatorRegistry?
    let projectID: String?
    private let unsavedIdentity = UUID().uuidString
    @Published var project = DataFlowProject(name: "Data preparation", graph: DataFlowDocument())
    @Published var selectedNodeIDs = Set<String>()
    @Published var selectedNodeID: String? { didSet { if oldValue != selectedNodeID { invalidateNumericEdits() } } }
    @Published var selectedEdgeID: String?
    @Published var pendingSourceID: String?
    @Published var statusMessage = "Choose a source and configure the preparation flow."
    @Published var preparationFailure: DataFlowFailure?
    @Published var isPreparing = false
    var isBusy: Bool { isPreparing || isInspectingSource || isRestoringResults || workspace.isAssessingLearning || (policy == .normalizationLesson && projectID != workspace.activeProjectID) }
    @Published var isRestoringResults = false
    @Published var restoredResult = false
    @Published var completedResult: DataFlowCompletedReference?
    var restorationOperation: DataFlowRestorationOperation?
    var resultRestorationExecute: PreparationInspectionExecutor?
    @Published var previews: [String: [String: Any]] = [:]
    @Published var cachePolicy = DataFlowCachePolicy.materialize
    @Published var lastCacheReference: DataFlowCacheReference?
    @Published var cacheReport: DataFlowCacheReport?
    @Published var manifestURL: URL?
    @Published var sourceHeaders: [String: SourceHeaderEvidence] = [:]
    @Published var isInspectingSource = false
    var sourceInspectionOperation: DataFlowSourceInspectionOperation?
    var sourceInspectionProjectID: String { projectID ?? unsavedIdentity }
    var inspectionGeneration = UUID()
    var queuedInspectionSourceID: String?
    var prepareAfterInspection = false
    var inspectionClosed = false
    var sourceInspectionExecute: SourceInspectionExecutor?

    var preparationWorker: WorkflowWorkerHandle?
    var preparationLaunchTask: Task<Void, Never>?
    var preparationToken: UUID?
    var preparationWorkerLauncher: WorkflowWorkerLauncher = WorkflowExecutionSession.launchWorker
    var cancelled = false
    var history = EditHistory<DataFlowProject>()
    var resultGraph: DataFlowDocument?
    var resultBindings: [String: String] = [:]
    var resultSourceDigests: [String: String] = [:]
    var previewWeightLabel: String { policy == .normalizationLesson ? "Lesson construction · no preparation run" : "Data preparation · training-only fit" }
    static let editableTypes: Set<String> = Set(["decide","gameRules","sourceCSV","sourceFiles","classifyAudio","transcribeAudio","extractText","prepareMedia","textEncode","sequenceWindows","columns","filterRows","partition","formula","preprocess","sampleRows","join","aggregate","union","sort","unique","reshapeTable","splitText","view","outputDelimited"].map {"kwip.data."+$0})
    var saveURL: URL { workspace.workspaceDirectory.appendingPathComponent("dataflow/projects/\(projectID ?? unsavedIdentity)/flow.kwipdata") }

    convenience init(workspace: WorkspaceStore) { self.init(workspace:workspace,policy:.preparation,initialProject:nil) }
    convenience init(workspace: WorkspaceStore, lessonProject: DataFlowProject) { self.init(workspace:workspace,policy:.normalizationLesson,initialProject:lessonProject) }
    private init(workspace: WorkspaceStore, policy: DataFlowEditorPolicy, initialProject: DataFlowProject?) {
        self.workspace = workspace
        self.policy = policy
        projectID = workspace.activeProjectID
        let directory: URL
        if let runtime = workspace.runtime {
            directory = runtime.kind == .bundled ? runtime.workingDirectoryURL.appendingPathComponent("runner") : runtime.entryURL.deletingLastPathComponent()
        } else { directory = workspace.projectRoot.appendingPathComponent("runner") }
        operators = try? DataFlowOperatorRegistry.load(from: directory.appendingPathComponent("dataflow-operators.json"))
        workspaceObservation = workspace.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }
        if operators == nil { statusMessage = "Data operators are unavailable in this runtime." }
        if let initialProject {
            project = initialProject
            cachePolicy = .off
            statusMessage = operators == nil ? "Data operators are unavailable in this runtime." : "Build the normalization construction using adapter-owned fixture observations."
        } else {
            if FileManager.default.fileExists(atPath: saveURL.path) { reopen() }
            else { loadExample(columns: ["x1", "x2"]) }
            restoreCachePreferences()
        }
    }
    var registry: ComponentRegistry {
        ComponentRegistry(components: (operators?.operators ?? []).filter { allowedOperatorIDs.contains($0.id) }.map {
            ComponentDefinition(id: $0.id, title: policy == .normalizationLesson && $0.id == "kwip.data.sourceCSV" ? "Fixture source" : $0.title, category: "Data", inputs: $0.inputs.map(\.id), outputs: $0.authoringOutputs.map(\.id))
        })
    }
    var graph: GraphDocument {
        GraphDocument(name: project.name, nodes: project.graph.nodes.map { item in
            let layout = project.editor[item.id] ?? DataFlowEditorNode(name: operators?.operatorDefinition(id: item.operatorID)?.title ?? "Component", x: 30, y: 100)
            return GraphNode(id: item.id, type: item.operatorID, name: layout.name, x: layout.x, y: layout.y,config:item.config)
        }, edges: project.graph.edges.map { GraphEdge(id: $0.id, source: $0.from.node, sourcePort: $0.from.port, target: $0.to.node, targetPort: $0.to.port) })
    }
    var validation: ValidationResult {
        var issues: [ValidationIssue] = []
        if policy == .normalizationLesson {
            if !project.sourceBindings.isEmpty { issues.append(ValidationIssue(message:"Lesson fixtures cannot use local source bindings.")) }
            do { try NormalizationLearningSubmission(graph:project.graph,outputs:[:]).validateDraft() }
            catch { issues.append(ValidationIssue(message:error.localizedDescription)) }
        }
        for node in project.graph.nodes {
            guard let definition = operators?.operatorDefinition(id: node.operatorID)?.configured(node.config) else {
                issues.append(ValidationIssue(message: "Operator unavailable", nodeID: node.id)); continue
            }
            for input in definition.inputs where input.required && !project.graph.edges.contains(where: { $0.to.node == node.id && $0.to.port == input.id }) {
                issues.append(ValidationIssue(message: "Connect \(input.id)", nodeID: node.id))
            }
            if policy == .preparation, node.operatorID == "kwip.data.sourceCSV", project.sourceBindings[node.id] == nil { issues.append(ValidationIssue(message: "Choose a local CSV", nodeID: node.id)) }
        }
        return ValidationResult(issues: issues)
    }
    var resultsCurrent: Bool {
        allowsPreparationOperations && !restoredResult && manifestURL != nil && resultGraph == project.graph && resultBindings == project.sourceBindings && !resultSourceDigests.isEmpty && resultSourceDigests.allSatisfy { workspace.sourceDigest($0.key) == $0.value }
    }
    func previewStatus(for id: String) -> String {
        guard policy == .preparation, previews[id] != nil else { return "Not evaluated" }
        if restoredResult { return resultMatchesEditor ? "Historical · sources unchecked" : "Historical · stale flow" }
        guard resultsCurrent else { return "Stale" }
        return cacheReport?.reusedNodeIDs.contains(id) == true ? "Reused" : "Computed"
    }
    func canvasSummary(_ node: GraphNode) -> String {
        if policy == .preparation, let outputs = previews[node.id] {
            let counts = outputs.keys.sorted().compactMap { port -> (String, Int)? in
                guard let output = outputs[port] as? [String: Any], output["kind"] as? String == "table",
                      let preview = output["preview"] as? [String: Any], let count = preview["rowCount"] as? Int else { return nil }
                return (port, count)
            }
            if counts.count == 1 { return "\(counts[0].1) rows · table" }
            if !counts.isEmpty { return counts.map { "\($0.0): \($0.1)" }.joined(separator: " · ") + " rows" }
        }
        return operators?.operatorDefinition(id: node.type)?.outputs.map(\.kind).joined(separator: " · ") ?? "Unavailable"
    }
    func canvasPortDescription(nodeID: String, port: String) -> String {
        guard let node = project.graph.nodes.first(where: { $0.id == nodeID }), let output = operators?.operatorDefinition(id: node.operatorID)?.configured(node.config).outputs.first(where: { $0.id == port }) else { return "Unknown port" }
        return [output.kind, output.partition, output.stateKind].compactMap { $0 }.joined(separator: " · ")
    }
    func selectNode(_ id: String, extending: Bool = false) {
        if extending { if selectedNodeIDs.contains(id) { selectedNodeIDs.remove(id) } else { selectedNodeIDs.insert(id) } }
        else { selectedNodeIDs = [id] }
        selectedNodeID = selectedNodeIDs.contains(id) ? id : selectedNodeIDs.sorted().first; selectedEdgeID = nil
    }
    func selectEdge(_ id: String) { clearSelection(); selectedEdgeID = id }
    func clearSelection() { selectedNodeIDs = []; selectedNodeID = nil; selectedEdgeID = nil }
    private func edit(_ body: (inout DataFlowProject) -> Void) {
        guard !isBusy else { return }
        let before = project; var candidate=project;body(&candidate)
        if policy == .normalizationLesson {
            guard candidate.sourceBindings.isEmpty else { statusMessage="Lesson fixtures cannot use local source bindings.";return }
            do { try NormalizationLearningSubmission(graph:candidate.graph,outputs:[:]).validateDraft() }
            catch { statusMessage=error.localizedDescription;return }
        }
        project=candidate;history.record(before:before,after:project)
    }
    func rename(_ name: String) { guard let id = selectedNodeID else { return }; edit { $0.editor[id]?.name = String(name.prefix(80)) } }
    func configure(_ config: [String: JSONValue]) { guard let id = selectedNodeID else { return }; configure(config, nodeID: id) }
    func configure(_ config: [String: JSONValue], nodeID: String) {
        guard selectedNodeID == nodeID else { return }
        edit { p in if let i = p.graph.nodes.firstIndex(where: { $0.id == nodeID }) { p.graph.nodes[i].config = config } }
    }
    func bindSource(_ path: String, nodeID: String) {
        guard requirePreparationOperation() else {return}
        guard project.graph.nodes.contains(where: { $0.id == nodeID && $0.operatorID == "kwip.data.sourceCSV" }) else { return }
        edit { $0.sourceBindings[nodeID] = path }; sourceHeaders[nodeID] = nil
    }
    @discardableResult func apply(_ command: GraphCommand) -> Bool {
        guard !isBusy else { return false }
        do {
            let before = project
            let result = try GraphCommandEngine.apply(command, to: GraphEditorState(document: graph, selection: GraphSelection(nodeIDs: selectedNodeIDs, edgeIDs: selectedEdgeID.map { [$0] } ?? [])), registry: registry)
            let old = Dictionary(uniqueKeysWithValues: project.graph.nodes.map { ($0.id, $0) })
            project.graph.nodes = result.after.document.nodes.map { item in
                let original = old[item.id] ?? result.copies.nodes.first(where: { $0.value == item.id }).flatMap { old[$0.key] }
                return DataFlowNode(id: item.id, operatorID: item.type, config: original?.config ?? operators?.operatorDefinition(id: item.type)?.defaultConfig ?? [:])
            }
            project.graph.edges = result.after.document.edges.map { DataFlowEdge(from: DataFlowEndpoint(node: $0.source, port: $0.sourcePort), to: DataFlowEndpoint(node: $0.target, port: $0.targetPort)) }
            project.editor = Dictionary(uniqueKeysWithValues: result.after.document.nodes.map { ($0.id, DataFlowEditorNode(name: $0.name, x: $0.x, y: $0.y)) })
            project.sourceBindings = project.sourceBindings.filter { key, _ in project.graph.nodes.contains { $0.id == key } }
            for (oldID, newID) in result.copies.nodes { if let binding = before.sourceBindings[oldID] { project.sourceBindings[newID] = binding } }
            selectedNodeIDs = result.after.selection.nodeIDs; selectedNodeID = selectedNodeIDs.sorted().first; selectedEdgeID = nil
            history.record(before: before, after: project)
            return true
        } catch { statusMessage = error.localizedDescription; return false }
    }
    func moveNodes(ids: Set<String>, dx: Double, dy: Double) { apply(.move(nodeIDs: ids, offset: GraphOffset(x: dx, y: dy))) }
    func duplicateSelection() { apply(.duplicate(nodeIDs: selectedNodeIDs, offset: GraphOffset(), semantics: .independent)) }
    func deleteSelection() { apply(.delete(GraphSelection(nodeIDs: selectedNodeIDs, edgeIDs: selectedEdgeID.map { [$0] } ?? []))) }
    func addNode(type: String, x: Double, y: Double) {
        guard let definition = operators?.operatorDefinition(id: type), allowedOperatorIDs.contains(type) else { return }
        let node = DataFlowNode(operatorID: type, config: definition.defaultConfig)
        edit { $0.graph.nodes.append(node); $0.editor[node.id] = DataFlowEditorNode(name: definition.title, x: x, y: y) }; selectNode(node.id)
    }
    func connectionError(source: String, sourcePort: String, target: String, targetPort: String) -> String? {
        guard let a = project.graph.nodes.first(where: { $0.id == source }), let b = project.graph.nodes.first(where: { $0.id == target }),
              let output = operators?.operatorDefinition(id: a.operatorID)?.configured(a.config).authoringOutputs.first(where: { $0.id == sourcePort }), let input = operators?.operatorDefinition(id: b.operatorID)?.configured(b.config).inputs.first(where: { $0.id == targetPort }) else { return "Choose declared data ports." }
        let tabular = Set(["table", "dataset"])
        if input.kind != output.kind && !(tabular.contains(input.kind) && tabular.contains(output.kind)) { return "Connect \(input.kind) to \(input.kind); table and fitted state are distinct." }
        if let required = input.partition, let known = knownPartition(nodeID: source, port: sourcePort), known != required { return "This input requires the \(required) partition. Validation or unsplit data must never fit state." }
        if let required = input.stateKind, output.stateKind != required { return "This input requires \(required) fitted state." }
        return nil
    }
    /// Table-preserving operators retain the incoming partition. Missing inputs stay unresolved until worker validation.
    func knownPartition(nodeID: String, port: String, visited: Set<String> = []) -> String? {
        guard !visited.contains(nodeID), let node = project.graph.nodes.first(where: { $0.id == nodeID }),
              let output = operators?.operatorDefinition(id: node.operatorID)?.configured(node.config).outputs.first(where: { $0.id == port }) else { return nil }
        if let partition = output.partition { return partition }
        if ["kwip.data.sourceCSV","kwip.data.sourceFiles"].contains(node.operatorID) { return "unassigned" }
        if node.operatorID == "kwip.data.splitText",node.config["outputMode"] == .string("rows") {return nil}
        let preserving: Set<String> = ["kwip.data.filterRows","kwip.data.columns","kwip.data.formula","kwip.data.preprocess","kwip.data.sampleRows","kwip.data.sort","kwip.data.unique","kwip.data.splitText","kwip.data.view","kwip.data.outputDelimited"]
        guard preserving.contains(node.operatorID), let incoming = project.graph.edges.first(where: { $0.to.node == nodeID && $0.to.port == "table" }) else { return nil }
        return knownPartition(nodeID: incoming.from.node, port: incoming.from.port, visited: visited.union([nodeID]))
    }
    func canConnect(source: String, sourcePort: String, target: GraphNode, port: String) -> Bool {
        connectionError(source: source, sourcePort: sourcePort, target: target.id, targetPort: port) == nil && (try? GraphCommandEngine.apply(.connect(source: GraphPortAddress(source, sourcePort), target: GraphPortAddress(target.id, port)), to: GraphEditorState(document: graph), registry: registry)) != nil
    }
    func connect(source: String, sourcePort: String, target: String, targetPort: String) {
        if let error = connectionError(source: source, sourcePort: sourcePort, target: target, targetPort: targetPort) { statusMessage = error; return }
        guard apply(.connect(source: GraphPortAddress(source, sourcePort), target: GraphPortAddress(target, targetPort))) else { return }
        pendingSourceID = nil
        statusMessage = connectionStatus(source: source, port: sourcePort, verb: "Connected")
    }
    func reconnect(edgeID: String, source: String, sourcePort: String, target: String, targetPort: String) {
        if let error = connectionError(source: source, sourcePort: sourcePort, target: target, targetPort: targetPort) { statusMessage = error; return }
        guard apply(.reconnect(edgeID: edgeID, source: GraphPortAddress(source, sourcePort), target: GraphPortAddress(target, targetPort))) else { return }
        statusMessage = connectionStatus(source: source, port: sourcePort, verb: "Reconnected")
    }
    private func connectionStatus(source: String, port: String, verb: String) -> String {
        knownPartition(nodeID: source, port: port) == nil ? "\(verb). Partition unresolved; the worker must verify lineage before fitting or publishing outputs." : "\(verb) declared data ports."
    }
    func insertNode(type: String, edgeID: String, x: Double, y: Double) {
        guard let op = operators?.operatorDefinition(id: type), op.inputs.count == 1, op.outputs.count == 1, op.inputs[0].kind == "table", op.outputs[0].kind == "table", let edge = graph.edges.first(where: { $0.id == edgeID }), canvasPortDescription(nodeID: edge.source, port: edge.sourcePort).hasPrefix("table") else { statusMessage = "Insert a component with one table input and output."; return }
        // Partition-changing insertion needs explicit reconnection, never an implicit leakage adapter.
        guard op.inputs[0].partition == nil, op.outputs[0].partition == nil, let target = project.graph.nodes.first(where: { $0.id == edge.target }), operators?.operatorDefinition(id: target.operatorID)?.inputs.first(where: { $0.id == edge.targetPort })?.partition == nil else { statusMessage = "Reconnect partition-constrained inputs explicitly."; return }
        apply(.insertOnWire(edgeID: edgeID, node: GraphNode(type: type, name: op.title, x: x, y: y), inputPort: op.inputs[0].id, outputPort: op.outputs[0].id))
    }
    func undo() { guard !isBusy, let value = history.undo(current: project) else { return }; project = value; clearSelection() }
    func redo() { guard !isBusy, let value = history.redo(current: project) else { return }; project = value; clearSelection() }
    func save() {
        guard requirePreparationOperation() else {return}
        guard projectID != nil else { statusMessage = "Save the model project before opening its preparation workspace."; return }
        do { try FileManager.default.createDirectory(at: saveURL.deletingLastPathComponent(), withIntermediateDirectories: true); try DataFlowCodec.encodeProject(project).write(to: saveURL, options: .atomic); statusMessage = "Preparation flow saved on this Mac." }
        catch { statusMessage = error.localizedDescription }
    }
    func reopen() {
        guard requirePreparationOperation() else {return}
        guard !isBusy else { return }
        do {
            let reopened = try DataFlowCodec.decodeProject(WorkspaceDocumentReader.read(saveURL))
            historySessionRevision = UUID(); project = reopened
            history = EditHistory(); clearSelection(); statusMessage = "Preparation flow reopened."
            if FileManager.default.fileExists(atPath: completedResultURL.path) { restoreCompletedOutputs() }
            else if let source = project.graph.nodes.first(where: { $0.operatorID == "kwip.data.sourceCSV" && project.sourceBindings[$0.id] != nil }) { inspectSource(id: source.id) }
        } catch { statusMessage = error.localizedDescription }
    }

    func loadExample(columns: [String]) {
        guard requirePreparationOperation() else {return}
        guard !isBusy else { return }
        let types = [("source", "sourceCSV", "Table Input", 30.0, 190.0), ("cast", "columns", "Numeric columns", 240, 190), ("split", "partition", "Three populations", 450, 190), ("fit", "preprocess", "Learn from Training", 660, 190), ("train", "view", "Prepared training", 880, 70), ("validation", "view", "Prepared validation", 880, 290), ("test", "view", "Protected test", 880, 510)]
        var next = DataFlowProject(name: "Training-only standardization", graph: DataFlowDocument())
        for (id, type, name, x, y) in types {
            var config = operators?.operatorDefinition(id: "kwip.data." + type)?.defaultConfig ?? [:]
            if type == "columns" {config = ["rows":.array(columns.map {.object(["source":.string($0),"included":true,"outputName":.string($0),"targetType":.string("float64")])}),"castFailure":.string("error"),"trimText":false,"newColumns":.string("keep"),"missingPolicy":.string("error")]}
            if type == "preprocess" {config = PreprocessConfiguration(columns:columns.map { name in
                var row=PreprocessConfiguration.Column(column:name);row.missing="error";row.scaling="standard";return row
            }).values}
            if type == "partition" {config = ["strategy":.string("random"),"seed":42,"validationPercent":20,"testPercent":20]}
            next.graph.nodes.append(DataFlowNode(id: id, operatorID: "kwip.data." + type, config: config)); next.editor[id] = DataFlowEditorNode(name: name, x: x, y: y)
        }
        for (a, ap, b, bp) in [("source","table","cast","table"),("cast","table","split","table"),("split","training","fit","training"),("split","validation","fit","validation"),("split","test","fit","test"),("fit","training","train","table"),("fit","validation","validation","table"),("fit","test","test","table")] {
            next.graph.edges.append(DataFlowEdge(from: DataFlowEndpoint(node: a, port: ap), to: DataFlowEndpoint(node: b, port: bp)))
        }
        historySessionRevision = UUID(); edit { $0 = next }; clearSelection()
    }
}
