import Foundation
import CryptoKit
import KwipMLCore

enum CompositeArchitectureIdentity {
    static func digest(_ graph: GraphDocument, registry: ComponentRegistry? = nil) throws -> String {
        hash(try GraphSemantics.canonicalData(graph, registry: registry ?? ComponentRegistry.builtIn()))
    }
    static func legacyDigest(_ graph: GraphDocument) throws -> String {
        var value = graph; value.name = ""; value.execution = nil
        for index in value.nodes.indices { value.nodes[index].name = ""; value.nodes[index].x = 0; value.nodes[index].y = 0 }
        value.nodes.sort { $0.id < $1.id }; value.edges.sort { $0.id < $1.id }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        return hash(try encoder.encode(value))
    }
    static func hash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
}

extension WorkspaceStore {
    var compositeSourceIsCurrent: Bool {
        guard let reference = execution.compositeSource,
              let data = try? WorkspaceDocumentReader.read(URL(fileURLWithPath: reference.path)), CompositeArchitectureIdentity.hash(data) == reference.sha256,
              let snapshot = try? JSONDecoder().decode(CompositeSourceSnapshot.self, from: data), (try? snapshot.verifiedExpansion(registry: registry)) != nil,
              let snapshotDigest = try? CompositeArchitectureIdentity.digest(snapshot.expandedGraph, registry: registry),
              let currentDigest = try? CompositeArchitectureIdentity.digest(graph, registry: registry) else { return false }
        let recordedIdentityMatches = reference.expandedSemanticSHA256 == snapshotDigest || reference.expandedSemanticSHA256 == (try? CompositeArchitectureIdentity.legacyDigest(snapshot.expandedGraph))
        return recordedIdentityMatches && currentDigest == snapshotDigest
    }
    var compositeSourceStatus: String {
        guard execution.compositeSource != nil else { return "No reusable source attached to the model." }
        return compositeSourceIsCurrent ? "Model matches its preserved reusable source." : "Model architecture changed after expansion. Reusable source is retained but will not be attributed to this edited model's run."
    }
    func adoptCompositeExpansion(project: CompositeProjectDocument, expansion: CompositeExpansion, sourceURL: URL) throws {
        guard !isBusy else { throw CompositeContractError(.invalidExpandedGraph, "Stop the active worker before using a reusable architecture.") }
        let checked = try CompositeExpander.expand(project, registry: registry)
        guard checked == expansion else { throw CompositeContractError(.invalidExpandedGraph, "Expansion does not match its source document.") }
        let saved = try JSONDecoder().decode(CompositeEditorState.self, from: WorkspaceDocumentReader.read(sourceURL))
        guard saved.project == project else { throw CompositeContractError(.invalidExpandedGraph, "Save the current composite source before expanding it.") }
        let snapshot = CompositeSourceSnapshot(project: project, expansion: checked)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(snapshot)
        guard data.count <= 2_000_000 else { throw CompositeContractError(.resourceLimit, "Composite provenance exceeds 2 MB.") }
        let hash = CompositeArchitectureIdentity.hash(data)
        let directory = workspaceDirectory.appendingPathComponent("composite/snapshots")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent(hash + ".json")
        if FileManager.default.fileExists(atPath: url.path) {
            guard try WorkspaceDocumentReader.read(url) == data else { throw CompositeContractError(.invalidExpandedGraph, "The immutable composite snapshot has changed.") }
        } else { try data.write(to: url, options: .atomic) }
        var settings = execution
        settings.compositeSource = CompositeSourceReference(path: url.path, sha256: hash, expandedSemanticSHA256: try CompositeArchitectureIdentity.digest(checked.graph))
        var expanded = checked.graph; expanded.name = graph.name; expanded.execution = settings
        edit { $0 = expanded }
        historySessionRevision = UUID()
        statusMessage = "Expanded architecture adopted. Its reusable source, exact pins and node origins are preserved."
    }
    func compositeTrainingArguments(requestDirectory: URL) throws -> [String] {
        if let owner = modelAuthoring {
            return try compositeTrainingArguments(request: ModelExecutionRequest.capture(from: owner, registry: registry), requestDirectory: requestDirectory)
        }
        return try legacyCompositeTrainingArguments(graph: graph, settings: execution, requestDirectory: requestDirectory)
    }
    /// Stages provenance from the same immutable request used to write the worker graph.
    func compositeTrainingArguments(request: ModelExecutionRequest, requestDirectory: URL) throws -> [String] {
        guard request.projectID == activeProjectID, request.isCurrent(in: modelAuthoring) else {
            throw CompositeContractError(.invalidExpandedGraph, "The parent model changed after capture. Capture the current model before staging its source.")
        }
        guard let provenance = request.snapshot.provenance else {
            return try legacyCompositeTrainingArguments(graph: request.graph, settings: request.settings, requestDirectory: requestDirectory)
        }
        _ = try provenance.verifiedExpansion(registry: registry)
        guard provenance.expandedGraph == request.graph,
              try CompositeArchitectureIdentity.digest(provenance.expandedGraph, registry: registry) == request.snapshot.semanticSHA256 else {
            throw CompositeContractError(.invalidExpandedGraph, "Captured Block provenance does not match its executable graph.")
        }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(provenance)
        guard data.count <= 2_000_000 else { throw CompositeContractError(.resourceLimit, "Captured Block provenance exceeds 2 MB.") }
        return try stageCompositeTrainingSource(data, sha256: CompositeArchitectureIdentity.hash(data), requestDirectory: requestDirectory)
    }
    private func legacyCompositeTrainingArguments(graph: GraphDocument, settings: ExecutionSettings, requestDirectory: URL) throws -> [String] {
        guard let reference = settings.compositeSource else { return [] }
        let data = try WorkspaceDocumentReader.read(URL(fileURLWithPath: reference.path))
        guard CompositeArchitectureIdentity.hash(data) == reference.sha256 else { throw CompositeContractError(.invalidExpandedGraph, "Composite source snapshot changed. Adopt the source again before training.") }
        let snapshot = try WorkflowArchiveCodec.decodeCompositeSource(data)
        _ = try snapshot.verifiedExpansion(registry: registry)
        let currentIdentity = try CompositeArchitectureIdentity.digest(snapshot.expandedGraph, registry: registry)
        let legacyIdentity = try CompositeArchitectureIdentity.legacyDigest(snapshot.expandedGraph)
        guard reference.expandedSemanticSHA256 == currentIdentity || reference.expandedSemanticSHA256 == legacyIdentity else { throw CompositeContractError(.invalidExpandedGraph, "Composite source architecture identity does not match the selected model.") }
        guard try CompositeArchitectureIdentity.digest(graph, registry: registry) == currentIdentity else { return [] }
        return try stageCompositeTrainingSource(data, sha256: reference.sha256, requestDirectory: requestDirectory)
    }
    private func stageCompositeTrainingSource(_ data: Data, sha256: String, requestDirectory: URL) throws -> [String] {
        let staged = requestDirectory.appendingPathComponent("composite-source.json")
        try data.write(to: staged, options: .atomic)
        return ["--composite-source", staged.path, "--composite-source-sha256", sha256]
    }

}

extension CompositeWorkspaceStore {
    func useExpandedArchitecture() {
        guard projectID == workspace.activeProjectID, projectID != nil, state.activeReference == nil else { statusMessage = "Return to this project's root composite workspace before adopting its architecture."; return }
        do {
            let expansion = try CompositeExpander.expand(project, registry: primitiveRegistry)
            save()
            try workspace.adoptCompositeExpansion(project: project, expansion: expansion, sourceURL: saveURL)
            statusMessage = workspace.statusMessage
        } catch { statusMessage = error.localizedDescription }
    }
    func validateExpansion() {
        do { let result = try CompositeExpander.expand(project, registry: primitiveRegistry); statusMessage = "Valid expansion · \(result.graph.nodes.count) primitives · \(result.consumedDefinitions.count) pinned revisions." }
        catch { statusMessage = error.localizedDescription }
    }
    func resetSelectedOverrides() {
        guard let id = selectedNodeID, state.activeReference == nil, !isBusy, let index = state.project.root.instances.firstIndex(where: { $0.id == id }) else { return }
        let before = state; state.project.root.instances[index].overrides = [:]; history.record(before: before, after: state)
    }
}
