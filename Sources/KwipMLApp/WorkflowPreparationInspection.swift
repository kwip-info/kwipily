import Foundation
import KwipMLCore

/// Owns a private copy of verified bytes; callers remove it after the worker exits.
/// No current source path, authoring owner or mutable canvas projection is retained.
struct WorkflowPreparationInspectionInput {
    struct Node: Equatable {
        let address: WorkflowNodeAddress
        let name: String
        let operatorID: String
        let operatorTitle: String
        let portKinds: [String: String]
        var ports: Set<String> { Set(portKinds.keys) }
    }
    let reference: WorkflowHistoryReference
    let runID: String
    let requestIdentity: String
    let terminalStatus: WorkflowExecutionPhase
    let capturedRevision: UInt64
    let directory: URL
    let manifestSHA256: String
    let graphDigest: String
    let registryDigest: String
    let numpyVersion: String
    let graph: DataFlowDocument
    let nodes: [Node]
    let cacheReport: DataFlowCacheReport
    var captureOwnership: CaptureOwnership? = nil
    var storageReservation: WorkflowCaptureStorageReservation? { captureOwnership?.storageReservation }
    func remove() {
        if let captureOwnership { captureOwnership.remove() }
        else { try? FileManager.default.removeItem(at: directory) }
    }
    /// Shared by copied display metadata; removal clears admission ownership for every copy.
    final class CaptureOwnership {
        private let lock = NSLock()
        private var capture: WorkflowPreviewCapture?
        private var reservation: WorkflowCaptureStorageReservation?
        init(capture: WorkflowPreviewCapture, reservation: WorkflowCaptureStorageReservation?) {
            self.capture = capture; self.reservation = reservation
        }
        var storageReservation: WorkflowCaptureStorageReservation? {
            lock.lock(); defer { lock.unlock() }
            return reservation
        }
        func remove() {
            lock.lock()
            let capture = self.capture, reservation = self.reservation
            self.capture = nil; self.reservation = nil
            lock.unlock()
            capture?.remove(); reservation?.close()
        }
        deinit { remove() }
    }
    var arguments: [String] {
        ["--inspect-dataflow", directory.appendingPathComponent("manifest.json").path,
         "--dataflow-manifest-sha256", manifestSHA256]
    }
}

struct WorkflowPreparationInspectionSnapshot {
    let input: WorkflowPreparationInspectionInput
    let outputs: [WorkflowNodeAddress: [String: JSONValue]]
    let cacheReport: DataFlowCacheReport
}

/// Pinned saved-output semantics shared by History and standalone restoration.
/// Current source bindings and History receipt identity are deliberately absent.
struct PreparationInspectionExpectation {
    let manifestSHA256: String
    let graphDigest: String
    let registryDigest: String
    let numpyVersion: String
    let graph: DataFlowDocument
    let nodes: [WorkflowPreparationInspectionInput.Node]
    let cacheReport: DataFlowCacheReport

    init(input: WorkflowPreparationInspectionInput) {
        manifestSHA256 = input.manifestSHA256; graphDigest = input.graphDigest
        registryDigest = input.registryDigest; numpyVersion = input.numpyVersion
        graph = input.graph; nodes = input.nodes; cacheReport = input.cacheReport
    }
    private init(manifestSHA256: String, graphDigest: String, registryDigest: String, numpyVersion: String,
                 graph: DataFlowDocument, nodes: [WorkflowPreparationInspectionInput.Node], cacheReport: DataFlowCacheReport) {
        self.manifestSHA256 = manifestSHA256; self.graphDigest = graphDigest; self.registryDigest = registryDigest
        self.numpyVersion = numpyVersion; self.graph = graph; self.nodes = nodes; self.cacheReport = cacheReport
    }
    /// Reads bounded pinned metadata only. The execution service verifies and stages
    /// the complete artifact inventory before the worker interprets table/state bytes.
    static func read(manifestURL: URL, manifestSHA256: String) throws -> Self {
        guard manifestURL.isFileURL, manifestURL.lastPathComponent == "manifest.json",
              WorkflowArchiveCodec.isHash(manifestSHA256) else { throw WorkflowFileLeaseError.unsafePath }
        try Task.checkCancellation()
        let root = manifestURL.deletingLastPathComponent()
        let bytes = try DiagnosticFileIO.read(root: root, parts: ["manifest.json"], limit: 2_000_000)
        guard WorkflowArchiveCodec.hash(bytes) == manifestSHA256 else { throw WorkflowArchiveError("Saved preparation manifest differs from its selected pin.") }
        struct Manifest: Decodable {
            let graphDigest: String; let registryDigest: String; let numpyVersion: String
            let artifacts: [WorkflowArchiveFile]
        }
        let manifest = try JSONDecoder().decode(Manifest.self, from: bytes)
        try WorkflowCompletedPreparationInventory.verify(data: bytes,
            files: manifest.artifacts + [.init(path: "manifest.json", bytes: bytes.count, sha256: manifestSHA256)], prefix: "")
        guard WorkflowArchiveCodec.isHash(manifest.registryDigest), !manifest.numpyVersion.isEmpty,
              manifest.numpyVersion.utf8.count <= 128 else { throw WorkflowArchiveError("Saved preparation provenance is invalid.") }
        func metadata(_ name: String) throws -> Data {
            try Task.checkCancellation()
            guard let file = manifest.artifacts.first(where: { $0.path == name }), file.bytes <= 2_000_000 else {
                throw WorkflowArchiveError("Saved preparation lacks bounded graph or registry metadata.")
            }
            let content = try DiagnosticFileIO.read(root: root, parts: [name], limit: file.bytes)
            guard content.count == file.bytes, WorkflowArchiveCodec.hash(content) == file.sha256 else {
                throw WorkflowArchiveError("Saved preparation metadata changed: " + name)
            }
            return content
        }
        let graph = try DataFlowCodec.decodeGraph(metadata("dataflow.json"))
        let registry = try JSONDecoder().decode(DataFlowOperatorRegistry.self, from: metadata("dataflow-operators.json"))
        guard registry.schemaVersion == 1, !registry.packageID.isEmpty,
              Set(registry.operators.map(\.id)).count == registry.operators.count else {
            throw WorkflowArchiveError("Saved preparation registry has unsupported or duplicate operator identities.")
        }
        let nodes = try graph.nodes.map { node -> WorkflowPreparationInspectionInput.Node in
            guard let definition = registry.operatorDefinition(id: node.operatorID)?.configuredForExecution(node.config,connectedInputs:Set(graph.edges.filter{$0.to.node==node.id}.map{$0.to.port})),
                  !definition.id.isEmpty, definition.id.count <= 128,
                  Set(definition.outputs.map(\.id)).count == definition.outputs.count,
                  definition.outputs.allSatisfy({ !$0.id.isEmpty && !$0.id.contains("\0") && ["table", "dataset", "state"].contains($0.kind) }) else {
                throw WorkflowArchiveError("Saved preparation component or output port is unsupported.")
            }
            return .init(address: .init(domain: .data, localID: node.id), name: node.id,
                operatorID: node.operatorID, operatorTitle: definition.title.isEmpty ? node.operatorID : definition.title,
                portKinds: Dictionary(uniqueKeysWithValues: definition.outputs.map { ($0.id, $0.kind) }))
        }
        let report = try DataFlowCacheReportCodec.decodeManifest(bytes, graph: graph)
        try Task.checkCancellation()
        // Canonical graph/registry digests differ from raw file hashes. Their semantic
        // equivalence is checked by the worker against these same pinned bytes.
        return .init(manifestSHA256: manifestSHA256, graphDigest: manifest.graphDigest,
            registryDigest: manifest.registryDigest, numpyVersion: manifest.numpyVersion,
            graph: graph, nodes: nodes, cacheReport: report)
    }
}

struct PreparationInspectionResult {
    let outputs: [WorkflowNodeAddress: [String: JSONValue]]
    let cacheReport: DataFlowCacheReport
}

/// History-specific presentation wraps the same strict saved-output stream.
struct WorkflowPreparationInspectionStream {
    let input: WorkflowPreparationInspectionInput
    private var stream: PreparationInspectionStream
    init(input: WorkflowPreparationInspectionInput) {
        self.input = input; stream = .init(expectation: .init(input: input))
    }
    mutating func receive(_ line: String) throws { try stream.receive(line) }
    mutating func finish(exitCode: Int32) throws -> WorkflowPreparationInspectionSnapshot {
        let result = try stream.finish(exitCode: exitCode)
        return .init(input: input, outputs: result.outputs, cacheReport: result.cacheReport)
    }
}

/// All-or-nothing protocol accumulator. Partial or malformed streams cannot be
/// promoted after an error; a successful terminal event alone is insufficient.
struct PreparationInspectionStream {
    let expectation: PreparationInspectionExpectation
    private var started = false, completed = false, failed = false, finished = false
    private var byteCount = 0
    private var outputs: [WorkflowNodeAddress: [String: JSONValue]] = [:]
    mutating func receive(_ line: String) throws {
        do { try accept(line) }
        catch { failed = true; throw error }
    }
    private mutating func accept(_ line: String) throws {
        try require(!failed && !completed && !finished, "Inspection emitted data after failure or completion.")
        try require(line.utf8.count <= 1_000_000, "Inspection event exceeds its byte limit.")
        let bytes = Data(line.utf8)
        let event = try JSONDecoder().decode([String: JSONValue].self, from: bytes)
        switch event["event"] {
        case .string("dataflowInspectionStarted"):
            try require(!started && event.count == 1, "Inspection start is invalid or duplicated.")
            started = true
        case .string("dataflowPreview"):
            try require(started, "Inspection preview preceded its start.")
            byteCount += bytes.count
            try require(byteCount <= 32_000_000, "Inspection previews exceed the aggregate byte limit.")
            guard case .string(let id) = event["nodeID"], case .object(let ports) = event["outputs"],
                  let node = expectation.nodes.first(where: { $0.address.localID == id }),
                  outputs[node.address] == nil, Set(ports.keys) == node.ports else {
                throw WorkflowArchiveError("Inspection outputs differ from captured node/port identities.")
            }
            for (port, value) in ports {
                try WorkflowPreparationPreviewValidation.validate(value, kind: node.portKinds[port] ?? "", operatorID: node.operatorID)
            }
            guard case .object(let execution) = event["execution"] else { throw WorkflowArchiveError("Inspection omitted its cache decision.") }
            let expected = expectation.cacheReport.nodes.first { $0.nodeID == id }
            try require(execution["nodeID"] == .string(id)
                && execution["status"] == .string(expected?.status ?? "executed")
                && execution["reason"] == .string(expected?.reason ?? "disabled")
                && execution["cacheKey"] == (expected?.cacheKey.map(JSONValue.string) ?? .null),
                "Inspection cache decision differs from the verified archive.")
            outputs[node.address] = ports
        case .string("dataflowInspectionCompleted"):
            try require(started && outputs.count == expectation.nodes.count, "Inspection did not return every captured node.")
            try require(event["manifestSHA256"] == .string(expectation.manifestSHA256)
                && event["graphDigest"] == .string(expectation.graphDigest), "Inspection completed with a different artifact identity.")
            guard case .object(let provenance) = event["provenance"],
                  provenance["historical"] == .bool(true), provenance["sourceRevalidated"] == .bool(false),
                  provenance["cacheReuseVerified"] == .bool(false),
                  provenance["registryDigest"] == .string(expectation.registryDigest),
                  provenance["numpyVersion"] == .string(expectation.numpyVersion),
                  let cache = event["cache"] else { throw WorkflowArchiveError("Inspection historical provenance is invalid.") }
            let report = try DataFlowCacheReportCodec.decodeVerified(JSONEncoder().encode(cache), graph: expectation.graph)
            try require(report == expectation.cacheReport, "Inspection cache report differs from the verified archive.")
            completed = true
        case .string("error"):
            let message: String
            if case .string(let value) = event["message"] { message = String(value.prefix(2000)) }
            else { message = "Preparation inspection failed." }
            throw WorkflowArchiveError(message)
        default: throw WorkflowArchiveError("Inspection returned an unsupported event.")
        }
    }
    mutating func finish(exitCode: Int32) throws -> PreparationInspectionResult {
        defer { finished = true }
        try require(!finished && !failed && started && completed && exitCode == 0, "Preparation inspection did not complete successfully; no previews accepted.")
        return .init(outputs: outputs, cacheReport: expectation.cacheReport)
    }
    private func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw WorkflowArchiveError(message) }
    }
}
