import Foundation
import Darwin
import KwipMLCore

/// Capture on a background task while the caller retains its workspace reservation.
/// Only one bounded external source is materialized at a time; the returned recipe holds private files.
final class PreparationLaunchPlan {
    let sourceDigests: [String: String]
    let storageReservation: WorkflowCaptureStorageReservation?
    private let capturedInputs: [WorkflowPreviewCapture.Input]
    private let sourcePaths: [String: String]
    private let preparationCacheLimitBytes: Int?
    private let cachePolicy: WorkflowPreparationCachePolicy
    private let reuseManifestSHA256: String?
    private let sourceStage: SourceStage

    private init(inputs: [WorkflowPreviewCapture.Input], sourcePaths: [String: String],
                 sourceDigests: [String: String], cachePolicy: WorkflowPreparationCachePolicy,
                 reuseManifestSHA256: String?, preparationCacheLimitBytes: Int?, sourceStage: SourceStage, storageReservation: WorkflowCaptureStorageReservation?) {
        self.preparationCacheLimitBytes = preparationCacheLimitBytes
        self.storageReservation = storageReservation
        capturedInputs = inputs; self.sourcePaths = sourcePaths; self.sourceDigests = sourceDigests
        self.cachePolicy = cachePolicy; self.reuseManifestSHA256 = reuseManifestSHA256; self.sourceStage = sourceStage
    }
    func removeSourceStage() { sourceStage.remove(); storageReservation?.close() }
    deinit { removeSourceStage() }

    static func capture(graph: DataFlowDocument, bindings: [String: String], labels: [String: String],
                        cachePolicy: WorkflowPreparationCachePolicy, cacheReference: DataFlowCacheReference?,
                        workspace: URL, accountingRoot: URL? = nil, sourcePermissions: URL, projectID: String? = nil, storageLimitBytes: Int64? = nil, preparationCacheLimitBytes: Int? = nil, workload: ExecutionWorkloadPolicy? = nil,
                        checkCancellation: () throws -> Void = { try Task.checkCancellation() }) throws -> Self {
        if let limit = preparationCacheLimitBytes, !ExecutionPreferences.preparationCacheByteBounds(workload: workload).contains(limit) { throw WorkflowArchiveError("Invalid preparation cache allowance.") }
        try checkCancellation()
        let graphBytes = try DataFlowCodec.encodeGraph(graph)
        let sources = Set(graph.nodes.filter { DataFlowSourceOperators.contains($0.operatorID) }.map(\.id))
        guard !graph.nodes.contains(where:{$0.operatorID == DataFlowSourceOperators.fileInput}) else {
            throw WorkflowArchiveError("File Input preparation uses the universal Build canvas, where selected collections are captured with their complete member inventory.")
        }
        let known = Set(graph.nodes.map(\.id))
        guard graphBytes.count <= 2_000_000, Set(bindings.keys) == sources,
              bindings.values.allSatisfy({ !$0.isEmpty && !$0.contains("\0") && $0.count <= 4096 }),
              labels.count <= 128, Set(labels.keys).isSubset(of: known),
              labels.values.allSatisfy({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && $0.count <= 200 }) else {
            throw WorkflowArchiveError("Preparation inputs or diagnostic labels do not match the captured graph.")
        }
        let labelsBytes = try JSONEncoder().encode(labels)
        guard labelsBytes.count <= 65_536 else { throw WorkflowArchiveError("Preparation labels exceed 64 KB.") }
        var inputs: [WorkflowPreviewCapture.Input] = [.bytes(path: "input/graph.json", data: graphBytes),
                                                     .bytes(path: "input/node-labels.json", data: labelsBytes)]
        var reusePin: String?
        if cachePolicy == .reuse {
            guard let cacheReference, WorkflowArchiveCodec.isHash(cacheReference.manifestSHA256) else {
                throw WorkflowArchiveError("No pinned materialization is selected. Recompute and retain preparation first.")
            }
            let source = URL(fileURLWithPath: cacheReference.manifestPath)
            let raw = try DiagnosticFileIO.read(root: source.deletingLastPathComponent(), parts: [source.lastPathComponent], limit: 2_000_000)
            guard WorkflowArchiveCodec.hash(raw) == cacheReference.manifestSHA256 else {
                throw WorkflowArchiveError("The selected preparation cache changed. Recompute before reusing it.")
            }
            let manifest = try JSONDecoder().decode(ReuseManifest.self, from: raw)
            let artifactPolicy = try PreparationArtifactPolicy(manifest: raw)
            guard manifest.schemaVersion == 1, manifest.kind == "dataflowExecution", manifest.status == "completed",
                  WorkflowArchiveCodec.isHash(manifest.graphDigest), (1...artifactPolicy.inventoryCount).contains(manifest.artifacts.count) else {
                throw WorkflowArchiveError("The selected cache does not contain a completed preparation inventory.")
            }
            var names = Set<String>(), total = 0
            for file in manifest.artifacts {
                try checkCancellation()
                let parts = try WorkflowArchiveCodec.pathParts(file.path)
                guard parts.count == 1, file.path != "manifest.json", names.insert(file.path).inserted,
                      (0...min(artifactPolicy.fileBytes(file.path),Int(workload?.limit(.retainedBytes) ?? 50_000_000))).contains(file.bytes), WorkflowArchiveCodec.isHash(file.sha256) else {
                    throw WorkflowFileLeaseError.unsafePath
                }
                let relative = "input/reuse/" + file.path
                _ = try WorkflowPreviewCapture.Purpose.preparation.fileLimit(relative, workload: workload)
                total += file.bytes; guard total <= Int(workload?.limit(.retainedBytes) ?? 200_000_000) else { throw WorkflowArchiveError("The selected reuse inventory exceeds 200 MB.") }
                inputs.append(.file(path: relative, source: source.deletingLastPathComponent().appendingPathComponent(file.path), expectedSHA256: file.sha256))
            }
            // Reuse deliberately allows a different current graph. The worker decides unchanged-node hits.
            inputs.append(.bytes(path: "input/reuse/manifest.json", data: raw)); reusePin = cacheReference.manifestSHA256
        }
        try checkCancellation()
        let access = SourceAccess(directory: sourcePermissions)
        let sourceStageID = UUID()
        var admittedSourceBytes: [String: Int] = [:]
        var storageReservation: WorkflowCaptureStorageReservation?
        if let storageLimitBytes {
            guard let projectID else { throw WorkflowArchiveError("Storage admission requires a project identity.") }
            var sourceBytes = 0
            for original in Set(bindings.values).sorted() {
                try checkCancellation()
                let count = try access.withAccess(to: original) { url in
                    try WorkflowPreviewCapture.byteCountBound(inputs: [
                        .file(path: "input/data/" + url.lastPathComponent, source: url, expectedSHA256: nil)
                    ], purpose: .sourceInspection)
                }
                admittedSourceBytes[original] = count
                sourceBytes += count
            }
            let existingBytes = try WorkflowPreviewCapture.byteCountBound(inputs: inputs, purpose: .preparation, workload: workload)
            // Bindings are rendered later against the final operation directory.
            let captureCeiling = existingBytes + sourceBytes + 1_000_000
            guard captureCeiling <= WorkflowPreviewCapture.Purpose.preparation.captureBytes(workload) else {
                throw WorkflowArchiveError("Preparation inputs exceed their capture allowance.")
            }
            storageReservation = try WorkflowCaptureStorageReservation(workspace: workspace, accountingRoot: accountingRoot, projectID: projectID,
                purpose: .preparation, maximumCaptureBytes: captureCeiling,
                additionalScopes: [try WorkflowStorageAdmission.Scope(namespace: .preparationSourceStaging, components: [sourceStageID.uuidString])],
                additionalBytes: Int64(sourceBytes), limitBytes: storageLimitBytes, workload: workload)
        }
        let stage = try SourceStage(workspace: workspace, stageID: sourceStageID)
        var retained = false
        defer { if !retained { stage.remove(); storageReservation?.close() } }
        var byOriginal: [String: (relative: String, hash: String)] = [:]
        var byNode: [String: String] = [:], digests: [String: String] = [:], captured = Set<String>()
        for node in sources.sorted() {
            try checkCancellation()
            guard let original = bindings[node] else { throw WorkflowFileLeaseError.unsafePath }
            if let prior = byOriginal[original] {
                byNode[node] = prior.relative; digests[original] = prior.hash; continue
            }
            let capturedSource = try access.withAccess(to: original) { url -> (relative: String, hash: String) in
                try checkCancellation()
                let sourceLimit = admittedSourceBytes[original] ?? 10_000_000
                let pending = "incoming-" + UUID().uuidString
                let receipt = try DiagnosticFileIO.copySelectedFile(url, root: stage.directory, parts: [pending], limit: sourceLimit, checkCancellation: checkCancellation)
                try checkCancellation()
                let hash = receipt.sha256
                let relative = "input/sources/" + hash + "/" + url.lastPathComponent
                _ = try WorkflowPreviewCapture.Purpose.preparation.fileLimit(relative, workload: workload)
                if !captured.contains(relative) {
                    let destination = stage.directory.appendingPathComponent(relative)
                    try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
                    try FileManager.default.moveItem(at: stage.directory.appendingPathComponent(pending), to: destination)
                    captured.insert(relative)
                    inputs.append(.file(path: relative, source: stage.directory.appendingPathComponent(relative), expectedSHA256: hash))
                } else { try FileManager.default.removeItem(at: stage.directory.appendingPathComponent(pending)) }
                try checkCancellation()
                return (relative, hash)
            }
            byOriginal[original] = capturedSource; byNode[node] = capturedSource.relative; digests[original] = capturedSource.hash
        }
        try checkCancellation()
        retained = true
        return Self(inputs: inputs, sourcePaths: byNode, sourceDigests: digests,
                    cachePolicy: cachePolicy, reuseManifestSHA256: reusePin, preparationCacheLimitBytes: cachePolicy == .off ? nil : preparationCacheLimitBytes, sourceStage: stage, storageReservation: storageReservation)
    }
    func inputs(in operationURL: URL) throws -> [WorkflowPreviewCapture.Input] {
        var bindings: [String: String] = [:]
        for (node, relative) in sourcePaths { bindings[node] = operationURL.appendingPathComponent(relative).path }
        let bytes = try JSONEncoder().encode(bindings)
        guard bytes.count <= 1_000_000 else { throw WorkflowArchiveError("Captured source bindings exceed 1 MB.") }
        return capturedInputs + [.bytes(path: "input/bindings.json", data: bytes)]
    }
    func arguments(in operationURL: URL) -> [String] {
        func path(_ relative: String) -> String { operationURL.appendingPathComponent(relative).path }
        var result = ["--execute-dataflow", path("input/graph.json"), "--source-bindings", path("input/bindings.json"),
                      "--dataflow-node-labels", path("input/node-labels.json"), "--output", path("artifacts"),
                      "--dataflow-cache-policy", cachePolicy.rawValue]
        if let limit = preparationCacheLimitBytes { result += ["--dataflow-cache-limit-bytes", String(limit)] }
        if let reuseManifestSHA256 {
            result += ["--dataflow-reuse-manifest", path("input/reuse/manifest.json"), "--dataflow-reuse-manifest-sha256", reuseManifestSHA256]
        }
        return result
    }
    private struct ReuseManifest: Decodable {
        let schemaVersion: Int
        let kind: String
        let status: String
        let graphDigest: String
        let artifacts: [WorkflowArchiveFile]
    }
    private final class SourceStage {
        let directory: URL
        private let device: dev_t
        private let inode: ino_t
        init(workspace: URL, stageID: UUID) throws {
            try FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: true)
            guard let resolved = realpath(workspace.path, nil) else { throw WorkflowFileLeaseError.unsafePath }
            let root = URL(fileURLWithPath: String(cString: resolved)); free(resolved)
            let rootFD = try WorkflowEvaluationOwnership.openDirectory(root)
            defer { Darwin.close(rootFD) }
            let namespace = "preparation-source-staging"
            if mkdirat(rootFD, namespace, 0o700) != 0, errno != EEXIST { throw WorkflowFileLeaseError.system(errno) }
            let stageRoot = root.appendingPathComponent(namespace)
            let stageFD = try WorkflowEvaluationOwnership.openDirectory(stageRoot)
            defer { Darwin.close(stageFD) }
            _ = try WorkflowEvaluationOwnership.directoryIdentity(stageFD)
            let name = stageID.uuidString
            try DiagnosticFileIO.createExclusiveDirectory(root: stageRoot, name: name)
            let path = stageRoot.appendingPathComponent(name)
            var metadata = stat()
            guard lstat(path.path, &metadata) == 0, metadata.st_mode & mode_t(S_IFMT) == mode_t(S_IFDIR) else {
                throw WorkflowFileLeaseError.unsafePath
            }
            directory = path; device = metadata.st_dev; inode = metadata.st_ino
        }
        func remove() {
            var metadata = stat()
            guard lstat(directory.path, &metadata) == 0, metadata.st_mode & mode_t(S_IFMT) == mode_t(S_IFDIR),
                  metadata.st_dev == device, metadata.st_ino == inode else { return }
            try? FileManager.default.removeItem(at: directory)
        }
        deinit { remove() }
    }
}
