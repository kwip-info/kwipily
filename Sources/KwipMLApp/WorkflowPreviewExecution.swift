import Foundation
import Darwin
import KwipMLCore

/// Preview-specific validation sits behind the shared multi-file execution lifecycle.
@MainActor enum WorkflowPreviewExecution {
    static func launch(workspace: URL, accountingRoot: URL? = nil, projectID: String, runtime: RuntimeLocation,
                       window: ExecutionBudgetWindow, storageLimitBytes: Int64? = nil, inputs: [WorkflowPreviewCapture.Input],
                       arguments: (URL) -> [String],
                       launcher: @escaping WorkflowWorkerLauncher = WorkflowExecutionSession.launchWorker,
                       onLine: @escaping (String) -> Void,
                       onExit: @escaping (Int32, String, URL?) -> Void) async throws -> WorkflowWorkerHandle {
        try await WorkflowCapturedToolExecution.launch(mode: .preview, workspace: workspace, accountingRoot: accountingRoot, projectID: projectID,
            runtime: runtime, window: window, storageLimitBytes: storageLimitBytes, inputs: { _ in inputs }, arguments: arguments,
            launcher: launcher, onLine: onLine, onExit: onExit)
    }
    nonisolated static func verifyResult(directory: URL, request: WorkflowToolRequest, completion: Data,
                                                  window: ExecutionBudgetWindow) throws -> URL {
        func check() throws {
            try Task.checkCancellation()
            _ = try window.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
        }
        try check()
        let output = directory.appendingPathComponent("artifacts")
        let outputFD = try WorkflowEvaluationOwnership.openDirectory(output)
        defer { Darwin.close(outputFD) }
        _ = try WorkflowEvaluationOwnership.directoryIdentity(outputFD)
        let resultData = try WorkflowEvaluationOwnership.readProtocol(output, name: "preview.json", limit: 1_000_000)
        let metadataData = try WorkflowEvaluationOwnership.readProtocol(output, name: "activation-cache.json", limit: 100_000)
        guard let result = try JSONSerialization.jsonObject(with: resultData) as? [String: Any],
              let observed = try JSONSerialization.jsonObject(with: completion) as? [String: Any],
              NSDictionary(dictionary: result).isEqual(to: observed), result["event"] as? String == "previewCompleted",
              result["outputDir"] as? String == output.path,
              result["cachePath"] as? String == output.appendingPathComponent("activation-cache.pt").path,
              let metadata = try JSONSerialization.jsonObject(with: metadataData) as? [String: Any] else {
            throw WorkflowArchiveError("Preview completion does not match its owned saved result.")
        }
        struct Metadata: Decodable { let schemaVersion: Int; let signature: String; let fileHash: String; let inferenceOnly: Bool }
        let decoded = try JSONDecoder().decode(Metadata.self, from: metadataData)
        guard [1, 2].contains(decoded.schemaVersion), decoded.inferenceOnly,
              WorkflowArchiveCodec.isHash(decoded.signature), WorkflowArchiveCodec.isHash(decoded.fileHash),
              WorkflowArchiveCodec.isHash(result["initialWeightsHash"] as? String ?? ""),
              WorkflowArchiveCodec.isHash(result["finalWeightsHash"] as? String ?? ""),
              ["cpu", "mps"].contains(result["device"] as? String ?? ""),
              result["weights"] as? String == (request.arguments.contains("--checkpoint") ? "checkpoint" : "initialized") else {
            throw WorkflowArchiveError("Preview cache metadata is unsupported or differs from its request.")
        }
        for key in ["recipeVersion", "weights", "device", "executedNodeIDs", "reusedNodeIDs", "stopAt", "inputProvenance"] {
            guard let lhs = result[key], let rhs = metadata[key],
                  NSDictionary(dictionary: [key: lhs]).isEqual(to: [key: rhs]) else { throw WorkflowArchiveError("Preview result differs from its cache metadata.") }
        }
        var cacheStat = stat()
        guard fstatat(outputFD, "activation-cache.pt", &cacheStat, AT_SYMLINK_NOFOLLOW) == 0,
              cacheStat.st_size >= 0, cacheStat.st_size <= 32_000_000 else { throw WorkflowFileLeaseError.unsafePath }
        try WorkflowPreviewToolContract.verifyFile(file: .init(path: "activation-cache.pt", bytes: Int(cacheStat.st_size), sha256: decoded.fileHash),
            directory: output, limit: 32_000_000)
        try check()
        let weights = request.inputFiles.filter { $0.path.hasPrefix("input/weights/") }
        if !weights.isEmpty {
            try WorkflowPreviewToolContract.verifyInputs(files: weights, directory: directory)
            try DiagnosticFileIO.createExclusiveDirectory(root: output, name: "weights")
            for file in weights {
                try check()
                try FileManager.default.moveItem(at: directory.appendingPathComponent(file.path),
                    to: output.appendingPathComponent("weights/" + URL(fileURLWithPath: file.path).lastPathComponent))
            }
        }
        try check()
        return output
    }
}
