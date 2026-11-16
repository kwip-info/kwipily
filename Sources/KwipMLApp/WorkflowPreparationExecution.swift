import Foundation
import Darwin
import KwipMLCore

/// Preparation-specific output acceptance; staging, deadlines and ownership are shared with preview.
@MainActor enum WorkflowPreparationExecution {
    static func launch(workspace: URL, projectID: String, runtime: RuntimeLocation,
                       window: ExecutionBudgetWindow, storageAdmission: WorkflowCaptureStorageReservation? = nil, inputs: @escaping (URL) throws -> [WorkflowPreviewCapture.Input],
                       arguments: (URL) -> [String],
                       launcher: @escaping WorkflowWorkerLauncher = WorkflowExecutionSession.launchWorker,
                       onLine: @escaping (String) -> Void,
                       onExit: @escaping (Int32, String, URL?) -> Void) async throws -> WorkflowWorkerHandle {
        try await WorkflowCapturedToolExecution.launch(mode: .preparation, workspace: workspace, projectID: projectID,
            runtime: runtime, window: window, operationID: storageAdmission?.operationID ?? UUID().uuidString,
            storageAdmission: storageAdmission, inputs: inputs, arguments: arguments,
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
        let fd = try WorkflowEvaluationOwnership.openDirectory(output)
        defer { Darwin.close(fd) }
        _ = try WorkflowEvaluationOwnership.directoryIdentity(fd)
        guard let event = try JSONSerialization.jsonObject(with: completion) as? [String: Any],
              event["event"] as? String == "dataflowCompleted", event["outputDir"] as? String == output.path,
              event["manifestPath"] as? String == output.appendingPathComponent("manifest.json").path else {
            throw WorkflowArchiveError("Preparation completion does not identify its owned output directory.")
        }
        let manifestData = try WorkflowEvaluationOwnership.readProtocol(output, name: "manifest.json", limit: 2_000_000)
        struct Manifest: Decodable { let graphDigest: String; let artifacts: [WorkflowArchiveFile] }
        let manifest = try JSONDecoder().decode(Manifest.self, from: manifestData)
        let manifestFile = WorkflowArchiveFile(path: "manifest.json", bytes: manifestData.count, sha256: WorkflowArchiveCodec.hash(manifestData))
        try WorkflowCompletedPreparationInventory.verify(data: manifestData, files: manifest.artifacts + [manifestFile], prefix: "")
        guard event["graphDigest"] as? String == manifest.graphDigest,
              event["artifacts"] as? [String] == manifest.artifacts.map(\.path),
              let graphFile = manifest.artifacts.first(where: { $0.path == "dataflow.json" }), graphFile.bytes <= 2_000_000 else {
            throw WorkflowArchiveError("Preparation completion differs from its complete artifact inventory.")
        }
        guard Set(try FileManager.default.contentsOfDirectory(atPath: output.path)) == Set(manifest.artifacts.map(\.path) + ["manifest.json"]) else {
            throw WorkflowArchiveError("Preparation output contains files outside its completed inventory.")
        }
        let artifactPolicy = try PreparationArtifactPolicy(manifest: manifestData)
        for file in manifest.artifacts {
            try check()
            try WorkflowPreviewToolContract.verifyFile(file: file, directory: output, limit: artifactPolicy.fileBytes(file.path))
        }
        func capturedOption(_ key: String) -> String? {
            guard let index = request.arguments.firstIndex(of: key), index + 1 < request.arguments.count else { return nil }
            return request.arguments[index + 1]
        }
        try WorkflowCompletedPreparationInventory.verifyWorkload(data: manifestData, expected: request.executionBudget.resources?.workloadPolicy)
        let modelInputs=try capturedOption("--dataflow-model-input-nodes").map {try JSONDecoder().decode([String].self,from:Data($0.utf8))} ?? []
        try WorkflowCompletedPreparationInventory.verifyModelInputs(data:manifestData,expected:modelInputs)
        let retention=try JSONDecoder().decode([String:JSONValue].self,from:manifestData)["observationRetention"]
        guard retention == (capturedOption("--dataflow-capture-outputs")=="false" ? .string("requiredOnly"):nil) else {throw WorkflowArchiveError("Preparation retention differs from the accepted request.")}
        if retention != nil {
            let expected=try capturedOption("--dataflow-required-nodes").map {try JSONDecoder().decode([String].self,from:Data($0.utf8))} ?? []
            guard try JSONDecoder().decode([String:JSONValue].self,from:manifestData)["requiredOutputNodes"] == .array(expected.sorted().map(JSONValue.string)) else {throw WorkflowArchiveError("Required preparation outputs differ from the captured selection.")}
        }
        let sampling: JSONValue = try capturedOption("--dataflow-sampling").map { try JSONDecoder().decode(JSONValue.self, from: Data($0.utf8)) } ?? .object(["schemaVersion":.integer(1), "mode":.string("all")])
        let fullNodes = try capturedOption("--dataflow-complete-nodes").map { Set(try JSONDecoder().decode([String].self, from: Data($0.utf8))) } ?? []
        let coverage = try WorkflowCompletedPreparationInventory.verifyCoverage(data: manifestData, files: manifest.artifacts + [manifestFile], prefix: "", directory: output, allowSample: true, expectedSampling: sampling, expectedCompleteNodes: fullNodes)
        if let coverage {
            guard let observed = event["executionCoverage"], try JSONDecoder().decode(JSONValue.self, from: JSONSerialization.data(withJSONObject: observed)) == coverage.value else { throw WorkflowArchiveError("Preparation completion differs from its verified data coverage.") }
        }
        let capturedBytes = try WorkflowEvaluationOwnership.readProtocol(directory.appendingPathComponent("input"), name: "graph.json", limit: 2_000_000)
        let savedBytes = try WorkflowEvaluationOwnership.readProtocol(output, name: "dataflow.json", limit: 2_000_000)
        guard request.inputFiles.contains(.init(path: "input/graph.json", bytes: capturedBytes.count, sha256: WorkflowArchiveCodec.hash(capturedBytes))),
              savedBytes.count == graphFile.bytes, WorkflowArchiveCodec.hash(savedBytes) == graphFile.sha256 else {
            throw WorkflowArchiveError("Preparation graph changed during result verification.")
        }
        let graph = try DataFlowCodec.decodeGraph(capturedBytes)
        guard try DataFlowCodec.decodeGraph(savedBytes) == graph else { throw WorkflowArchiveError("Prepared graph differs from the captured request.") }
        func option(_ key: String) -> String? {
            guard let index = request.arguments.firstIndex(of: key), index + 1 < request.arguments.count else { return nil }
            return request.arguments[index + 1]
        }
        guard let rawPolicy = option("--dataflow-cache-policy"), let policy = DataFlowCachePolicy(rawValue: rawPolicy), let observed = event["cache"] else {
            throw WorkflowArchiveError("Preparation completion omitted its requested cache provenance.")
        }
        let expectation = DataFlowCacheExpectation(policy: policy, sourceManifestSHA256: option("--dataflow-reuse-manifest-sha256"), materializedBytesLimit: option("--dataflow-cache-limit-bytes").flatMap(Int.init), workload: request.executionBudget.resources?.workloadPolicy)
        let report = try DataFlowCacheReportCodec.decodeVerified(JSONSerialization.data(withJSONObject: observed), graph: graph, expectation: expectation)
        _ = try DataFlowCacheReportCodec.decodeManifest(manifestData, graph: graph, expectation: expectation, observedReport: report)
        try check()
        return output
    }
}
