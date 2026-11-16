import Foundation
import KwipMLCore

enum BackendFitOperationToolContract {
    static func fileLimit(path: String, workload: ExecutionWorkloadPolicy? = nil) throws -> Int {
        if path == "input/model-preparation.json" {return PrebuiltPreparationAttachment.maximumBytes}
        if path == "input/model-state.json" {return 32_000_000}
        if path == "input/backend-request.json" { return 100_000 }
        if path == "input/preparation/manifest.json" { return 2_000_000 }
        let parts = try WorkflowArchiveCodec.pathParts(path)
        guard parts.count == 3, parts[0] == "input", parts[1] == "preparation" else { throw WorkflowFileLeaseError.unsafePath }
        if let workload, PreparationArtifactPolicy.isCapacityFileName(parts[2]) {return Int(workload.limit(.retainedBytes))}
        return 50_000_000
    }
    static func validate(arguments: [String], files: [WorkflowArchiveFile], directory: URL?, workload: ExecutionWorkloadPolicy? = nil) throws {
        guard arguments.count == 6, arguments[0] == "--fit-backend", arguments[2] == "--backend-request-sha256", arguments[4] == "--output",
              arguments.allSatisfy({ !$0.contains("\0") && $0.utf8.count <= 4096 }),
              arguments[1].hasPrefix("/") else { throw WorkflowFileLeaseError.unsafePath }
        let root = directory ?? URL(fileURLWithPath: arguments[1]).deletingLastPathComponent().deletingLastPathComponent()
        guard arguments[1] == root.appendingPathComponent("input/backend-request.json").path,
              arguments[5] == root.appendingPathComponent("artifacts").path,
              WorkflowArchiveCodec.isHash(arguments[3]), (2...WorkflowPreviewCapture.Purpose.backendFit.captureCount(workload)).contains(files.count),
              Set(files.map(\.path)).count == files.count else { throw WorkflowFileLeaseError.unsafePath }
        var total = 0
        for file in files {
            guard file.bytes >= 0, file.bytes <= (try fileLimit(path: file.path, workload: workload)), WorkflowArchiveCodec.isHash(file.sha256) else { throw WorkflowFileLeaseError.unsafePath }
            total += file.bytes
            guard total <= WorkflowPreviewCapture.Purpose.backendFit.captureBytes(workload) else { throw WorkflowFileLeaseError.unsafePath }
        }
        guard files.contains(where: { $0.path == "input/backend-request.json" && $0.sha256 == arguments[3] }),
              files.contains(where: { $0.path == "input/preparation/manifest.json" }) else { throw WorkflowFileLeaseError.unsafePath }
    }
    static func verifyInputs(files: [WorkflowArchiveFile], directory: URL, workload: ExecutionWorkloadPolicy? = nil) throws {
        for file in files { try WorkflowPreviewToolContract.verifyFile(file: file, directory: directory, limit: fileLimit(path: file.path, workload: workload)) }
        let bytes = try DiagnosticFileIO.read(root: directory, parts: ["input", "backend-request.json"], limit: 100_000)
        guard let root = try JSONSerialization.jsonObject(with: bytes) as? [String: Any],
              let backend = root["backendID"] as? String, ["sklearn", "tensorflow"].contains(backend),
              let prepared = root["prepared"] as? [String: Any],
              prepared["manifest"] as? String == directory.appendingPathComponent("input/preparation/manifest.json").path,
              let pin = prepared["manifestSHA256"] as? String, let graph = prepared["graphDigest"] as? String,
              files.contains(where: { $0.path == "input/preparation/manifest.json" && $0.sha256 == pin }) else { throw WorkflowFileLeaseError.unsafePath }
        if let model=root["model"] as? [String:Any] {
            guard Set(model.keys)==["path","sha256"] || Set(model.keys)==["path","sha256","preparationPath","preparationSHA256"] else {throw WorkflowFileLeaseError.unsafePath}
            if let path=model["preparationPath"] as? String {
                guard path==directory.appendingPathComponent("input/model-preparation.json").path,files.contains(where:{$0.path=="input/model-preparation.json" && $0.sha256==model["preparationSHA256"] as? String}) else {throw WorkflowFileLeaseError.unsafePath}
            } else if files.contains(where:{$0.path=="input/model-preparation.json"}) {throw WorkflowFileLeaseError.unsafePath}
            guard model["path"] as? String==directory.appendingPathComponent("input/model-state.json").path,files.contains(where:{$0.path=="input/model-state.json" && $0.sha256==model["sha256"] as? String}) else {throw WorkflowFileLeaseError.unsafePath}
        }
        if backend == "sklearn" { _ = try SklearnPreparedWorkerRequest.decode(bytes) }
        else { _ = try TensorFlowPreparedWorkerRequest.decode(bytes) }
        let manifest = try DiagnosticFileIO.read(root: directory, parts: ["input", "preparation", "manifest.json"], limit: 2_000_000)
        try WorkflowCompletedPreparationInventory.verify(data: manifest, files: files, prefix: "input/preparation/", expectedGraphDigest: graph, workload: workload)
        guard let training = prepared["trainingNode"] as? String, let validation = prepared["validationNode"] as? String else { throw WorkflowFileLeaseError.unsafePath }
        _ = try WorkflowCompletedPreparationInventory.verifyCoverage(data: manifest, files: files, prefix: "input/preparation/", directory: directory, selectedNodes: [training, validation])
    }
}
