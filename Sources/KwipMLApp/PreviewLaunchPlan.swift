import Foundation
import KwipMLCore

/// A private, captured inference recipe. Build off-main while the workspace is reserved.
struct PreviewLaunchPlan {
    let inputs: [WorkflowPreviewCapture.Input]
    private let settings: ExecutionSettings
    private let selected: String
    private let resume: Bool
    private let csvName: String?
    let hasCheckpoint: Bool

    static func capture(graph: GraphDocument, settings: ExecutionSettings, selected: String, resume: Bool,
                        cache: String?, checkpoint: String?, hashes: [String: String], sourcePermissions: URL) throws -> Self {
        try Task.checkCancellation()
        var inputs: [WorkflowPreviewCapture.Input] = [.bytes(path: "input/graph.json", data: try JSONEncoder().encode(graph))]
        let data = try ModelDataLaunchInputs.capture(settings: settings, sourcePermissions: sourcePermissions)
        inputs += data.inputs
        let csvName = data.csvName
        func retained(_ source: String, path: String) throws {
            guard let hash = hashes[source], WorkflowArchiveCodec.isHash(hash) else {
                throw WorkflowArchiveError("Preview inputs have no retained integrity pin. Preview the flow again.")
            }
            inputs.append(.file(path: path, source: URL(fileURLWithPath: source), expectedSHA256: hash))
        }
        if resume {
            guard let cache else { throw WorkflowArchiveError("No activation cache is available. Preview the flow first.") }
            try retained(cache, path: "input/cache/activation-cache.pt")
            try retained(URL(fileURLWithPath: cache).deletingPathExtension().appendingPathExtension("json").path, path: "input/cache/activation-cache.json")
        }
        if let checkpoint {
            let source = URL(fileURLWithPath: checkpoint), parent = source.deletingLastPathComponent()
            if resume {
                try retained(checkpoint, path: "input/weights/checkpoint.pt")
                try retained(parent.appendingPathComponent("graph.json").path, path: "input/weights/graph.json")
            } else {
                let raw = try DiagnosticFileIO.read(root: parent, parts: ["manifest.json"], limit: 20_000_000)
                let graphBytes = try DiagnosticFileIO.read(root: parent, parts: ["graph.json"], limit: 2_000_000)
                guard let manifest = try JSONSerialization.jsonObject(with: raw) as? [String: Any], manifest["status"] as? String == "completed",
                      let checkpointInfo = manifest["checkpoint"] as? [String: Any], checkpointInfo["schemaVersion"] as? Int == 1,
                      let expected = checkpointInfo["sha256"] as? String, WorkflowArchiveCodec.isHash(expected),
                      let expectedSize = checkpointInfo["bytes"] as? Int, (0...100_000_000).contains(expectedSize),
                      (try source.resourceValues(forKeys: [.fileSizeKey])).fileSize == expectedSize,
                      let metricsHash = checkpointInfo["metricsSHA256"] as? String, WorkflowArchiveCodec.isHash(metricsHash),
                      let graphObject = manifest["graph"],
                      try JSONDecoder().decode(GraphDocument.self, from: graphBytes) == JSONDecoder().decode(GraphDocument.self, from: JSONSerialization.data(withJSONObject: graphObject)),
                      WorkflowArchiveCodec.hash(try DiagnosticFileIO.read(root: parent, parts: ["metrics.json"], limit: 4_000_000)) == metricsHash else {
                    throw WorkflowArchiveError("Completed checkpoint evidence changed. Run again to create verified inference weights.")
                }
                inputs.append(.file(path: "input/weights/checkpoint.pt", source: source, expectedSHA256: expected))
                inputs.append(.bytes(path: "input/weights/graph.json", data: graphBytes))
            }
        }
        try Task.checkCancellation()
        return Self(inputs: inputs, settings: settings, selected: selected, resume: resume, csvName: csvName, hasCheckpoint: checkpoint != nil)
    }
    func arguments(in directory: URL) -> [String] {
        func path(_ suffix: String) -> String { directory.appendingPathComponent(suffix).path }
        var args = ["--graph", path("input/graph.json"), "--output", path("artifacts"), "--device", settings.device, "--seed", String(settings.seed)]
        args += ModelDataLaunchInputs.arguments(settings: settings, csvName: csvName, directory: directory)
        if hasCheckpoint { args += ["--checkpoint", path("input/weights/checkpoint.pt")] }
        args += resume ? ["--resume-from", selected, "--cache", path("input/cache/activation-cache.pt")] : ["--preview", "--stop-at", selected]
        return args
    }
}

/// Shared captured data selection for preview and standalone training.
enum ModelDataLaunchInputs {
    static func capture(settings: ExecutionSettings, sourcePermissions: URL,
                        checkCancellation: () throws -> Void = { try Task.checkCancellation() }) throws -> (inputs: [WorkflowPreviewCapture.Input], csvName: String?) {
        try checkCancellation()
        var inputs: [WorkflowPreviewCapture.Input] = []
        var csvName: String?
        if let preparation = settings.preparedTraining {
            try preparation.validate()
            let manifestURL = URL(fileURLWithPath: preparation.manifestPath)
            let bytes = try DiagnosticFileIO.read(root: manifestURL.deletingLastPathComponent(), parts: [manifestURL.lastPathComponent], limit: 2_000_000)
            guard WorkflowArchiveCodec.hash(bytes) == preparation.manifestSHA256,
                  let object = try JSONSerialization.jsonObject(with: bytes) as? [String: Any],
                  object["status"] as? String == "completed", object["kind"] as? String == "dataflowExecution",
                  object["graphDigest"] as? String == preparation.graphDigest,
                  let artifacts = object["artifacts"] as? [[String: Any]], (1...300).contains(artifacts.count) else {
                throw WorkflowArchiveError("The selected preparation changed or is incomplete. Prepare data again.")
            }
            inputs.append(.bytes(path: "input/preparation/manifest.json", data: bytes))
            var names = Set<String>(), total = 0
            for artifact in artifacts {
                try checkCancellation()
                guard let name = artifact["path"] as? String, name != "manifest.json", !name.contains("/"),
                      names.insert(name).inserted, let size = artifact["bytes"] as? Int, (0...50_000_000).contains(size),
                      let hash = artifact["sha256"] as? String, WorkflowArchiveCodec.isHash(hash) else { throw WorkflowFileLeaseError.unsafePath }
                _ = try WorkflowPreviewToolContract.fileLimit(path: "input/preparation/" + name)
                total += size; guard total <= 200_000_000 else { throw WorkflowFileLeaseError.unsafePath }
                inputs.append(.file(path: "input/preparation/" + name, source: manifestURL.deletingLastPathComponent().appendingPathComponent(name), expectedSHA256: hash))
            }
        } else if let csv = settings.datasetCSV {
            guard settings.featureColumns.count == 2, settings.targetColumn != nil else {
                throw WorkflowArchiveError("Map two feature columns and one target before running.")
            }
            // Keep the selected source scope open until its bounded bytes are captured.
            let access = SourceAccess(directory: sourcePermissions)
            let captured = try access.withAccess(to: csv) { url -> (String, Data) in
                (url.lastPathComponent, try DiagnosticFileIO.readSelectedFile(url, limit: 10_000_000))
            }
            csvName = captured.0
            inputs.append(.bytes(path: "input/data/" + captured.0, data: captured.1))
        }
        try checkCancellation()
        return (inputs, csvName)
    }
    static func arguments(settings: ExecutionSettings, csvName: String?, directory: URL) -> [String] {
        func path(_ suffix: String) -> String { directory.appendingPathComponent(suffix).path }
        func json(_ columns: [String]) -> String { String(decoding: try! JSONEncoder().encode(columns), as: UTF8.self) }
        var args: [String] = []
        if let p = settings.preparedTraining {
            args += ["--prepared-dataflow", path("input/preparation/manifest.json"), "--prepared-manifest-sha256", p.manifestSHA256,
                     "--prepared-graph-digest", p.graphDigest, "--prepared-training-node", p.trainingNode, "--prepared-training-port", p.trainingPort,
                     "--prepared-validation-node", p.validationNode, "--prepared-validation-port", p.validationPort,
                     "--feature-columns-json", json(p.featureColumns), "--target-column", p.targetColumn]
        } else {
            args += ["--sample-limit", String(settings.sampleLimit), "--split-seed", String(settings.splitSeed), "--validation-percent", String(settings.validationPercent)]
            if let csvName, let target = settings.targetColumn {
                args += ["--dataset-csv", path("input/data/" + csvName), "--feature-columns-json", json(settings.featureColumns), "--target-column", target]
            }
        }
        return args
    }
}
