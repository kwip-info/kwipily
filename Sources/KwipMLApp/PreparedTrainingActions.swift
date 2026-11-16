import Foundation
import CryptoKit
import KwipMLCore

extension WorkspaceStore {
    func verifiedPreparationManifest(_ selection: PreparedTrainingSelection) throws -> [String: Any] {
        try selection.validate()
        let url = URL(fileURLWithPath: selection.manifestPath)
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey])
        guard values.isRegularFile == true, values.isSymbolicLink != true, (values.fileSize ?? Int.max) <= 2_000_000 else { throw preparationError("Preparation manifest is unavailable or exceeds its size limit. Prepare the data again.") }
        let handle = try FileHandle(forReadingFrom: url); defer { try? handle.close() }
        let data = try handle.read(upToCount: 2_000_001) ?? Data()
        let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        guard data.count <= 2_000_000, hash == selection.manifestSHA256,
              let manifest = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              manifest["status"] as? String == "completed", manifest["kind"] as? String == "dataflowExecution",
              manifest["graphDigest"] as? String == selection.graphDigest else {
            throw preparationError("The selected preparation changed or is incomplete. Prepare data and choose its outputs again.")
        }
        return manifest
    }
    func usePreparedTraining(_ selection: PreparedTrainingSelection) throws {
        guard !isBusy else { throw preparationError("Stop the current model run before changing its data.") }
        _ = try verifiedPreparationManifest(selection)
        setExecution { value in
            value.preparedTraining = selection
            value.datasetCSV = nil; value.featureColumns = []; value.targetColumn = nil
        }
        statusMessage = "Prepared partitions selected. Training uses this frozen preparation without re-splitting."
    }
    func trainingDataArguments() throws -> [String] { try trainingDataArguments(settings: execution) }
    func trainingDataArguments(settings execution: ExecutionSettings) throws -> [String] {
        if let selected = execution.preparedTraining {
            _ = try verifiedPreparationManifest(selected)
            return ["--prepared-dataflow", selected.manifestPath,
                    "--prepared-manifest-sha256", selected.manifestSHA256, "--prepared-graph-digest", selected.graphDigest,
                    "--prepared-training-node", selected.trainingNode, "--prepared-training-port", selected.trainingPort,
                    "--prepared-validation-node", selected.validationNode, "--prepared-validation-port", selected.validationPort,
                    "--feature-columns-json", try encodedFeatureColumns(selected.featureColumns), "--target-column", selected.targetColumn]
        }
        var arguments = ["--sample-limit", String(execution.sampleLimit), "--split-seed", String(execution.splitSeed), "--validation-percent", String(execution.validationPercent)]
        if let csv = execution.datasetCSV {
            guard execution.featureColumns.count == 2, let target = execution.targetColumn else { throw preparationError("Map two feature columns and one target in Data before running.") }
            arguments += ["--dataset-csv", try workerSourcePath(csv), "--feature-columns-json", try encodedFeatureColumns(execution.featureColumns), "--target-column", target]
        }
        return arguments
    }
    func encodedFeatureColumns(_ names: [String]) throws -> String {
        String(decoding: try JSONEncoder().encode(names), as: UTF8.self)
    }
    private func preparationError(_ message: String) -> NSError { NSError(domain: "KwipML.Preparation", code: 1, userInfo: [NSLocalizedDescriptionKey: message]) }
}
