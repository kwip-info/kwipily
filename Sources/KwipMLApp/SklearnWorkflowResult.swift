import Foundation
import KwipMLCore

struct SklearnWorkflowResult {
    struct Evaluation: Codable, Equatable { let count: Int; let accuracy: Double; let logLoss: Double }
    let evaluation: Evaluation
    let provenance: [String: JSONValue]
    let artifacts: [WorkflowArchiveFile]
    let manifestSHA256: String
    private struct Manifest: Decodable {
        let schemaVersion: Int; let backendID: String; let runtimeVersion: String; let device: String; let task: String
        let stateFormat: String; let configuration: [String: JSONValue]; let provenance: [String: JSONValue]
        let evaluation: Evaluation; let artifacts: [WorkflowArchiveFile]
    }
    static func verify(directory: URL, request: SklearnPreparedWorkerRequest) throws -> Self {
        try request.validate()
        if PrebuiltModelConfiguration.isPrebuilt(request.configuration) {return try PrebuiltWorkflowResult.verify(directory:directory,request:request)}
        if DocumentClassificationConfiguration.isDocument(request.configuration) {
            return try DocumentClassificationWorkflowResult.verify(directory:directory,request:request)
        }
        let bytes = try DiagnosticFileIO.read(root: directory, parts: ["manifest.json"], limit: 2_000_000)
        let manifest = try JSONDecoder().decode(Manifest.self, from: bytes)
        let canonical = try JSONDecoder().decode(SklearnPreparedWorkerRequest.self, from: request.encoded())
        guard manifest.schemaVersion == 1, manifest.backendID == "sklearn", manifest.runtimeVersion == "1.7.2", manifest.device == "cpu",
              manifest.task == "binaryClassification", manifest.stateFormat == "kwipml.logistic-regression.v1",
              manifest.configuration == canonical.configuration, (1...100000).contains(manifest.evaluation.count),
              manifest.evaluation.accuracy.isFinite, (0...1).contains(manifest.evaluation.accuracy),
              manifest.evaluation.logLoss.isFinite, manifest.evaluation.logLoss >= 0 else { throw WorkflowArchiveError("Sklearn result differs from the captured estimator.") }
        let p = manifest.provenance
        guard case .object(let preparation) = p["preparation"],
              preparation["manifestSHA256"] == .string(request.prepared.manifestSHA256), preparation["graphDigest"] == .string(request.prepared.graphDigest),
              preparation["trainingNode"] == .string(request.prepared.trainingNode), preparation["trainingPort"] == .string(request.prepared.trainingPort),
              preparation["validationNode"] == .string(request.prepared.validationNode), preparation["validationPort"] == .string(request.prepared.validationPort),
              p["featureColumns"] == .array(request.prepared.featureColumns.map(JSONValue.string)), p["targetColumn"] == .string(request.prepared.targetColumn),
              case .array(let train) = p["trainingRowIDs"], case .array(let validation) = p["validationRowIDs"],
              !train.isEmpty, train.count <= 100000, validation.count == manifest.evaluation.count,
              p["trainingSamples"] == .integer(Int64(train.count)), p["validationSamples"] == .integer(Int64(validation.count)),
              train.allSatisfy(rowID), validation.allSatisfy(rowID), Set(train).count == train.count, Set(validation).count == validation.count,
              Set(train).isDisjoint(with: Set(validation)) else { throw WorkflowArchiveError("Sklearn partition provenance differs from captured preparation.") }
        let paths = Set(manifest.artifacts.map(\.path))
        guard (7...512).contains(paths.count), paths.count == manifest.artifacts.count,
              Set(["model-state.json", "preparation/manifest.json", "portable/state.json", "portable/inference.py", "portable/sklearn_backend.py", "portable/manifest.json"]).isSubset(of: paths),
              paths.allSatisfy({ $0 == "model-state.json" || $0.hasPrefix("preparation/") || $0.hasPrefix("portable/") }) else { throw WorkflowFileLeaseError.unsafePath }
        var total = 0
        for file in manifest.artifacts {
            guard file.bytes >= 0, file.bytes <= 50_000_000 else { throw WorkflowFileLeaseError.unsafePath }
            total += file.bytes; guard total <= 202_100_000 else { throw WorkflowFileLeaseError.unsafePath }
            try WorkflowPreviewToolContract.verifyFile(file: file, directory: directory, limit: 50_000_000)
        }
        guard manifest.artifacts.first(where: { $0.path == "preparation/manifest.json" })?.sha256 == request.prepared.manifestSHA256,
              manifest.artifacts.first(where: { $0.path == "model-state.json" })?.sha256 == manifest.artifacts.first(where: { $0.path == "portable/state.json" })?.sha256 else { throw WorkflowArchiveError("Sklearn exported state or preparation does not match the run.") }
        try verifyPredictions(directory: directory, request: request, manifest: JSONDecoder().decode([String: JSONValue].self, from: bytes))
        return .init(evaluation: manifest.evaluation, provenance: p, artifacts: manifest.artifacts, manifestSHA256: WorkflowArchiveCodec.hash(bytes))
    }
    private static func verifyPredictions(directory: URL, request: SklearnPreparedWorkerRequest, manifest: [String: JSONValue]) throws {
        guard request.capturePredictions == true else {
            guard manifest["predictions"] == nil else { throw WorkflowArchiveError("Sklearn result captured predictions despite the disabled policy.") }; return
        }
        guard case .object(let sample) = manifest["predictions"], Set(sample.keys) == ["rowIDs", "classes", "probabilities", "decisionScores"],
              case .array(let rowIDs) = sample["rowIDs"], case .array(let classes) = sample["classes"],
              case .array(let scores) = sample["decisionScores"], case .array(let probabilities) = sample["probabilities"],
              case .object(let provenance) = manifest["provenance"], case .array(let validationIDs) = provenance["validationRowIDs"],
              rowIDs == Array(validationIDs.prefix(8)), classes.count == rowIDs.count, scores.count == rowIDs.count, probabilities.count == rowIDs.count else { throw WorkflowArchiveError("Sklearn prediction sample differs from the captured validation prefix.") }
        let stateBytes = try DiagnosticFileIO.read(root: directory, parts: ["model-state.json"], limit: 100_000)
        let state = try JSONDecoder().decode([String: JSONValue].self, from: stateBytes)
        guard case .array(let weights) = state["coefficients"], weights.count == 2, let w0 = weights[0].doubleValue, let w1 = weights[1].doubleValue,
              let intercept = state["intercept"]?.doubleValue, [w0,w1,intercept].allSatisfy(\.isFinite) else { throw WorkflowArchiveError("Invalid fitted logistic coefficients.") }
        let preparedBytes = try DiagnosticFileIO.read(root: directory, parts: ["preparation", "manifest.json"], limit: 2_000_000)
        let preparation = try JSONDecoder().decode([String: JSONValue].self, from: preparedBytes)
        guard case .array(let artifacts) = preparation["artifacts"], let entry = artifacts.first(where: { value in
            if case .object(let file) = value { return file["nodeID"] == .string(request.prepared.validationNode) && file["port"] == .string(request.prepared.validationPort) }; return false
        }), case .object(let file) = entry, let path = file["path"]?.stringValue else { throw WorkflowFileLeaseError.unsafePath }
        let table = try JSONDecoder().decode([String: JSONValue].self, from: DiagnosticFileIO.read(root: directory, parts: ["preparation"] + WorkflowArchiveCodec.pathParts(path), limit: 50_000_000))
        guard case .array(let columns) = table["columns"], case .array(let rows) = table["rows"], rows.count >= rowIDs.count else { throw WorkflowFileLeaseError.unsafePath }
        let names = columns.map { value -> String? in if case .object(let column) = value { return column["name"]?.stringValue }; return nil }
        guard let first = names.firstIndex(of: request.prepared.featureColumns[0]), let second = names.firstIndex(of: request.prepared.featureColumns[1]) else { throw WorkflowFileLeaseError.unsafePath }
        func close(_ actual: Double, _ expected: Double) -> Bool { actual.isFinite && abs(actual-expected) <= 1e-9 * max(1,abs(expected)) }
        for index in rowIDs.indices {
            guard case .array(let row) = rows[index], row.indices.contains(first), row.indices.contains(second),
                  let x0 = row[first].doubleValue, let x1 = row[second].doubleValue, let actual = scores[index].doubleValue,
                  case .integer(let classification) = classes[index], case .array(let pair) = probabilities[index], pair.count == 2,
                  let p0 = pair[0].doubleValue, let p1 = pair[1].doubleValue else { throw WorkflowFileLeaseError.unsafePath }
            let expected = Double(Float(x0))*w0 + Double(Float(x1))*w1 + intercept
            let probability = expected >= 0 ? 1/(1+exp(-expected)) : exp(expected)/(1+exp(expected))
            guard expected.isFinite, close(actual,expected), classification == (expected > 0 ? 1 : 0), close(p0,1-probability), close(p1,probability) else { throw WorkflowArchiveError("Sklearn prediction values differ from independent affine/sigmoid evaluation.") }
        }
    }
    private static func rowID(_ value: JSONValue) -> Bool { if case .string(let id) = value { return !id.isEmpty && id.count <= 200 }; return false }
    static func validateCompletion(_ data: Data, directory: URL, request: SklearnPreparedWorkerRequest) throws -> Self {
        guard data.count <= 1_000_000, var event = try JSONSerialization.jsonObject(with: data) as? [String: Any], event.removeValue(forKey: "event") as? String == "backendCompleted" else { throw WorkflowFileLeaseError.unsafePath }
        let manifest = try DiagnosticFileIO.read(root: directory, parts: ["manifest.json"], limit: 2_000_000)
        guard NSDictionary(dictionary: event).isEqual(to: try JSONSerialization.jsonObject(with: manifest) as? [String: Any] ?? [:]) else { throw WorkflowArchiveError("Sklearn completion differs from its manifest.") }
        return try verify(directory: directory, request: request)
    }
}

@MainActor enum BackendFitWorkflowExecution {
    nonisolated static func verifyResult(directory: URL, request: WorkflowToolRequest, completion: Data, window: ExecutionBudgetWindow) throws -> URL {
        try Task.checkCancellation(); _ = try window.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
        try request.validate(directory: directory, verifyInputs: true)
        let bytes = try DiagnosticFileIO.read(root: directory, parts: ["input", "backend-request.json"], limit: 100_000)
        let output = directory.appendingPathComponent("artifacts")
        let object = try JSONSerialization.jsonObject(with: bytes) as? [String: Any]
        if object?["backendID"] as? String == "sklearn" {
            _ = try SklearnWorkflowResult.validateCompletion(completion, directory: output, request: SklearnPreparedWorkerRequest.decode(bytes))
        } else if object?["backendID"] as? String == "tensorflow" {
            _ = try TensorFlowWorkflowResult.validateCompletion(completion, directory: output, request: TensorFlowPreparedWorkerRequest.decode(bytes))
        } else { throw WorkflowArchiveError("Unsupported backend completion.") }
        try Task.checkCancellation(); return output
    }
}
