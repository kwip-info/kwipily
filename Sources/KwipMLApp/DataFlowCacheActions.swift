import Foundation
import CryptoKit
import KwipMLCore

typealias DataFlowCachePolicy = WorkflowPreparationCachePolicy
struct DataFlowCacheReference: Codable, Equatable {
    var manifestPath: String
    var manifestSHA256: String
}
struct DataFlowCacheReport: Codable, Equatable {
    struct Decision: Codable, Equatable {
        var nodeID: String
        var status: String
        var reason: String
        var cacheKey: String? = nil
    }
    var policy: String
    var executedNodeIDs: [String]
    var reusedNodeIDs: [String]
    var nodes: [Decision]
    // Optional for legacy saved projections and existing initializer compatibility.
    // The verified codec separately requires the complete worker report schema.
    var schemaVersion: Int? = nil
    var implementationDigest: String? = nil
    var sourceManifestSHA256: String? = nil
    var materializedBytesLimit: Int? = nil
    var materializedBytes: Int? = nil
}
private struct DataFlowCachePreferences: Codable {
    var schemaVersion = 1
    var policy: DataFlowCachePolicy
    var reference: DataFlowCacheReference?
}

extension DataFlowWorkspaceStore {
    private var cachePreferencesURL: URL { saveURL.deletingLastPathComponent().appendingPathComponent("cache-selection.json") }
    func setCachePolicy(_ policy: DataFlowCachePolicy) {
        guard requirePreparationOperation() else {return}
        guard !isBusy else { return }
        cachePolicy = policy
        persistCachePreferences()
    }
    func restoreCachePreferences() {
        guard requirePreparationOperation() else {return}
        guard FileManager.default.fileExists(atPath: cachePreferencesURL.path) else { return }
        do {
            let value = try JSONDecoder().decode(DataFlowCachePreferences.self, from: WorkspaceDocumentReader.read(cachePreferencesURL))
            guard value.schemaVersion == 1 else { throw DataFlowContractError("Unsupported cache preference version.") }
            cachePolicy = value.policy; lastCacheReference = value.reference
        } catch { statusMessage = "Cache selection could not be restored: \(error.localizedDescription). Choose Recompute to create a new materialization." }
    }
    func persistCachePreferences() {
        guard requirePreparationOperation() else {return}
        guard projectID != nil, projectID == workspace.activeProjectID else { return }
        do {
            try FileManager.default.createDirectory(at: cachePreferencesURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try JSONEncoder().encode(DataFlowCachePreferences(policy: cachePolicy, reference: lastCacheReference)).write(to: cachePreferencesURL, options: .atomic)
        } catch { statusMessage = "Cache preference save failed: \(error.localizedDescription)" }
    }
    func preparationCacheArguments() throws -> [String] {
        try requirePreparationAccess()
        var arguments = ["--dataflow-cache-policy", cachePolicy.rawValue]
        if cachePolicy == .reuse {
            guard let reference = lastCacheReference else { throw DataFlowContractError("No materialized preparation is selected. Choose Recompute first.") }
            let data = try WorkspaceDocumentReader.read(URL(fileURLWithPath: reference.manifestPath))
            guard Self.cacheDigest(data) == reference.manifestSHA256 else { throw DataFlowContractError("The selected cache manifest changed. Choose Recompute; modified cache data will not be reused.") }
            arguments += ["--dataflow-reuse-manifest", reference.manifestPath, "--dataflow-reuse-manifest-sha256", reference.manifestSHA256]
        }
        return arguments
    }
    func acceptCacheReport(manifestURL: URL, report: DataFlowCacheReport?) throws {
        try requirePreparationAccess()
        guard let report else { throw DataFlowContractError("Preparation worker omitted cache provenance.") }
        let data = try WorkspaceDocumentReader.read(manifestURL)
        guard let manifest = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              manifest["status"] as? String == "completed", manifest["kind"] as? String == "dataflowExecution",
              let stored = manifest["cache"] as? [String: Any], stored["policy"] as? String == report.policy else {
            throw DataFlowContractError("Preparation did not publish matching cache provenance.")
        }
        cacheReport = report
        if report.policy == "materialize" || report.policy == "reuse" {
            lastCacheReference = DataFlowCacheReference(manifestPath: manifestURL.path, manifestSHA256: Self.cacheDigest(data))
            persistCachePreferences()
        }
    }
    var cacheSummary: String {
        guard let report = cacheReport else { return "Sources are always read again. Reuse validates the selected artifacts and recomputes changed descendants." }
        if restoredResult { return "Historical execution: \(report.executedNodeIDs.count) computed · \(report.reusedNodeIDs.count) reused. Current sources and cache reuse were not checked." }
        return "Last preparation: \(report.executedNodeIDs.count) computed · \(report.reusedNodeIDs.count) reused · sources checked"
    }
    static func cacheDigest(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
}
