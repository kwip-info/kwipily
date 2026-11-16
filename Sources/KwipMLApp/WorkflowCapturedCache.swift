import Foundation
import KwipMLCore

/// A pinned managed preparation selection, not a claim that its artifacts match
/// today's runtime or that every unused artifact has been read. The worker remains
/// authoritative for implementation compatibility and each requested cache hit.
struct WorkflowCapturedCache: Equatable {
    let reference: WorkflowArchiveCacheReference
    let manifestURL: URL
    private let workspaceDirectory: URL
    private let receiptSHA256: String

    private init(reference: WorkflowArchiveCacheReference, workspaceDirectory: URL, receiptSHA256: String) {
        self.reference = reference
        self.workspaceDirectory = workspaceDirectory
        self.receiptSHA256 = receiptSHA256
        self.manifestURL = workspaceDirectory.appendingPathComponent("workflow-runs")
            .appendingPathComponent(reference.projectID).appendingPathComponent(reference.directoryID)
            .appendingPathComponent("preparation/artifacts/manifest.json")
    }

    static func capture(reference: WorkflowArchiveCacheReference, workspaceDirectory: URL) throws -> Self {
        let receiptHash = try verify(reference: reference, root: workspaceDirectory)
        return Self(reference: reference, workspaceDirectory: workspaceDirectory, receiptSHA256: receiptHash)
    }

    /// Reopens the same managed evidence before launch. A completed receipt is
    /// immutable too; replacing it with another valid receipt cannot retarget a run.
    func revalidate() throws {
        let current = try Self.verify(reference: reference, root: workspaceDirectory)
        try Self.require(current == receiptSHA256, "Selected cache receipt changed after capture.")
    }

    private struct Manifest: Decodable {
        let schemaVersion: Int
        let kind: String
        let status: String
        let graphDigest: String
        let artifacts: [Artifact]
    }
    private struct Artifact: Decodable {
        let path: String
        let bytes: Int
        let sha256: String
    }

    private static func verify(reference: WorkflowArchiveCacheReference, root: URL) throws -> String {
        // Validate identities before deriving any path. Preserve recorded UUID case.
        try require(UUID(uuidString: reference.projectID) != nil && UUID(uuidString: reference.directoryID) != nil,
                    "Selected cache has an invalid managed project or directory identity.")
        try require(hash(reference.preparationManifestSHA256), "Selected cache requires a lowercase SHA-256 manifest pin.")
        let base = ["workflow-runs", reference.projectID, reference.directoryID]
        let receiptBytes = try DiagnosticFileIO.read(root: root, parts: base + ["outcome.json"], limit: 256_000)
        let receipt = try WorkflowArchiveCodec.decodeOutcome(receiptBytes)
        try require(receipt.projectID == reference.projectID && receipt.directoryID == reference.directoryID,
                    "Selected cache receipt belongs to a different managed run.")
        try require(receipt.preparationManifestSHA256 == reference.preparationManifestSHA256,
                    "Selected cache receipt differs from its preparation manifest pin.")
        let prefix = "preparation/artifacts/"
        guard let manifestEntry = receipt.artifacts.first(where: { $0.path == prefix + "manifest.json" }) else {
            throw WorkflowArchiveError("Selected cache receipt has no preparation manifest inventory.")
        }
        let manifestBytes = try DiagnosticFileIO.read(root: root, parts: base + ["preparation", "artifacts", "manifest.json"], limit: 2_000_000)
        try require(manifestBytes.count == manifestEntry.bytes && WorkflowArchiveCodec.hash(manifestBytes) == reference.preparationManifestSHA256 && manifestEntry.sha256 == reference.preparationManifestSHA256,
                    "Selected cache preparation manifest bytes changed.")
        let manifest = try JSONDecoder().decode(Manifest.self, from: manifestBytes)
        let artifactPolicy = try PreparationArtifactPolicy(manifest: manifestBytes)
        try require(manifest.schemaVersion == 1 && manifest.kind == "dataflowExecution" && manifest.status == "completed" && hash(manifest.graphDigest),
                    "Selected cache requires completed preparation evidence.")
        try require((1...artifactPolicy.inventoryCount).contains(manifest.artifacts.count), "Selected cache artifact inventory exceeds its count bound.")
        var names = Set<String>(), total = 0
        for entry in manifest.artifacts {
            try require(!entry.path.isEmpty && entry.path != "." && entry.path != ".." && !entry.path.contains("/") && !entry.path.contains("\0") && names.insert(entry.path).inserted && hash(entry.sha256) && (0...artifactPolicy.fileBytes(entry.path)).contains(entry.bytes),
                        "Selected cache contains an invalid artifact inventory entry.")
            try require(entry.bytes <= artifactPolicy.totalBytes - total, "Selected cache inventory exceeds its byte budget.")
            total += entry.bytes
        }
        guard let graphEntry = manifest.artifacts.first(where: { $0.path == "dataflow.json" }),
              let archivedGraph = receipt.artifacts.first(where: { $0.path == prefix + "dataflow.json" }) else {
            throw WorkflowArchiveError("Selected cache has no inventoried preparation graph.")
        }
        try require(graphEntry.bytes == archivedGraph.bytes && graphEntry.sha256 == archivedGraph.sha256,
                    "Selected cache graph inventories disagree.")
        let graphBytes = try DiagnosticFileIO.read(root: root, parts: base + ["preparation", "artifacts", "dataflow.json"], limit: 2_000_000)
        try require(graphBytes.count == graphEntry.bytes && WorkflowArchiveCodec.hash(graphBytes) == graphEntry.sha256,
                    "Selected cache preparation graph bytes changed.")
        let graph = try DataFlowCodec.decodeGraph(graphBytes)
        let report = try DataFlowCacheReportCodec.decodeManifest(manifestBytes, graph: graph)
        try require(report.policy == "materialize" || report.policy == "reuse",
                    "Selected preparation did not retain reusable cache evidence. Run with materialize first.")
        return WorkflowArchiveCodec.hash(receiptBytes)
    }
    private static func hash(_ value: String) -> Bool {
        value.utf8.count == 64 && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }
    private static func require(_ condition: Bool, _ message: String) throws {
        guard condition else { throw WorkflowArchiveError(message) }
    }
}
