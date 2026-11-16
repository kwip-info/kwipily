import Foundation
import Darwin
import KwipMLCore

/// Exact private worker request. Destination is always the export owner's package child.
struct WorkflowEvaluationExportRequest: Codable, Equatable {
    var schemaVersion: Int = 1
    var kind: String = "portableEvaluationExport"
    let exportID: String
    let projectID: String
    let evaluationID: String
    let requestIdentity: String
    let evaluationDirectory: String
    let evaluationManifestSHA256: String
    let includeTestRows: Bool
    var executionBudget: ResolvedExecutionBudget? = nil

    func validate() throws {
        try WorkflowWorkerLaunchBudget.validatePolicy(schemaVersion: schemaVersion, executionBudget: executionBudget)
        guard kind == "portableEvaluationExport",
              UUID(uuidString: exportID)?.uuidString == exportID,
              UUID(uuidString: projectID)?.uuidString == projectID,
              UUID(uuidString: evaluationID)?.uuidString == evaluationID,
              WorkflowArchiveCodec.isHash(requestIdentity), WorkflowArchiveCodec.isHash(evaluationManifestSHA256),
              evaluationDirectory.utf8.count <= 4096, evaluationDirectory.hasPrefix("/"),
              !evaluationDirectory.contains("\0") else { throw WorkflowFileLeaseError.unsafePath }
        let components = evaluationDirectory.split(separator: "/")
        guard !components.contains("."), !components.contains(".."),
              Array(components.suffix(3)).map(String.init) == ["workflow-evaluations", projectID, evaluationID],
              let canonical = realpath(evaluationDirectory, nil) else { throw WorkflowFileLeaseError.unsafePath }
        defer { free(canonical) }
        guard String(cString: canonical) == evaluationDirectory else { throw WorkflowFileLeaseError.unsafePath }
        let fd = try WorkflowEvaluationOwnership.openDirectory(URL(fileURLWithPath: evaluationDirectory))
        defer { Darwin.close(fd) }
        _ = try WorkflowEvaluationOwnership.directoryIdentity(fd)
    }
    func encoded() throws -> Data { try validate(); return try WorkflowArchiveCodec.encode(self) }
    func requestSHA256() throws -> String { try WorkflowArchiveCodec.hash(encoded()) }
    static func decode(_ bytes: Data) throws -> Self {
        guard bytes.count <= 256_000,
              let object = try JSONSerialization.jsonObject(with: bytes) as? [String: Any] else {
            throw WorkflowFileLeaseError.unsafePath
        }
        let request = try JSONDecoder().decode(Self.self, from: bytes)
        try WorkflowWorkerLaunchBudget.validatePolicyFields(object, required: ["schemaVersion", "kind", "exportID", "projectID", "evaluationID", "requestIdentity", "evaluationDirectory", "evaluationManifestSHA256", "includeTestRows"], schemaVersion: request.schemaVersion, executionBudget: request.executionBudget)
        try request.validate()
        return request
    }
}
