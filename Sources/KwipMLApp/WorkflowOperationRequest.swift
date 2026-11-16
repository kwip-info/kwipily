import Foundation
import Darwin
import KwipMLCore

/// Immutable launch envelope. Remaining time is captured from the operation's shared clock.
struct WorkflowOperationRequest: Codable, Equatable {
    var schemaVersion: Int = 1
    var kind: String = "workflowOperation"
    let operationID: String
    let projectID: String
    let runID: String
    let directoryID: String
    let requestIdentity: String
    let phase: WorkflowLaunchIntent.Phase
    var arguments: [String]
    var executionBudget: ResolvedExecutionBudget? = nil
    var remainingSeconds: Double? = nil

    func validate(workspace: URL) throws {
        guard schemaVersion == 1, kind == "workflowOperation",
              [operationID, projectID, runID, directoryID].allSatisfy({ UUID(uuidString: $0)?.uuidString == $0 }),
              WorkflowArchiveCodec.isHash(requestIdentity), workspace.isFileURL,
              !workspace.path.contains("\0"), let canonical = realpath(workspace.path, nil) else {
            throw WorkflowFileLeaseError.unsafePath
        }
        defer { free(canonical) }
        guard String(cString: canonical) == workspace.path else { throw WorkflowFileLeaseError.unsafePath }
        switch (executionBudget, remainingSeconds) {
        case (nil, nil): break
        case let (budget?, remaining?):
            guard remaining.isFinite, remaining > 0, remaining <= Double(budget.maximumRunSeconds) else {
                throw WorkflowFileLeaseError.unsafePath
            }
        default: throw WorkflowFileLeaseError.unsafePath
        }
        try WorkflowPhaseArguments.validate(arguments, workspace: workspace, projectID: projectID,
                                            directoryID: directoryID, phase: phase)
    }
    private func validate() throws {
        guard let outputIndex = arguments.firstIndex(of: "--output"), outputIndex + 1 < arguments.count else {
            throw WorkflowFileLeaseError.unsafePath
        }
        let path = arguments[outputIndex + 1]
        guard path.hasPrefix("/"), !path.contains("\0"),
              !path.split(separator: "/").contains("."), !path.split(separator: "/").contains("..") else {
            throw WorkflowFileLeaseError.unsafePath
        }
        var workspace = URL(fileURLWithPath: path)
        for _ in 0..<5 { workspace.deleteLastPathComponent() }
        try validate(workspace: workspace)
    }
    func encoded() throws -> Data { try validate(); return try WorkflowArchiveCodec.encode(self) }
    func requestSHA256() throws -> String { try WorkflowArchiveCodec.hash(encoded()) }
    static func decode(_ bytes: Data) throws -> Self {
        let required: Set<String> = ["schemaVersion", "kind", "operationID", "projectID", "runID", "directoryID", "requestIdentity", "phase", "arguments"]
        guard bytes.count <= 256_000,
              let object = try JSONSerialization.jsonObject(with: bytes) as? [String: Any],
              Set(object.keys) == required || Set(object.keys) == required.union(["executionBudget", "remainingSeconds"]),
              !(object["executionBudget"] is NSNull), !(object["remainingSeconds"] is NSNull) else {
            throw WorkflowFileLeaseError.unsafePath
        }
        let request = try JSONDecoder().decode(Self.self, from: bytes)
        try request.validate()
        if let budget = request.executionBudget {
            guard let original = object["executionBudget"] as? [String: Any],
                  let normalized = try JSONSerialization.jsonObject(with: WorkflowArchiveCodec.encode(budget)) as? [String: Any],
                  Set(original.keys) == Set(normalized.keys) else { throw WorkflowFileLeaseError.unsafePath }
        }
        return request
    }
}
