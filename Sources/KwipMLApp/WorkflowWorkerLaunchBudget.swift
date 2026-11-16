import Foundation
import Darwin
import KwipMLCore

/// Post-staging allowance, separately pinned so staging consumes the operation budget.
struct WorkflowWorkerLaunchBudget: Codable, Equatable {
    let schemaVersion: Int
    let requestSHA256: String
    let remainingSeconds: Double

    static func validatePolicy(schemaVersion: Int, executionBudget: ResolvedExecutionBudget?) throws {
        guard (schemaVersion == 1 && executionBudget == nil) || (schemaVersion == 2 && executionBudget != nil) else {
            throw WorkflowFileLeaseError.unsafePath
        }
    }
    static func validatePolicyFields(_ object: [String: Any], required: Set<String>,
                                     schemaVersion: Int, executionBudget: ResolvedExecutionBudget?) throws {
        try validatePolicy(schemaVersion: schemaVersion, executionBudget: executionBudget)
        guard Set(object.keys) == (schemaVersion == 2 ? required.union(["executionBudget"]) : required),
              !(object["executionBudget"] is NSNull) else { throw WorkflowFileLeaseError.unsafePath }
        if let budget = executionBudget {
            guard let original = object["executionBudget"] as? [String: Any],
                  let normalized = try JSONSerialization.jsonObject(with: WorkflowArchiveCodec.encode(budget)) as? [String: Any],
                  Set(original.keys) == Set(normalized.keys) else { throw WorkflowFileLeaseError.unsafePath }
        }
    }
    private func validate(requestSHA256: String, budget: ResolvedExecutionBudget) throws {
        guard schemaVersion == 1, WorkflowArchiveCodec.isHash(self.requestSHA256), self.requestSHA256 == requestSHA256,
              remainingSeconds.isFinite, remainingSeconds > 0,
              remainingSeconds <= Double(budget.maximumRunSeconds) else { throw WorkflowFileLeaseError.unsafePath }
    }
    static func decode(_ data: Data, requestSHA256: String, budget: ResolvedExecutionBudget) throws -> Self {
        guard data.count <= 16_384,
              let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              Set(object.keys) == ["schemaVersion", "requestSHA256", "remainingSeconds"] else { throw WorkflowFileLeaseError.unsafePath }
        let value = try JSONDecoder().decode(Self.self, from: data)
        try value.validate(requestSHA256: requestSHA256, budget: budget)
        return value
    }
    static func publish(directory: URL, requestSHA256: String, budget: ResolvedExecutionBudget,
                        remainingSeconds: Double) throws -> String {
        let value = Self(schemaVersion: 1, requestSHA256: requestSHA256, remainingSeconds: remainingSeconds)
        try value.validate(requestSHA256: requestSHA256, budget: budget)
        let data = try WorkflowArchiveCodec.encode(value)
        try DiagnosticFileIO.writeImmutable(data, root: directory, parts: ["launch-budget.json"], limit: 16_384)
        return WorkflowArchiveCodec.hash(data)
    }
    static func verify(directory: URL, requestSHA256: String, budget: ResolvedExecutionBudget?, expectedSHA256: String?) throws {
        if let budget {
            guard let expectedSHA256 else { throw WorkflowFileLeaseError.unsafePath }
            let data = try WorkflowEvaluationOwnership.readProtocol(directory, name: "launch-budget.json", limit: 16_384)
            guard WorkflowArchiveCodec.hash(data) == expectedSHA256 else { throw WorkflowFileLeaseError.unsafePath }
            _ = try decode(data, requestSHA256: requestSHA256, budget: budget)
        } else {
            guard expectedSHA256 == nil else { throw WorkflowFileLeaseError.unsafePath }
            let fd = try WorkflowEvaluationOwnership.openDirectory(directory)
            defer { Darwin.close(fd) }
            var info = stat()
            guard fstatat(fd, "launch-budget.json", &info, AT_SYMLINK_NOFOLLOW) != 0 else { throw WorkflowFileLeaseError.unsafePath }
            guard errno == ENOENT else { throw WorkflowFileLeaseError.system(errno) }
        }
    }
}
