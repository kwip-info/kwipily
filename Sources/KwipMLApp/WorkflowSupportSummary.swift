import Foundation
import KwipMLCore

/// Allowlisted projection only. Never serialize the archive, raw errors or arbitrary logs.
struct WorkflowSupportSummary: Encodable {
    let schemaVersion = 1
    let operation = "workflowTraining"
    let status: WorkflowExecutionPhase
    let terminalPhase: WorkflowExecutionPhase
    let artifactCount: Int
    let failureCode: WorkflowFailureCode?
    let executionBudget: ResolvedExecutionBudget?
    let sampleCapture: TrainingSampleCapturePolicy?

    init(outcome: WorkflowArchiveOutcome, budget: ResolvedExecutionBudget?,
         sampleCapture: TrainingSampleCapturePolicy?, includeExecutionPolicy: Bool = true) {
        status = outcome.status; terminalPhase = outcome.terminalPhase
        artifactCount = outcome.artifacts.count; failureCode = outcome.failureCode
        executionBudget = includeExecutionPolicy ? budget : nil
        self.sampleCapture = includeExecutionPolicy ? sampleCapture : nil
    }
    func encoded() throws -> Data {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        let bytes = try encoder.encode(self)
        guard bytes.count <= 16_384 else { throw WorkflowArchiveError("Support summary exceeds its size bound.") }
        return bytes
    }
}
