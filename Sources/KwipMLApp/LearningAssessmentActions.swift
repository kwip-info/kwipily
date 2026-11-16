import Foundation
import Combine
import CryptoKit
import KwipMLCore

/// Executes app-owned fixtures under retained supervision; evidence never grants launch authority.
@MainActor final class LearningAssessmentAdapter: ObservableObject {
    @Published private(set) var runtimeVersions: [String: String] = [:]
    @Published private(set) var error: String?
    @Published private(set) var grade: LearningGrade?
    private(set) var evidence: LearningEvidence?
    @Published private(set) var isRunning = false
    var completion: (() -> Void)?
    typealias ToolLauncher = @MainActor (URL, String, RuntimeLocation, ExecutionBudgetWindow, WorkflowToolRequest.Mode,
        [String: Data], Double, Int64?, @escaping (String) -> Void, @escaping (Int32, String) -> Void) async throws -> WorkflowWorkerHandle
    var launchTool: ToolLauncher?
    var accountingRoot: URL?
    private var compositionRegistry: Data?
    private var stagingTask: Task<Void, Never>?
    private var terminalReceived = false
    private var worker: WorkflowWorkerHandle?
    private var token = UUID()
    private var wasCancelled = false
    private var context: (workspace: URL, projectID: String, window: ExecutionBudgetWindow, storageLimitBytes: Int64?)?
    private struct Event: Decodable { var event: String; var versions: [String: String]?; var evidence: LearningEvidence?; var message: String? }
    private struct Fixture: Encodable { var id: String; var inputs: [String: LearningTensor] }
    private struct Request: Encodable {
        var assessmentID: String; var assessmentVersion: ContractVersion; var fixtureDigest: String
        var runtime: LearningRuntime; var budget: LearningBudget; var fixtures: [Fixture]
    }
    func configureExecution(workspace: URL, projectID: String, budget: ResolvedExecutionBudget, storageLimitBytes: Int64? = nil) throws {
        guard !isRunning, UUID(uuidString: projectID)?.uuidString == projectID else { throw LearningContractError("Choose an idle lesson project before assessment.") }
        let policy = try ResolvedExecutionBudget.resolve(global: .init(maximumRunSeconds: budget.requestedSeconds),
            projectMaximumRunSeconds: budget.source == .project ? budget.requestedSeconds : nil,
            capabilityMaximumSeconds: min(65, budget.capabilityMaximumSeconds ?? 65), resources: budget.resources, provenance: budget.provenance)
        context = (workspace, projectID, try ExecutionBudgetWindow(policy: policy, startedAtUptime: ProcessInfo.processInfo.systemUptime), storageLimitBytes)
    }
    func configureCompositionRegistry(_ data: Data?) throws {
        guard !isRunning else { throw LearningContractError("Finish the current lesson check before changing its captured registry.") }
        compositionRegistry = nil
        if let data {
            guard data.count <= 2_000_000 else { throw LearningContractError("Captured lesson registry exceeds 2 MB.") }
            let registry = try JSONDecoder().decode(ComponentRegistry.self, from: data)
            guard [1,2].contains(registry.schemaVersion) else { throw LearningContractError("Unsupported captured lesson registry.") }
            compositionRegistry = data
        }
    }
    func discoverRuntime(_ runtime: RuntimeLocation) {
        guard !isRunning else { return }
        runtimeVersions = [:]
        launch(runtime: runtime, toolMode: .learningRuntime, inputs: [:], plan: nil, digest: nil)
    }
    func assess(plan: LearningAssessmentPlan, submission: Data, runtime: RuntimeLocation) {
        guard !isRunning else { return }
        do {
            try plan.validate()
            guard submission.count <= 2_000_000 else { throw LearningContractError("Submission exceeds 2 MB.") }
            let request = Request(assessmentID: plan.id, assessmentVersion: plan.assessmentVersion, fixtureDigest: plan.fixtureDigest,
                runtime: plan.runtime, budget: plan.budget, fixtures: plan.fixtures.map { Fixture(id: $0.id, inputs: $0.inputs) })
            let bytes = try JSONEncoder().encode(request)
            let digest = WorkflowArchiveCodec.hash(submission)
            var inputs = ["input/assessment.json": bytes, "input/submission.json": submission]
            if let compositionRegistry { inputs["input/model-registry.json"] = compositionRegistry }
            guard plan.runtime.capabilities["moduleComposition"] == nil || compositionRegistry != nil else {
                throw LearningContractError("Capture the verified registry before checking this composition lesson.")
            }
            launch(runtime: runtime, toolMode: .learningAssessment, inputs: inputs, plan: plan, digest: digest)
        } catch { fail(error) }
    }
    func cancel() { guard isRunning, !wasCancelled else { return }; wasCancelled = true; stagingTask?.cancel(); worker?.cancel() }
    private func launch(runtime: RuntimeLocation, toolMode: WorkflowToolRequest.Mode, inputs: [String: Data], plan: LearningAssessmentPlan?, digest: String?) {
        let generation = UUID(); token = generation; isRunning = true; error = nil; grade = nil; evidence = nil; wasCancelled = false; terminalReceived = false
        var received: LearningEvidence?
        stagingTask = Task { [self] in
          defer { if token == generation { stagingTask = nil } }
          do {
            guard let context else { throw LearningContractError("Assessment requires its captured project execution context.") }
            let accountingRoot = self.accountingRoot
            let execute: ToolLauncher = launchTool ?? { workspace, projectID, runtime, window, mode, inputs, maximum, storage, onLine, onExit in
                try await WorkflowToolExecution.launchAsync(workspace: workspace, accountingRoot: accountingRoot, projectID: projectID, runtime: runtime,
                    window: window, toolMode: mode, inputs: inputs, maximumSeconds: maximum, storageLimitBytes: storage, onLine: onLine, onExit: onExit)
            }
            let handle = try await execute(context.workspace, context.projectID,
                runtime, context.window, toolMode, inputs,
                Double(plan.map { max(10, $0.budget.milliseconds / 1000 + 5) } ?? 10), context.storageLimitBytes,
                 { [weak self] line in
                    guard let self, self.token == generation, !self.terminalReceived, !self.wasCancelled,
                          let event = try? JSONDecoder().decode(Event.self, from: Data(line.utf8)) else { return }
                    if event.event == "learningRuntime", let versions = event.versions { self.runtimeVersions = versions }
                    if event.event == "learningEvidence" { received = event.evidence }
                    if event.event == "error" { self.error = event.message ?? "Assessment worker failed." }
                }, { [self] status, diagnostics in
                    guard self.token == generation, !self.terminalReceived else { return }
                    self.terminalReceived = true
                    self.worker = nil; self.isRunning = false
                    defer { self.completion?() }
                    if self.wasCancelled {
                        self.grade = LearningGrade(status: .cancelled, summary: "Assessment cancelled; no correctness judgment recorded."); self.error = nil; return
                    }
                    if status == 124 {
                        self.grade = LearningGrade(status: .budgetExceeded, summary: "Assessment exceeded its captured execution allowance."); self.evidence = nil; return
                    }
                    do {
                        guard status == 0 else { throw LearningContractError(self.error ?? String(diagnostics.prefix(500))) }
                        if let plan {
                            guard let evidence = received, evidence.submissionDigest == digest else { throw LearningContractError("Worker returned no matching submission evidence.") }
                            self.grade = try LearningAssessor.evaluate(plan, evidence: evidence); self.evidence = evidence
                        } else if self.runtimeVersions.isEmpty { throw LearningContractError("Worker returned no runtime versions.") }
                    } catch {
                        self.error = error.localizedDescription; self.grade = LearningGrade(status: .executionFailed, summary: error.localizedDescription)
                    }
                })
            if token == generation, isRunning { worker = handle; if wasCancelled { handle.cancel() } }
          } catch {
            guard token == generation, !terminalReceived else { return }
            terminalReceived = true
            if wasCancelled {
                self.error = nil; evidence = nil; isRunning = false
                grade = LearningGrade(status: .cancelled, summary: "Assessment cancelled; no correctness judgment recorded.")
                completion?()
            } else { fail(error) }
          }
        }
    }
    private func fail(_ failure: Error) {
        error = failure.localizedDescription; evidence = nil; isRunning = false
        let status: LearningGradeStatus
        if case ExecutionBudgetError.wallTimeLimitExceeded = failure { status = .budgetExceeded } else { status = .executionFailed }
        grade = LearningGrade(status: status, summary: failure.localizedDescription); completion?()
    }
}
