import Foundation
import Combine
import KwipMLCore

protocol WorkflowPreparationStaging: Actor {
    func stage(_ reference: WorkflowHistoryReference, root: URL, window: ExecutionBudgetWindow) async throws -> WorkflowPreparationInspectionInput
}
actor WorkflowPreparationStager: WorkflowPreparationStaging {
    let accountingRoot: URL?
    let storageLimitBytes: Int64?
    init(accountingRoot: URL? = nil, storageLimitBytes: Int64? = nil) { self.accountingRoot = accountingRoot; self.storageLimitBytes = storageLimitBytes }
    func stage(_ reference: WorkflowHistoryReference, root: URL, window: ExecutionBudgetWindow) async throws -> WorkflowPreparationInspectionInput {
        func check() throws {
            try Task.checkCancellation()
            _ = try window.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
        }
        try check()
        let accountingRoot=accountingRoot,limit=storageLimitBytes
        let input = try await WorkflowHistoryVerificationWork.run(discard: {$0.remove()}) {
            try WorkflowHistoryReader.stagePreparationInspection(reference, root: root, accountingRoot: accountingRoot, storageLimitBytes: limit, workload: window.policy.resources?.workloadPolicy, checkCancellation: check)
        }
        do { try check(); return input }
        catch { input.remove(); throw error }
    }
}

/// One accepted operation survives view/session replacement until its worker exits.
/// Cancellation is sticky; only its own reservation and private copy are released.
@MainActor final class WorkflowPreparationInspectionSession: ObservableObject {
    let reference: WorkflowHistoryReference
    @Published private(set) var snapshot: WorkflowPreparationInspectionSnapshot?
    @Published private(set) var isBusy = false
    @Published private(set) var statusMessage = "Inspect saved preparation to load its archived rows and state."
    private let executionBudget: ResolvedExecutionBudget?
    private let root: URL, runtime: RuntimeLocation
    private let reservation: WorkflowExecutionReservation
    private let stager: any WorkflowPreparationStaging
    private let execute: PreparationInspectionExecutor?
    private let launch: WorkflowWorkerLauncher
    private let accepts: (WorkflowPreparationInspectionInput) -> Bool
    private var token: UUID?
    private var started = false, terminal = false, cancelled = false
    private var failure: String?
    private var task: Task<Void, Never>?
    private var worker: WorkflowWorkerHandle?
    private var input: WorkflowPreparationInspectionInput?
    private var retained: WorkflowPreparationInspectionSession?

    init(reference: WorkflowHistoryReference, root: URL, runtime: RuntimeLocation,
         reservation: WorkflowExecutionReservation, accountingRoot: URL? = nil, storageLimitBytes: Int64? = nil, executionBudget: ResolvedExecutionBudget? = nil, stager: (any WorkflowPreparationStaging)? = nil,
         launch: @escaping WorkflowWorkerLauncher = WorkflowExecutionSession.launchWorker,
         execute: PreparationInspectionExecutor? = nil,
         previous: WorkflowPreparationInspectionSnapshot? = nil,
         accepts: @escaping (WorkflowPreparationInspectionInput) -> Bool = { _ in true }) {
        self.reference = reference; self.root = root; self.runtime = runtime
        self.reservation = reservation; self.executionBudget = executionBudget
        self.stager = stager ?? WorkflowPreparationStager(accountingRoot: accountingRoot, storageLimitBytes: storageLimitBytes)
        self.accepts = accepts; self.execute = execute; self.launch = launch
        self.snapshot = previous?.input.reference == reference ? previous : nil
    }
    @discardableResult func start() -> Bool {
        guard !started else { return false }
        guard let token = reservation.acquire() else { statusMessage = "Wait for the current workspace operation before inspecting saved preparation."; return false }
        let window: ExecutionBudgetWindow
        do {
            let policy = try executionBudget ?? ResolvedExecutionBudget.resolve(global: .factory, projectMaximumRunSeconds: nil)
            window = try ExecutionBudgetWindow(policy: policy, startedAtUptime: ProcessInfo.processInfo.systemUptime)
        } catch {
            reservation.release(token); statusMessage = error.localizedDescription; return false
        }
        started = true; self.token = token; retained = self; isBusy = true
        statusMessage = "Verifying and staging saved preparation; current sources are not read…"
        task = Task { [self] in
            defer { task = nil }
            do {
                let captured = try await stager.stage(reference, root: root, window: window)
                input = captured
                guard !cancelled, !Task.isCancelled else { finish(code: -1, stderr: "", result: nil); return }
                guard captured.reference == reference, accepts(captured) else { throw WorkflowArchiveError("Inspection staging returned another history entry.") }
                _ = try window.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
                statusMessage = "Reading verified historical tables and fitted state…"
                let completion: (Int32, String, PreparationInspectionResult?) -> Void = { [self] code, stderr, result in
                    finish(code: code, stderr: stderr, result: result)
                }
                let handle: WorkflowWorkerHandle
                if let execute {
                    handle = try await execute(root, reference.projectID, runtime, window,
                        captured.directory.appendingPathComponent("manifest.json"), captured.manifestSHA256, completion)
                } else {
                    handle = try await PreparationInspectionExecution.launch(workspace: root, projectID: reference.projectID,
                        runtime: runtime, window: window, manifestURL: captured.directory.appendingPathComponent("manifest.json"),
                        manifestSHA256: captured.manifestSHA256, storageAdmission: captured.storageReservation,
                        launcher: launch, onExit: completion)
                }
                // Test and process adapters may report a terminal exit synchronously.
                if !terminal {
                    worker = handle
                    if cancelled || failure != nil { handle.cancel() }
                }
            } catch {
                if !cancelled { failure = error.localizedDescription }
                finish(code: -1, stderr: "", result: nil)
            }
        }
        return true
    }
    func cancel() {
        guard isBusy, !cancelled else { return }
        cancelled = true
        statusMessage = "Cancelling saved preparation inspection…"
        task?.cancel(); worker?.cancel()
    }
    private func finish(code: Int32, stderr: String, result: PreparationInspectionResult?) {
        guard !terminal else { return }
        terminal = true; worker = nil
        defer {
            input?.remove()
            input = nil
            if let token { reservation.release(token) }
            token = nil; isBusy = false; retained = nil
        }
        if cancelled || code == 130 { statusMessage = "Inspection cancelled; no new outputs accepted."; return }
        if let failure { statusMessage = "Saved preparation was not loaded: " + failure; return }
        if code == 124 {
            statusMessage = "Saved-output inspection exceeded its time allowance. Reduce the archive size or adjust the execution budget."
            return
        }
        do {
            guard let input, input.reference == reference, accepts(input) else {
                throw WorkflowArchiveError("Selected history changed before inspection completed; no new outputs accepted.")
            }
            guard code == 0, let result, result.cacheReport == input.cacheReport,
                  Set(result.outputs.keys) == Set(input.nodes.map(\.address)) else {
                throw WorkflowArchiveError("Inspection did not return a complete verified saved-output result.")
            }
            snapshot = .init(input: input, outputs: result.outputs, cacheReport: result.cacheReport)
            statusMessage = "Historical preparation loaded. Current sources and cache reuse eligibility were not checked."
        } catch {
            statusMessage = "Saved preparation was not loaded: " + error.localizedDescription
            if !stderr.isEmpty { statusMessage += " " + String(stderr.prefix(1000)) }
        }
    }
}
