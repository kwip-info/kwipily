import Foundation
import Combine

/// Selection-scoped presentation state. No kernel recovery capability is retained
/// while the user considers an action; every operation acquires its own authority.
@MainActor final class WorkflowExperimentRecoveryStore:ObservableObject {
    struct Scope:Hashable,Sendable {
        let workspace:URL
        let projectID:String
        let sessionID:UUID
        let groupID:String
        var directory:URL {workspace.appendingPathComponent("workflow-experiments/\(projectID)/\(groupID)")}
    }
    enum State:Equatable {
        case idle,checking,inUse,recoverable,recovering,cancelling,recovered,terminal,unavailable(String)
    }
    typealias Probe = (Scope) async throws -> WorkflowExperimentRecoveryCoordinator.Availability
    typealias Recover = (Scope) async throws -> WorkflowExperimentGroupRecord
    @Published private(set) var scope:Scope?
    @Published private(set) var state:State = .idle
    var isBusy:Bool {state == .checking || state == .recovering || state == .cancelling}
    private let probe:Probe
    private let performRecovery:Recover
    private let isCurrent:(Scope)->Bool
    private let onRefresh:(Scope)->Void
    private var operation=UUID()
    private var task:Task<Void,Never>?
    private var active=true

    init(probe:@escaping Probe = WorkflowExperimentRecoveryStore.defaultProbe,
         recover:@escaping Recover = WorkflowExperimentRecoveryStore.defaultRecover,
         isCurrent:@escaping (Scope)->Bool = {_ in true},
         onRefresh:@escaping (Scope)->Void = {_ in}) {
        self.probe=probe;self.performRecovery=recover;self.isCurrent=isCurrent;self.onRefresh=onRefresh
    }
    deinit {task?.cancel()}
    func select(scope:Scope) {
        // SwiftUI can retain this store while the detail disappears. A current
        // selection explicitly reactivates it; a retired host cannot do so.
        guard isCurrent(scope) else {return}
        active=true
        task?.cancel();operation=UUID();task=nil
        self.scope=scope;state = .idle
        refresh()
    }
    func refresh() {
        guard active,let scope,isCurrent(scope),state != .recovering,state != .cancelling else {return}
        startProbe(scope)
    }
    private func startProbe(_ scope:Scope) {
        task?.cancel();let token=UUID();operation=token;state = .checking
        let probe=self.probe
        task=Task {[weak self] in
            do {
                let value=try await probe(scope)
                guard let self,self.accepts(scope,token) else {return}
                self.task=nil
                if Task.isCancelled {self.state = .idle;return}
                self.state=Self.state(value)
            } catch {
                guard let self,self.accepts(scope,token) else {return}
                self.task=nil
                self.state=error is CancellationError || Task.isCancelled ? .idle:.unavailable(error.localizedDescription)
            }
        }
    }
    func recover() {
        guard active,let scope,isCurrent(scope),state == .recoverable else {return}
        let token=UUID();operation=token;state = .recovering
        let recover=performRecovery
        task=Task {[weak self] in
            var failure:Error?
            do {_ = try await recover(scope)} catch {failure=error}
            guard let self,self.accepts(scope,token) else {return}
            self.task=nil
            // A cancelled/erroring operation may have committed the ledger or a
            // revocation. Refresh actual evidence; never advertise a rollback.
            let cancelled=Task.isCancelled || failure is CancellationError
            if cancelled {
                self.state = .idle
                self.onRefresh(scope)
                guard self.accepts(scope,token) else {return}
                self.startProbe(scope)
            } else if let failure {
                self.state = .unavailable(failure.localizedDescription)
                self.onRefresh(scope)
            } else {
                self.state = .recovered
                self.onRefresh(scope)
            }
        }
    }
    func cancel() {
        guard task != nil else {return}
        if state == .recovering || state == .cancelling {state = .cancelling}
        task?.cancel()
    }
    func invalidate() {
        active=false;operation=UUID();task?.cancel();task=nil;state = .idle;scope=nil
    }
    private func accepts(_ scope:Scope,_ token:UUID)->Bool {
        guard active,operation == token,self.scope == scope else {return false}
        guard isCurrent(scope) else {task=nil;state = .idle;return false}
        return true
    }
    private static func state(_ value:WorkflowExperimentRecoveryCoordinator.Availability)->State {
        switch value {
        case .inUse:return .inUse
        case .recoverable:return .recoverable
        case .terminal:return .terminal
        case .unavailable(let reason):return .unavailable(reason)
        }
    }
    nonisolated static func defaultProbe(_ scope:Scope)async throws->WorkflowExperimentRecoveryCoordinator.Availability {
        try await WorkflowHistoryVerificationWork.runCancellable {check in
            try WorkflowExperimentRecoveryCoordinator.probe(directory:scope.directory,workspaceDirectory:scope.workspace,checkCancellation:check)
        }
    }
    nonisolated static func defaultRecover(_ scope:Scope)async throws->WorkflowExperimentGroupRecord {
        try await WorkflowHistoryVerificationWork.runCancellable {check in
            try WorkflowExperimentRecoveryCoordinator.recover(directory:scope.directory,workspaceDirectory:scope.workspace,checkCancellation:check)
        }
    }
}
