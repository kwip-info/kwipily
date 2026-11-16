import Foundation
import Combine
import KwipMLCore

protocol WorkflowHistoryLoading:Actor {
    func list(root:URL,projectID:String)async throws->WorkflowHistoryListing
    func load(_ reference:WorkflowHistoryReference,root:URL)async throws->WorkflowHistoryLoad
}

/// Bounded verification runs serially on the shared explicit-stack worker. Cancelled
/// queued requests do no I/O; cancellation during a read discards its result.
actor WorkflowHistoryLoader:WorkflowHistoryLoading {
    private let verify:(WorkflowHistoryReference,URL)->WorkflowHistoryLoad
    private var pending:Task<WorkflowHistoryLoad,Error>?
    private var pendingID=UUID()
    init(verify:@escaping (WorkflowHistoryReference,URL)->WorkflowHistoryLoad = {WorkflowHistoryReader.load($0,root:$1)}) {self.verify=verify}

    func list(root:URL,projectID:String)async throws->WorkflowHistoryListing {
        try Task.checkCancellation()
        let result=try WorkflowHistoryReader.list(root:root,projectID:projectID,limit:1000)
        try Task.checkCancellation()
        return result
    }
    func load(_ reference:WorkflowHistoryReference,root:URL)async throws->WorkflowHistoryLoad {
        try Task.checkCancellation()
        let prior=pending,verify=self.verify,id=UUID()
        let work=Task {
            if let prior {_ = try? await prior.value}
            try Task.checkCancellation()
            return try await WorkflowHistoryVerificationWork.run {verify(reference,root)}
        }
        pending=work;pendingID=id
        defer {if pendingID==id {pending=nil}}
        return try await withTaskCancellationHandler(operation:{try await work.value},onCancel:{work.cancel()})
    }
}

/// One project's recorded history. Loading never reads the current external CSV,
/// changes the workflow source, or substitutes live metrics for archived evidence.
@MainActor final class WorkflowHistoryStore:ObservableObject {
    let projectID:String
    let sessionID:UUID
    private let root:URL
    private weak var workspace:WorkspaceStore?
    private weak var host:WorkflowWorkspaceSession?
    private let loader:any WorkflowHistoryLoading
    var preparationLaunch: WorkflowWorkerLauncher = WorkflowExecutionSession.launchWorker
    var preparationExecute: PreparationInspectionExecutor?
    @Published private(set) var listing=WorkflowHistoryListing(entries:[],issues:[],truncated:false)
    @Published private(set) var selectedReference:WorkflowHistoryReference?
    @Published private(set) var selection:WorkflowHistoryLoad?
    @Published private(set) var comparisonBaseline: VerifiedWorkflowOutcome?
    @Published private(set) var trainedExport: WorkflowTrainedExportStore?
    @Published private(set) var programExport: WorkflowProgramExportStore?
    @Published private(set) var preparationInspection: WorkflowPreparationInspectionSession?
    @Published private(set) var isRefreshing=false
    @Published private(set) var isLoading=false
    @Published private(set) var statusMessage="Refresh to read this project's recorded history."
    private var active=true
    private var refreshToken=UUID(),selectionToken=UUID()
    private var refreshTask:Task<Void,Never>?,selectionTask:Task<Void,Never>?

    init(workspace:WorkspaceStore,host:WorkflowWorkspaceSession,loader:any WorkflowHistoryLoading=WorkflowHistoryLoader()) {
        self.workspace=workspace;self.host=host;self.loader=loader
        projectID=host.owner.projectID;sessionID=host.owner.sessionID;root=workspace.workspaceDirectory
    }
    var isActive:Bool {
        guard active,let workspace,let host else {return false}
        return workspace.workflowSession === host && workspace.activeProjectID == projectID
            && host.owner.sessionID == sessionID && !host.isRetired && host.owner.isCurrent(host.owner.lease())
    }
    var modelContent:WorkflowVerifiedModelContent? {
        guard isActive,case .verified(let value)=selection else {return nil}
        return value.modelContent
    }
    func refresh() {
        guard isActive else {rejectInactive();return}
        refreshTask?.cancel();let token=UUID();refreshToken=token
        isRefreshing=true;statusMessage="Reading recorded history…"
        let loader=self.loader,root=self.root,projectID=self.projectID
        refreshTask=Task {[weak self] in
            do {
                let value=try await loader.list(root:root,projectID:projectID)
                guard !Task.isCancelled,let self,self.refreshToken == token else {return}
                self.isRefreshing=false;self.refreshTask=nil
                guard self.isActive else {self.rejectInactive();return}
                self.listing=value
                self.statusMessage=value.truncated ? "History listing is bounded and incomplete; review the scan notice." : "Read \(value.entries.count) recorded history entries. Select one to verify its captured evidence."
            } catch {
                guard let self,self.refreshToken == token else {return}
                self.isRefreshing=false;self.refreshTask=nil
                guard self.isActive else {self.rejectInactive();return}
                self.statusMessage=error is CancellationError ? "History refresh cancelled." : "History could not be read: \(error.localizedDescription)"
            }
        }
    }
    func select(_ reference:WorkflowHistoryReference) {
        guard isActive else {rejectInactive();return}
        guard reference.projectID == projectID else {statusMessage="This history entry belongs to another project.";return}
        programExport?.invalidate();programExport=nil
        trainedExport?.invalidate();trainedExport=nil
        preparationInspection?.cancel();preparationInspection=nil
        selectionTask?.cancel();let token=UUID();selectionToken=token
        selectedReference=reference;selection=nil;isLoading=true;statusMessage="Verifying captured workflow and artifacts…"
        let loader=self.loader,root=self.root
        selectionTask=Task {[weak self] in
            do {
                let value=try await loader.load(reference,root:root)
                guard !Task.isCancelled,let self,self.selectionToken == token else {return}
                self.isLoading=false;self.selectionTask=nil
                guard self.isActive,self.selectedReference == reference else {self.rejectInactive();return}
                // Reader results are immutable values; refreshing the list does not
                // replace a selected verified snapshot or silently reread its files.
                self.selection=value
                switch value {
                case .verified:self.statusMessage="Captured evidence verified. Current local sources were not revalidated."
                case .legacyUnverifiable(_,let reason):self.statusMessage=reason
                case .incomplete(let reason),.unavailable(let reason):self.statusMessage=reason
                }
            } catch {
                guard let self,self.selectionToken == token else {return}
                self.isLoading=false;self.selectionTask=nil
                guard self.isActive else {self.rejectInactive();return}
                self.statusMessage=error is CancellationError ? "History loading cancelled." : "History could not be loaded: \(error.localizedDescription)"
            }
        }
    }
    @discardableResult func pinComparisonBaseline() -> Bool {
        guard isActive, !isLoading, case .verified(let value) = selection,
              value.outcome.status == .completed, value.modelContent != nil else {
            statusMessage = "Select a verified completed model to set the comparison baseline."
            return false
        }
        comparisonBaseline = value
        statusMessage = "Baseline pinned from verified evidence. Select another run to compare."
        return true
    }
    func clearComparisonBaseline() { comparisonBaseline = nil }
    func cancelLoading() {
        programExport?.cancel()
        trainedExport?.cancel()
        preparationInspection?.cancel()
        refreshToken=UUID();selectionToken=UUID()
        refreshTask?.cancel();selectionTask?.cancel();refreshTask=nil;selectionTask=nil
        isRefreshing=false;isLoading=false;statusMessage="History loading cancelled."
    }
    @discardableResult func inspectPreparation() -> Bool {
        guard isActive, let workspace, !workspace.isBusy, !isLoading,
              case .verified(let verified) = selection, verified.preparationCacheReport != nil,
              let pin = verified.outcome.preparationManifestSHA256, let runtime = workspace.runtime else {
            statusMessage = "Select a verified preparation result and wait for the workspace to be idle."
            return false
        }
        let policy: ResolvedExecutionPreferences
        do { policy = try workspace.executionPolicy(for: projectID) }
        catch { statusMessage = error.localizedDescription; return false }
        let reference = verified.reference
        let reservation = WorkflowExecutionReservation(acquire: { [weak self, weak workspace] in
            guard let self, let workspace, self.isActive, self.selectedReference == reference, !workspace.isBusy else { return nil }
            workspace.cancelDataPreparation = { [weak self] in self?.preparationInspection?.cancel() }
            let token = workspace.dataPreparationReservationID
            workspace.isPreparingData = true
            return token
        }, release: { [weak workspace] token in
            guard let workspace, workspace.dataPreparationReservationID == token else { return }
            workspace.isPreparingData = false; workspace.cancelDataPreparation = nil
        })
        let next = WorkflowPreparationInspectionSession(reference: reference, root: root, runtime: runtime,
            reservation: reservation, accountingRoot: workspace.applicationDirectory, storageLimitBytes: policy.storageLimitBytes, executionBudget: policy.budget, launch: preparationLaunch, execute: preparationExecute, previous: preparationInspection?.snapshot,
            accepts: { [weak self] input in
                guard let self, self.isActive, self.selectedReference == reference else { return false }
                return input.runID == verified.outcome.runID && input.requestIdentity == verified.outcome.requestIdentity
                    && input.manifestSHA256 == pin && input.capturedRevision == verified.capture.revision
                    && input.terminalStatus == verified.outcome.status
            })
        preparationInspection = next
        return next.start()
    }
    @discardableResult func prepareProgramExport()->Bool {
        guard let workspace,case .verified(let verified)=selection,let pin=verified.outcome.modelManifestSHA256 else {
            statusMessage="Select a verified completed model before exporting its inference program.";return false
        }
        let selected=WorkflowHistoryProgramExportSelection(reference:verified.reference,identity:.init(runID:verified.outcome.runID,requestIdentity:verified.outcome.requestIdentity,modelManifestSHA256:pin),completed:verified.outcome.status == .completed,hasModel:verified.modelContent != nil,isBag:verified.plan.bagInput != nil || verified.modelContent?.bagContext != nil)
        let policy:ResolvedExecutionPreferences
        do {
            try WorkflowHistoryProgramExportFactory.validate(selected,currentReference:selectedReference,active:isActive,busy:workspace.isBusy,loading:isLoading,capability:workspace.programExportRuntimeCapability,runtimeVersion:workspace.runtimeContractVersion,runtime:workspace.runtime)
            policy=try workspace.executionPolicy(for:projectID)
        } catch {statusMessage=error.localizedDescription;return false}
        guard let runtime=workspace.runtime,let version=workspace.runtimeContractVersion else {return false}
        programExport?.invalidate();trainedExport?.invalidate();trainedExport=nil
        let reference=selected.reference
        let reservation=WorkflowExecutionReservation(acquire:{[weak self,weak workspace] in
            guard let self,let workspace,self.isActive,self.selectedReference == reference,!workspace.isBusy,
                  workspace.programExportRuntimeCapability.isAvailable,workspace.runtimeContractVersion == version,workspace.runtime == runtime else {return nil}
            workspace.cancelDataPreparation={[weak self] in self?.programExport?.cancel()}
            let token=workspace.dataPreparationReservationID;workspace.isPreparingData=true;return token
        },release:{[weak workspace] token in
            guard let workspace,workspace.dataPreparationReservationID == token else {return}
            workspace.isPreparingData=false;workspace.cancelDataPreparation=nil
        })
        let transport=WorkflowProgramExportManagedTransport(expectedTorchVersion:version,accountingRoot:workspace.applicationDirectory,storageLimitBytes:policy.storageLimitBytes)
        let source=WorkflowTrainedExportWriter(accountingRoot:workspace.applicationDirectory,storageLimitBytes:policy.storageLimitBytes)
        programExport=WorkflowProgramExportStore(reference:reference,root:root,runtime:runtime,expected:selected.identity,reservation:reservation,io:WorkflowProgramExportWriter(transport:transport,source:source),executionBudget:policy.budget,isCurrent:{[weak self,weak workspace] in
            guard let self,let workspace,self.isActive,self.selectedReference == reference,
                  workspace.programExportRuntimeCapability.isAvailable,workspace.runtimeContractVersion == version,workspace.runtime == runtime,
                  case .verified(let current)=self.selection else {return false}
            return current.reference == reference && current.outcome.status == .completed && current.modelContent != nil
                && current.outcome.runID == selected.identity.runID && current.outcome.requestIdentity == selected.identity.requestIdentity
                && current.outcome.modelManifestSHA256 == selected.identity.modelManifestSHA256
        })
        statusMessage="Configure the saved inference program, then prepare it for review."
        return true
    }
    func closeProgramExport() {programExport?.invalidate();programExport=nil}
    @discardableResult func prepareTrainedExport() -> Bool {
        guard isActive, let workspace, !workspace.isBusy, !isLoading,
              case .verified(let verified) = selection, verified.outcome.status == .completed,
              verified.modelContent != nil, let pin = verified.outcome.modelManifestSHA256,
              selectedReference == verified.reference else {
            statusMessage = "Select a verified completed model and wait for the workspace to be idle."
            return false
        }
        let policy: ResolvedExecutionPreferences
        do { policy = try workspace.executionPolicy(for: projectID) }
        catch { statusMessage = error.localizedDescription; return false }
        programExport?.invalidate();programExport=nil
        trainedExport?.invalidate()
        let reference = verified.reference
        let reservation = WorkflowExecutionReservation(acquire: { [weak self, weak workspace] in
            guard let self, let workspace, self.isActive, self.selectedReference == reference, !workspace.isBusy else { return nil }
            workspace.cancelDataPreparation = { [weak self] in self?.trainedExport?.cancel() }
            let token = workspace.dataPreparationReservationID
            workspace.isPreparingData = true
            return token
        }, release: { [weak workspace] token in
            guard let workspace, workspace.dataPreparationReservationID == token else { return }
            workspace.isPreparingData = false; workspace.cancelDataPreparation = nil
        })
        let next = WorkflowTrainedExportStore(reference: reference, root: root, runID: verified.outcome.runID,
            requestIdentity: verified.outcome.requestIdentity, modelManifestSHA256: pin, reservation: reservation,
            io: WorkflowTrainedExportWriter(accountingRoot: workspace.applicationDirectory, storageLimitBytes: policy.storageLimitBytes), isCurrent: { [weak self] in
                guard let self, self.isActive, self.selectedReference == reference,
                      case .verified(let selected) = self.selection else { return false }
                return selected.outcome.runID == verified.outcome.runID
                    && selected.outcome.requestIdentity == verified.outcome.requestIdentity
                    && selected.outcome.modelManifestSHA256 == pin && selected.outcome.status == .completed
            })
        trainedExport = next
        return next.prepare()
    }
    func invalidate() {
        programExport?.invalidate();programExport=nil
        trainedExport?.invalidate();trainedExport=nil
        active=false;comparisonBaseline=nil;cancelLoading();preparationInspection=nil;selection=nil;selectedReference=nil
        listing = .init(entries:[],issues:[],truncated:false)
        statusMessage="This history workspace is no longer active."
    }
    private func rejectInactive() {invalidate()}
}
