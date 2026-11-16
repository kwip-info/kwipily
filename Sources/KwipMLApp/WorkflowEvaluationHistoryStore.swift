import Foundation
import Combine
import KwipMLCore

protocol WorkflowEvaluationHistoryLoading:Actor {
    func list(root:URL,projectID:String)async throws->WorkflowEvaluationHistoryListing
    func disclose(_ reference:WorkflowEvaluationHistoryReference,root:URL)async throws->WorkflowEvaluationDisclosure.DisclosedEvaluation
}
actor WorkflowEvaluationHistoryLoader:WorkflowEvaluationHistoryLoading {
    private let exposureRoot: URL?
    init(exposureRoot: URL? = nil) { self.exposureRoot = exposureRoot }
    func list(root:URL,projectID:String)async throws->WorkflowEvaluationHistoryListing {
        try Task.checkCancellation()
        let value=try WorkflowEvaluationHistoryReader.list(root:root,projectID:projectID)
        try Task.checkCancellation();return value
    }
    func disclose(_ reference:WorkflowEvaluationHistoryReference,root:URL)async throws->WorkflowEvaluationDisclosure.DisclosedEvaluation {
        try Task.checkCancellation()
        let value=try WorkflowEvaluationDisclosure.load(reference,root:root,exposureRoot:exposureRoot)
        try Task.checkCancellation();return value
    }
}

/// Project-scoped asynchronous presentation. Only disclosure-authorized values
/// enter observable UI state; listing alone never exposes test metrics.
@MainActor final class WorkflowEvaluationHistoryStore:ObservableObject {
    let projectID:String
    private let root:URL
    private let current:()->Bool
    private let loader:any WorkflowEvaluationHistoryLoading
    private var active=true
    private var refreshID=UUID(),selectionID=UUID()
    private var refreshTask:Task<Void,Never>?,selectionTask:Task<Void,Never>?
    weak var workspace:WorkspaceStore?
    @Published private(set) var export:WorkflowEvaluationExportStore?
    @Published private(set) var listing=WorkflowEvaluationHistoryListing(entries:[],issues:[],truncated:false)
    @Published private(set) var selectedReference:WorkflowEvaluationHistoryReference?
    @Published private var selectedDisclosure:WorkflowEvaluationDisclosure.DisclosedEvaluation?
    @Published private(set) var isLoading=false
    @Published private(set) var isRefreshing=false
    @Published private(set) var statusMessage="Select an evaluation to verify and inspect its saved results."
    var isActive:Bool {active && current()}
    var disclosure:WorkflowEvaluationDisclosure.DisclosedEvaluation? {isActive ? selectedDisclosure:nil}
    init(root:URL,projectID:String,exposureRoot:URL? = nil,isCurrent:@escaping()->Bool,loader:(any WorkflowEvaluationHistoryLoading)?=nil) {
        self.root=root;self.projectID=projectID;current=isCurrent;self.loader=loader ?? WorkflowEvaluationHistoryLoader(exposureRoot:exposureRoot)
    }
    func refresh() {
        guard isActive else {return}
        refreshTask?.cancel();let token=UUID();refreshID=token
        isRefreshing=true
        let root=self.root,project=projectID,loader=self.loader
        refreshTask=Task {[weak self] in
            do {
                let value=try await loader.list(root:root,projectID:project)
                guard let self,self.refreshID==token else {return}
                self.isRefreshing=false;self.refreshTask=nil
                guard !Task.isCancelled,self.isActive else {return}
                self.listing=value
            } catch {
                guard let self,self.refreshID==token else {return}
                self.isRefreshing=false;self.refreshTask=nil
                guard self.isActive,!Task.isCancelled else {return}
                self.statusMessage="Evaluation history could not be read: "+error.localizedDescription
            }
        }
    }
    func select(_ reference:WorkflowEvaluationHistoryReference) {
        guard isActive,reference.projectID==projectID else {return}
        closeExport()
        selectionTask?.cancel();let token=UUID();selectionID=token
        selectedReference=reference;selectedDisclosure=nil;isLoading=true
        statusMessage="Verifying saved evaluation and recording inspection…"
        let root=self.root,loader=self.loader
        selectionTask=Task {[weak self] in
            do {
                let value=try await loader.disclose(reference,root:root)
                guard let self,self.selectionID==token else {return}
                self.isLoading=false;self.selectionTask=nil
                guard !Task.isCancelled,self.isActive,self.selectedReference==reference else {return}
                guard value.outcome.projectID==reference.projectID,value.outcome.evaluationID==reference.evaluationID.uuidString else {
                    self.statusMessage="The selected evaluation changed. Refresh and select it again.";return
                }
                self.selectedDisclosure=value
                self.statusMessage="Verified saved evaluation. Inspection is recorded in local holdout history."
            } catch {
                guard let self,self.selectionID==token else {return}
                self.isLoading=false;self.selectionTask=nil
                guard self.isActive,!Task.isCancelled else {return}
                self.statusMessage="Results remain undisclosed: "+error.localizedDescription
            }
        }
    }
    func cancelLoading() {
        refreshID=UUID();selectionID=UUID()
        refreshTask?.cancel();selectionTask?.cancel();refreshTask=nil;selectionTask=nil
        isLoading=false;isRefreshing=false
        statusMessage="Loading cancelled. Any inspection already recorded remains in history."
    }
    @discardableResult func prepareExport()->Bool {
        guard isActive,!isLoading,let workspace,!workspace.isBusy,let runtime=workspace.runtime,
              let reference=selectedReference,let disclosed=disclosure,
              case .string(let stateDigest)=disclosed.value.manifest.frozenLineage["stateDigest"] else {
            statusMessage="Select a verified evaluation and wait for the workspace and runtime to be ready.";return false
        }
        let policy: ResolvedExecutionPreferences
        do { policy = try workspace.executionPolicy(for: projectID) }
        catch { statusMessage = error.localizedDescription; return false }
        closeExport()
        let manifest=disclosed.value.manifest
        let expected=WorkflowEvaluationExportIdentity(evaluationID:reference.evaluationID.uuidString,evaluationManifestSHA256:disclosed.value.manifestSHA256,
            checkpointSHA256:manifest.parent.checkpointSHA256,stateDigest:stateDigest,snapshotReceiptSHA256:manifest.snapshotReceiptSHA256)
        let reservation=WorkflowExecutionReservation(acquire:{[weak self,weak workspace] in
            guard let self,let workspace,self.isActive,self.selectedReference==reference,!workspace.isBusy else {return nil}
            workspace.cancelDataPreparation={[weak self] in self?.export?.cancel()}
            let token=workspace.dataPreparationReservationID;workspace.isPreparingData=true;return token
        },release:{[weak workspace] token in
            guard let workspace,workspace.dataPreparationReservationID==token else {return}
            workspace.isPreparingData=false;workspace.cancelDataPreparation=nil
        })
        let next=WorkflowEvaluationExportStore(reference:reference,root:root,runtime:runtime,expected:expected,reservation:reservation,
            io:WorkflowEvaluationExportWriter(accountingRoot:workspace.applicationDirectory,storageLimitBytes:policy.storageLimitBytes),executionBudget:policy.budget,partitionTitle:manifest.settings.partitionTitle,isCurrent:{[weak self] in
            guard let self,self.isActive,self.selectedReference==reference,let current=self.disclosure else {return false}
            return current.value.manifestSHA256==expected.evaluationManifestSHA256
        })
        export=next
        return next.prepare()
    }
    func closeExport() {export?.invalidate();export=nil}
    func clearSelection() {closeExport();cancelLoading();selectedReference=nil;selectedDisclosure=nil}
    func invalidate() {
        closeExport()
        active=false;cancelLoading();selectedReference=nil;selectedDisclosure=nil
        listing = .init(entries:[],issues:[],truncated:false)
    }
}
