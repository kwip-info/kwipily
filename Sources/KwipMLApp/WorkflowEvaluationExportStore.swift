import Foundation
import Combine
import KwipMLCore

/// Review owns its private bytes until staging/publication actually terminates.
/// Changed privacy options or selection can never reuse a prior prepared package.
@MainActor final class WorkflowEvaluationExportStore:ObservableObject {
    @Published private(set) var package:WorkflowEvaluationExportPackage?
    @Published private(set) var includesTestRows=false
    @Published private(set) var isBusy=false
    @Published private(set) var savedURL:URL?
    @Published private(set) var statusMessage="Choose what to include, then prepare a verified package for review."
    let partitionTitle:String
    private let reference:WorkflowEvaluationHistoryReference
    private let root:URL
    private let runtime:RuntimeLocation
    private let expected:WorkflowEvaluationExportIdentity
    private let reservation:WorkflowExecutionReservation
    private let io:any WorkflowEvaluationExportIO
    private let executionBudget: ResolvedExecutionBudget?
    private let isCurrent:()->Bool
    private var files:EvaluationExportFiles?
    private var active=true,cancelled=false
    private var task:Task<Void,Never>?
    init(reference:WorkflowEvaluationHistoryReference,root:URL,runtime:RuntimeLocation,expected:WorkflowEvaluationExportIdentity,
         reservation:WorkflowExecutionReservation,io:any WorkflowEvaluationExportIO=WorkflowEvaluationExportWriter(),executionBudget:ResolvedExecutionBudget?=nil,partitionTitle:String="Test",isCurrent:@escaping()->Bool) {
        self.partitionTitle=partitionTitle;self.reference=reference;self.root=root;self.runtime=runtime;self.expected=expected
        self.reservation=reservation;self.io=io;self.isCurrent=isCurrent;self.executionBudget=executionBudget
    }
    func setIncludesTestRows(_ value:Bool) {
        guard active,value != includesTestRows else {return}
        includesTestRows=value;savedURL=nil
        if isBusy {cancel();statusMessage="Options changed. Waiting for the current export operation to stop…"}
        else {clean();statusMessage="Options changed. Prepare a new package to review."}
    }
    @discardableResult func prepare()->Bool {
        guard active,isCurrent(),!isBusy else {return false}
        guard let token=reservation.acquire() else {statusMessage="Wait for the current workspace operation.";return false}
        clean();savedURL=nil;cancelled=false;isBusy=true
        let includeRows=includesTestRows
        statusMessage="Verifying saved evaluation and preparing export…"
        task=Task {[self] in
            defer {finish(token)}
            do {
                let staged=try await io.stage(reference:reference,root:root,runtime:runtime,expected:expected,includesTestRows:includeRows,executionBudget:executionBudget)
                files=EvaluationExportFiles(package:staged);package=staged
                guard active,!cancelled,!Task.isCancelled,isCurrent(),includesTestRows==includeRows else {throw CancellationError()}
                guard staged.identity==expected,staged.includesTestRows==includeRows else {throw WorkflowArchiveError("Export differs from the selected evaluation or privacy option.")}
                statusMessage="Package ready. Review its contents before choosing a destination."
            } catch {
                clean()
                statusMessage=error is CancellationError ? "Export preparation cancelled. Prepare again when ready.":"Could not prepare export: "+error.localizedDescription
            }
        }
        return true
    }
    @discardableResult func save(to destination:URL,access:URL?=nil)->Bool {
        guard active,isCurrent(),!isBusy,let package,package.includesTestRows==includesTestRows else {return false}
        guard let token=reservation.acquire() else {statusMessage="Wait for the current workspace operation.";return false}
        let scoped=access?.startAccessingSecurityScopedResource() ?? false
        cancelled=false;isBusy=true;savedURL=nil;statusMessage="Saving reviewed evaluation package…"
        task=Task {[self] in
            defer {
                if scoped {access?.stopAccessingSecurityScopedResource()}
                if cancelled || !active {clean()}
                finish(token)
            }
            do {
                try await io.publish(package,to:destination)
                // A committed rename is success even when cancellation arrives afterward.
                savedURL=destination
                statusMessage=package.includesTestRows ? "Evaluation package saved with the selected archived rows.":"Evaluation package saved without source rows or predictions."
            } catch {
                statusMessage=error is CancellationError ? "Export cancelled.":"Could not save export: "+error.localizedDescription
            }
        }
        return true
    }
    func cancel() {
        if isBusy {cancelled=true;task?.cancel();statusMessage="Cancelling; waiting for export to stop…"}
        else {clean();statusMessage="Export review closed."}
    }
    func invalidate() {active=false;cancel()}
    private func clean() {package=nil;files=nil}
    private func finish(_ token:UUID) {reservation.release(token);task=nil;isBusy=false}
}
private final class EvaluationExportFiles {
    let package: WorkflowEvaluationExportPackage
    init(package: WorkflowEvaluationExportPackage) { self.package = package }
    deinit { package.remove() }
}
