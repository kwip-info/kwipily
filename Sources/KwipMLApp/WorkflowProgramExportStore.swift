import Foundation
import Combine
import KwipMLCore

/// One atomic draft and one retained review; neither selection nor configuration
/// may change silently underneath an asynchronous capture/publication operation.
@MainActor final class WorkflowProgramExportStore:ObservableObject {
    @Published private(set) var package:WorkflowProgramExportPackage?
    @Published private(set) var isBusy=false
    @Published private(set) var savedURL:URL?
    @Published private(set) var statusMessage="Configure an inference program, then prepare it for review."
    @Published var draft:[String:JSONValue]=ProgramExportConfiguration.defaults
    @Published private(set) var configuration:ProgramExportConfiguration
    private let reference:WorkflowHistoryReference,root:URL,runtime:RuntimeLocation
    private let expected:WorkflowProgramExportIdentity,reservation:WorkflowExecutionReservation
    private let io:any WorkflowProgramExportIO
    private let executionBudget:ResolvedExecutionBudget?
    private let isCurrent:()->Bool
    private var files:ProgramExportFiles?
    private var active=true,cancelled=false
    private var task:Task<Void,Never>?
    init(reference:WorkflowHistoryReference,root:URL,runtime:RuntimeLocation,expected:WorkflowProgramExportIdentity,
         reservation:WorkflowExecutionReservation,io:any WorkflowProgramExportIO,executionBudget:ResolvedExecutionBudget?=nil,isCurrent:@escaping()->Bool) {
        self.reference=reference;self.root=root;self.runtime=runtime;self.expected=expected
        self.reservation=reservation;self.io=io;self.executionBudget=executionBudget;self.isCurrent=isCurrent
        configuration=try! ProgramExportConfiguration(values:ProgramExportConfiguration.defaults)
    }
    var hasDraftChanges:Bool {draft != configuration.values}
    @discardableResult func applyDraft()->Bool {
        guard active,isCurrent() else {return false}
        do {
            let value=try ProgramExportConfiguration(values:draft)
            guard value != configuration else {return true}
            configuration=value;savedURL=nil
            if isBusy {cancel();statusMessage="Configuration changed. Waiting for the owned export operation to stop…"}
            else {clean();statusMessage="Configuration applied. Prepare a new program for review."}
            return true
        } catch {statusMessage=error.localizedDescription;return false}
    }
    func discardDraft() {draft=configuration.values}
    @discardableResult func prepare()->Bool {
        guard active,isCurrent(),!isBusy,!hasDraftChanges else {return false}
        guard let token=reservation.acquire() else {statusMessage="Wait for the current workspace operation.";return false}
        clean();savedURL=nil;cancelled=false;isBusy=true
        let capturedConfiguration=configuration
        statusMessage="Verifying the saved model and capturing an inference program…"
        task=Task {[self] in
            defer {finish(token)}
            do {
                let staged=try await io.stage(reference:reference,root:root,runtime:runtime,expected:expected,configuration:capturedConfiguration,executionBudget:executionBudget)
                files=ProgramExportFiles(staged);package=staged
                guard active,!cancelled,!Task.isCancelled,isCurrent(),configuration == capturedConfiguration,!hasDraftChanges else {throw CancellationError()}
                guard staged.identity == expected,staged.configuration == capturedConfiguration else {throw WorkflowArchiveError("The program differs from the selected saved model or configuration.")}
                statusMessage="Program ready. Review its shapes, runtime and files before choosing a destination."
            } catch {
                clean();statusMessage=error is CancellationError ? "Program preparation cancelled. Prepare again when ready.":"Could not prepare program: "+error.localizedDescription
            }
        }
        return true
    }
    @discardableResult func save(to destination:URL,access:URL?=nil)->Bool {
        guard active,isCurrent(),!isBusy,!hasDraftChanges,let package,package.configuration == configuration else {return false}
        guard let token=reservation.acquire() else {statusMessage="Wait for the current workspace operation.";return false}
        let scoped=access?.startAccessingSecurityScopedResource() ?? false
        cancelled=false;isBusy=true;savedURL=nil;statusMessage="Saving the reviewed inference program…"
        task=Task {[self] in
            defer {
                if scoped {access?.stopAccessingSecurityScopedResource()}
                if cancelled || !active {clean()}
                finish(token)
            }
            do {
                try await io.publish(package,to:destination)
                // A committed rename remains success if cancellation arrives afterward.
                savedURL=destination;statusMessage="Inference program saved. Use its exact Torch version and captured device."
            } catch {statusMessage=error is CancellationError ? "Program publication cancelled.":"Could not save program: "+error.localizedDescription}
        }
        return true
    }
    func cancel() {
        if isBusy {cancelled=true;task?.cancel();statusMessage="Cancelling; waiting for the owned export operation to stop…"}
        else {clean();statusMessage="Program review closed."}
    }
    func invalidate() {active=false;cancel()}
    private func clean() {package=nil;files=nil}
    private func finish(_ token:UUID) {reservation.release(token);task=nil;isBusy=false}
}
private final class ProgramExportFiles {
    let package:WorkflowProgramExportPackage
    init(_ package:WorkflowProgramExportPackage) {self.package=package}
    deinit {package.remove()}
}
