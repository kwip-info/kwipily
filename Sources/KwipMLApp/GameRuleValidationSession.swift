import Foundation
import Combine
import KwipMLCore

/// A disposable, source-free preparation job; never runs the surrounding workflow or its writes.
@MainActor final class GameRuleValidationSession:ObservableObject {
    struct Environment {let root:URL;let accounting:URL;let projectID:String;let runtime:RuntimeLocation;let policy:ResolvedExecutionPreferences}
    @Published private(set) var isBusy=false
    @Published private(set) var message="Rules have not been validated."
    @Published private(set) var validated:[String:JSONValue]?
    @Published private(set) var lastTested:[String:JSONValue]?
    private var worker:WorkflowWorkerHandle?
    private var task:Task<Void,Never>?
    private var token:UUID?
    private var generation=UUID()
    private var cancelRequested=false
    private let environment:()->Environment?
    private let acquire:(@escaping()->Void)->UUID?
    private let release:(UUID)->Void
    init(environment:@escaping()->Environment?,acquire:@escaping(@escaping()->Void)->UUID?,release:@escaping(UUID)->Void){self.environment=environment;self.acquire=acquire;self.release=release}
    func run(_ values:[String:JSONValue]) {
        guard !isBusy,let env=environment(),let reservation=acquire({[weak self] in self?.cancel()}) else {message="Wait for the current workspace operation.";return}
        token=reservation;cancelRequested=false;isBusy=true;lastTested=values;validated=nil;message="Validating definition and running its tests…";let id=UUID();generation=id
        task=Task { [self] in
            var plan:PreparationLaunchPlan?
            do {
                try GameRuleConfiguration.validate(values)
                let graph=DataFlowDocument(nodes:[.init(id:"rules",operatorID:GameRuleConfiguration.operatorID,config:values)])
                let window=try ExecutionBudgetWindow(policy:env.policy.budget,startedAtUptime:ProcessInfo.processInfo.systemUptime)
                let captured=try await Task.detached(priority:.userInitiated) {
                    try PreparationLaunchPlan.capture(graph:graph,bindings:[:],labels:["rules":"Game rule validation"],cachePolicy:.off,cacheReference:nil,workspace:env.root,accountingRoot:env.accounting,sourcePermissions:env.root,projectID:env.projectID,storageLimitBytes:env.policy.storageLimitBytes,workload:env.policy.budget.resources?.workloadPolicy)
                }.value
                plan=captured
                try Task.checkCancellation()
                var summary:[Any]?
                var workerError:String?
                worker=try await WorkflowPreparationExecution.launch(workspace:env.root,projectID:env.projectID,runtime:env.runtime,window:window,storageAdmission:captured.storageReservation,inputs:{try captured.inputs(in:$0)},arguments:captured.arguments,onLine:{line in
                    guard let data=line.data(using:.utf8),let event=try? JSONSerialization.jsonObject(with:data) as? [String:Any] else{return}
                    if event["event"] as? String=="error" {workerError=event["message"] as? String ?? event["error"] as? String;return}
                    guard event["event"] as? String=="dataflowPreview",let outputs=event["outputs"] as? [String:[String:Any]],let preview=outputs["summary"]?["preview"] as? [String:Any],let rows=preview["rows"] as? [[Any]] else{return};summary=rows.first
                },onExit:{[self] code,detail,directory in
                    defer {
                        captured.removeSourceStage()
                        if let directory {try? FileManager.default.removeItem(at:directory.deletingLastPathComponent())}
                    }
                    guard self.generation==id else{return}
                    if self.cancelRequested {self.message="Rule validation cancelled."}
                    else if code==0,let summary,summary.count==4,let passed=summary[1] as? Int,let failed=summary[2] as? Int {
                        self.message="\(passed) passed · \(failed) failed. Run Flow for complete test and move-trace tables."
                        // A semantically valid definition can contain a failing authored test;
                        // keep it editable, but don't allow this validation to approve Apply.
                        if failed==0 {self.validated=values}
                    } else {self.message=workerError ?? (detail.isEmpty ? "Rule validation failed; inspect the expressions and types.":detail)}
                    self.finish()
                })
            } catch {plan?.removeSourceStage();if generation==id {message=cancelRequested ? "Rule validation cancelled.":error.localizedDescription;finish()}}
        }
    }
    private func finish(){isBusy=false;worker=nil;task=nil;if let token {release(token)};token=nil}
    func cancel(){guard isBusy else{return};cancelRequested=true;message="Cancelling rule validation…";if let worker {worker.cancel()} else {task?.cancel()}}
}
