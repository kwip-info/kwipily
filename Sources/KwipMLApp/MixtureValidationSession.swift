import Foundation
import Combine
import KwipMLCore

/// Synthetic, cancellable worker check. Never runs the surrounding workflow or an optimizer.
@MainActor final class MixtureValidationSession:ObservableObject {
    typealias Environment=GameRuleValidationSession.Environment
    @Published private(set) var isBusy=false
    @Published private(set) var message="Not validated."
    @Published private(set) var validated:[String:JSONValue]?
    private var worker:WorkflowWorkerHandle?,task:Task<Void,Never>?,token:UUID?
    private var cancelled=false
    private let environment:()->Environment?,acquire:(@escaping()->Void)->UUID?,release:(UUID)->Void
    init(environment:@escaping()->Environment?,acquire:@escaping(@escaping()->Void)->UUID?,release:@escaping(UUID)->Void){self.environment=environment;self.acquire=acquire;self.release=release}
    func run(_ values:[String:JSONValue]) {
        guard !isBusy,let env=environment(),let token=acquire({[weak self] in self?.cancel()}) else{message="Wait for the current workspace operation.";return}
        self.token=token;isBusy=true;cancelled=false;validated=nil;message="Checking synthetic CPU values and gradients…"
        task=Task {[self] in
            do {
                let bytes=try WorkflowArchiveCodec.encode(values)
                let window=try ExecutionBudgetWindow(policy:env.policy.budget,startedAtUptime:ProcessInfo.processInfo.systemUptime)
                var qualified=false,detail=""
                worker=try await WorkflowToolExecution.launchAsync(workspace:env.root,accountingRoot:env.accounting,projectID:env.projectID,runtime:env.runtime,window:window,toolMode:.mixtureValidation,inputs:["input/mixture.json":bytes],maximumSeconds:30,storageLimitBytes:env.policy.storageLimitBytes,onLine:{line in
                    guard let raw=line.data(using:.utf8),let event=try? JSONDecoder().decode([String:JSONValue].self,from:raw) else{return}
                    if event["event"] == .string("error"){detail=event["message"]?.stringValue ?? "Validation failed."}
                    if event["event"] == .string("mixtureValidated"),event["device"] == .string("cpu"),event["optimizerUpdates"]==0{qualified=true;detail="CPU synthetic values/gradients verified · \(event["parameters"]?.integerValue ?? 0) resident parameters · no optimizer updates."}
                },onExit:{[self] code,error in
                    if cancelled{message="Validation cancelled."}else if code==0 && qualified{validated=values;message=detail}else{message=detail.isEmpty ? (error.isEmpty ? "Validation failed.":error):detail}
                    finish()
                })
                if cancelled{worker?.cancel()}
            }catch{message=cancelled ? "Validation cancelled.":error.localizedDescription;finish()}
        }
    }
    private func finish(){isBusy=false;worker=nil;task=nil;if let token{release(token)};token=nil}
    func cancel(){guard isBusy else{return};cancelled=true;validated=nil;message="Cancelling validation…";if let worker{worker.cancel()}else{task?.cancel()}}
}
