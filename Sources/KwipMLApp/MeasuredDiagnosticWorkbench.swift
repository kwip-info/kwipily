import SwiftUI
import AppKit
import UniformTypeIdentifiers
import KwipMLCore

/// A local re-evaluation of verified historical evidence. This never overwrites
/// archived diagnostics and never executes an experiment on the user's behalf.
struct MeasuredDiagnosticWorkbench:View {
    let verified:VerifiedWorkflowOutcome
    @ObservedObject var host:WorkflowWorkspaceSession
    @ObservedObject var workspace:WorkspaceStore
    let inspectSource:()->Void
    let inspectPreparation:()->Void
    @StateObject private var suppressions:MeasuredDiagnosticSuppressionStore
    @State private var input:MeasuredDiagnosticInput?
    @State private var report:MeasuredDiagnosticReport?
    @State private var message:String?
    @State private var loading=true
    @State private var suppressionTarget:MeasuredDiagnosticFinding?
    @State private var reason=""
    @State private var expiryDays=1
    @State private var proposal:MeasuredDiagnosticRepair?
    @State private var proposedBindings:UInt64?
    @State private var repairRevision=0
    init(verified:VerifiedWorkflowOutcome,host:WorkflowWorkspaceSession,workspace:WorkspaceStore,inspectSource:@escaping()->Void,inspectPreparation:@escaping()->Void) {
        self.verified=verified;self.host=host;self.workspace=workspace
        self.inspectSource=inspectSource;self.inspectPreparation=inspectPreparation
        _suppressions=StateObject(wrappedValue:MeasuredDiagnosticSuppressionStore(root:workspace.workspaceDirectory,projectID:verified.capture.projectID))
    }
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            HStack {
                Text("Measured diagnostics").font(.headline)
                Spacer()
                if let report {Button("Export redacted summary…") {export(report)}.buttonStyle(FlatButtonStyle())}
            }
            Text("Current evaluation of this verified capture · measured rules v2. Findings are observations and hypotheses, not proof of a cause or of model health.")
                .font(.caption).foregroundStyle(MLTheme.muted)
            if loading {ProgressView("Evaluating captured measurements…").controlSize(.small)}
            if let message {Text(message).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled)}
            if let error=suppressions.error {Text("Annotations unavailable: "+error).font(.caption).foregroundStyle(MLTheme.rose)}
            if let report {
                TimelineView(.periodic(from:.now,by:60)) {context in
                    VStack(alignment:.leading,spacing:10) {
                        ForEach(report.findings) {finding in
                            if let rule=(try? MeasuredDiagnostics.rules(forVersion:report.registryVersion))?.first(where:{$0.id==finding.ruleID}) {
                                if let item=suppressions.entries.first(where:{$0.applies(to:report,finding:finding,at:context.date)}) {
                                    ResultsSection(title:"Suppressed · "+rule.title,symbol:"eye.slash") {
                                        Text(item.reason).font(.caption).textSelection(.enabled)
                                        Text("Expires "+item.expiresAt.formatted()).font(.caption2).foregroundStyle(MLTheme.muted)
                                        Button("Restore finding") {perform{try suppressions.remove(item.id)}}.buttonStyle(FlatButtonStyle())
                                    }
                                } else {findingCard(finding,rule:rule)}
                            }
                        }
                    }
                }
                if report.findings.isEmpty {Text("No findings from the available captured evidence. Review withheld checks below.").font(.caption).foregroundStyle(MLTheme.muted)}
                if let proposal {repairPanel(proposal).id(repairRevision)}
                coverage(report)
            }
        }
        .sheet(item:$suppressionTarget) {finding in suppressionSheet(finding)}
        .task(id:verified.reference) {
            loading=true;report=nil;input=nil;message=nil
            let captured=verified
            let task=Task.detached(priority:.userInitiated) { ()throws -> (MeasuredDiagnosticInput,MeasuredDiagnosticReport) in
                try Task.checkCancellation()
                let input=try MeasuredDiagnosticInput.from(captured)
                let report=try MeasuredDiagnostics.evaluateCurrent(input, context:MeasuredDiagnosticInput.generalizationContext(from:captured))
                try Task.checkCancellation();return(input,report)
            }
            do {
                let result=try await withTaskCancellationHandler(operation:{try await task.value},onCancel:{task.cancel()})
                guard !Task.isCancelled else {return}
                input=result.0;report=result.1;loading=false
            } catch {
                guard !Task.isCancelled else {return};loading=false;message="Evaluation unavailable: "+error.localizedDescription
            }
        }
    }
    private func findingCard(_ finding:MeasuredDiagnosticFinding,rule:MeasuredDiagnosticRule)->some View {
        VStack(alignment:.leading,spacing:8) {
            Text(rule.title).font(.system(size:13,weight:.semibold))
            DiagnosticExplanationView(guide: .make(ruleID: finding.ruleID, observedFact: finding.observation))
            HStack {
                Button(resolvedComponentTarget(finding) == nil ? "Show Train component":"Show component on canvas") {navigate(finding)}.buttonStyle(FlatButtonStyle())
                if verified.modelContent != nil {Button("Captured source",action:inspectSource).buttonStyle(FlatButtonStyle())}
                if finding.ruleID.hasPrefix("data."),verified.outcome.preparationManifestSHA256 != nil {Button("Captured preparation",action:inspectPreparation).buttonStyle(FlatButtonStyle())}
            }
            HStack {
                if ["gradient.growth.v1","gradient.nonfinite.v1"].contains(finding.ruleID) {
                    Button("Propose lower learning rate…") {propose(finding)}.buttonStyle(FlatButtonStyle()).disabled(proposal?.hasAppliedChange == true)
                }
                Button("Suppress with reason…") {reason="";expiryDays=1;suppressionTarget=finding}.buttonStyle(FlatButtonStyle()).disabled(suppressions.error != nil)
            }
            ResultsSection(title:"Technical evidence & original guidance",symbol:"function") {
                VStack(alignment:.leading,spacing:8) {
                    labeled("Captured interpretation",finding.interpretation)
                    labeled("Original proposed experiment",rule.experiment)
                    labeled("Evidence references",finding.references.joined(separator:"\n"))
                    labeled("Measured signals",finding.values.keys.sorted().map{"\($0): \(finding.values[$0]!)"}.joined(separator:" · "))
                    labeled("Thresholds",rule.thresholds.keys.sorted().map{"\($0): \(rule.thresholds[$0]!)"}.joined(separator:" · "))
                    labeled("Alternative explanations",rule.confounders.joined(separator:"\n"))
                    labeled("Why this matters",rule.lesson)
                }.padding(.top,6)
            }.font(.caption)
        }.padding(12).frame(maxWidth:.infinity,alignment:.leading).background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))
    }
    private func coverage(_ report:MeasuredDiagnosticReport)->some View {
        ResultsSection(title:"Measured coverage · \(report.checks.count) checks",symbol:"checklist") {
            VStack(alignment:.leading,spacing:10) {
                Text("Ruleset \(report.registryID) v\(report.registryVersion) · evidence \(report.inputSHA256.prefix(12))").font(.caption2).foregroundStyle(MLTheme.muted)
                ForEach(report.checks,id:\.ruleID) {check in
                    if let rule=(try? MeasuredDiagnostics.rules(forVersion:report.registryVersion))?.first(where:{$0.id==check.ruleID}) {
                        VStack(alignment:.leading,spacing:3) {
                            Text(rule.title+" · "+check.status.rawValue).font(.caption.bold())
                            Text(check.reason).font(.caption)
                            Text("Requires: "+rule.prerequisites.joined(separator:"; ")).font(.caption2).foregroundStyle(MLTheme.muted)
                        }
                    }
                }
                Text("Calibration: the original nine checks have 27 designed controls. The separate divergence check requires verified fixed-data context and consecutive observations. Passing controlled fixtures does not estimate field precision. Gradient capture is global and sampled; activation evidence has at most eight scalars per node. No unsupported observation is inferred.")
                    .font(.caption).foregroundStyle(MLTheme.muted)
            }.padding(.top,8)
        }
    }
    private func repairPanel(_ proposal:MeasuredDiagnosticRepair)->some View {
        VStack(alignment:.leading,spacing:8) {
            Text("Learning-rate experiment").font(.headline)
            Text("\(proposal.originalLearningRate) → \(proposal.proposedLearningRate)").font(.system(.body,design:.monospaced))
            Text("One configuration change, recorded locally and undoable. This does not verify external data, run training, or guarantee an improvement. Preserve split and seed when comparing the next run.").font(.caption).foregroundStyle(MLTheme.muted)
            HStack {
                if proposal.hasAppliedChange {
                    Button("Revert this change") {
                        perform {
                            try proposal.revert(in: host.owner)
                            self.proposal = nil
                            proposedBindings = nil
                            repairRevision += 1
                        }
                    }.buttonStyle(FlatButtonStyle())
                } else {
                    Button("Apply learning-rate change") {
                        perform {
                            guard host.bindings.revision==proposedBindings else {throw DiagnosticContractError("Data bindings changed after this proposal. Review the current sources first.")}
                            try proposal.apply(to:host.owner);repairRevision+=1
                        }
                    }.buttonStyle(FlatButtonStyle(prominent:true))
                    Button("Dismiss proposal") {self.proposal=nil}.buttonStyle(FlatButtonStyle())
                }
            }
        }.padding(12).overlay(Rectangle().stroke(MLTheme.rule))
    }
    private func suppressionSheet(_ finding:MeasuredDiagnosticFinding)->some View {
        VStack(alignment:.leading,spacing:14) {
            Text("Suppress this captured finding").font(.headline)
            Text("Applies only to this run and evidence fingerprint. New evidence and expiry restore the finding.").font(.caption).foregroundStyle(MLTheme.muted)
            if let message {Text(message).font(.caption).foregroundStyle(MLTheme.rose)}
            TextField("Reason for suppressing",text:$reason).textFieldStyle(.roundedBorder)
            Picker("Expires in",selection:$expiryDays) {Text("1 day").tag(1);Text("7 days").tag(7);Text("30 days").tag(30)}
            HStack {
                Button("Cancel") {suppressionTarget=nil}
                Spacer()
                Button("Save suppression") {
                    guard let report else {return}
                    perform {
                        let now=Date()
                        try suppressions.save(.init(report:report,finding:finding,reason:reason,now:now,expiresAt:now.addingTimeInterval(Double(expiryDays)*86400)),now:now)
                        suppressionTarget=nil
                    }
                }.disabled(reason.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty || reason.count>512)
            }
        }.padding(24).frame(width:460)
    }
    private func labeled(_ label:String,_ value:String)->some View {
        VStack(alignment:.leading,spacing:3) {Text(label).font(.caption.bold());Text(value).font(.caption).textSelection(.enabled)}
    }
    private func perform(_ action:()throws->Void) {do {try action();message=nil} catch {message=error.localizedDescription;repairRevision+=1}}
    private func propose(_ finding:MeasuredDiagnosticFinding) {
        guard let input else {return}
        perform {
            proposal=try MeasuredDiagnosticRepair.propose(input:input,findingID:finding.id,train:verified.capture.train,owner:host.owner,journalRoot:workspace.workspaceDirectory)
            proposedBindings=host.bindings.revision
        }
    }
    private func resolvedComponentTarget(_ finding:MeasuredDiagnosticFinding)->WorkflowNodeAddress? {
        guard let node=finding.nodeID,let origin=verified.plan.model.origins[node] else {return nil}
        return .init(domain:.model,localID:origin.instancePath.first ?? origin.localNodeID)
    }
    private func navigate(_ finding:MeasuredDiagnosticFinding) {
        perform {
            let target=resolvedComponentTarget(finding) ?? verified.capture.train
            try MeasuredDiagnosticNavigation.show(target,workflowSHA256:verified.capture.workflow.sha256,host:host,workspace:workspace)
        }
    }
    private func export(_ report:MeasuredDiagnosticReport) {
        perform {
            let bytes=try MeasuredDiagnosticExport.redacted(report)
            let panel=NSSavePanel();panel.allowedContentTypes=[.json];panel.nameFieldStringValue="diagnostic-summary.json"
            guard panel.runModal() == .OK,let destination=panel.url else {return}
            try bytes.write(to:destination,options:.atomic)
        }
    }
}
