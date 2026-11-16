import SwiftUI
import KwipMLCore

/// A presentation of the actual editable graph, never a second template executor.
struct WorkflowTemplateSetupStep: Equatable, Identifiable {
    enum Action: Equatable { case chooseSource, chooseOutput, configure }
    let address: WorkflowNodeAddress
    let title: String
    var action: Action
    var id: String { address.domain.rawValue + ":" + address.localID }
}

/// Explicit, independent project-setting actions; never edits application defaults or old captures.
@MainActor enum WorkflowTemplateRunOptionsPlan {
    private static func snapshot(_ entry:WorkflowTemplate,owner:WorkflowAuthoringStore,workspace:WorkspaceStore)throws->ProjectSettingsSnapshot {
        guard owner.document.templateOrigin == .init(id:entry.id,version:entry.version),
              workspace.workflowSession?.owner === owner else {throw ProjectSettingsFailure(code:"project_unavailable")}
        return try workspace.projectSettingsSnapshot(projectID:owner.projectID)
    }
    static func applyWorkingSpace(_ entry:WorkflowTemplate,owner:WorkflowAuthoringStore,workspace:WorkspaceStore)throws->ProjectSettingsSnapshot {
        guard let preset=entry.suggestedWorkload,preset != .custom else {throw ProjectSettingsFailure(code:"project_unavailable")}
        let baseline=try snapshot(entry,owner:owner,workspace:workspace),current=baseline.overrides
        let next=try ProjectExecutionOverrides(maximumRunSeconds:current.maximumRunSeconds,cpuThreads:current.cpuThreads,
            memory:current.memory,sampleCapture:current.sampleCapture,derivativeCapture:current.derivativeCapture,
            workloadPolicy:.init(preset:preset))
        return try workspace.applyProjectSettings(workspace.previewProjectSettings(.execution(next),baseline:baseline))
    }
    static func applyCompactHistory(_ entry:WorkflowTemplate,owner:WorkflowAuthoringStore,workspace:WorkspaceStore)throws->ProjectSettingsSnapshot {
        guard entry.suggestCompactHistory == true else {throw ProjectSettingsFailure(code:"project_unavailable")}
        let baseline=try snapshot(entry,owner:owner,workspace:workspace)
        return try workspace.applyProjectSettings(workspace.previewProjectSettings(.historyCapture(.init(metricsAndDiagnostics:true,componentOutputs:false)),baseline:baseline))
    }
}

@MainActor enum WorkflowTemplateSetupPlan {
    static func steps(in document: WorkflowDocument) -> [WorkflowTemplateSetupStep] {
        guard let origin = document.templateOrigin,
              let entries = try? WorkflowTemplates.catalog(),
              entries.contains(where: { $0.id == origin.id && $0.version == origin.version }) else { return [] }
        return document.nodes.compactMap { node in
            let action: WorkflowTemplateSetupStep.Action
            let fallback: String
            var title: String?
            switch node.payload {
            case .dataOperator(let id, let config):
                switch id {
                case "kwip.data.sourceCSV":
                    action = ["api", "database"].contains(config["format"]?.stringValue ?? "") ? .configure : .chooseSource
                    fallback = action == .configure ? "Data connection" : "Your table file"
                case "kwip.data.sourceFiles": action = .chooseSource; fallback = "Your documents"
                case "kwip.data.outputDelimited":
                    action = APITableSettings.isExternal(config) ? .configure : .chooseOutput
                    fallback = "Report destination"
                case "kwip.data.extractText": action = .configure; fallback = "OCR model"
                case "kwip.data.partition" where config["strategy"]?.stringValue == "time":
                    action = .configure; fallback = "Training and validation dates"; title = fallback
                default: return nil
                }
            case .frozenModel: action = .configure; fallback = "Model source"
            case .saveModel: action = .configure; fallback = "Model destination"
            default: return nil
            }
            return .init(address: node.address, title: title ?? (node.name.isEmpty ? fallback : node.name), action: action)
        }
    }

    static func configurationSummary(_ step:WorkflowTemplateSetupStep,in document:WorkflowDocument)->String {
        if let node=document.nodes.first(where:{$0.address==step.address}),
           case .dataOperator("kwip.data.partition",let config)=node.payload,
           config["strategy"]?.stringValue == "time" {
            let training=config["trainingEnd"]?.stringValue ?? "not configured"
            let validation=config["validationEnd"]?.stringValue ?? "not configured"
            return "Train through \(training) · validate through \(validation). Review these dates for your data."
        }
        return "Review this component's connection, model or destination settings."
    }

    /// Header guidance only. A cached source preview does not validate downstream data or a run.
    static func missingFields(for step:WorkflowTemplateSetupStep,in document:WorkflowDocument,observedColumns:[String]?)->[String]? {
        guard step.action == .chooseSource,let observedColumns,
              let node=document.nodes.first(where:{$0.address==step.address}),
              case .dataOperator("kwip.data.sourceCSV",let config)=node.payload else {return nil}
        var declared=Set<String>()
        for key in ["schemaOverrides","schema"] {
            if case .object(let fields)=config[key] {declared.formUnion(fields.keys)}
        }
        guard !declared.isEmpty else {return nil}
        return declared.subtracting(Set(observedColumns)).sorted()
    }
    @discardableResult static func focus(_ step:WorkflowTemplateSetupStep,owner:WorkflowAuthoringStore,
                                         sources:any WorkflowDataSources,lease:WorkflowEditorLease)->Bool {
        guard owner.isCurrent(lease),!owner.isBusy,!sources.isBusy,!owner.hasPendingInspectorEdits,
              owner.document.drafts.isEmpty,sources.projectID==owner.projectID,sources.sessionID==owner.sessionID,
              steps(in:owner.document).contains(step) else {return false}
        return owner.select(.init(nodeAddresses:[step.address]),lease:lease)
    }

    @discardableResult static func chooseReportFolder(owner:WorkflowAuthoringStore,sources:any WorkflowDataSources,lease:WorkflowEditorLease)->Bool {
        let addresses=Set(steps(in:owner.document).filter{$0.action == .chooseOutput}.map(\.address))
        guard owner.isCurrent(lease),!owner.isBusy,!sources.isBusy,!owner.hasPendingInspectorEdits,
              owner.document.drafts.isEmpty,!addresses.isEmpty,sources.projectID==owner.projectID,
              sources.sessionID==owner.sessionID,owner.select(.init(nodeAddresses:addresses),lease:lease) else {return false}
        return sources.chooseOutputDestinations(for:addresses,lease:lease)
    }

    @discardableResult static func perform(_ step: WorkflowTemplateSetupStep, owner: WorkflowAuthoringStore,
                                          sources: any WorkflowDataSources, lease: WorkflowEditorLease) -> Bool {
        guard focus(step,owner:owner,sources:sources,lease:lease) else {return false}
        switch step.action {
        case .chooseSource: return sources.chooseSource(for: step.address, lease: lease)
        case .chooseOutput: return sources.chooseOutputDestination(for: step.address, lease: lease)
        case .configure: return true
        }
    }
}

/// Choosers retain the existing source service's post-picker lease and sandbox checks.
/// Only cached presentation is read; this view never probes files or starts a run.
struct WorkflowTemplateSetupView: View {
    @ObservedObject var owner: WorkflowAuthoringStore
    let entry: WorkflowTemplate
    let sources: any WorkflowDataSources
    let revealConfiguration: () -> Void
    let close: () -> Void
    var workspace: WorkspaceStore? = nil
    var expanded=false
    var registerDiscard:((@escaping ()->Void,@escaping ()->Bool)->Void)?
    @State private var sourceRevision: UInt = 0
    @State private var message: String?
    @State private var reportNamesExpanded=false
    @State private var fieldMatchSession:WorkflowTemplateFieldMatchSession?
    private var unavailable: Bool {
        owner.isBusy || sources.isBusy || owner.hasPendingInspectorEdits || !owner.document.drafts.isEmpty
    }
    var body: some View {
        let _ = sourceRevision
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 5) {
                    Text(entry.title).font(.system(size: 19, weight: .semibold))
                    Text(entry.summary).font(.system(size: 12)).foregroundStyle(MLTheme.muted)
                }
                Spacer()
                Button(expanded ? "Close":"Show canvas", action: close).buttonStyle(FlatButtonStyle())
                    .keyboardShortcut(.cancelAction)
                    .accessibilityIdentifier("template.setup.close").disabled(!expanded && owner.hasPendingInspectorEdits)
            }.fixedSize(horizontal:false,vertical:true)
            ThemedScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    let steps=WorkflowTemplateSetupPlan.steps(in:owner.document)
                    let reports=steps.filter{$0.action == .chooseOutput}
                    ForEach(steps.filter{reports.count <= 1 || $0.action != .chooseOutput}) { step in
                        HStack(spacing: 16) {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(step.title).font(.system(size: 12, weight: .medium))
                                Text(summary(step)).font(.caption).foregroundStyle(MLTheme.muted)
                                    .lineLimit(2).frame(maxWidth: .infinity, alignment: .leading)
                            }
                            Spacer(minLength: 8)
                            if canReviewFields(step) {
                                Button((missingFields(step)?.isEmpty == false) ? "Match fields":"Review fields") {
                                    do {fieldMatchSession=try .init(step:step,owner:owner,sources:sources);message=nil}
                                    catch {message=error.localizedDescription}
                                }.buttonStyle(FlatButtonStyle()).disabled(unavailable)
                                    .accessibilityIdentifier("template.setup.fields."+step.id)
                            }
                            Button(buttonTitle(step)) {
                                if WorkflowTemplateSetupPlan.perform(step, owner: owner, sources: sources, lease: owner.lease()) {
                                    message = nil
                                    if step.action == .configure { revealConfiguration(); close() }
                                } else { message = sources.statusMessage.isEmpty ? "No change was made.":sources.statusMessage }
                            }.buttonStyle(FlatButtonStyle()).disabled(unavailable)
                                .accessibilityIdentifier("template.setup." + step.id)
                        }.padding(.vertical, 10)
                        if let session=fieldMatchSession,session.plan.source.address==step.address {
                            WorkflowTemplateFieldMatchView(session:session) {applied in fieldMatchSession=nil;message=applied ? "Field matches saved on the canvas. Refresh the source preview or run the flow.":"Field draft closed. Your saved matches are unchanged."}
                                .padding(.bottom,10)
                        }
                        Divider()
                    }
                    if reports.count > 1 {
                        HStack {
                            Text("Save \(reports.count) reports").font(.system(size:12,weight:.medium))
                            Spacer()
                            Button("Choose report folder…") {
                                let chosen=WorkflowTemplateSetupPlan.chooseReportFolder(owner:owner,sources:sources,lease:owner.lease())
                                message=chosen ? nil:sources.statusMessage
                            }.buttonStyle(FlatButtonStyle()).disabled(unavailable)
                                .accessibilityIdentifier("template.setup.reports")
                        }.padding(.vertical,10)
                        Text(reportFolderSummary(reports)).font(.caption).foregroundStyle(MLTheme.muted).lineLimit(2)
                            .textSelection(.enabled).padding(.bottom,6)
                        Button(reportNamesExpanded ? "Hide report names":"Show report names") {reportNamesExpanded.toggle()}
                            .buttonStyle(.plain).font(.caption).foregroundStyle(MLTheme.rose)
                            .accessibilityIdentifier("template.setup.reportNames")
                        if reportNamesExpanded {
                            ForEach(reports) {step in
                                Text(reportFilename(step)).font(.caption.monospaced()).foregroundStyle(MLTheme.muted)
                                    .textSelection(.enabled).padding(.vertical,3)
                            }
                        }
                        Divider()
                    }
                    if let workspace {
                        if let preset=entry.suggestedWorkload {
                            HStack {
                                VStack(alignment:.leading,spacing:3) {
                                    Text("Working space for larger tables").font(.system(size:12,weight:.medium))
                                    Text("Use the existing \(preset.rawValue.capitalized) preset for this project.").font(.caption).foregroundStyle(MLTheme.muted)
                                }
                                Spacer()
                                Button("Use \(preset.rawValue.capitalized)") {
                                    do {_ = try WorkflowTemplateRunOptionsPlan.applyWorkingSpace(entry,owner:owner,workspace:workspace);message="Working space updated for this project."}
                                    catch {message=error.localizedDescription}
                                }.buttonStyle(FlatButtonStyle()).disabled(unavailable)
                                    .accessibilityIdentifier("template.setup.workingSpace")
                            }.padding(.vertical,10)
                            Divider()
                        }
                        if entry.suggestCompactHistory == true {
                            HStack {
                                VStack(alignment:.leading,spacing:3) {
                                    Text("Compact run history").font(.system(size:12,weight:.medium))
                                    Text("Keep run summaries and live previews; skip optional intermediate History copies.").font(.caption).foregroundStyle(MLTheme.muted)
                                }
                                Spacer()
                                Button("Use compact History") {
                                    do {_ = try WorkflowTemplateRunOptionsPlan.applyCompactHistory(entry,owner:owner,workspace:workspace);message="Compact History applied. Exports stay in your report folder."}
                                    catch {message=error.localizedDescription}
                                }.buttonStyle(FlatButtonStyle()).disabled(unavailable)
                                    .accessibilityIdentifier("template.setup.compactHistory")
                            }.padding(.vertical,10)
                            Divider()
                        }
                    }
                }.padding(.trailing, 6)
            }.frame(minHeight:80,idealHeight:expanded ? 480:128,maxHeight:expanded ? .infinity:160).clipped()
            HStack(alignment: .top) {
                Text(message ?? (expanded ? "Choose your data and match fields. Close setup, then use Run Flow on the canvas. File selection alone does not validate the workflow.":"Choose your data. Match differing column names here, then use Run Flow. File selection alone is not validation; reports appear in Results."))
                    .font(.caption).foregroundStyle(MLTheme.muted)
                Spacer(minLength: 12)
                Button("Requirements") { ApplicationNavigationWindows.shared.showTemplates(id: entry.id) }
                    .buttonStyle(FlatButtonStyle()).disabled(unavailable)
            }
        }.padding(16).background(MLTheme.panel)
            .overlay(Rectangle().stroke(MLTheme.rule))
            .onReceive(sources.changes) { sourceRevision &+= 1 }
            .onAppear {registerDiscard?({fieldMatchSession?.discard();fieldMatchSession=nil},{fieldMatchSession?.dirty == true})}
            .onDisappear {fieldMatchSession?.discard();fieldMatchSession=nil}
    }
    private func canReviewFields(_ step:WorkflowTemplateSetupStep)->Bool {
        guard step.action == .chooseSource,sources.sourceInspectionPreview(for:step.address) != nil,
              let node=owner.document.nodes.first(where:{$0.address==step.address}),
              case .dataOperator("kwip.data.sourceCSV",let config)=node.payload else {return false}
        return (config["format"]?.stringValue ?? "csv")=="csv"
    }
    private func summary(_ step: WorkflowTemplateSetupStep) -> String {
        switch step.action {
        case .chooseSource:
            let selected = sources.bindingDisplayName(for: step.address)
            if let missing=missingFields(step),!missing.isEmpty {
                let names=missing.prefix(6).joined(separator:", ")
                return "Missing configured fields: "+names+(missing.count>6 ? " · \(missing.count-6) more":"")
            }
            if let status = sources.sourceInspectionStatus(for: step.address) {
                return (selected.map { $0 + " · " } ?? "") + status.title
            }
            return selected.map { $0 + " · selected" } ?? "Choose a compatible file or folder on this Mac."
        case .chooseOutput: return sources.outputDestinationSummary(for: step.address)
        case .configure: return WorkflowTemplateSetupPlan.configurationSummary(step,in:owner.document)
        }
    }
    private func reportFolderSummary(_ reports:[WorkflowTemplateSetupStep])->String {
        let summaries=Set(reports.map {sources.outputDestinationSummary(for:$0.address)})
        return summaries.count == 1 ? (summaries.first ?? "Choose a report folder.") : "Reports currently use separate folders. Choose one folder for all reports, or review individual destinations in Configuration."
    }
    private func reportFilename(_ step:WorkflowTemplateSetupStep)->String {
        guard let node=owner.document.nodes.first(where:{$0.address==step.address}),
              case .dataOperator(_,let config)=node.payload else {return step.title}
        return config["filename"]?.stringValue ?? step.title
    }
    private func missingFields(_ step:WorkflowTemplateSetupStep)->[String]? {
        WorkflowTemplateSetupPlan.missingFields(for:step,in:owner.document,
            observedColumns:sources.sourceInspectionPreview(for:step.address)?.columns)
    }
    private func buttonTitle(_ step: WorkflowTemplateSetupStep) -> String {
        switch step.action {
        case .chooseSource: return sources.bindingDisplayName(for: step.address) == nil ? "Choose data…" : "Change data…"
        case .chooseOutput: return "Choose folder…"
        case .configure: return "Configure"
        }
    }
}
