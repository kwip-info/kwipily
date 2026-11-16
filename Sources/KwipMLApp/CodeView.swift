import SwiftUI
import KwipMLCore

struct CodeView: View {
    @ObservedObject var store: WorkspaceStore
    @State private var recorded = false
    @State private var selectedFile = "workflow.py"
    @State private var selectedRecord = ""
    @State private var recordedSnapshot: RecordedWorkflowCode.Snapshot?
    @State private var recordedError: String?
    @State private var loadingRecord = false
    @State private var historyChoices: [RecordedWorkflowCode.Choice] = []
    private var files: [String: String] {
        recorded ? recordedSnapshot?.files ?? [:] : store.currentCodeReady ? store.draftCodeFiles : [:]
    }
    private var fileName: String { files[selectedFile] != nil ? selectedFile : files["workflow.py"] != nil ? "workflow.py" : files.keys.sorted().first ?? "" }
    private var source: String { files[fileName] ?? "" }
    private var choices: [RecordedWorkflowCode.Choice] { historyChoices }
    private func title(_ scope: WorkflowCodePlan.Scope) -> String {
        switch scope { case .workflow: return "Entire workflow"; case .preparation: return "Data preparation"; case .model: return "Model"; case .training: return "Training" }
    }
    private func available(_ scope: WorkflowCodePlan.Scope) -> Bool {
        guard let document = store.workflowSession?.owner.document else { return scope == .model }
        switch scope {
        case .workflow: return !document.nodes.isEmpty
        case .preparation: return document.nodes.contains { $0.address.domain == .data }
        case .model: return document.nodes.contains { WorkflowModelBoundary.isVisibleDefinition($0,in:document) }
        case .training: return document.nodes.contains { switch $0.payload { case .train, .estimator: return true; default: return false } }
        }
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 5) {
                    Text("Workflow source").font(.system(size: 26, weight: .semibold))
                    Text(recorded ? "Captured source from a verified run. Read only." : "Generated from your canvas. Read only.")
                        .font(.system(size: 12)).foregroundStyle(MLTheme.muted)
                }
                Spacer()
                Button("Copy") { store.copyPythonCode(source) }.buttonStyle(FlatButtonStyle()).disabled(source.isEmpty)
                Button("Save file…") { store.savePythonCode(source, filename: fileName) }.buttonStyle(FlatButtonStyle()).disabled(source.isEmpty)
                Button("Export package…") { store.exportWorkflowCodePackage() }.buttonStyle(FlatButtonStyle(prominent: true))
                    .disabled(recorded || files["workflow.py"] == nil || store.isGeneratingCode)
            }
            if !recorded, let message = store.codePackageMessage {
                Text(message).font(.caption).foregroundStyle(message.hasPrefix("Code package export failed") ? MLTheme.rose : MLTheme.muted).textSelection(.enabled)
            }
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 16) { originControls; if !recorded { scopeControls }; Spacer() }
                VStack(alignment: .leading, spacing: 10) { originControls; if !recorded { scopeControls } }
            }
            if recorded {
                if choices.isEmpty {
                    HStack { Text("No recorded source is available yet.").font(.caption).foregroundStyle(MLTheme.muted)
                        Button("Open History") { store.activeSection = "History" }.buttonStyle(FlatButtonStyle()) }
                } else {
                    FlatChoiceControl(title: "Recorded run", selection: selectedRecord,
                        options: choices.map { .init(id: $0.id, title: $0.label) }) { selectedRecord = $0 }
                }
            } else {
                Text(store.workflowSession.map { "Current workflow · Revision \($0.owner.document.revision)" } ?? "Current model")
                    .font(.system(size: 11)).foregroundStyle(MLTheme.muted)
            }
            Divider()
            if loadingRecord || (!recorded && store.isGeneratingCode) {
                HStack { ProgressView().controlSize(.small); Text(recorded ? "Verifying recorded source…" : "Generating source from your workflow…").font(.caption) }
                    .padding(20)
                Spacer()
            } else if let error = recorded ? recordedError : store.draftCodeError {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Source is not available").font(.headline)
                    Text(error).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled)
                    HStack {
                        Button(recorded ? "Open History" : "Open canvas") { store.activeSection = recorded ? "History" : "Build" }.buttonStyle(FlatButtonStyle())
                        if !recorded { Button("Retry generation") { store.refreshDraftCode() }.buttonStyle(FlatButtonStyle()) }
                    }
                }.padding(20)
                Spacer()
            } else if source.isEmpty {
                ContentUnavailableView("Source is not available", systemImage: "curlybraces", description: Text(recorded ? "Choose a recorded run to verify its saved source." : "Add and connect components, then choose a scope."))
            } else {
                codePanel
                Text(recorded ? (recordedSnapshot?.inferenceOnly == true ? "Recorded backend source is an inference export; it does not replay training." : "Verified archived copy. Current canvas edits do not rewrite it.") : "Run the exported package with its declared dependencies and separate local bindings. Source files contain no input data or file permissions.")
                    .font(.system(size: 10)).foregroundStyle(MLTheme.muted)
            }
        }.padding(24)
        .task(id: store.codeGenerationContextID) {
            historyChoices = []
            if !available(store.workflowCodeScope) { store.workflowCodeScope = store.workflowSession == nil ? .model : .workflow }
            guard let host = store.workflowSession else { return }
            let project = host.owner.projectID, root = store.workspaceDirectory
            let listing = try? await Task.detached(priority: .utility) { try WorkflowHistoryReader.list(root: root, projectID: project) }.value
            await host.dataPreparationHistory.reloadAndWait()
            await host.backendHistory.reloadAndWait()
            guard !Task.isCancelled, store.workflowSession === host else { return }
            let training = (listing?.entries ?? []).filter { $0.verification == .recordedV2 }.map {
                RecordedWorkflowCode.Choice(id: "training:" + $0.reference.directoryID.uuidString, label: "Training · " + ($0.finishedAt ?? "Recorded run"))
            }
            historyChoices = training + RecordedWorkflowCode.choices(workspace: store).filter { !$0.id.hasPrefix("training:") }
            if recorded, selectedRecord.isEmpty, let first = historyChoices.first { selectedRecord = first.id }
        }
        .task(id: store.currentCodeGenerationKey) {
            try? await Task.sleep(nanoseconds: 200_000_000)
            guard !Task.isCancelled else { return }
            store.refreshDraftCode()
        }
        .task(id: "\(recorded):\(selectedRecord):\(store.codeGenerationContextID)") {
            recordedSnapshot = nil; recordedError = nil
            guard recorded else { return }
            if selectedRecord.isEmpty, let first = choices.first { selectedRecord = first.id; return }
            guard !selectedRecord.isEmpty else { return }
            loadingRecord = true
            defer { loadingRecord = false }
            do {
                let snapshot = try await RecordedWorkflowCode.load(workspace: store, selection: selectedRecord)
                guard !Task.isCancelled else { return }
                recordedSnapshot = snapshot
            } catch { guard !Task.isCancelled else { return }; recordedError = error.localizedDescription }
        }
    }
    private var codePanel: some View {
        HStack(alignment: .top, spacing: 0) {
            VStack(alignment: .leading, spacing: 6) {
                Text("FILES").font(.system(size: 10, weight: .semibold)).foregroundStyle(MLTheme.muted).padding(.bottom, 6)
                ThemedScrollView {
                    VStack(alignment: .leading, spacing: 6) {
                        ForEach(files.keys.sorted(), id: \.self) { name in fileRow(name) }
                    }.frame(maxWidth: .infinity, alignment: .leading)
                }
            }.padding(12).frame(width: 205)
            Rectangle().fill(MLTheme.rule).frame(width: 1)
            SyntaxHighlightedCode(source: source).frame(maxWidth: .infinity, maxHeight: .infinity)
        }.overlay(Rectangle().stroke(MLTheme.rule))
    }
    private func fileRow(_ name: String) -> some View {
        let selected = name == fileName
        let icon = name.hasSuffix(".py") ? "doc.text" : "doc"
        return Button { selectedFile = name } label: {
            HStack(spacing: 8) {
                Image(systemName: icon)
                Text(name).lineLimit(2).help(name)
                Spacer(minLength: 0)
            }
            .font(.system(size: 12)).padding(8).contentShape(Rectangle())
            .foregroundStyle(selected ? MLTheme.rose : Color.primary)
            .background(selected ? MLTheme.rose.opacity(0.10) : Color.clear)
        }.buttonStyle(.plain).accessibilityLabel("Show \(name)")
    }
    private var originControls: some View {
        HStack(spacing: 4) {
            Text("Source origin").font(.system(size: 11, weight: .semibold)).padding(.trailing, 5)
            Button("Current workflow") { recorded = false }.buttonStyle(FlatButtonStyle(prominent: !recorded))
            Button("Recorded run") { recorded = true }.buttonStyle(FlatButtonStyle(prominent: recorded))
        }
    }
    private var scopeControls: some View {
        HStack(spacing: 4) {
            Text("Scope").font(.system(size: 11, weight: .semibold)).padding(.trailing, 5)
            ForEach(WorkflowCodePlan.Scope.allCases, id: \.rawValue) { scope in
                Button(title(scope)) { store.workflowCodeScope = scope; store.refreshDraftCode() }
                    .buttonStyle(FlatButtonStyle(prominent: store.workflowCodeScope == scope))
                    .disabled(!available(scope))
            }
        }
    }
}
