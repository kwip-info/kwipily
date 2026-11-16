import SwiftUI
import AppKit
import KwipMLCore

struct WorkspaceView: View {
    @Environment(\.appearanceTheme) private var appearanceTheme
    @State private var paneFocus=WorkspacePaneFocus()
    @ObservedObject var store: WorkspaceStore
    @ObservedObject var preparationSession: PreparationSession
    @ObservedObject var affineLearning: AffineLearningStore
    @ObservedObject var normalizationLearning: NormalizationLearningStore
    @ObservedObject var tensorLearning: TensorLearningStore
    private var sections: [String] {
        store.projectSections
    }
    private var isRetiredLesson: Bool {
        store.lessonContext?.kind == .normalization ||
            (store.lessonContext?.kind == .tensor && RetiredTrainingLessonDestination.contains(tensorLearning.draft?.operation))
    }
    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                projectNavigation
                if !isRetiredLesson {
                if tensorLearning.isActive, let canvas = tensorLearning.canvas {
                    WorkspaceHistoryButtons(editor:canvas)
                } else if !tensorLearning.isActive {
                    WorkspaceHistoryButtons(editor:store)
                }
                Spacer(minLength: 8)
                if CheckpointLearningTask(rawValue:tensorLearning.draft?.operation ?? "") == nil || !tensorLearning.isActive {
                Button { store.showLibrary.toggle() } label: { Image(systemName: "rectangle.topthird.inset.filled") }
                    .help("Show or hide the component ribbon").accessibilityLabel("Toggle components")
                }
                Button { store.showInspector.toggle() } label: { Image(systemName: "sidebar.left") }
                    .help("Show or hide configuration").accessibilityLabel("Toggle configuration")
                if store.activeSection == "Build" {
                    if store.workflowSession == nil {
                        Button { store.showPreview.toggle() } label: { Image(systemName: "rectangle.bottomthird.inset.filled") }
                            .help("Show or hide results").accessibilityLabel("Toggle results")
                    }
                }
                if store.lessonRailEntryID != nil {
                    Button {store.showLessonRail.toggle()} label: {Image(systemName:"sidebar.right")}
                        .accessibilityLabel("Toggle lesson")
                }
                Divider().frame(height: 18)
                SaveFeedbackButton {
                    if tensorLearning.isActive { return tensorLearning.save() }
                    else if normalizationLearning.isActive { return normalizationLearning.save() }
                    else if affineLearning.isActive { return affineLearning.save() }
                    else { return store.saveManagedProject() }
                }.disabled(store.lessonContext?.restoreState == .failed)
                if !store.isLearningLessonActive {
                    WorkspaceRunControl(store: store)
                }
                } else {
                    Spacer()
                    Text("Historical lesson").font(.caption).foregroundStyle(MLTheme.muted)
                }
            }.padding(.horizontal, 14).padding(.vertical, 9)
            Divider()
            if store.workflowSession != nil, let error = store.saveError {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "exclamationmark.circle").foregroundStyle(Color.orange)
                    Text(error).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
                }.font(.system(size: 12)).padding(.horizontal, 14).padding(.vertical, 8)
                    .background(MLTheme.panel)
                    .accessibilityElement(children: .combine).accessibilityLabel("Save needs attention: " + error)
                Divider()
            }
            HSplitView {
                content.frame(maxWidth: .infinity, maxHeight: .infinity)
                    .environment(\.lessonRailPresented, store.lessonRailEntryID != nil && store.showLessonRail)
                    .environment(\.lessonNodeFocus, store.lessonRailEntryID != nil && store.showLessonRail ? store.lessonNodeFocus:CanvasLessonFocus())
                if let entryID=store.lessonRailEntryID,store.showLessonRail,
                   store.activeSection == "Build" || store.activeSection == "Learn" {
                    LessonRailView(workspace:store,affine:affineLearning,normalization:normalizationLearning,tensor:tensorLearning,reading:store.lessonReading,entryID:entryID)
                        .id((store.activeProjectID ?? "")+":"+entryID)
                        .frame(minWidth:300,idealWidth:350,maxWidth:460).workspacePane(.lesson)
                }
            }
            Divider()
            HStack(spacing: 9) {
                if let host = store.workflowSession { WorkflowStatusView(host: host) } else {
                Circle().fill(store.isLearningLessonActive ? MLTheme.muted : (store.validation.isValid ? MLTheme.teal : Color.orange)).frame(width: 6, height: 6)
                Text(store.isLearningLessonActive ? "Lesson workspace" : (store.validation.isValid ? "Graph valid" : "\(store.validation.issues.count) issues"))
                Divider().frame(height: 12)
                Text(store.saveError ?? store.statusMessage).foregroundStyle(MLTheme.muted).lineLimit(1)
                Spacer()
                Text(store.isLearningLessonActive ? "Local · learning" : "Local · \(store.isRunning || store.lastCompleted ? store.resolvedDevice.uppercased() : store.execution.device.uppercased())")
                    .foregroundStyle(MLTheme.muted)
            }
            }.font(.system(size: 10)).padding(.horizontal, 14).padding(.vertical, 6)
        }.background(MLTheme.background).buttonStyle(FlatButtonStyle()).scrollIndicators(.hidden)
        .environment(\.workspacePaneFocus,paneFocus)
            .environment(\.configureTensorCapture,{store.activeSection="Settings"}).windowActivatingControls()
        .onAppear { preparationSession.attach(store); store.learningProjects.restoreCurrentProject(); store.restoreLessonRail() }
        .onChange(of:store.activeProjectID) {_,_ in store.restoreLessonRail()}
        .onChange(of: store.isLearningLessonActive) { _, active in if active { store.activeSection = "Learn" } }
    }
    private var projectNavigation: some View {
        HStack(spacing: 14) {
            ForEach(sections, id: \.self) { section in
                Button { store.activeSection = section } label: {
                    Label(WorkspaceToolbarSection.title(section), systemImage: WorkspaceToolbarSection.symbol(section))
                        .font(.system(size: 12, weight: .medium)).fixedSize()
                        .foregroundStyle(store.activeSection == section ? appearanceTheme.accentColor : MLTheme.muted)
                        .padding(.vertical, 8)
                        .overlay(alignment: .bottom) {
                            if store.activeSection == section { Rectangle().fill(appearanceTheme.accentColor).frame(height: 2) }
                        }
                }.buttonStyle(.plain)
            }
        }.layoutPriority(1)
    }
    @ViewBuilder private var content: some View {
        if store.isLearningLessonActive && store.activeSection == "Learn" {
            if store.lessonContext?.restoreState == .failed {
                VStack(alignment: .leading, spacing: 16) {
                    Text("Lesson needs recovery").font(.title2)
                    Text(store.statusMessage).textSelection(.enabled)
                    Text("Saved lesson files remain on this Mac. Open Projects from the menu bar to choose another flow while preserving this lesson for recovery.").foregroundStyle(MLTheme.muted)
                    Button("Open projects") { store.refreshProjects(); store.activeSection = "Projects" }
                    Spacer()
                }.padding(28).frame(maxWidth: .infinity, alignment: .leading)
            } else if store.lessonContext?.kind == .tensor {
                TensorLearningView(lesson:tensorLearning,workspace:store)
            } else if store.lessonContext?.kind == .normalization {
                NormalizationLearningView(lesson: normalizationLearning, workspace: store)
            } else { AffineLearningView(lesson: affineLearning, workspace: store) }
        } else if let host = store.workflowSession, store.activeSection != "Projects" {
            workflowContent(host)
        } else if store.activeSection == "Build" { modelWorkbench }
        else if store.activeSection == "Run" {
            HSplitView {
                if store.showInspector { InspectorView(store: store).frame(minWidth: 225, idealWidth: 280, maxWidth: 380).workspacePane(.inspector) }
                RunView(store: store).frame(minWidth: 500)
            }
        } else if store.activeSection == "Projects" { ProjectHomeView(store: store) }
        else if store.activeSection == "Data" { DataView(store: store, session: preparationSession) }
        else if store.activeSection == "Experiments" { ExperimentsView(store: store) }
        else if store.activeSection == "Settings" { SettingsView(store: store) }
        else if store.activeSection == "Code" { CodeView(store: store) }
        else { modelWorkbench }
    }
    @ViewBuilder private func workflowContent(_ host: WorkflowWorkspaceSession) -> some View {
        if store.activeSection == "History", let history = store.workflowHistory {
            WorkflowHistoryView(host: host, history: history, workspace: store)
                .id(host.owner.sessionID)
        } else if store.activeSection == "Code" { CodeView(store: store) }
        else if store.activeSection == "Settings" {
            WorkflowProjectSettingsView(workspace: store, owner: host.owner).id(host.owner.sessionID)
        } else {
            WorkflowRuntimeWorkbenchView(store:store,host:host,capture:store.gradientCapture)
                .id(host.owner.sessionID)
        }
    }
    private var modelWorkbench: some View {
        VStack(spacing: 0) {
            if let canvas = store.modelCanvas {
                ModelScopeEditorView(store: canvas, showLibrary: store.showLibrary, showInspector: store.showInspector)
            } else {
                if store.showLibrary { ComponentPaletteView(store: store); Divider() }
                HSplitView {
                    if store.showInspector {
                        InspectorView(store: store).frame(minWidth: 225, idealWidth: 280, maxWidth: 380).workspacePane(.inspector)
                            .workspaceSplitLayout(store: store, scope: .modelRibbonInspector)
                    }
                    VSplitView {
                        VStack(spacing: 0) {
                            GraphCanvasView(store: store).id("\(store.activeProjectID ?? "draft")-\(store.tutorialStep ?? -1)")

                        }.frame(minHeight: 260)
                            .workspaceSplitLayout(store: store, scope: store.showPreview ? .modelPreview : nil)
                        if store.showPreview { preview.frame(minHeight: 100, idealHeight: 170, maxHeight: 300).workspacePane(.results) }
                    }.frame(minWidth: 500)
                }
            }
        }
    }
    private var previewHeader: String {
        if store.isPreviewing { return "Evaluating upstream flow…" }
        let id = store.selectedNodeID ?? store.graph.edges.first(where: { $0.id == store.selectedEdgeID })?.source
        guard let id else { return "Select a block or wire" }
        let status = store.previewStatus(for: id)
        return status == "Not evaluated" ? "Schema · not evaluated" : "\(status) · \(store.previewWeightLabel)"
    }

    private var preview: some View {
        ThemedScrollView {
            VStack(alignment: .leading, spacing: 12) {
                HStack { SectionLabel(title: "Data inspector"); Spacer(); Text(previewHeader).font(.system(size: 10)).foregroundStyle(previewHeader.hasPrefix("Stale") ? Color.orange : MLTheme.muted) }
                if let input = store.graph.nodes.first(where: { ["input","index_input"].contains($0.type) }), let values = store.previews[input.id] {
                    PreviewProvenanceView(values: values)
                }
                if let node = store.selectedNode {
                    HStack(alignment: .top, spacing: 38) {
                        VStack(alignment: .leading, spacing: 5) { Text(node.name).font(.system(size: 13, weight: .medium)); Text(node.type).font(.caption).foregroundStyle(MLTheme.muted) }
                        VStack(alignment: .leading, spacing: 5) { SectionLabel(title: "Inferred output"); Text(store.validation.tensorLayouts[node.id].map { $0.displayShape } ?? "Unresolved").font(.system(size: 12, design: .monospaced)) }
                        if let values = store.previews[node.id] {
                            TensorOutputPreviewView(preview: values,family:.resolve(type:store.selectedNode?.type ?? ""))
                        }
                    }
                } else if let edge = store.graph.edges.first(where: { $0.id == store.selectedEdgeID }) {
                    Text("\(store.graph.nodes.first(where: {$0.id == edge.source})?.name ?? edge.source) → \(store.graph.nodes.first(where: {$0.id == edge.target})?.name ?? edge.target) · \(edge.targetPort)")
                    Text(store.validation.outputTensorContracts[.init(edge.source,edge.sourcePort)].map { "Tensor · \($0.dtype.rawValue) · \($0.layout.displayShape)" } ?? "Shape unresolved").font(.system(size: 12, design: .monospaced))
                    if let values = store.previews[edge.source] {
                        TensorOutputPreviewView(preview: values, selectedPort: edge.sourcePort,family:.resolve(type:store.graph.nodes.first(where:{$0.id==edge.source})?.type ?? ""))
                    }
                } else { Text("Select a block or connection to inspect its shape and captured data.").font(.system(size: 12)).foregroundStyle(MLTheme.muted) }
            }.padding(18).frame(maxWidth: .infinity, alignment: .leading)
        }.scrollIndicators(.hidden).appearanceChrome()
    }
}

/// Observe the same editor that owns the visible graph, including lesson history and busy state.
private struct WorkspaceHistoryButtons<Editor: ObservableObject & EditorHistory>: View {
    @ObservedObject var editor: Editor
    var body: some View {
        HStack(spacing:4) {
            Button { editor.undo() } label: { Image(systemName:"chevron.left") }
                .disabled(!editor.canUndo).accessibilityLabel("Undo edit").help("Undo edit")
            Button { editor.redo() } label: { Image(systemName:"chevron.right") }
                .disabled(!editor.canRedo).accessibilityLabel("Redo edit").help("Redo edit")
        }.buttonStyle(.borderless)
    }
}

/// Observe capability changes directly rather than waiting for an unrelated
/// workspace publication to enable or revoke a component control.
private struct WorkflowRuntimeWorkbenchView:View {
    @ObservedObject var store:WorkspaceStore
    @ObservedObject var host:WorkflowWorkspaceSession
    @ObservedObject var capture:GradientCaptureStore
    var body:some View {
        WorkflowWorkbenchView(owner:host.owner,canvas:host.canvas,sources:host.sourceProvider,outputs:host,
            executionHost:host,history:store.workflowHistory,workspace:store,showComponents:store.showLibrary,
            showInspector:store.showInspector,canCaptureGradients:capture.canCapture,openHistory:{store.activeSection="History"})
    }
}
