import SwiftUI
import KwipMLCore

struct CanvasVersionHistoryView:View {
    @ObservedObject var owner:WorkflowAuthoringStore
    @ObservedObject var workspace:WorkspaceStore
    @State private var selectedChange:Int?
    @State private var selected:String?
    @State private var search=""
    @State private var label=""
    @State private var issue:String?
    @State private var confirmRestore=false
    @State private var captureExpanded:Bool
    @State private var informationExpanded:Bool
    init(owner:WorkflowAuthoringStore,workspace:WorkspaceStore,captureExpanded:Bool=false,informationExpanded:Bool=false) {
        self.owner=owner;self.workspace=workspace
        _captureExpanded=State(initialValue:captureExpanded)
        _informationExpanded=State(initialValue:informationExpanded)
    }
    private var journal:CanvasVersionJournal? {owner.canvasVersions}
    private var versions:[CanvasVersionJournal.Version] {
        (journal?.versions ?? []).filter {search.isEmpty || ("Version \($0.number) "+$0.label).localizedCaseInsensitiveContains(search)}
    }
    private var currentChanges:[CanvasVersionJournal.Change] {
        journal?.lastDocument.map {CanvasVersionJournal.changes(from:$0,to:owner.document)} ?? []
    }
    private var chosen:CanvasVersionJournal.Version? {journal?.versions.first {$0.id==selected}}
    private var changes:[CanvasVersionJournal.Change] {chosen?.changes ?? currentChanges}
    private var canChange:Bool {!workspace.isBusy && !owner.isBusy && !owner.hasPendingInspectorEdits && owner.document.drafts.isEmpty}
    var body:some View {
        GeometryReader { viewport in
        VStack(spacing:0) {
            HStack {
                Button {selected=nil} label: {Label("Changes since last save · \(currentChanges.count)",systemImage:"arrow.triangle.branch")}
                    .buttonStyle(FlatButtonStyle(prominent:selected==nil))
                Spacer()
                TextField("Search versions",text:$search).flatField().frame(maxWidth:240)
                Button {do {try journal?.reload();owner.objectWillChange.send();issue=nil} catch {issue=error.localizedDescription}} label:{Image(systemName:"arrow.clockwise")}
                    .buttonStyle(FlatButtonStyle()).accessibilityLabel("Refresh canvas versions")
            }.padding(12)
            if journal==nil {
                Text("Canvas History unavailable. The editable canvas and existing runs are preserved.").font(.caption).foregroundStyle(MLTheme.rose).padding(12)
                Button("Retry canvas History") {do {try owner.configureCanvasVersions(root:workspace.workspaceDirectory);owner.objectWillChange.send();issue=nil} catch {issue=error.localizedDescription}}.buttonStyle(FlatButtonStyle())
            }
            // Bound AppKit's split to the space left by the header and footer.
            // Its children's intrinsic minima must not enlarge the hosting window.
            GeometryReader { tables in
            VSplitView {
                CapturedNativeTable(columns:WorkflowResultTableData.columns(["Version","Saved","Changes"],numeric:[2]),rows:versions.map {version in
                    WorkflowResultTableData.row(version.number-1,["Version \(version.number)"+(version.label.isEmpty ? "":" · "+version.label),version.savedAt.formatted(date:.abbreviated,time:.shortened),String(version.changes.count)])
                },selectedRow:Binding(get:{versions.first {$0.id==selected}.map {$0.number-1}},set:{number in selected=versions.first {$0.number-1==number}?.id}),rowHeaderTitle:"#",fillsWidth:true)
                    .frame(minHeight:0,idealHeight:220,maxHeight:.infinity).clipped()
                VStack(alignment:.leading,spacing:0) {
                    HStack {
                        Text(chosen.map {"Changes saved in version \($0.number)"} ?? "Changes since last save").font(.system(size:13,weight:.semibold))
                        Spacer()
                        if selected != nil {Button("Current changes") {selected=nil}.buttonStyle(FlatButtonStyle())}
                    }.padding(12).background(MLTheme.panel)
                    if changes.isEmpty {
                        Text(chosen == nil ? "No changes since last save":"Initial canvas").font(.caption).foregroundStyle(MLTheme.muted).padding(16)
                        Spacer()
                    } else {
                        CapturedNativeTable(columns:WorkflowResultTableData.columns(["Component","Change","Before","After"]),rows:changes.enumerated().map {index,row in
                            WorkflowResultTableData.row(index,[row.component,row.field,row.before,row.after])
                        },selectedRow:$selectedChange,rowHeaderTitle:"#",fillsWidth:true)

                    }
                }.frame(minHeight:0,idealHeight:300,maxHeight:.infinity).clipped()
            }
            .frame(width:tables.size.width,height:tables.size.height)
            .clipped()
            }.frame(minHeight:0,maxHeight:.infinity)
            Rectangle().fill(MLTheme.rule).frame(height:1)
            HStack {
                TextField("Version label (optional)",text:$label).flatField().frame(maxWidth:280)
                Button("Save version") {save()}.buttonStyle(FlatButtonStyle(prominent:true)).disabled(!canChange || journal==nil)
                Spacer()
                if selected != nil {Button("Restore as new version") {confirmRestore=true}.buttonStyle(FlatButtonStyle()).disabled(!canChange)}
            }.padding(12)
            if let issue {Text(issue).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled).padding(.horizontal,12)}
            ThemedScrollView {
            VStack(alignment:.leading,spacing:0) {
            DisclosureGroup("Run-detail capture",isExpanded:$captureExpanded) {
                RunHistoryCaptureControls(owner:owner,workspace:workspace).padding(.vertical,8)
            }.font(.system(size:11)).padding(.horizontal,12).padding(.top,8)
            DisclosureGroup("Information",isExpanded:$informationExpanded) {
                Text("Save records a canvas version; edits are also persisted for recovery. Versions store verified differences, bounded to 512 versions and 8 MB per project; reaching a limit preserves existing history and reports a save error. Restore preserves the current canvas first and creates a new version. Source files, private connections and captured run results are not rolled back. Rerunning can produce different results. Layout changes do not invalidate execution outputs. Historical run details use their exact captured canvas, including any unsaved changes. Capture controls collection and storage on future runs across this project. Turning it off skips optional tensor snapshots, prediction previews, diagnostic analysis, performance samples and intermediate preparation copies. Epoch metrics, model weights, source/replay evidence, final prepared tables, fitted state, explicit View/file outputs, evaluation tables and requested caches remain. Connected prediction outputs remain required workflow data. Existing runs are unchanged; this is not an archive purge.")
                    .font(.caption).foregroundStyle(MLTheme.muted).fixedSize(horizontal:false,vertical:true).padding(.vertical,8)
            }.font(.system(size:11)).padding(12)
            }.frame(maxWidth:.infinity,alignment:.leading)
            }
            .frame(height:captureExpanded || informationExpanded ? min(260,max(100,viewport.size.height*0.38)):68)
        }
        .frame(width:viewport.size.width,height:viewport.size.height)
        .clipped()
        }
        .confirmationDialog("Restore this canvas version?",isPresented:$confirmRestore,titleVisibility:.visible) {
            Button("Preserve current canvas and restore") {
                guard let selected else {return}
                let token=owner.lease()
                if owner.restoreCanvasVersion(selected,lease:token) {self.selected=nil;issue=owner.error;workspace.hasUnsavedChanges=false}
                else {issue=owner.error}
            }
        } message: {Text("Current changes will be saved first. External files and model results stay unchanged.")}
    }
    private func save() {
        do {try owner.saveCanvasVersion(label:label);label="";selected=nil;issue=nil;_ = workspace.saveManagedProject()}
        catch {issue=error.localizedDescription}
    }
    private func symbol(_ kind:CanvasVersionJournal.ChangeKind)->String {
        switch kind {case .added:return "plus.circle";case .removed:return "minus.circle";case .configuration:return "slider.horizontal.3";case .connection:return "point.3.connected.trianglepath.dotted";case .layout:return "arrow.up.left.and.arrow.down.right";case .metadata:return "pencil"}
    }
}
