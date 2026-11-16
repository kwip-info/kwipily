import SwiftUI
import KwipMLCore

struct MediaLibrarySettingsView:View {
    @ObservedObject var store:WorkspaceStore
    var projectID:String?=nil
    @State private var report:MediaLibraryInventory.Report?
    @State private var job:Task<Void,Never>?
    @State private var message=""
    @State private var projectOnly=false
    @State private var selectedProjectID:String?
    private var scope:String? {projectID ?? (projectOnly ? selectedProjectID:nil)}
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            SectionLabel(title:"Captured media & prepared tensors")
            Text("Owned image pixels, PCM samples and prepared tensors referenced by preparation or training History in this workspace. Source files stay in their original locations; this inventory grants no source access.").font(.caption).foregroundStyle(MLTheme.muted)
            if projectID==nil {
                FlatChoiceControl(title:"Asset scope",selection:projectOnly ? "project":"workspace",options:[.init(id:"workspace",title:"Current workspace"),.init(id:"project",title:"Saved project",isEnabled:!store.projects.isEmpty)]) {value in
                    projectOnly=value=="project"
                    if projectOnly && selectedProjectID==nil {selectedProjectID=store.projects.first(where:{$0.id==store.activeProjectID})?.id ?? store.projects.first?.id}
                    refresh()
                }
                if projectOnly {
                    FlatChoiceControl(title:"Project",selection:selectedProjectID ?? "",options:store.projects.map {.init(id:$0.id,title:$0.name+" · "+String($0.id.prefix(8)))}) {value in selectedProjectID=value;refresh()}
                }
            }
            Button(job==nil ? "Refresh media inventory":"Reviewing captured assets…",action:refresh).buttonStyle(FlatButtonStyle()).disabled(job != nil)
            if let report {
                Text("\(report.items.count) captured assets · \(ByteCountFormatter.string(fromByteCount:report.items.reduce(0){$0+$1.bytes},countStyle:.file))").font(.callout)
                Text("Metadata checked on refresh; content pins are verified when opened. All listed copies are protected by retained evidence.").font(.caption).foregroundStyle(MLTheme.muted)
                ForEach(["imagePixels","audioSamples","preparedTensor"],id:\.self) {kind in
                    let items=report.items.filter{$0.kind==kind}
                    if !items.isEmpty {
                        HStack {
                            Label(kind=="imagePixels" ? "Captured images":kind=="audioSamples" ? "PCM audio":"Prepared tensors",systemImage:kind=="audioSamples" ? "waveform":"square.stack.3d.up")
                            Spacer();Text("\(items.count) · \(ByteCountFormatter.string(fromByteCount:items.reduce(0){$0+$1.bytes},countStyle:.file))")
                        }.font(.caption).padding(10).overlay(Rectangle().stroke(MLTheme.rule))
                    }
                }
                Text("\(report.inspectedRecords) operation/run records checked · \(report.completedRecordsWithoutMedia) completed preparation records without captured media").font(.caption).foregroundStyle(MLTheme.muted)
                if report.protectedRecords>0 {
                    VStack(alignment:.leading,spacing:4) {
                        Label("\(report.protectedRecords) records outside the media inventory remain protected.",systemImage:"lock.shield")
                        if report.recordsWithoutManifest>0 {Text("\(report.recordsWithoutManifest) without a media manifest at the expected location")}
                        if report.otherManifestRecords>0 {Text("\(report.otherManifestRecords) with other manifest formats or states")}
                        if report.unreadableRecords>0 {Text("\(report.unreadableRecords) with unreadable or invalid media metadata")}
                    }.font(.caption).foregroundStyle(MLTheme.muted)
                }
            }
            if !message.isEmpty {Text(message).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled)}
            Text("Retained media cannot be removed here. Active, unknown and archived references stay protected. Use the temporary-files review in Settings → Models & assets to remove only registered, inactive staging data. Cleanup rechecks ownership and leases before deleting.").font(.caption).foregroundStyle(MLTheme.muted)
        }.task(id:store.workspaceDirectory) {refresh()}.onChange(of:projectID){_,_ in refresh()}.onDisappear{job?.cancel()}
    }
    private func refresh() {
        job?.cancel();let root=store.workspaceDirectory,selected=scope
        message="";report=nil
        if projectID==nil && projectOnly && selected==nil {message="Select a saved project to review its captured assets.";job=nil;return}
        job=Task {
            let worker=Task.detached(priority:.utility) {try MediaLibraryInventory.scan(workspace:root,projectID:selected)}
            do {
                let result=try await withTaskCancellationHandler(operation:{try await worker.value},onCancel:{worker.cancel()})
                guard !Task.isCancelled,store.workspaceDirectory==root,scope==selected else {return}
                report=result;job=nil
            } catch {if !Task.isCancelled {message=error.localizedDescription;job=nil}}
        }
    }
}
