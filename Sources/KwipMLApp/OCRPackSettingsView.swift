import SwiftUI
import KwipMLCore

struct OCRPackSettingsView:View {
    @ObservedObject var store:WorkspaceStore
    @State private var installed=false
    @State private var status=""
    @State private var job:Task<Void,Never>?
    @State private var showReview=false
    @State private var references:Int?
    @State private var confirmRemove=false
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            SectionLabel(title:"Optional OCR model packs")
            Text("English printed text · CPU").fontWeight(.semibold)
            Text(installed ? "Verified archives · available offline":"Not installed").foregroundStyle(installed ? MLTheme.teal:MLTheme.muted)
            Text("CRAFT detector + English recognizer · "+ByteCountFormatter.string(fromByteCount:Int64(OCRPackCatalog.bytes),countStyle:.file)+" download. Not included at first installation.").font(.caption).foregroundStyle(MLTheme.muted)
            if !installed {
                Button("Review model pack…"){showReview=true}.buttonStyle(FlatButtonStyle()).disabled(job != nil)
            } else {
                Text(references.map {"\($0) retained graph/reference records · required copies are protected"} ?? "Reference inventory unavailable; removal is protected").font(.caption).foregroundStyle(MLTheme.muted)
                Button("Remove unused pack…"){confirmRemove=true}.buttonStyle(FlatButtonStyle()).disabled(job != nil || references != 0 || store.isBusy || currentProjectUsesOCR)
                Button("Verify installed pack",action:verify).buttonStyle(FlatButtonStyle()).disabled(job != nil)
            }
            if job != nil {Button("Cancel download"){job?.cancel()}.buttonStyle(FlatButtonStyle())}
            if !status.isEmpty {Text(status).font(.caption).foregroundStyle(MLTheme.muted).textSelection(.enabled)}
        }.confirmationDialog("Remove the unused English OCR pack?",isPresented:$confirmRemove) {
            Button("Remove pack",role:.destructive){remove()}
        } message:{Text("Only these downloaded, integrity-verified model archives are removed. You can download the same pack again. Source files, projects and History are retained.")}
        .task(id:store.workspaceDirectory){verify()}
        .sheet(isPresented:$showReview) {
            VStack(alignment:.leading,spacing:14) {
                Text("English OCR pack").font(.headline)
                Text("Download only the pinned detector and recognizer from EasyOCR’s official GitHub releases. The isolated worker loads only reviewed tensors; it cannot download models or execute repository code.")
                Text("Download: "+ByteCountFormatter.string(fromByteCount:Int64(OCRPackCatalog.bytes),countStyle:.file)+". Allow another 99 MB of transient uncompressed weights plus model/processing memory when running. Existing project resource limits still apply.")
                Text("EasyOCR1.7.2 · Apache2.0; CRAFT detector code · MIT. Exact asset sources and hashes are retained in the application catalog. Printed English only; handwriting and complex page layouts are not qualified.").font(.caption).foregroundStyle(MLTheme.muted)
                Link("Review upstream license",destination:URL(string:"https://github.com/JaidedAI/EasyOCR/blob/v1.7.2/LICENSE")!)
                HStack {Button("Cancel"){showReview=false}.buttonStyle(FlatButtonStyle());Button("Download pack"){showReview=false;download()}.buttonStyle(FlatButtonStyle(prominent:true))}
            }.padding(20).frame(width:470)
        }.onDisappear {job?.cancel()}
    }
    private var currentProjectUsesOCR:Bool {store.workflowSession?.owner.document.nodes.contains(where:{node in if case .dataOperator(let id,_)=node.payload {return id==OCRConfiguration.operatorID};return false}) == true}
    private func verify(){guard job==nil else{return};let root=store.workspaceDirectory;job=Task {
        let value=await Task.detached(priority:.utility){(OCRPackStore.installed(workspace:root),try? OCRPackStore.referenceCount(workspace:root))}.value
        guard !Task.isCancelled else {job=nil;return};installed=value.0;references=value.1;job=nil
    }}
    private func remove(){guard job==nil,!store.isBusy,!currentProjectUsesOCR else{return};let root=store.workspaceDirectory;job=Task {
        do {try await Task.detached(priority:.utility){try OCRPackStore.removeUnreferenced(workspace:root)}.value;installed=false;status="Unused English OCR pack removed. Projects and History were retained."} catch {status=error.localizedDescription}
        job=nil
    }}
    private func download(){guard job==nil else{return};let root=store.workspaceDirectory,accounting=store.applicationDirectory
        job=Task {
            do {
                let policy=try store.executionPolicy(for:nil)
                try await OCRPackStore.download(workspace:root,accountingRoot:accounting,limit:policy.storageLimitBytes,maximumSeconds:Double(policy.budget.maximumRunSeconds)){text in Task {@MainActor in status=text}}
                installed=true;references=nil;status="English OCR verified. Available offline in this workspace. Refresh verification to review retained references."
            } catch {status=Task.isCancelled ? "Download cancelled; no partial pack was installed.":error.localizedDescription}
            job=nil
        }
    }
}
