import SwiftUI
import AppKit
import KwipMLCore

struct ModelLibraryItem:Identifiable,Equatable,Sendable {
    let id:String
    let directory:URL
    let projectID:String
    let architecture:String
    let projectName:String?
    let bytes:Int64
    let fileCount:Int
}

/// Metadata-only inventory. Never hashes weights, follows links, grants sources or deletes assets.
enum ModelLibraryInventory {
    static func scan(application:URL,projectNames:[String:String]=[:])throws->[ModelLibraryItem] {
        var result:[ModelLibraryItem]=[],visited=0
        func directories(_ parent:URL)throws->[URL] {
            guard FileManager.default.fileExists(atPath:parent.path) else {return []}
            let values=try parent.resourceValues(forKeys:[.isDirectoryKey,.isSymbolicLinkKey])
            guard values.isDirectory == true,values.isSymbolicLink != true else {throw WorkflowFileLeaseError.unsafePath}
            let entries=try FileManager.default.contentsOfDirectory(at:parent,includingPropertiesForKeys:[.isDirectoryKey,.isSymbolicLinkKey])
            visited+=entries.count;guard visited<=10000 else {throw WorkflowArchiveError("Model inventory exceeds its bounded scan. Existing assets were retained.")}
            return try entries.filter {entry in
                try Task.checkCancellation()
                let value=try entry.resourceValues(forKeys:[.isDirectoryKey,.isSymbolicLinkKey])
                guard value.isSymbolicLink != true else {throw WorkflowFileLeaseError.unsafePath}
                return value.isDirectory == true && UUID(uuidString:entry.lastPathComponent) != nil
            }
        }
        let workspaces=[application]+(try directories(application.appendingPathComponent("external-projects")))
        for workspace in workspaces {
            for project in try directories(workspace.appendingPathComponent("model-assets")) {
                for asset in try directories(project) {
                    try Task.checkCancellation()
                    let leaves=try FileManager.default.contentsOfDirectory(at:asset,includingPropertiesForKeys:[.isRegularFileKey,.isSymbolicLinkKey,.fileSizeKey])
                    visited+=leaves.count;guard visited<=10000,leaves.count<=129 else {throw WorkflowArchiveError("Model inventory exceeds its bounded scan. Existing assets were retained.")}
                    var size:Int64=0
                    for file in leaves {
                        let value=try file.resourceValues(forKeys:[.isRegularFileKey,.isSymbolicLinkKey,.fileSizeKey])
                        guard value.isRegularFile == true,value.isSymbolicLink != true,let bytes=value.fileSize,bytes>=0 else {throw WorkflowFileLeaseError.unsafePath}
                        let (next,overflow)=size.addingReportingOverflow(Int64(bytes));guard !overflow else {throw WorkflowFileLeaseError.unsafePath};size=next
                    }
                    var architecture="Model snapshot"
                    if let data=try DiagnosticFileIO.readIfPresent(root:asset,parts:["config.json"],limit:8_000_000),let config=try JSONSerialization.jsonObject(with:data) as? [String:Any],let name=config["model_type"] as? String,name.utf8.count<=100 {architecture=name.uppercased()}
                    result.append(.init(id:asset.path,directory:asset,projectID:project.lastPathComponent,architecture:architecture,projectName:projectNames[project.lastPathComponent],bytes:size,fileCount:leaves.count))
                }
            }
        }
        return result.sorted{$0.id<$1.id}
    }
}

struct ModelLibrarySettingsView:View {
    @ObservedObject var store:WorkspaceStore
    @State private var items:[ModelLibraryItem]=[]
    @State private var loaded=false
    @State private var scanning=false
    @State private var job:Task<Void,Never>?
    @State private var message=""
    @State private var credentialNames:[String]=[]
    @State private var credentialSheet=false
    @State private var name=""
    @State private var token=""
    @State private var removing:String?
    var body:some View {
        VStack(alignment:.leading,spacing:16) {
            SectionLabel(title:"Local model snapshots")
            Text("Imported and downloaded models are copied into managed storage. Existing projects and recorded generations may need these exact weights and tokenizers for offline use.").font(.caption).foregroundStyle(MLTheme.muted)
            Button(scanning ? "Reviewing models…":"Refresh model inventory",action:refresh).disabled(scanning)
            if loaded {
                Text("\(items.count) snapshots · \(ByteCountFormatter.string(fromByteCount:items.reduce(0){$0+$1.bytes},countStyle:.file))").font(.callout)
                LazyVStack(alignment:.leading,spacing:8) {ForEach(items) {item in
                    HStack(alignment:.top,spacing:12) {
                        Image(systemName:"shippingbox").foregroundStyle(MLTheme.rose)
                        VStack(alignment:.leading,spacing:4) {
                            Text(item.projectName ?? "Project \(item.projectID.prefix(8))").font(.callout.weight(.semibold))
                            Text("\(item.architecture) · \(item.fileCount) files · \(ByteCountFormatter.string(fromByteCount:item.bytes,countStyle:.file))").font(.caption).foregroundStyle(MLTheme.muted)
                            Text(item.directory.lastPathComponent).font(.caption2.monospaced()).foregroundStyle(MLTheme.muted)
                        }
                        Spacer()
                        Button("Show in Finder") {NSWorkspace.shared.activateFileViewerSelecting([item.directory])}
                    }.padding(10).frame(maxWidth:.infinity,alignment:.leading).overlay(Rectangle().stroke(MLTheme.rule))
                }}
            }
            Text("These retained copies are not disposable cache. Verification runs when a model is used; this inventory does not load weights. Removing temporary downloads below preserves completed model snapshots and History.").font(.caption).foregroundStyle(MLTheme.muted)
            Divider()
            OCRPackSettingsView(store:store)
            Divider()
            MediaLibrarySettingsView(store:store)
            Divider()
            SectionLabel(title:"Hugging Face credentials")
            Text("Saved privately in this Mac’s Keychain. Removing a credential prevents future requests using it; admitted requests and local model copies remain. Revoke the token with Hugging Face to revoke it at the provider.").font(.caption).foregroundStyle(MLTheme.muted)
            ForEach(credentialNames,id:\.self) {credential in
                HStack {Text(credential).font(.callout);Spacer();Button("Update") {name=credential;token="";credentialSheet=true};Button("Remove") {removing=credential}}
                    .padding(9).overlay(Rectangle().stroke(MLTheme.rule))
            }
            Button("Add credential…") {name="";token="";credentialSheet=true}
            if !message.isEmpty {Text(message).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled)}
            Divider()
            SectionLabel(title:"Model task limits")
            Text("GPT-2/Llama causal models, Whisper speech recognition, AST audio classification and reviewed ModernBERT Laya typed decisions, each with its matching tokenizer or processor. Safe weights and matching tokenizers are required. Whisper is CPU inference through Transcribe audio; it does not use causal fine-tuning. Quantized formats, GGUF and repository code are not supported.").font(.caption).foregroundStyle(MLTheme.muted)
            Text("Model source / trained weights: 2 GB. With no worker RSS threshold, imported tasks use a separate 2 GB memory-admission allowance. Increasing the RSS threshold does not raise the file-format ceiling. Non-training model tasks may be limited to 300 seconds; training uses the captured run deadline.").font(.caption).foregroundStyle(MLTheme.muted)
            Text("Hub requests allow only the built-in HTTPS provider/CDN hosts and reviewed model files. Private access does not grant arbitrary URLs or runtime network access.").font(.caption).foregroundStyle(MLTheme.muted)
            Divider()
            ScratchCleanupSettingsView(store:store)
        }.task {refresh()}.onDisappear{job?.cancel()}
        .sheet(isPresented:$credentialSheet) {
            VStack(alignment:.leading,spacing:12) {
                Text("Hugging Face credential").font(.headline)
                TextField("Credential name",text:$name).flatField()
                SecureField("Access token",text:$token).flatField()
                Text("The token is never included in a project, History or MCP response.").font(.caption).foregroundStyle(MLTheme.muted)
                if !message.isEmpty {Text(message).font(.caption).foregroundStyle(MLTheme.rose)}
                HStack {Button("Cancel") {token="";credentialSheet=false};Button("Save credential") {do {try ModelHubCredentials.save(name:name,token:token);token="";credentialSheet=false;credentialNames=try ModelHubCredentials.names();message=""} catch {message=error.localizedDescription}}}
            }.padding(20).frame(width:360).onDisappear{token=""}
        }
        .confirmationDialog("Remove this saved Hub credential?",isPresented:Binding(get:{removing != nil},set:{if !$0 {removing=nil}}),titleVisibility:.visible) {
            Button("Remove credential",role:.destructive) {if let removing {do {try ModelHubCredentials.remove(name:removing);credentialNames=try ModelHubCredentials.names();message="Credential removed from this Mac."} catch {message=error.localizedDescription}};removing=nil}
        } message:{Text("Local models and captured results remain available. Projects using this credential will need another selection before private downloads.")}
    }
    private func refresh() {
        guard !scanning else{return};scanning=true
        let root=store.applicationDirectory
        let names=Dictionary(store.projects.map{($0.id,$0.name)}+store.externalProjects.map{($0.projectID,$0.title)},uniquingKeysWith:{first,_ in first})
        job=Task {
            defer {scanning=false;job=nil}
            do {
                let worker=Task.detached(priority:.utility){try ModelLibraryInventory.scan(application:root,projectNames:names)}
                let items=try await withTaskCancellationHandler(operation:{try await worker.value},onCancel:{worker.cancel()})
                try Task.checkCancellation()
                self.items=items;loaded=true;credentialNames=try ModelHubCredentials.names();message=""
            } catch {message=error.localizedDescription}
        }
    }
}
