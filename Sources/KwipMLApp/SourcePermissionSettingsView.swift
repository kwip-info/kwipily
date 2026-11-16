import SwiftUI

struct SourcePermissionSettingsView:View {
    @ObservedObject var store:WorkspaceStore
    @State private var listing:SourceAccess.PermissionListing?
    @State private var search=""
    @State private var message=""
    @State private var refresh=0
    @State private var selected:SourceAccess.PermissionEntry?
    private var entries:[SourceAccess.PermissionEntry] {
        (listing?.entries ?? []).filter {search.isEmpty || ($0.path ?? $0.id).localizedCaseInsensitiveContains(search)}
    }
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            HStack {
                SectionLabel(title:"Source permissions")
                Spacer()
                Button("Refresh permissions") {message="";refresh+=1}
            }
            Text("Saved access to local files on this Mac. Revoking blocks future reads of that source path across projects until you choose the file again. It does not delete source files, project references, cached data or captured run artifacts.")
                .font(.caption).foregroundStyle(MLTheme.muted)
            if store.isBusy {Text("Stop the active workspace operation before revoking access.").font(.caption).foregroundStyle(MLTheme.rose)}
            TextField("Search source paths",text:$search).flatField()
                .accessibilityLabel("Search source permissions")
            if let listing {
                if entries.isEmpty {Text(search.isEmpty ? "No saved source permissions.":"No matching source permissions.").font(.caption).foregroundStyle(MLTheme.muted)}
                else {
                    ThemedScrollView {
                        LazyVStack(alignment:.leading,spacing:8) {
                            ForEach(entries) {entry in
                                HStack(alignment:.top,spacing:12) {
                                    VStack(alignment:.leading,spacing:4) {
                                        Text(entry.path.map {URL(fileURLWithPath:$0).lastPathComponent} ?? "Unresolved source · \(entry.id.prefix(8))").font(.caption.bold())
                                        if let path=entry.path {Text(path).font(.caption2).foregroundStyle(MLTheme.muted).textSelection(.enabled)}
                                        Text(stateLabel(entry.state)).font(.caption)
                                        if let detail=entry.detail {Text(detail).font(.caption2).foregroundStyle(MLTheme.muted)}
                                    }.frame(maxWidth:.infinity,alignment:.leading)
                                    if entry.state != .revoked {
                                        Button("Revoke…") {selected=entry}.disabled(store.isBusy)
                                            .accessibilityLabel("Revoke access to \(entry.path ?? String(entry.id.prefix(8)))")
                                    }
                                }.padding(10).background(MLTheme.panel)
                            }
                        }
                    }.frame(height:260)
                }
                if listing.truncated {Text("Permission scan limit reached. This list is incomplete.").font(.caption).foregroundStyle(MLTheme.rose)}
                ForEach(Array(listing.issues.enumerated()),id:\.offset) {_,issue in Text(issue).font(.caption2).foregroundStyle(MLTheme.rose)}
            } else {Text("Reading saved permissions…").font(.caption).foregroundStyle(MLTheme.muted)}
            if !message.isEmpty {Text(message).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled)}
        }
        .onReceive(NotificationCenter.default.publisher(for:SourceAccess.permissionsDidChange).receive(on:RunLoop.main)) {event in
            guard event.object as? String == store.sourceAccess.directory.standardizedFileURL.path else {return}
            message="";refresh+=1
        }
        .task(id:refresh) {
            let access=store.sourceAccess
            let worker=Task.detached(priority:.utility) {try access.listPermissions()}
            do {
                let result=try await withTaskCancellationHandler(operation:{try await worker.value},onCancel:{worker.cancel()})
                guard !Task.isCancelled else {return};listing=result
            } catch {guard !Task.isCancelled else {return};message=error.localizedDescription}
        }
        .alert("Revoke source access?",isPresented:Binding(get:{selected != nil},set:{if !$0 {selected=nil}}),presenting:selected) {entry in
            Button("Revoke access",role:.destructive) {
                do {try store.revokeSourcePermission(entry);message="Access revoked. Choose the source again in its component to restore access.";refresh+=1}
                catch {message=error.localizedDescription}
                selected=nil
            }
            Button("Cancel",role:.cancel) {selected=nil}
        } message: {entry in
            Text("\(entry.path ?? "Unresolved permission \(entry.id.prefix(8))")\n\nFuture reads of this path will be blocked across projects. Existing files, cached data and run artifacts remain. Selecting the source again restores access.")
        }
    }
    private func stateLabel(_ state:SourceAccess.PermissionEntry.State)->String {
        switch state {
        case .saved:return "Saved permission · access checked on use"
        case .revoked:return "Revoked · choose source again to restore"
        case .unavailable:return "Permission unavailable · choose source again or revoke"
        }
    }
}
