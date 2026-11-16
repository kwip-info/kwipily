import SwiftUI
import AppKit
import KwipMLCore

struct WorkflowGenerationHistoryView:View {
    let record:WorkflowGenerationRecord
    @ObservedObject var workspace:WorkspaceStore
    let isActive:()->Bool
    @State private var query:TableViewQuerySession?
    @State private var message="Verifying captured generation…"
    @State private var folderName="KwipML generation"
    @State private var exporting=false
    var body:some View {
        VStack(alignment:.leading,spacing:8) {
            HStack {
                TextField("Package folder name",text:$folderName).flatField().frame(maxWidth:220)
                    .accessibilityLabel("Generation package folder name").disabled(exporting)
                Spacer()
                Button("Export replay package…") {export()}.buttonStyle(FlatButtonStyle())
                    .disabled(query==nil || workspace.isBusy || exporting || !validFolderName)
            }
            if let query {TableViewOutputView(session:query,executionBusy:workspace.isBusy)}
            if !message.isEmpty {Text(message).font(.caption).foregroundStyle(MLTheme.muted)}
        }.task(id:record.id) {
            do {
                let root=workspace.workspaceDirectory,verified=try await WorkflowHistoryVerificationWork.run {try record.verify(root:root)}
                guard !Task.isCancelled,isActive() else {return}
                query=workspace.makeTableViewQuerySession(artifact:verified.artifact,projectID:record.document.projectID,accepts:isActive)
                message=""
            } catch {message=error.localizedDescription}
        }.onDisappear {query?.cancelIfUnowned()}
    }
    private var validFolderName:Bool {
        (try? WorkflowExportPublication.destination(in:URL(fileURLWithPath:"/"),folderName:folderName)) != nil
    }
    private func export() {
        guard !exporting,!workspace.isBusy,validFolderName,isActive() else {return}
        let name=folderName
        let panel=NSOpenPanel();panel.title="Export generation replay"
        panel.canChooseDirectories=true;panel.canChooseFiles=false;panel.allowsMultipleSelection=false;panel.canCreateDirectories=true
        panel.prompt="Save package";panel.message="Create a new folder named “"+name+"”. Existing folders will not be replaced."
        exporting=true
        guard panel.runModal() == .OK,let access=panel.url else {exporting=false;return}
        let started=access.startAccessingSecurityScopedResource()
        message="Verifying and exporting…"
        Task {
            defer {exporting=false;if started {access.stopAccessingSecurityScopedResource()}}
            do {
                let destination=try WorkflowExportPublication.destination(in:access,folderName:name)
                let root=workspace.workspaceDirectory,pending=try await WorkflowHistoryVerificationWork.run {try WorkflowGenerationExport.stage(record,root:root,to:destination)}
                defer {pending.close()};guard isActive(),!Task.isCancelled else {throw CancellationError()}
                try pending.commit();message="Replay package saved"
            } catch {message=error.localizedDescription}
        }
    }
}
