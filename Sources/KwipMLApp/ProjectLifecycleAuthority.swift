import Foundation
import KwipMLCore

/// Managed workflow lifecycle reads and transitions borrow native owners. Status
/// never opens a window; close never saves a dirty canvas or discards a draft.
@MainActor enum ProjectLifecycleAuthority {
    struct SavedVersion:Codable,Equatable {let id:String;let number:Int;let hash:String}
    struct Snapshot:Codable,Equatable {
        let projectID:String;let name:String;let open:Bool;let revision:UInt64
        let sessionID:String?;let busy:Bool;let pendingInspectorEdits:Bool
        let hasDrafts:Bool;let dirty:Bool;let nativeUnsavedChanges:Bool;let savedVersion:SavedVersion?
        var lifecycleRevision:String=""
    }
    struct Failure:Error {let code:String}
    private static func opened(_ id:String,root:URL,windows:ProjectWindowRegistry)->ProjectWindowController? {
        windows.openControllers.first {$0.session.workspace.activeProjectID==id && $0.session.workspace.applicationDirectory==root && $0.session.workspace.externalLocation==nil}
    }
    static func snapshot(projectID:String,root:URL,windows:ProjectWindowRegistry)throws->Snapshot {
        guard UUID(uuidString:projectID)?.uuidString==projectID else {throw Failure(code:"project_unavailable")}
        let window=opened(projectID,root:root,windows:windows),workspace=window?.session.workspace
        let owner=workspace?.workflowSession?.owner
        let document:WorkflowDocument
        if let owner {document=owner.document}
        else {
            guard case .workflow(let source)=try ProjectLibrary(directory:root.appendingPathComponent("projects")).loadSource(id:projectID) else {throw Failure(code:"project_lifecycle_unsupported")}
            document=source
        }
        let journal=try CanvasVersionJournal(root:root,projectID:projectID)
        let version=journal.versions.first.map {SavedVersion(id:$0.id,number:$0.number,hash:$0.hash)}
        var value=Snapshot(projectID:projectID,name:document.name,open:window != nil,revision:document.revision,
            sessionID:owner?.sessionID.uuidString,busy:workspace.map {$0.isBusy || $0.workflowSession?.hasActiveOperation==true || owner?.isBusy==true} ?? false,
            pendingInspectorEdits:owner?.hasPendingInspectorEdits ?? false,hasDrafts:!document.drafts.isEmpty,
            dirty:!(try journal.matchesSavedDocument(document)),nativeUnsavedChanges:workspace?.hasUnsavedChanges ?? false,savedVersion:version)
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        value.lifecycleRevision=WorkflowArchiveCodec.hash(try encoder.encode(value))
        return value
    }
    private static func current(projectID:String,revision:String,root:URL,windows:ProjectWindowRegistry)throws->(Snapshot,ProjectWindowController) {
        let value=try snapshot(projectID:projectID,root:root,windows:windows)
        guard value.lifecycleRevision==revision else {throw Failure(code:"project_lifecycle_stale")}
        guard let window=opened(projectID,root:root,windows:windows),window.session.workspace.workflowSession != nil else {throw Failure(code:"project_not_open")}
        guard !value.busy,!value.pendingInspectorEdits,!value.hasDrafts else {throw Failure(code:"project_lifecycle_busy_or_draft")}
        return(value,window)
    }
    static func rename(projectID:String,name:String,revision:String,root:URL,windows:ProjectWindowRegistry)throws->Snapshot {
        guard !name.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,name.count<=80,!name.contains("\0"),!name.contains("\n"),!name.contains("\r") else {throw Failure(code:"project_name_invalid")}
        let (_,window)=try current(projectID:projectID,revision:revision,root:root,windows:windows)
        let owner=window.session.workspace.workflowSession!.owner
        guard owner.rename(name,lease:owner.lease()) else {throw Failure(code:"project_rename_refused")}
        window.session.workspace.invalidateProjectCatalogSnapshot();window.session.workspace.refreshProjects()
        return try snapshot(projectID:projectID,root:root,windows:windows)
    }
    static func close(projectID:String,revision:String,root:URL,windows:ProjectWindowRegistry)throws->Snapshot {
        let (value,window)=try current(projectID:projectID,revision:revision,root:root,windows:windows)
        guard !value.dirty,!value.nativeUnsavedChanges else {throw Failure(code:"project_unsaved_version")}
        guard windows.close(window) else {throw Failure(code:"project_close_refused")}
        let result=try snapshot(projectID:projectID,root:root,windows:windows)
        guard !result.open else {throw Failure(code:"project_close_incomplete")}
        return result
    }
}
