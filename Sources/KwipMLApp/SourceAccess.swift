import Foundation
import CryptoKit
import KwipMLCore

/// Grants are private application state, never embedded in portable project documents.
final class SourceAccess {
    static let permissionsDidChange=Notification.Name("KwipML.sourcePermissionsDidChange")
    private func notifyPermissionChange() {
        NotificationCenter.default.post(name:Self.permissionsDidChange,object:directory.standardizedFileURL.path)
    }
    private static let revokedRecord=Data("KwipML source access revoked v1\n".utf8)
    let directory: URL
    init(directory: URL) { self.directory = directory }
    private func recordURL(_ path: String) -> URL {
        let key = SHA256.hash(data: Data(path.utf8)).map { String(format: "%02x", $0) }.joined()
        return directory.appendingPathComponent(key + ".bookmark")
    }
    func remember(_ url: URL) throws {
        let started = url.startAccessingSecurityScopedResource()
        defer { if started { url.stopAccessingSecurityScopedResource() } }
        let bookmark = try url.bookmarkData(options: [.withSecurityScope, .securityScopeAllowOnlyReadAccess], includingResourceValuesForKeys: nil, relativeTo: nil)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try bookmark.write(to: recordURL(url.path), options: .atomic)
        notifyPermissionChange()
    }
    /// Replaces only this path's private bookmark with a durable denial marker.
    /// Existing captured artifacts and active scopes are not deleted or recalled.
    func revoke(path:String)throws {
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        try (Self.revokedRecord+Data(path.utf8)).write(to:recordURL(path),options:.atomic)
        notifyPermissionChange()
    }
    struct PermissionEntry:Identifiable,Equatable {
        enum State:Equatable {case saved,revoked,unavailable}
        let id:String
        let path:String?
        let state:State
        let recordSHA256:String
        let detail:String?
    }
    struct PermissionListing {
        let entries:[PermissionEntry]
        let issues:[String]
        let truncated:Bool
    }
    private func validRecordID(_ id:String)->Bool {
        id.hasSuffix(".bookmark") && id.count == 73 && id.dropLast(9).allSatisfy {"0123456789abcdef".contains($0)}
    }
    /// Enumerates private permission records without reading the source data or starting scopes.
    func listPermissions(limit:Int=1000)throws->PermissionListing {
        guard (1...1000).contains(limit) else {throw accessError("Permission listing limit must be between 1 and 1000.")}
        guard FileManager.default.fileExists(atPath:directory.path) else {return .init(entries:[],issues:[],truncated:false)}
        guard let scan=FileManager.default.enumerator(at:directory,includingPropertiesForKeys:nil,options:[.skipsSubdirectoryDescendants]) else {
            throw accessError("Saved permissions could not be listed.")
        }
        var entries:[PermissionEntry]=[],issues:[String]=[],seen=0,truncated=false
        while let file=scan.nextObject() as? URL {
            try Task.checkCancellation()
            seen+=1
            if seen>limit {truncated=true;break}
            let id=file.lastPathComponent
            guard validRecordID(id) else {issues.append("Unrecognized permission record: \(id)");continue}
            do {
                let bytes=try DiagnosticFileIO.read(root:directory,parts:[id],limit:1_000_000)
                let hash=WorkflowArchiveCodec.hash(bytes)
                if bytes.starts(with:Self.revokedRecord) {
                    let text=String(data:bytes.dropFirst(Self.revokedRecord.count),encoding:.utf8)
                    let path=text.flatMap {$0.isEmpty ? nil:$0}
                    guard path == nil || recordURL(path!).lastPathComponent == id else {throw accessError("Revoked record path does not match its identity.")}
                    entries.append(.init(id:id,path:path,state:.revoked,recordSHA256:hash,detail:nil))
                } else {
                    do {
                        var stale=false
                        let url=try URL(resolvingBookmarkData:bytes,options:[.withSecurityScope,.withoutUI],relativeTo:nil,bookmarkDataIsStale:&stale)
                        let candidates=[url.path,url.standardizedFileURL.path]
                        guard let path=candidates.first(where:{recordURL($0).lastPathComponent == id}) else {
                            throw accessError("Source moved; choose it again in its component.")
                        }
                        entries.append(.init(id:id,path:path,state:.saved,recordSHA256:hash,
                            detail:stale ? "Saved permission needs refresh on the next read.":nil))
                    } catch {
                        entries.append(.init(id:id,path:nil,state:.unavailable,recordSHA256:hash,detail:error.localizedDescription))
                    }
                }
            } catch {issues.append("Permission \(id.prefix(8)): \(error.localizedDescription)")}
        }
        entries.sort {($0.path ?? $0.id).localizedStandardCompare($1.path ?? $1.id) == .orderedAscending}
        return .init(entries:entries,issues:issues,truncated:truncated)
    }
    /// Optimistic same-process guard: the settings host must also prohibit active operations.
    func revoke(entry:PermissionEntry)throws {
        guard validRecordID(entry.id) else {throw accessError("Invalid permission selection.")}
        let bytes=try DiagnosticFileIO.read(root:directory,parts:[entry.id],limit:1_000_000)
        guard WorkflowArchiveCodec.hash(bytes) == entry.recordSHA256 else {throw accessError("Permission changed. Refresh the list before revoking it.")}
        if let path=entry.path,recordURL(path).lastPathComponent != entry.id {throw accessError("Permission identity changed.")}
        try (Self.revokedRecord+Data((entry.path ?? "").utf8)).write(to:directory.appendingPathComponent(entry.id),options:.atomic)
        notifyPermissionChange()
    }
    func withAccess<T>(to path: String, _ body: (URL) throws -> T) throws -> T {
        let record = recordURL(path)
        var url = URL(fileURLWithPath: path)
        if FileManager.default.fileExists(atPath: record.path) {
            let size = try record.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
            guard size <= 1_000_000 else { throw accessError("Saved source permission is unreadable. Choose the source again.") }
            let bookmark=try Data(contentsOf:record)
            guard !bookmark.starts(with:Self.revokedRecord) else {
                throw accessError("Access to this source was revoked. Choose the file again to restore access.")
            }
            var stale = false
            url = try URL(resolvingBookmarkData: bookmark, options: [.withSecurityScope, .withoutUI], relativeTo: nil, bookmarkDataIsStale: &stale)
            // Keep project identity explicit when a bookmark follows a moved file.
            guard url.standardizedFileURL.path == URL(fileURLWithPath: path).standardizedFileURL.path else { throw accessError("Source moved. Choose its new location in the source component’s Configuration to update the project reference.") }
            if stale { try remember(url) }
        }
        let started = url.startAccessingSecurityScopedResource()
        defer { if started { url.stopAccessingSecurityScopedResource() } }
        return try body(url)
    }
    /// Off-main, bounded group capture. Security scope is held through each whole
    /// read; only private copies are returned to the eventual execution host.
    func captureExperimentInputs(paths:[WorkflowNodeAddress:String],parent:URL,fileConfigurations:[WorkflowNodeAddress:[String:JSONValue]]=[:],workspace:URL?=nil,accountingRoot:URL?=nil,storageLimitBytes:Int64?=nil) async throws->WorkflowExperimentInputs {
        try await captureExperimentInputs(paths:paths,destination:parent,callerOwned:false,fileConfigurations:fileConfigurations,workspace:workspace,accountingRoot:accountingRoot,storageLimitBytes:storageLimitBytes)
    }
    /// The host retains ownership until detached reads finish, including cancellation.
    /// Caller-owned directories and stable ownership files are never cleaned up here.
    func captureExperimentInputs(paths:[WorkflowNodeAddress:String],directory:URL,fileConfigurations:[WorkflowNodeAddress:[String:JSONValue]]=[:],workspace:URL?=nil,accountingRoot:URL?=nil,storageLimitBytes:Int64?=nil) async throws->WorkflowExperimentInputs {
        try await captureExperimentInputs(paths:paths,destination:directory,callerOwned:true,fileConfigurations:fileConfigurations,workspace:workspace,accountingRoot:accountingRoot,storageLimitBytes:storageLimitBytes)
    }
    private func captureExperimentInputs(paths:[WorkflowNodeAddress:String],destination:URL,callerOwned:Bool,fileConfigurations:[WorkflowNodeAddress:[String:JSONValue]],workspace:URL?,accountingRoot:URL?,storageLimitBytes:Int64?) async throws->WorkflowExperimentInputs {
        try Task.checkCancellation()
        if !fileConfigurations.isEmpty {
            guard let workspace else {throw WorkflowArchiveError("File collection experiments require owned source staging.")}
            let captured=try await WorkflowRunSourceCapture.capture(paths:paths,workspace:workspace,accountingRoot:accountingRoot,sourcePermissions:directory,storageLimitBytes:storageLimitBytes,fileConfigurations:fileConfigurations)
            defer{captured.remove()}
            let worker=Task.detached(priority:.userInitiated) {
                if callerOwned {return try WorkflowExperimentInputs.captureSources(addresses:Array(paths.keys),directory:destination,sources:captured.sources)}
                return try WorkflowExperimentInputs.captureSources(addresses:Array(paths.keys),parent:destination,sources:captured.sources)
            }
            return try await withTaskCancellationHandler(operation:{try await worker.value},onCancel:{worker.cancel()})
        }
        let access=SourceAccess(directory:directory)
        let worker=Task.detached(priority:.userInitiated) {
            let read:(WorkflowNodeAddress)throws->Data = {address in
                try access.withAccess(to:paths[address]!) {url in
                    do { return try DiagnosticFileIO.readSelectedFile(url,limit:10_000_000) }
                    catch {
                        throw access.accessError("Cannot read the selected source as a stable, bounded regular file. Choose the source again to restore access, then retry the experiment.")
                    }
                }
            }
            if callerOwned {return try WorkflowExperimentInputs.captureBytes(addresses:Array(paths.keys),directory:destination,sourceURLs:paths.mapValues { URL(fileURLWithPath:$0) },read:read)}
            return try WorkflowExperimentInputs.captureBytes(addresses:Array(paths.keys),parent:destination,sourceURLs:paths.mapValues { URL(fileURLWithPath:$0) },read:read)
        }
        let result=try await withTaskCancellationHandler(operation:{try await worker.value},onCancel:{worker.cancel()})
        // Await above joins the detached capture before the host can release ownership.
        guard !Task.isCancelled else {
            if !callerOwned {try? FileManager.default.removeItem(at:result.directory)}
            throw CancellationError()
        }
        return result
    }
    private func accessError(_ message: String) -> NSError { NSError(domain: "KwipML.SourceAccess", code: 1, userInfo: [NSLocalizedDescriptionKey: message]) }
    func snapshot(to path: String, directory: URL) throws -> String {
        do {
            return try withAccess(to: path) { try SourceSnapshot.prepare(source: $0, directory: directory).url.path }
        } catch {
            throw NSError(domain: "KwipML.SourceAccess", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Cannot read this source. Choose local CSV in Data to select an available file and grant access. Details: \(error.localizedDescription)",
                NSUnderlyingErrorKey: error
            ])
        }
    }
}

extension WorkspaceStore {
    var sourceAccess: SourceAccess { SourceAccess(directory: applicationDirectory.appendingPathComponent("source-permissions")) }
    func revokeSourcePermission(_ entry:SourceAccess.PermissionEntry)throws {
        guard !isBusy else {throw DataFlowContractError("Stop the active workspace operation before revoking source access.")}
        try sourceAccess.revoke(entry:entry)
    }
    func workerSourcePath(_ path: String) throws -> String {
        guard runtime?.kind == .bundled else { return try sourceAccess.withAccess(to:path) {$0.path} }
        return try sourceAccess.snapshot(to: path, directory: workspaceDirectory.appendingPathComponent("source-snapshots"))
    }
}
