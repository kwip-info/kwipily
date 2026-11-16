import Foundation
import Combine
import Darwin
import KwipMLCore

/// Per-Save model parent bookmarks and leaf names. Never part of portable source.
@MainActor final class WorkflowPrivateModelDestinations:ObservableObject {
    private struct Entry:Codable {let address:WorkflowNodeAddress;let parent:String;let leaf:String;let bookmark:Data}
    private struct Envelope:Codable {let schemaVersion:Int;let projectID:String;let revision:UInt64;let entries:[Entry]}
    let owner:WorkflowAuthoringStore
    @Published private(set) var revision:UInt64=0
    @Published private(set) var statusMessage="Choose a new local destination before saving a model."
    private let directory:URL
    private let parts:[String]
    private var entries:[WorkflowNodeAddress:Entry]=[:]
    init(owner:WorkflowAuthoringStore,directory:URL)throws {
        guard UUID(uuidString:owner.projectID) != nil,owner.isCurrent(owner.lease()) else {throw WorkflowArchiveError("Model destinations require a current project.")}
        self.owner=owner;self.directory=directory;parts=["workflow-model-destinations",owner.projectID,"bindings.json"]
        if let raw=try DiagnosticFileIO.readIfPresent(root:directory,parts:parts,limit:1_000_000) {
            let value=try JSONDecoder().decode(Envelope.self,from:raw)
            guard value.schemaVersion==1,value.projectID==owner.projectID,value.entries.count<=256,Set(value.entries.map(\.address)).count==value.entries.count else {throw WorkflowArchiveError("Private model destinations are malformed; existing records were preserved.")}
            for entry in value.entries {try Self.validate(entry)}
            revision=value.revision;entries=Dictionary(uniqueKeysWithValues:value.entries.map {($0.address,$0)})
        }
    }
    func path(for address:WorkflowNodeAddress)->String? {
        guard isSave(address),let entry=entries[address] else{return nil}
        return URL(fileURLWithPath:entry.parent,isDirectory:true).appendingPathComponent(entry.leaf).path
    }
    func summary(for address:WorkflowNodeAddress)->String {path(for:address) ?? "No destination selected · write permission required"}
    @discardableResult func set(destination:URL,for address:WorkflowNodeAddress,lease:WorkflowEditorLease,expectedRevision:UInt64?=nil)->Bool {
        do {
            try editable(address,lease:lease,revision:expectedRevision)
            guard destination.isFileURL else {throw WorkflowArchiveError("Choose a local model destination.")}
            let parent=destination.deletingLastPathComponent(),leaf=destination.lastPathComponent
            try WorkflowExportPublication.validateExclusiveLeaf(leaf)
            let started=parent.startAccessingSecurityScopedResource();defer {if started {parent.stopAccessingSecurityScopedResource()}}
            let canonical=try DataOutputDestinationProvider.canonicalGrantedPath(parent)
            let fd=try WorkflowEvaluationOwnership.openDirectory(URL(fileURLWithPath:canonical,isDirectory:true));defer {Darwin.close(fd)}
            try WorkflowExportPublication.requireAbsent(leaf,parent:fd)
            let bookmark=try parent.bookmarkData(options:[.withSecurityScope],includingResourceValuesForKeys:nil,relativeTo:nil)
            let entry=Entry(address:address,parent:canonical,leaf:leaf,bookmark:bookmark);try Self.validate(entry)
            try editable(address,lease:lease,revision:expectedRevision)
            var next=entries;next[address]=entry;try persist(next)
            statusMessage="Destination selected on this Mac · "+leaf;return true
        } catch {statusMessage=error.localizedDescription;return false}
    }
    @discardableResult func clear(for address:WorkflowNodeAddress,lease:WorkflowEditorLease,expectedRevision:UInt64?=nil)->Bool {
        do {try editable(address,lease:lease,revision:expectedRevision);var next=entries;next[address]=nil;try persist(next);statusMessage="Model destination cleared.";return true}
        catch {statusMessage=error.localizedDescription;return false}
    }
    func acquire(for address:WorkflowNodeAddress,lease:WorkflowEditorLease)throws->WorkflowModelDestinationAccess {
        guard owner.isCurrent(lease),isSave(address),let entry=entries[address] else {throw WorkflowArchiveError("Choose this Save model component's destination again.")}
        var stale=false
        let parent=try URL(resolvingBookmarkData:entry.bookmark,options:[.withSecurityScope,.withoutUI],relativeTo:nil,bookmarkDataIsStale:&stale)
        let started=parent.startAccessingSecurityScopedResource()
        do {
            guard !stale,try DataOutputDestinationProvider.canonicalGrantedPath(parent)==entry.parent else {throw WorkflowArchiveError("Model destination moved or its permission is stale. Choose again.")}
            let destination=URL(fileURLWithPath:entry.parent,isDirectory:true).appendingPathComponent(entry.leaf)
            return try WorkflowModelDestinationAccess(destination:destination,accessURL:parent,revision:revision,scopeStarted:started)
        } catch {if started {parent.stopAccessingSecurityScopedResource()};throw error}
    }
    private func isSave(_ address:WorkflowNodeAddress)->Bool {
        owner.isCurrent(owner.lease()) && address.domain == .control && owner.document.nodes.contains {node in
            guard node.address==address,case .saveModel = node.payload else{return false};return true
        }
    }
    private func editable(_ address:WorkflowNodeAddress,lease:WorkflowEditorLease,revision expected:UInt64?)throws {
        guard owner.isCurrent(lease),!owner.isBusy,!owner.hasPendingInspectorEdits,isSave(address),expected == nil || expected==revision else {throw WorkflowArchiveError("Destination selection is stale or the workflow is busy. Choose the destination again.")}
    }
    private func persist(_ next:[WorkflowNodeAddress:Entry])throws {
        guard next.count<=256,revision<UInt64.max else {throw WorkflowArchiveError("Private model destination allowance reached.")}
        let envelope=Envelope(schemaVersion:1,projectID:owner.projectID,revision:revision+1,entries:next.values.sorted {$0.address.localID<$1.address.localID})
        let raw=try WorkflowArchiveCodec.encode(envelope);guard raw.count<=1_000_000 else {throw WorkflowArchiveError("Private model destination metadata exceeds its allowance.")}
        try DiagnosticFileIO.writeAtomic(raw,root:directory,parts:parts);entries=next;revision+=1
    }
    private static func validate(_ entry:Entry)throws {
        guard entry.address.domain == .control,!entry.address.localID.isEmpty,entry.address.localID.utf8.count<=128,!entry.address.localID.contains("\0"),entry.parent.hasPrefix("/"),entry.parent.utf8.count<=4096,!entry.parent.contains("\0"),!entry.bookmark.isEmpty,entry.bookmark.count<=64_000 else {throw WorkflowArchiveError("Invalid private model destination.")}
        try WorkflowExportPublication.validateExclusiveLeaf(entry.leaf)
    }
}

/// Scope and parent-directory identity remain held through final commit/cancellation.
final class WorkflowModelDestinationAccess {
    let destination:URL
    let accessURL:URL
    let revision:UInt64
    private let scopeStarted:Bool
    private let parentIdentity:WorkflowFileIdentity
    private var descriptor:Int32
    private let lock=NSLock()
    init(destination:URL,accessURL:URL,revision:UInt64,scopeStarted:Bool=false)throws {
        try WorkflowExportPublication.validateExclusiveLeaf(destination.lastPathComponent)
        let parent=destination.deletingLastPathComponent()
        let canonical=try WorkflowExportPublication.canonicalParentPath(parent)
        guard canonical == (try WorkflowExportPublication.canonicalParentPath(accessURL)) else {throw WorkflowFileLeaseError.unsafePath}
        let canonicalParent=URL(fileURLWithPath:canonical,isDirectory:true)
        let fd=try WorkflowEvaluationOwnership.openDirectory(canonicalParent)
        do {
            self.parentIdentity=try WorkflowEvaluationOwnership.directoryIdentity(fd)
            try WorkflowExportPublication.requireAbsent(destination.lastPathComponent,parent:fd)
        } catch {Darwin.close(fd);throw error}
        self.destination=canonicalParent.appendingPathComponent(destination.lastPathComponent);self.accessURL=accessURL;self.revision=revision;self.scopeStarted=scopeStarted;descriptor=fd
    }
    func check()throws {
        lock.lock();defer {lock.unlock()}
        guard descriptor>=0 else {throw WorkflowFileLeaseError.closed}
        let fd=try WorkflowEvaluationOwnership.openDirectory(destination.deletingLastPathComponent());defer {Darwin.close(fd)}
        guard try WorkflowEvaluationOwnership.directoryIdentity(fd)==parentIdentity else {throw WorkflowFileLeaseError.unsafePath}
    }
    func close() {
        lock.lock();defer {lock.unlock()}
        guard descriptor>=0 else{return};Darwin.close(descriptor);descriptor = -1
        if scopeStarted {accessURL.stopAccessingSecurityScopedResource()}
    }
    deinit {close()}
}
