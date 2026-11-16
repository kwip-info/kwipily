import AppKit
import Darwin
import Combine
import KwipMLCore

/// Write-scoped private bookmarks. Portable workflow documents contain neither
/// destinations nor permission data; a binding is required for each output node.
@MainActor final class DataOutputDestinationProvider:ObservableObject {
    private struct Entry:Codable {let address:WorkflowNodeAddress;let path:String;let bookmark:Data}
    private struct Envelope:Codable {let schemaVersion:Int;let projectID:String;let revision:UInt64;let entries:[Entry];let databases:[Entry]?}
    var projectID:String {owner.projectID}
    var sessionID:UUID {owner.sessionID}
    private let owner:WorkflowAuthoringStore
    private weak var workspace:WorkspaceStore?
    lazy var databaseOutputs:DatabaseOutputStore = .init(workspace:workspace!,owner:owner,destinations:self)
    private let directory:URL
    private let parts:[String]
    private var entries:[WorkflowNodeAddress:Entry]=[:]
    private var databaseEntries:[WorkflowNodeAddress:Entry]=[:]
    @Published private(set) var revision:UInt64=0
    @Published private(set) var statusMessage="Choose a destination folder on this Mac before saving a table."
    var pickDirectory:()->URL? = {
        let panel=NSOpenPanel();panel.title="Choose table output folder";panel.canChooseDirectories=true;panel.canChooseFiles=false;panel.allowsMultipleSelection=false;panel.canCreateDirectories=true
        return panel.runModal() == .OK ? panel.url:nil
    }
    init(workspace:WorkspaceStore,owner:WorkflowAuthoringStore)throws {
        self.workspace=workspace;self.owner=owner;directory=workspace.workspaceDirectory
        parts=["workflow-output-destinations",owner.projectID,"bindings.json"]
        if let bytes=try DiagnosticFileIO.readIfPresent(root:directory,parts:parts,limit:1_000_000) {
            let envelope=try JSONDecoder().decode(Envelope.self,from:bytes)
            guard [1,2].contains(envelope.schemaVersion),envelope.projectID==owner.projectID,envelope.entries.count<=128,Set(envelope.entries.map(\.address)).count==envelope.entries.count else {throw DataFlowContractError("Output destination permissions are malformed. Existing records were preserved.")}
            for entry in envelope.entries {try Self.validate(entry)}
            let db=envelope.databases ?? [];guard db.count<=128,Set(db.map(\.address)).count==db.count else {throw WorkflowFileLeaseError.unsafePath};for entry in db {try Self.validate(entry)};databaseEntries=Dictionary(uniqueKeysWithValues:db.map {($0.address,$0)})
            revision=envelope.revision;entries=Dictionary(uniqueKeysWithValues:envelope.entries.map {($0.address,$0)})
        }
    }
    var pickDatabaseFile:()->URL? = {
        let p=NSOpenPanel();p.title="Choose an existing SQLite database";p.canChooseDirectories=false;p.canChooseFiles=true;p.allowsMultipleSelection=false
        return p.runModal() == .OK ? p.url:nil
    }
    var pickDatabaseFolder:(URL)->URL? = {expected in
        let p=NSOpenPanel();p.title="Allow SQLite database writes";p.message="Choose the database’s containing folder so SQLite can maintain its transaction journal.";p.directoryURL=expected;p.canChooseDirectories=true;p.canChooseFiles=false;p.allowsMultipleSelection=false
        return p.runModal() == .OK ? p.url:nil
    }
    func chooseDatabase(for address:WorkflowNodeAddress,lease:WorkflowEditorLease)throws->String? {
        func valid()->Bool {owner.isCurrent(lease) && !owner.isBusy && !(workspace?.isBusy ?? true) && owner.selection.nodeAddresses==[address] && isOutput(address)}
        guard valid(),let file=pickDatabaseFile() else {return nil}
        let started=file.startAccessingSecurityScopedResource();defer {if started {file.stopAccessingSecurityScopedResource()}}
        let values=try file.resourceValues(forKeys:[.isRegularFileKey,.isSymbolicLinkKey]);guard values.isRegularFile==true,values.isSymbolicLink != true,!file.lastPathComponent.contains("\0"),file.lastPathComponent.utf8.count<=200 else {throw WorkflowFileLeaseError.unsafePath}
        let parent=file.deletingLastPathComponent();guard let folder=pickDatabaseFolder(parent),valid(),try Self.canonicalGrantedPath(folder)==Self.canonicalGrantedPath(parent) else {return nil}
        let access=folder.startAccessingSecurityScopedResource();defer {if access {folder.stopAccessingSecurityScopedResource()}}
        let bookmark=try folder.bookmarkData(options:[.withSecurityScope],includingResourceValuesForKeys:nil,relativeTo:nil)
        let entry=Entry(address:address,path:try Self.canonicalGrantedPath(folder),bookmark:bookmark);try Self.validate(entry)
        var next=databaseEntries;next[address]=entry;try persist(entries,databases:next);statusMessage="SQLite database folder linked on this Mac";return file.lastPathComponent
    }
    func privateAuthorityChanged(){guard revision<UInt64.max else{return};revision+=1;objectWillChange.send()}
    func bookmark(for address:WorkflowNodeAddress)throws->Data {
        guard isOutput(address),let entry=databaseEntries[address] else {throw DataFlowContractError("Choose the SQLite database folder before testing or writing.")}
        let access=try acquire([address],databaseOverride:true);defer {access.close()}
        var stale=false
        let url=try URL(resolvingBookmarkData:entry.bookmark,options:[.withSecurityScope,.withoutUI],relativeTo:nil,bookmarkDataIsStale:&stale)
        guard !stale else {throw WorkflowFileLeaseError.unsafePath}
        // Persistent app-scoped grants stay here. Only an ephemeral implicit
        // scope is transferred to the authenticated helper for this operation.
        return try url.bookmarkData(options:[],includingResourceValuesForKeys:nil,relativeTo:nil)
    }
    func databasePath(for address:WorkflowNodeAddress)->String? {isOutput(address) ? databaseEntries[address]?.path:nil}
    func path(for address:WorkflowNodeAddress)->String? {isOutput(address) ? entries[address]?.path:nil}
    func destinationPath(for address:WorkflowNodeAddress)->String? {path(for:address)}
    func summary(for address:WorkflowNodeAddress)->String {path(for:address) ?? "No destination selected · write permission required"}
    private func isOutput(_ address:WorkflowNodeAddress)->Bool {
        owner.isCurrent(owner.lease()) && address.domain == .data && owner.document.nodes.contains {node in
            guard node.address==address,case .dataOperator("kwip.data.outputDelimited",_)=node.payload else {return false};return true
        }
    }
    private func canChoose(_ address:WorkflowNodeAddress,lease:WorkflowEditorLease,revision:UInt64)->Bool {
        guard let workspace else {return false}
        return owner.isCurrent(lease) && !owner.isBusy && !workspace.isBusy && !owner.hasPendingInspectorEdits && workspace.activeProjectID==owner.projectID && self.revision==revision && isOutput(address) && owner.selection.nodeAddresses==[address]
    }
    @discardableResult func chooseDestination(for address:WorkflowNodeAddress,lease:WorkflowEditorLease)->Bool {
        chooseDestinations(for: [address], lease: lease)
    }
    /// One explicit folder selection grants only this exact set of local table outputs.
    /// Every binding is published together, after picker-time owner and private-revision checks.
    @discardableResult func chooseDestinations(for addresses:Set<WorkflowNodeAddress>,lease:WorkflowEditorLease)->Bool {
        let capturedRevision=revision
        func valid()->Bool {
            guard let workspace,!addresses.isEmpty,addresses.count<=128,
                  owner.isCurrent(lease),!owner.isBusy,!workspace.isBusy,!owner.hasPendingInspectorEdits,
                  owner.document.drafts.isEmpty,workspace.activeProjectID==owner.projectID,
                  revision==capturedRevision,owner.selection.nodeAddresses==addresses,owner.selection.edgeIDs.isEmpty else {return false}
            return addresses.allSatisfy {address in
                guard isOutput(address),let node=owner.document.nodes.first(where:{$0.address==address}),
                      case .dataOperator(_,let config)=node.payload else {return false}
                return !APITableSettings.isExternal(config)
            }
        }
        guard valid() else {statusMessage="Select idle local table outputs and finish pending edits before choosing a folder.";return false}
        guard let url=pickDirectory() else {statusMessage="Folder selection cancelled; existing destinations are unchanged.";return false}
        do {
            guard valid() else {throw DataFlowContractError("The workflow changed while choosing a destination. Choose again.")}
            let started=url.startAccessingSecurityScopedResource();defer {if started {url.stopAccessingSecurityScopedResource()}}
            let canonical=try Self.canonicalGrantedPath(url)
            guard try url.resourceValues(forKeys:[.isDirectoryKey]).isDirectory==true else {throw DataFlowContractError("Choose an existing directory without symlinks for output.")}
            let bookmark=try url.bookmarkData(options:[.withSecurityScope],includingResourceValuesForKeys:nil,relativeTo:nil)
            var next=entries
            for address in addresses {
                let entry=Entry(address:address,path:canonical,bookmark:bookmark);try Self.validate(entry);next[address]=entry
            }
            guard valid(),revision<UInt64.max else {throw DataFlowContractError("Destination authority changed. Choose the folder again.")}
            try persist(next)
            statusMessage="Folder linked to \(addresses.count) table report\(addresses.count == 1 ? "":"s") on this Mac · "+url.lastPathComponent;return true
        } catch {statusMessage=error.localizedDescription;return false}
    }
    @discardableResult func clearDestination(for address:WorkflowNodeAddress,lease:WorkflowEditorLease)->Bool {
        guard canChoose(address,lease:lease,revision:revision) else {return false}
        do {var next=entries;next[address]=nil;try persist(next);statusMessage="Destination cleared; no output permission is implied.";return true}
        catch {statusMessage=error.localizedDescription;return false}
    }
    private func persist(_ next:[WorkflowNodeAddress:Entry],databases:[WorkflowNodeAddress:Entry]?=nil)throws {
        let nextDatabases=databases ?? databaseEntries
        guard next.count<=128,revision<UInt64.max else {throw DataFlowContractError("Output destination binding limit reached.")}
        let envelope=Envelope(schemaVersion:2,projectID:owner.projectID,revision:revision+1,entries:next.values.sorted {$0.address.localID<$1.address.localID},databases:nextDatabases.values.sorted {$0.address.localID<$1.address.localID})
        let bytes=try WorkflowArchiveCodec.encode(envelope);guard bytes.count<=1_000_000 else {throw DataFlowContractError("Output permissions exceed their private storage allowance.")}
        try DiagnosticFileIO.writeAtomic(bytes,root:directory,parts:parts);entries=next;databaseEntries=nextDatabases;revision+=1
    }
    /// Foundation may fold /private/var back to /var when producing URL.path.
    /// Keep the actual POSIX canonical spelling for O_NOFOLLOW worker traversal.
    static func canonicalGrantedPath(_ url:URL)throws->String {
        guard url.isFileURL else {throw DataFlowContractError("Choose a local output directory.")}
        let raw=url.path
        guard raw.hasPrefix("/"),raw.utf8.count<=4096,!raw.contains("\0"),let pointer=realpath(raw,nil) else {throw DataFlowContractError("Output directory is unavailable.")}
        defer {free(pointer)}
        let canonical=String(cString:pointer)
        var expected=raw.count>1 && raw.hasSuffix("/") ? String(raw.dropLast()):raw
        // macOS itself exposes these three root aliases. No user-created link or
        // additional link within those trees is accepted by this exception.
        for alias in ["/var","/tmp","/etc"] where expected==alias || expected.hasPrefix(alias+"/") {
            expected="/private"+expected;break
        }
        guard canonical==expected else {throw DataFlowContractError("Choose an existing directory without user symlinks for output.")}
        return canonical
    }
    private static func validate(_ entry:Entry)throws {
        guard entry.address.domain == .data,!entry.address.localID.isEmpty,entry.address.localID.count<=128,entry.path.hasPrefix("/"),entry.path.utf8.count<=4096,!entry.path.contains("\0"),!entry.bookmark.isEmpty,entry.bookmark.count<=64_000 else {throw DataFlowContractError("Invalid private output destination record.")}
    }
    func acquire(_ addresses:Set<WorkflowNodeAddress>,databaseOverride:Bool?=nil)throws->DataOutputDestinationAccess {
        var urls:[URL]=[],paths:[String:String]=[:]
        do {
            for address in addresses.sorted(by:{$0.localID<$1.localID}) {
                let db=databaseOverride ?? owner.document.nodes.first(where:{$0.address==address}).map {if case .dataOperator(_,let c)=$0.payload {return APITableSettings.isExternal(c)};return false} ?? false
                guard isOutput(address),let entry=(db ? databaseEntries:entries)[address] else {throw DataFlowContractError(db ? "Choose an existing SQLite database and grant its containing folder.":"Choose a destination folder for output component \(address.localID).")}
                var stale=false
                let url=try URL(resolvingBookmarkData:entry.bookmark,options:[.withSecurityScope,.withoutUI],relativeTo:nil,bookmarkDataIsStale:&stale)
                if url.startAccessingSecurityScopedResource() {urls.append(url)}
                guard !stale,try Self.canonicalGrantedPath(url)==entry.path,try url.resourceValues(forKeys:[.isDirectoryKey]).isDirectory==true else {throw DataFlowContractError("Output folder moved or permission is stale. Choose its destination again.")}
                paths[address.localID]=entry.path
            }
            return DataOutputDestinationAccess(paths:paths,scopes:urls)
        } catch {urls.forEach {$0.stopAccessingSecurityScopedResource()};throw error}
    }
}

/// Held until the supervised worker exits and final verification joins.
final class DataOutputDestinationAccess {
    let paths:[String:String]
    private var scopes:[URL]
    init(paths:[String:String],scopes:[URL]) {self.paths=paths;self.scopes=scopes}
    func close() {let closing=scopes;scopes=[];closing.forEach {$0.stopAccessingSecurityScopedResource()}}
    deinit {close()}
}
