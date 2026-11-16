import Foundation
import Darwin

@_silgen_name("flock")
private func workflowFlock(_ descriptor:Int32,_ operation:Int32)->Int32

public struct WorkflowFileIdentity:Codable,Equatable,Sendable {
    public let device:UInt64
    public let inode:UInt64
    public init(device:UInt64,inode:UInt64) {self.device=device;self.inode=inode}
}

public struct WorkflowFileSystemInfo:Equatable,Sendable {
    public let type:String
    public let isLocal:Bool
}

public enum WorkflowFileLeaseError:Error,Equatable,LocalizedError {
    case busy,unsafePath,closed,system(Int32)
    public var errorDescription:String? {
        switch self {
        case .busy:return "Workflow ownership is held by another active operation."
        case .unsafePath:return "Workflow ownership requires unchanged regular lock files in real directories."
        case .closed:return "Workflow ownership lease has already closed."
        case .system(let code):return "Workflow ownership could not be checked (system error \(code))."
        }
    }
}

/// Nonblocking kernel lease. Callers serialize access to this non-Sendable owner.
/// Closing never unlinks/replaces the stable lock and never explicitly unlocks a shared
/// file description that may also be retained by another process.
public final class WorkflowFileLease {
    public enum Mode {case shared,exclusive}
    public let identity:WorkflowFileIdentity
    private let directory:URL
    private let name:String
    private let mode:Mode
    private let directoryIdentity:WorkflowFileIdentity
    private var descriptor:Int32
    private var parentDescriptor:Int32

    public init(directory:URL,name:String,mode:Mode,create:Bool=false,createExclusive:Bool=false,expectedIdentity:WorkflowFileIdentity?=nil,requireLocalAPFS:Bool=false)throws {
        guard ["owner.lock","activity.lock"].contains(name),!createExclusive || create else {throw WorkflowFileLeaseError.unsafePath}
        let parent=try Self.openDirectory(directory)
        var file:Int32 = -1
        do {
            let parentIdentity=try Self.directoryIdentity(parent)
            if requireLocalAPFS {
                let filesystem=try Self.fileSystemInfo(parent)
                guard filesystem.type == "apfs",filesystem.isLocal else {throw WorkflowFileLeaseError.unsafePath}
            }
            let flags=O_RDWR|O_NOFOLLOW|O_CLOEXEC|O_NONBLOCK|(create ? O_CREAT:0)|(createExclusive ? O_EXCL:0)
            file=openat(parent,name,flags,0o600)
            guard file >= 0 else {throw Self.openError(errno)}
            let fileIdentity=try Self.fileIdentity(file)
            if let expectedIdentity,expectedIdentity != fileIdentity {throw WorkflowFileLeaseError.unsafePath}
            let operation=(mode == .shared ? LOCK_SH:LOCK_EX)|LOCK_NB
            while workflowFlock(file,operation) != 0 {
                let code=errno
                if code == EINTR {continue}
                if code == EWOULDBLOCK || code == EAGAIN {throw WorkflowFileLeaseError.busy}
                throw WorkflowFileLeaseError.system(code)
            }
            self.directory=directory;self.name=name;self.mode=mode;self.directoryIdentity=parentIdentity
            self.identity=fileIdentity;self.descriptor=file;self.parentDescriptor=parent
        } catch {
            if file >= 0 {Darwin.close(file)}
            Darwin.close(parent)
            throw error
        }
        do {try checkIdentity()} catch {close();throw error}
    }
    public func checkIdentity()throws {
        guard descriptor >= 0,parentDescriptor >= 0 else {throw WorkflowFileLeaseError.closed}
        guard try Self.fileIdentity(descriptor) == identity,try Self.directoryIdentity(parentDescriptor) == directoryIdentity else {throw WorkflowFileLeaseError.unsafePath}
        let current=try Self.openDirectory(directory)
        defer {Darwin.close(current)}
        guard try Self.directoryIdentity(current) == directoryIdentity else {throw WorkflowFileLeaseError.unsafePath}
        var entry=stat()
        guard fstatat(current,name,&entry,AT_SYMLINK_NOFOLLOW) == 0 else {throw Self.openError(errno)}
        guard entry.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG),entry.st_nlink == 1,entry.st_uid == geteuid(),entry.st_mode & 0o022 == 0,Self.identity(entry) == identity else {throw WorkflowFileLeaseError.unsafePath}
    }
    public var parentIdentity:WorkflowFileIdentity {directoryIdentity}

    public func fileSystemInfo()throws->WorkflowFileSystemInfo {
        try checkIdentity()
        return try Self.fileSystemInfo(parentDescriptor)
    }
    private static func fileSystemInfo(_ descriptor:Int32)throws->WorkflowFileSystemInfo {
        var info=statfs()
        guard fstatfs(descriptor,&info)==0 else {throw WorkflowFileLeaseError.system(errno)}
        let name=withUnsafeBytes(of:info.f_fstypename) {String(decoding:$0.prefix(while:{$0 != 0}),as:UTF8.self)}
        return .init(type:name,isLocal:info.f_flags & UInt32(MNT_LOCAL) != 0)
    }
    /// Presence includes symlinks and nonregular files: initialization must reject
    /// partial or foreign state rather than silently replace it.
    public func containsSibling(_ name:String)throws->Bool {
        guard ["group.json","ownership.json","owner.lock","activity.lock"].contains(name) else {throw WorkflowFileLeaseError.unsafePath}
        try checkIdentity()
        var value=stat()
        if fstatat(parentDescriptor,name,&value,AT_SYMLINK_NOFOLLOW)==0 {return true}
        if errno==ENOENT {return false}
        throw WorkflowFileLeaseError.system(errno)
    }
    public func readOwnershipMetadata()throws->Data? {try readSibling("ownership.json",limit:16384)}
    public func readGroupRecord()throws->Data? {try readSibling("group.json",limit:256000)}
    private func requireExclusiveOwner()throws {
        try checkIdentity()
        guard name == "owner.lock",mode == .exclusive else {throw WorkflowFileLeaseError.unsafePath}
    }
    public func publishGroupRecord(_ data:Data)throws {
        try requireExclusiveOwner()
        try publishSibling(data,name:"group.json",limit:256000)
    }
    public func replaceGroupRecord(expected:Data,data:Data)throws {
        try requireExclusiveOwner()
        guard expected.count<=256000 else {throw WorkflowFileLeaseError.unsafePath}
        try publishSibling(data,name:"group.json",limit:256000,replacing:expected)
    }
    public func publishGroupAuthoring(_ data:Data)throws {
        try requireExclusiveOwner()
        try publishSibling(data,name:"authoring.json",limit:2_000_000)
    }
    public func readLaunch(nonce:String)throws->Data? {try readSibling(try launchName("launch",nonce),limit:256000)}
    public func readLaunchClaim(nonce:String)throws->Data? {try readSibling(try launchName("claim",nonce),limit:4096)}
    public func readPhaseSlot(runID:String,phase:String)throws->Data? {try readSibling(try phaseName(runID,phase),limit:256000)}
    public func publishLaunch(nonce:String,data:Data)throws {try publishSibling(data,name:launchName("launch",nonce),limit:256000)}
    public func publishLaunchClaim(nonce:String,data:Data)throws {try publishSibling(data,name:launchName("claim",nonce),limit:4096)}
    public func publishPhaseSlot(runID:String,phase:String,data:Data)throws {try publishSibling(data,name:phaseName(runID,phase),limit:256000)}
    private func launchName(_ prefix:String,_ nonce:String)throws->String {
        guard UUID(uuidString:nonce)?.uuidString == nonce else {throw WorkflowFileLeaseError.unsafePath}
        return "\(prefix)-\(nonce).json"
    }
    private func phaseName(_ runID:String,_ phase:String)throws->String {
        guard UUID(uuidString:runID)?.uuidString == runID,["preparation","training"].contains(phase) else {throw WorkflowFileLeaseError.unsafePath}
        return "phase-\(runID)-\(phase).json"
    }
    private func readSibling(_ name:String,limit:Int)throws->Data? {
        try checkIdentity()
        let file=openat(parentDescriptor,name,O_RDONLY|O_NOFOLLOW|O_CLOEXEC|O_NONBLOCK)
        if file<0 {if errno==ENOENT {return nil};throw Self.openError(errno)}
        defer {Darwin.close(file)}
        let metadataIdentity=try Self.fileIdentity(file)
        var info=stat()
        guard fstat(file,&info)==0 else {throw WorkflowFileLeaseError.system(errno)}
        guard (0...limit).contains(Int(info.st_size)) else {throw WorkflowFileLeaseError.unsafePath}
        var bytes=[UInt8](repeating:0,count:limit+1),count=0
        while count<bytes.count {
            let readCount=bytes.withUnsafeMutableBytes {Darwin.read(file,$0.baseAddress!.advanced(by:count),$0.count-count)}
            if readCount<0 {if errno==EINTR {continue};throw WorkflowFileLeaseError.system(errno)}
            if readCount==0 {break}
            count+=readCount
        }
        guard count<=limit,try Self.fileIdentity(file)==metadataIdentity else {throw WorkflowFileLeaseError.unsafePath}
        var after=stat(),entry=stat()
        guard fstat(file,&after)==0,fstatat(parentDescriptor,name,&entry,AT_SYMLINK_NOFOLLOW)==0 else {throw Self.openError(errno)}
        func unchanged(_ a:stat,_ b:stat)->Bool {
            Self.identity(a)==Self.identity(b) && a.st_size==b.st_size && a.st_mtimespec.tv_sec==b.st_mtimespec.tv_sec && a.st_mtimespec.tv_nsec==b.st_mtimespec.tv_nsec && a.st_ctimespec.tv_sec==b.st_ctimespec.tv_sec && a.st_ctimespec.tv_nsec==b.st_ctimespec.tv_nsec
        }
        guard count==info.st_size,unchanged(info,after),unchanged(after,entry),entry.st_mode & mode_t(S_IFMT)==mode_t(S_IFREG),entry.st_nlink==1,entry.st_uid==geteuid(),entry.st_mode & 0o022==0 else {throw WorkflowFileLeaseError.unsafePath}
        try checkIdentity()
        return Data(bytes.prefix(count))
    }
    /// Publish complete immutable bytes through the retained directory descriptor.
    /// The random temporary entry is ours; stable lock files are never replaced.
    public func publishOwnershipMetadata(_ data:Data)throws {try publishSibling(data,name:"ownership.json",limit:16384)}
    private func publishSibling(_ data:Data,name:String,limit:Int,replacing expected:Data?=nil)throws {
        guard data.count<=limit else {throw WorkflowFileLeaseError.unsafePath}
        try checkIdentity()
        var replacedIdentity:WorkflowFileIdentity?
        if let expected {
            guard try readSibling(name,limit:limit) == expected else {throw WorkflowFileLeaseError.unsafePath}
            var entry=stat()
            guard fstatat(parentDescriptor,name,&entry,AT_SYMLINK_NOFOLLOW)==0 else {throw Self.openError(errno)}
            replacedIdentity=Self.identity(entry)
        }
        let staged = try makePublicationTemporary(target: name, maximumBytes: limit)
        let temporary = staged.name
        defer { try? staged.discard() }
        try staged.write(data)
        try checkIdentity()
        if let expected {
            // The retained exclusive owner serializes cooperating ledger writers.
            // Verify target identity and bytes again immediately before atomic rename.
            guard try readSibling(name,limit:limit) == expected else {throw WorkflowFileLeaseError.unsafePath}
            var entry=stat()
            guard fstatat(parentDescriptor,name,&entry,AT_SYMLINK_NOFOLLOW)==0 else {throw Self.openError(errno)}
            guard Self.identity(entry)==replacedIdentity else {throw WorkflowFileLeaseError.unsafePath}
            guard renameat(parentDescriptor,temporary,parentDescriptor,name)==0 else {throw Self.openError(errno)}
        } else {
            guard linkat(parentDescriptor,temporary,parentDescriptor,name,0)==0 else {throw Self.openError(errno)}
            guard unlinkat(parentDescriptor,temporary,0)==0 else {throw Self.openError(errno)}
        }
        guard fsync(parentDescriptor)==0 else {throw WorkflowFileLeaseError.system(errno)}
        try checkIdentity()
    }
    func makePublicationTemporary(target: String, maximumBytes: Int) throws -> WorkflowPublicationTemporary {
        try requireExclusiveOwner()
        return try WorkflowPublicationTemporary(parent: parentDescriptor, parentIdentity: directoryIdentity,
            ownerIdentity: identity, target: target, maximumBytes: maximumBytes, check: { [weak self] in
                guard let self else { throw WorkflowFileLeaseError.closed }; try self.requireExclusiveOwner()
            })
    }
    /// Only callers holding both unchanged exclusive descriptions may reclaim
    /// registered metadata scratch. Missing evidence is never inferred by name.
    public func cleanupPublicationTemporaries(activity: WorkflowFileLease) throws -> WorkflowPublicationCleanupReport {
        func validate() throws {
            try requireExclusiveOwner(); try activity.checkIdentity()
            guard activity.name == "activity.lock", activity.mode == .exclusive,
                  activity.directoryIdentity == directoryIdentity else { throw WorkflowFileLeaseError.unsafePath }
            let filesystem = try Self.fileSystemInfo(parentDescriptor)
            guard filesystem.type == "apfs", filesystem.isLocal else { throw WorkflowFileLeaseError.unsafePath }
        }
        try validate()
        return try WorkflowPublicationTemporary.cleanup(parent: parentDescriptor, parentIdentity: directoryIdentity,
            ownerIdentity: identity, check: validate)
    }
    public func close() {
        if descriptor >= 0 {Darwin.close(descriptor);descriptor = -1}
        if parentDescriptor >= 0 {Darwin.close(parentDescriptor);parentDescriptor = -1}
    }
    deinit {close()}

    private static func identity(_ info:stat)->WorkflowFileIdentity {
        .init(device:UInt64(UInt32(bitPattern:info.st_dev)),inode:UInt64(info.st_ino))
    }
    private static func fileIdentity(_ descriptor:Int32)throws->WorkflowFileIdentity {
        var info=stat()
        guard fstat(descriptor,&info) == 0 else {throw WorkflowFileLeaseError.system(errno)}
        guard info.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG),info.st_nlink == 1,info.st_uid == geteuid(),info.st_mode & 0o022 == 0 else {throw WorkflowFileLeaseError.unsafePath}
        return identity(info)
    }
    private static func directoryIdentity(_ descriptor:Int32)throws->WorkflowFileIdentity {
        var info=stat()
        guard fstat(descriptor,&info) == 0 else {throw WorkflowFileLeaseError.system(errno)}
        guard info.st_mode & mode_t(S_IFMT) == mode_t(S_IFDIR),info.st_uid == geteuid(),info.st_mode & 0o022 == 0 else {throw WorkflowFileLeaseError.unsafePath}
        return identity(info)
    }
    private static func openError(_ code:Int32)->WorkflowFileLeaseError {
        [ELOOP,ENOTDIR].contains(code) ? .unsafePath:.system(code)
    }
    /// Reject symbolic links in every component without requiring ancestor read access.
    private static func openDirectory(_ url:URL)throws->Int32 {
        let path=url.path
        guard url.isFileURL,path.hasPrefix("/"),!path.contains("\0"),url.host == nil || url.host == "" || url.host == "localhost" else {throw WorkflowFileLeaseError.unsafePath}
        let parts=path.split(separator:"/",omittingEmptySubsequences:true).map(String.init)
        guard !parts.contains("."),!parts.contains("..") else {throw WorkflowFileLeaseError.unsafePath}
        // O_NOFOLLOW_ANY includes the final component; combining it with
        // O_NOFOLLOW is an invalid flag combination on macOS.
        let descriptor=Darwin.open(path,O_RDONLY|O_DIRECTORY|O_NOFOLLOW_ANY|O_CLOEXEC)
        guard descriptor >= 0 else {throw openError(errno)}
        return descriptor
    }
}
