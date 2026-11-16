import Foundation
import Darwin
import KwipMLCore

struct WorkflowEvaluationOwnershipMetadata:Codable,Equatable {
    let schemaVersion:Int
    let kind:String
    let evaluationID:String
    let projectID:String
    let requestIdentity:String
    let requestSHA256:String
    let directoryIdentity:WorkflowFileIdentity
    let ownerLockIdentity:WorkflowFileIdentity
    let activityLockIdentity:WorkflowFileIdentity
}

enum WorkflowEvaluationOwnership {
    enum Probe:Equatable {case inUse,inactive}
    static func create(workspace:URL,request:WorkflowEvaluationWorkerRequest)throws->WorkflowEvaluationOwner {
        try validate(request)
        guard workspace.isFileURL,!workspace.path.contains("\0"),let canonical=realpath(workspace.path,nil) else {throw WorkflowFileLeaseError.unsafePath}
        defer {free(canonical)}
        let root=URL(fileURLWithPath:String(cString:canonical),isDirectory:true)
        let fd=try openDirectory(root)
        defer {Darwin.close(fd)}
        let rootIdentity=try directoryIdentity(fd)
        func child(_ parent:Int32,_ name:String,exclusive:Bool=false)throws->Int32 {
            if mkdirat(parent,name,0o700) != 0 {
                guard !exclusive,errno==EEXIST else {throw WorkflowFileLeaseError.system(errno)}
            } else if fsync(parent) != 0 {throw WorkflowFileLeaseError.system(errno)}
            let result=openat(parent,name,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC)
            guard result>=0 else {throw WorkflowFileLeaseError.unsafePath}
            do {_ = try directoryIdentity(result);return result} catch {Darwin.close(result);throw error}
        }
        let namespace=try child(fd,"workflow-evaluations")
        defer {Darwin.close(namespace)}
        let project=try child(namespace,request.projectID)
        defer {Darwin.close(project)}
        let leaf=try child(project,request.evaluationID,exclusive:true)
        defer {Darwin.close(leaf)}
        let identity=try directoryIdentity(leaf)
        let directory=root.appendingPathComponent("workflow-evaluations").appendingPathComponent(request.projectID).appendingPathComponent(request.evaluationID)
        let owner=try initialize(directory:directory,request:request)
        do {
            let current=try openDirectory(root)
            defer {Darwin.close(current)}
            guard try directoryIdentity(current)==rootIdentity,owner.metadata.directoryIdentity==identity else {throw WorkflowFileLeaseError.unsafePath}
            try owner.check();return owner
        } catch {owner.close();throw error}
    }
    static func initialize(directory:URL,request:WorkflowEvaluationWorkerRequest)throws->WorkflowEvaluationOwner {
        try validate(request);try validateLeaf(directory,request)
        let fd=try openDirectory(directory)
        defer {Darwin.close(fd)}
        _ = try directoryIdentity(fd)
        guard try FileManager.default.contentsOfDirectory(atPath:directory.path).isEmpty else {throw WorkflowFileLeaseError.unsafePath}
        let owner=try WorkflowFileLease(directory:directory,name:"owner.lock",mode:.exclusive,create:true,createExclusive:true,requireLocalAPFS:true)
        do {
            let activity=try WorkflowFileLease(directory:directory,name:"activity.lock",mode:.exclusive,create:true,createExclusive:true,requireLocalAPFS:true)
            do {
                guard owner.parentIdentity==activity.parentIdentity,owner.parentIdentity == (try directoryIdentity(fd)) else {throw WorkflowFileLeaseError.unsafePath}
                let metadata=WorkflowEvaluationOwnershipMetadata(schemaVersion:1,kind:"frozenEvaluation",evaluationID:request.evaluationID,projectID:request.projectID,requestIdentity:request.requestIdentity,requestSHA256:try request.requestSHA256(),directoryIdentity:owner.parentIdentity,ownerLockIdentity:owner.identity,activityLockIdentity:activity.identity)
                let bytes=try WorkflowArchiveCodec.encode(metadata)
                try owner.publishOwnershipMetadata(bytes)
                let result=WorkflowEvaluationOwner(directory:directory,metadata:metadata,bytes:bytes,owner:owner,activity:activity)
                try result.check();return result
            } catch {activity.close();throw error}
        } catch {owner.close();throw error}
    }
    static func probe(directory:URL,request:WorkflowEvaluationWorkerRequest)throws->Probe {
        try validate(request);try validateLeaf(directory,request)
        do {
            try inspect(directory:directory,projectID:request.projectID,evaluationID:request.evaluationID) { metadata in
                guard metadata.requestIdentity==request.requestIdentity,metadata.requestSHA256 == (try request.requestSHA256()) else {throw WorkflowFileLeaseError.unsafePath}
                try verifyRequest(directory,metadata)
            }
            return .inactive
        } catch WorkflowFileLeaseError.busy {
            // Validate the caller's exact request even when another owner is active.
            let metadata=try readMetadata(directory)
            guard metadata.requestIdentity==request.requestIdentity,metadata.requestSHA256 == (try request.requestSHA256()) else {throw WorkflowFileLeaseError.unsafePath}
            return .inUse
        }
    }
    /// The body runs while both exclusive descriptions remain retained. No files are created.
    static func inspect<T>(directory:URL,projectID:String,evaluationID:String,_ body:(WorkflowEvaluationOwnershipMetadata)throws->T)throws->T {
        let inspection=try acquireInspection(directory:directory,projectID:projectID,evaluationID:evaluationID)
        defer {inspection.close()}
        let value=try body(inspection.metadata)
        try inspection.check()
        return value
    }
    /// Retain across asynchronous export staging. The caller serializes use and closes
    /// only after its worker exits; acquiring this lease does not verify result artifacts.
    static func acquireInspection(directory:URL,projectID:String,evaluationID:String)throws->WorkflowEvaluationInspection {
        let bytes=try readProtocol(directory,name:"ownership.json",limit:16384)
        let metadata=try decodeMetadata(bytes)
        guard metadata.projectID==projectID,metadata.evaluationID==evaluationID,directory.lastPathComponent==evaluationID else {throw WorkflowFileLeaseError.unsafePath}
        try checkPaths(directory,metadata)
        let owner=try WorkflowFileLease(directory:directory,name:"owner.lock",mode:.exclusive,expectedIdentity:metadata.ownerLockIdentity,requireLocalAPFS:true)
        do {
            guard try owner.readOwnershipMetadata()==bytes else {throw WorkflowFileLeaseError.unsafePath}
            let activity=try WorkflowFileLease(directory:directory,name:"activity.lock",mode:.exclusive,expectedIdentity:metadata.activityLockIdentity,requireLocalAPFS:true)
            let inspection=WorkflowEvaluationInspection(directory:directory,metadata:metadata,bytes:bytes,owner:owner,activity:activity)
            do {try inspection.check();return inspection}
            catch {inspection.close();throw error}
        } catch {owner.close();throw error}
    }
    private static func readMetadata(_ directory:URL)throws->WorkflowEvaluationOwnershipMetadata {
        try decodeMetadata(readProtocol(directory,name:"ownership.json",limit:16384))
    }
    private static func decodeMetadata(_ bytes:Data)throws->WorkflowEvaluationOwnershipMetadata {
        guard let object=try JSONSerialization.jsonObject(with:bytes) as? [String:Any],Set(object.keys)==["schemaVersion","kind","evaluationID","projectID","requestIdentity","requestSHA256","directoryIdentity","ownerLockIdentity","activityLockIdentity"] else {throw WorkflowFileLeaseError.unsafePath}
        for key in ["directoryIdentity","ownerLockIdentity","activityLockIdentity"] {
            guard let identity=object[key] as? [String:Any],Set(identity.keys)==["device","inode"] else {throw WorkflowFileLeaseError.unsafePath}
        }
        let value=try JSONDecoder().decode(WorkflowEvaluationOwnershipMetadata.self,from:bytes)
        guard value.schemaVersion==1,value.kind=="frozenEvaluation",UUID(uuidString:value.evaluationID)?.uuidString==value.evaluationID,UUID(uuidString:value.projectID)?.uuidString==value.projectID,WorkflowArchiveCodec.isHash(value.requestIdentity),WorkflowArchiveCodec.isHash(value.requestSHA256) else {throw WorkflowFileLeaseError.unsafePath}
        return value
    }
    fileprivate static func requireUnclaimed(_ directory:URL,_ metadata:WorkflowEvaluationOwnershipMetadata)throws {
        let fd=try openDirectory(directory)
        defer {Darwin.close(fd)}
        guard try directoryIdentity(fd)==metadata.directoryIdentity else {throw WorkflowFileLeaseError.unsafePath}
        var entry=stat()
        guard fstatat(fd,"claim.json",&entry,AT_SYMLINK_NOFOLLOW) != 0 else {throw WorkflowFileLeaseError.unsafePath}
        guard errno==ENOENT else {throw WorkflowFileLeaseError.system(errno)}
    }
    @discardableResult fileprivate static func verifyRequest(_ directory:URL,_ metadata:WorkflowEvaluationOwnershipMetadata)throws->WorkflowEvaluationWorkerRequest {
        let bytes=try readProtocol(directory,name:"request.json",limit:256000)
        let request=try WorkflowEvaluationWorkerRequest.decode(bytes)
        guard WorkflowArchiveCodec.hash(bytes)==metadata.requestSHA256,request.evaluationID==metadata.evaluationID,request.projectID==metadata.projectID,request.requestIdentity==metadata.requestIdentity else {throw WorkflowFileLeaseError.unsafePath}
        return request
    }
    static func readProtocol(_ directory:URL,name:String,limit:Int)throws->Data {
        let parent=try openDirectory(directory)
        defer {Darwin.close(parent)}
        let fd=openat(parent,name,O_RDONLY|O_NOFOLLOW|O_CLOEXEC|O_NONBLOCK)
        guard fd>=0 else {throw WorkflowFileLeaseError.unsafePath}
        defer {Darwin.close(fd)}
        func safe(_ value:stat)->Bool {value.st_mode & mode_t(S_IFMT)==mode_t(S_IFREG) && value.st_nlink==1 && value.st_uid==geteuid() && value.st_mode & 0o022==0}
        var before=stat()
        guard fstat(fd,&before)==0,safe(before),before.st_size>=0,before.st_size<=limit else {throw WorkflowFileLeaseError.unsafePath}
        var bytes=[UInt8](repeating:0,count:limit+1),count=0
        while count<bytes.count {
            let n=bytes.withUnsafeMutableBytes {Darwin.read(fd,$0.baseAddress!.advanced(by:count),$0.count-count)}
            if n<0 {if errno==EINTR {continue};throw WorkflowFileLeaseError.system(errno)}
            if n==0 {break};count+=n
        }
        var after=stat(),entry=stat()
        func same(_ a:stat,_ b:stat)->Bool {identity(a)==identity(b) && a.st_size==b.st_size && a.st_mtimespec.tv_sec==b.st_mtimespec.tv_sec && a.st_mtimespec.tv_nsec==b.st_mtimespec.tv_nsec && a.st_ctimespec.tv_sec==b.st_ctimespec.tv_sec && a.st_ctimespec.tv_nsec==b.st_ctimespec.tv_nsec}
        guard count<=limit,count==before.st_size,fstat(fd,&after)==0,fstatat(parent,name,&entry,AT_SYMLINK_NOFOLLOW)==0,safe(after),safe(entry),same(before,after),same(after,entry) else {throw WorkflowFileLeaseError.unsafePath}
        return Data(bytes.prefix(count))
    }
    private static func validate(_ request:WorkflowEvaluationWorkerRequest)throws {
        try request.validate()
        guard UUID(uuidString:request.projectID)?.uuidString==request.projectID,UUID(uuidString:request.evaluationID)?.uuidString==request.evaluationID else {throw WorkflowFileLeaseError.unsafePath}
    }
    private static func validateLeaf(_ directory:URL,_ request:WorkflowEvaluationWorkerRequest)throws {
        guard directory.lastPathComponent==request.evaluationID else {throw WorkflowFileLeaseError.unsafePath}
    }
    fileprivate static func checkPaths(_ directory:URL,_ metadata:WorkflowEvaluationOwnershipMetadata)throws {
        let fd=try openDirectory(directory)
        defer {Darwin.close(fd)}
        guard try directoryIdentity(fd)==metadata.directoryIdentity else {throw WorkflowFileLeaseError.unsafePath}
        for (name,expected) in [("owner.lock",metadata.ownerLockIdentity),("activity.lock",metadata.activityLockIdentity)] {
            var info=stat()
            guard fstatat(fd,name,&info,AT_SYMLINK_NOFOLLOW)==0,info.st_mode & mode_t(S_IFMT)==mode_t(S_IFREG),info.st_nlink==1,info.st_uid==geteuid(),info.st_mode & 0o022==0,identity(info)==expected else {throw WorkflowFileLeaseError.unsafePath}
        }
    }
    static func openDirectory(_ directory:URL)throws->Int32 {
        let path=directory.path
        guard directory.isFileURL,path.hasPrefix("/"),!path.contains("\0"),directory.host == nil || directory.host == "" || directory.host == "localhost" else {throw WorkflowFileLeaseError.unsafePath}
        let parts=path.split(separator:"/",omittingEmptySubsequences:true)
        guard !parts.contains("."),!parts.contains("..") else {throw WorkflowFileLeaseError.unsafePath}
        let fd=open(path,O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW_ANY)
        guard fd>=0 else {throw WorkflowFileLeaseError.unsafePath}
        return fd
    }
    static func identity(_ info:stat)->WorkflowFileIdentity {.init(device:UInt64(UInt32(bitPattern:info.st_dev)),inode:UInt64(info.st_ino))}
    static func directoryIdentity(_ fd:Int32)throws->WorkflowFileIdentity {
        var info=stat()
        guard fstat(fd,&info)==0,info.st_mode & mode_t(S_IFMT)==mode_t(S_IFDIR),info.st_uid==geteuid(),info.st_mode & 0o022==0 else {throw WorkflowFileLeaseError.unsafePath}
        return identity(info)
    }
}

/// Inactive ownership held for a reader's lifetime. Result verification and disclosure
/// remain separate requirements; this object only keeps the saved directory reserved.
final class WorkflowEvaluationInspection {
    let directory:URL
    let metadata:WorkflowEvaluationOwnershipMetadata
    private let bytes:Data
    private let owner:WorkflowFileLease
    private let activity:WorkflowFileLease
    private var closed=false
    fileprivate init(directory:URL,metadata:WorkflowEvaluationOwnershipMetadata,bytes:Data,owner:WorkflowFileLease,activity:WorkflowFileLease) {
        self.directory=directory;self.metadata=metadata;self.bytes=bytes;self.owner=owner;self.activity=activity
    }
    func check()throws {
        guard !closed else {throw WorkflowFileLeaseError.closed}
        try owner.checkIdentity();try activity.checkIdentity()
        try WorkflowEvaluationOwnership.checkPaths(directory,metadata)
        guard owner.parentIdentity==metadata.directoryIdentity,activity.parentIdentity==metadata.directoryIdentity,
              try owner.readOwnershipMetadata()==bytes else {throw WorkflowFileLeaseError.unsafePath}
    }
    func close() {guard !closed else {return};closed=true;activity.close();owner.close()}
    deinit {close()}
}

final class WorkflowEvaluationOwner {
    let directory:URL
    let metadata:WorkflowEvaluationOwnershipMetadata
    let ownershipSHA256:String
    private let bytes:Data
    private let owner:WorkflowFileLease
    private var activity:WorkflowFileLease?
    private var closed=false
    private var launched=false
    private var launchBudgetSHA256:String?
    fileprivate init(directory:URL,metadata:WorkflowEvaluationOwnershipMetadata,bytes:Data,owner:WorkflowFileLease,activity:WorkflowFileLease) {
        self.directory=directory;self.metadata=metadata;self.bytes=bytes;self.owner=owner;self.activity=activity
        ownershipSHA256=WorkflowArchiveCodec.hash(bytes)
    }
    func check()throws {
        guard !closed else {throw WorkflowFileLeaseError.closed}
        try owner.checkIdentity()
        try WorkflowEvaluationOwnership.checkPaths(directory,metadata)
        guard owner.parentIdentity==metadata.directoryIdentity,try owner.readOwnershipMetadata()==bytes else {throw WorkflowFileLeaseError.unsafePath}
        if let activity {try activity.checkIdentity()}
        else {
            let probe=try WorkflowFileLease(directory:directory,name:"activity.lock",mode:.shared,expectedIdentity:metadata.activityLockIdentity,requireLocalAPFS:true)
            defer {probe.close()}
            try probe.checkIdentity()
        }
    }
    func publishLaunchBudget(remainingSeconds:Double)throws->String {
        try check()
        guard !launched else {throw WorkflowFileLeaseError.unsafePath}
        try WorkflowEvaluationOwnership.requireUnclaimed(directory,metadata)
        let request=try WorkflowEvaluationOwnership.verifyRequest(directory,metadata)
        guard let budget=request.executionBudget else {throw WorkflowFileLeaseError.unsafePath}
        let pin=try WorkflowWorkerLaunchBudget.publish(directory:directory,requestSHA256:metadata.requestSHA256,budget:budget,remainingSeconds:remainingSeconds)
        launchBudgetSHA256=pin
        try check()
        return pin
    }
    @discardableResult func prepareLaunch()throws->String {
        try check()
        guard !launched else {throw WorkflowFileLeaseError.unsafePath}
        try WorkflowEvaluationOwnership.requireUnclaimed(directory,metadata)
        let request=try WorkflowEvaluationOwnership.verifyRequest(directory,metadata)
        try WorkflowWorkerLaunchBudget.verify(directory:directory,requestSHA256:metadata.requestSHA256,budget:request.executionBudget,expectedSHA256:launchBudgetSHA256)
        try check()
        launched=true;activity?.close();activity=nil
        try check()
        return ownershipSHA256
    }
    func validateDirectory(_ url:URL)throws {
        guard url.isFileURL,url.path==directory.path else {throw WorkflowFileLeaseError.unsafePath}
        try check()
    }
    func close() {guard !closed else {return};closed=true;activity?.close();activity=nil;owner.close()}
    deinit {close()}
}
