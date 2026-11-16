import Foundation
import Darwin
import CryptoKit
import KwipMLCore

struct WorkflowGroupOwnershipMetadata:Codable,Equatable {
    let schemaVersion:Int
    let groupID:String
    let generation:String
    let directoryIdentity:WorkflowFileIdentity
    let ownerLockIdentity:WorkflowFileIdentity
    let activityLockIdentity:WorkflowFileIdentity
    let createdAtUnixSeconds:Double
}

enum WorkflowGroupOwnership {
    /// Resolve aliases only at the caller's known workspace boundary. Every
    /// descendant is created/opened relative to retained directory descriptors.
    static func createGroup(workspace:URL,projectID:String,groupID:String=UUID().uuidString)throws->(directory:URL,owner:WorkflowGroupOwner) {
        try validateID(projectID)
        try validateID(groupID)
        guard workspace.isFileURL,!workspace.path.contains("\0"),let canonical=realpath(workspace.path,nil) else {throw WorkflowFileLeaseError.unsafePath}
        defer {free(canonical)}
        let root=URL(fileURLWithPath:String(cString:canonical),isDirectory:true)
        let rootFD=open(root.path,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC)
        guard rootFD>=0 else {throw WorkflowFileLeaseError.system(errno)}
        defer {Darwin.close(rootFD)}
        func identity(_ fd:Int32)throws->WorkflowFileIdentity {
            var info=stat()
            guard fstat(fd,&info)==0 else {throw WorkflowFileLeaseError.system(errno)}
            guard info.st_mode & mode_t(S_IFMT)==mode_t(S_IFDIR),info.st_uid==geteuid(),info.st_mode & 0o022==0 else {throw WorkflowFileLeaseError.unsafePath}
            return .init(device:UInt64(UInt32(bitPattern:info.st_dev)),inode:UInt64(info.st_ino))
        }
        let rootIdentity=try identity(rootFD)
        func child(_ parent:Int32,_ name:String,exclusive:Bool=false)throws->Int32 {
            if mkdirat(parent,name,0o700) != 0 {
                guard !exclusive,errno==EEXIST else {throw WorkflowFileLeaseError.system(errno)}
            } else {
                guard fsync(parent)==0 else {throw WorkflowFileLeaseError.system(errno)}
            }
            let fd=openat(parent,name,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC)
            guard fd>=0 else {throw WorkflowFileLeaseError.unsafePath}
            do {_ = try identity(fd);return fd} catch {Darwin.close(fd);throw error}
        }
        let experiments=try child(rootFD,"workflow-experiments")
        defer {Darwin.close(experiments)}
        let project=try child(experiments,projectID)
        defer {Darwin.close(project)}
        let group=try child(project,groupID,exclusive:true)
        defer {Darwin.close(group)}
        let createdIdentity=try identity(group)
        guard fsync(project)==0 else {throw WorkflowFileLeaseError.system(errno)}
        let directory=root.appendingPathComponent("workflow-experiments",isDirectory:true).appendingPathComponent(projectID,isDirectory:true).appendingPathComponent(groupID,isDirectory:true)
        // Detect replacement between descriptor creation and the ownership service's
        // own no-follow walk. Partial owned directories remain recovery evidence.
        let owner=try initialize(directory:directory,groupID:groupID)
        do {
            let currentRoot=open(root.path,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC)
            guard currentRoot>=0 else {throw WorkflowFileLeaseError.unsafePath}
            defer {Darwin.close(currentRoot)}
            guard try identity(currentRoot)==rootIdentity,owner.metadata.directoryIdentity==createdIdentity else {throw WorkflowFileLeaseError.unsafePath}
            try owner.validateDirectory(directory)
            return (directory,owner)
        } catch {owner.close();throw error}
    }
    static func initialize(directory:URL,groupID:String)throws->WorkflowGroupOwner {
        try validateID(groupID)
        try validateDirectoryPath(directory)
        for name in ["group.json","ownership.json","owner.lock","activity.lock"] {
            var info=stat()
            if lstat(directory.appendingPathComponent(name).path,&info) == 0 {throw WorkflowFileLeaseError.unsafePath}
            guard errno == ENOENT else {throw WorkflowFileLeaseError.system(errno)}
        }
        let owner=try WorkflowFileLease(directory:directory,name:"owner.lock",mode:.exclusive,create:true,createExclusive:true,requireLocalAPFS:true)
        do {
            let activity=try WorkflowFileLease(directory:directory,name:"activity.lock",mode:.exclusive,create:true,createExclusive:true,requireLocalAPFS:true)
            defer {activity.close()}
            try requireLocalAPFS(owner)
            guard owner.parentIdentity == activity.parentIdentity,try !owner.containsSibling("group.json"),try !owner.containsSibling("ownership.json") else {throw WorkflowFileLeaseError.unsafePath}
            let metadata=WorkflowGroupOwnershipMetadata(schemaVersion:1,groupID:groupID,generation:UUID().uuidString,directoryIdentity:owner.parentIdentity,ownerLockIdentity:owner.identity,activityLockIdentity:activity.identity,createdAtUnixSeconds:Date().timeIntervalSince1970)
            let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
            let bytes=try encoder.encode(metadata)
            try owner.publishOwnershipMetadata(bytes)
            try activity.checkIdentity()
            return WorkflowGroupOwner(directory:directory,metadata:metadata,bytes:bytes,owner:owner)
        } catch {owner.close();throw error}
    }
    static func readMetadata(directory:URL,groupID:String)throws->WorkflowGroupOwnershipMetadata {
        try validateID(groupID);try validateDirectoryPath(directory)
        let bytes=try DiagnosticFileIO.read(root:directory,parts:["ownership.json"],limit:16384)
        return try decode(bytes,groupID:groupID)
    }
    static func acquireRecovery(directory:URL,groupID:String)throws->WorkflowGroupRecovery {
        try validateID(groupID);try validateDirectoryPath(directory)
        let bytes=try DiagnosticFileIO.read(root:directory,parts:["ownership.json"],limit:16384)
        let metadata=try decode(bytes,groupID:groupID)
        let owner=try WorkflowFileLease(directory:directory,name:"owner.lock",mode:.exclusive,expectedIdentity:metadata.ownerLockIdentity,requireLocalAPFS:true)
        do {
            let activity=try WorkflowFileLease(directory:directory,name:"activity.lock",mode:.exclusive,expectedIdentity:metadata.activityLockIdentity,requireLocalAPFS:true)
            do {
                try requireLocalAPFS(owner)
                guard owner.parentIdentity == metadata.directoryIdentity,activity.parentIdentity == metadata.directoryIdentity,try owner.readOwnershipMetadata() == bytes else {throw WorkflowFileLeaseError.unsafePath}
                let result=WorkflowGroupRecovery(directory:directory,metadata:metadata,bytes:bytes,owner:owner,activity:activity)
                try result.check()
                return result
            } catch {activity.close();throw error}
        } catch {owner.close();throw error}
    }
    fileprivate static func requireLocalAPFS(_ lease:WorkflowFileLease)throws {
        let filesystem=try lease.fileSystemInfo()
        guard filesystem.type == "apfs",filesystem.isLocal else {throw WorkflowFileLeaseError.unsafePath}
    }
    fileprivate static func digest(_ bytes:Data)->String {SHA256.hash(data:bytes).map {String(format:"%02x",$0)}.joined()}
    private static func validateID(_ id:String)throws {
        guard let uuid=UUID(uuidString:id),uuid.uuidString == id else {throw WorkflowFileLeaseError.unsafePath}
    }
    private static func decode(_ bytes:Data,groupID:String)throws->WorkflowGroupOwnershipMetadata {
        guard bytes.count <= 16384,let object=try JSONSerialization.jsonObject(with:bytes) as? [String:Any],Set(object.keys) == ["schemaVersion","groupID","generation","directoryIdentity","ownerLockIdentity","activityLockIdentity","createdAtUnixSeconds"] else {throw WorkflowFileLeaseError.unsafePath}
        let metadata=try JSONDecoder().decode(WorkflowGroupOwnershipMetadata.self,from:bytes)
        try validateID(metadata.groupID);try validateID(metadata.generation)
        guard metadata.schemaVersion == 1,metadata.groupID == groupID,metadata.createdAtUnixSeconds.isFinite,(0...253402300799).contains(metadata.createdAtUnixSeconds) else {throw WorkflowFileLeaseError.unsafePath}
        return metadata
    }
    /// Read-only preflight. Mutation and final authority always use anchored lease descriptors.
    private static func validateDirectoryPath(_ directory:URL)throws {
        guard directory.isFileURL,!directory.path.contains("\0"),let resolved=realpath(directory.path,nil) else {throw WorkflowFileLeaseError.unsafePath}
        defer {free(resolved)}
        guard String(cString:resolved) == directory.path else {throw WorkflowFileLeaseError.unsafePath}
    }
}

final class WorkflowGroupOwner {
    let metadata:WorkflowGroupOwnershipMetadata
    private let directory:URL
    private let bytes:Data
    private let digest:String
    private let owner:WorkflowFileLease
    private var closed=false
    fileprivate init(directory:URL,metadata:WorkflowGroupOwnershipMetadata,bytes:Data,owner:WorkflowFileLease) {
        self.directory=directory;self.metadata=metadata;self.bytes=bytes;self.digest=WorkflowGroupOwnership.digest(bytes);self.owner=owner
    }
    func check()throws {
        guard !closed else {throw WorkflowFileLeaseError.closed}
        let activity=try WorkflowFileLease(directory:directory,name:"activity.lock",mode:.shared,expectedIdentity:metadata.activityLockIdentity,requireLocalAPFS:true)
        defer {activity.close()}
        try check(activity:activity)
    }
    private func check(activity:WorkflowFileLease)throws {
        guard !closed else {throw WorkflowFileLeaseError.closed}
        try owner.checkIdentity();try activity.checkIdentity();try WorkflowGroupOwnership.requireLocalAPFS(owner)
        guard owner.parentIdentity == metadata.directoryIdentity,activity.parentIdentity == metadata.directoryIdentity,activity.identity == metadata.activityLockIdentity,let current=try owner.readOwnershipMetadata(),current == bytes,WorkflowGroupOwnership.digest(current) == digest else {throw WorkflowFileLeaseError.unsafePath}
    }
    func validateDirectory(_ url:URL)throws {
        guard url.isFileURL,url.path == directory.path else {throw WorkflowFileLeaseError.unsafePath}
        try check()
    }
    func publishGroupRecord(_ data:Data)throws {
        try check()
        try owner.publishGroupRecord(data)
        try check()
    }
    func replaceGroupRecord(expected:Data,data:Data)throws {
        try check()
        try owner.replaceGroupRecord(expected:expected,data:data)
        try check()
    }
    func publishGroupAuthoring(_ data:Data)throws {
        try check()
        try owner.publishGroupAuthoring(data)
        try check()
    }
    func publishLaunch(projectID:String,ordinal:Int,runID:String,directoryID:String,requestIdentity:String,phase:WorkflowLaunchIntent.Phase,arguments:[String],nonce:String=UUID().uuidString,executionBudget:ResolvedExecutionBudget?=nil,remainingSeconds:Double?=nil)throws->WorkflowPublishedLaunch {
        guard !closed else {throw WorkflowFileLeaseError.closed}
        let activity=try WorkflowFileLease(directory:directory,name:"activity.lock",mode:.exclusive,expectedIdentity:metadata.activityLockIdentity,requireLocalAPFS:true)
        defer {activity.close()}
        try check(activity:activity)
        let intent=WorkflowLaunchIntent(schemaVersion:executionBudget == nil && remainingSeconds == nil ? 1 : 2,projectID:projectID,groupID:metadata.groupID,generation:metadata.generation,runID:runID,directoryID:directoryID,nonce:nonce,ordinal:ordinal,requestIdentity:requestIdentity,ownershipSHA256:digest,phase:phase,arguments:arguments,executionBudget:executionBudget,remainingSeconds:remainingSeconds)
        try WorkflowLaunchIntent.validate(intent,groupDirectory:directory)
        guard let ledger=try owner.readGroupRecord() else {throw WorkflowFileLeaseError.unsafePath}
        try WorkflowLaunchIntent.matchesLedger(ledger,intent:intent)
        if phase == .training {
            guard let preparation=try owner.readPhaseSlot(runID:runID,phase:"preparation") else {throw WorkflowFileLeaseError.unsafePath}
            let prior=try WorkflowLaunchIntent.decode(preparation)
            try WorkflowLaunchIntent.validate(prior,groupDirectory:directory)
            guard prior.schemaVersion == intent.schemaVersion,prior.executionBudget == intent.executionBudget else {throw WorkflowFileLeaseError.unsafePath}
            if let remaining = intent.remainingSeconds {
                guard let previous = prior.remainingSeconds,remaining <= previous else {throw WorkflowFileLeaseError.unsafePath}
            }
            guard prior.phase == .preparation,prior.projectID == projectID,prior.groupID == metadata.groupID,prior.generation == metadata.generation,prior.runID == runID,prior.directoryID == directoryID,prior.ordinal == ordinal,prior.requestIdentity == requestIdentity,prior.ownershipSHA256 == digest,try owner.readLaunch(nonce:prior.nonce) == preparation,let claim=try owner.readLaunchClaim(nonce:prior.nonce),try WorkflowLaunchClaim.validClaimed(claim,nonce:prior.nonce,intentSHA256:WorkflowPhaseArtifacts.hash(preparation)) else {throw WorkflowFileLeaseError.unsafePath}
        }
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        let data=try encoder.encode(intent)
        try check(activity:activity)
        guard try owner.readGroupRecord() == ledger else {throw WorkflowFileLeaseError.unsafePath}
        try owner.publishPhaseSlot(runID:runID,phase:phase.rawValue,data:data)
        try owner.publishLaunch(nonce:nonce,data:data)
        try check(activity:activity)
        return WorkflowPublishedLaunch(groupDirectory:directory,intent:intent,bytes:data)
    }
    func close() {guard !closed else {return};closed=true;owner.close()}
    deinit {close()}
}

final class WorkflowGroupRecovery {
    let metadata:WorkflowGroupOwnershipMetadata
    private let directory:URL
    private let bytes:Data
    private let digest:String
    var ownershipSHA256:String {digest}
    private let owner:WorkflowFileLease
    private let activity:WorkflowFileLease
    private var closed=false
    fileprivate init(directory:URL,metadata:WorkflowGroupOwnershipMetadata,bytes:Data,owner:WorkflowFileLease,activity:WorkflowFileLease) {
        self.directory=directory;self.metadata=metadata;self.bytes=bytes;self.digest=WorkflowGroupOwnership.digest(bytes);self.owner=owner;self.activity=activity
    }
    func check()throws {
        guard !closed else {throw WorkflowFileLeaseError.closed}
        try owner.checkIdentity();try activity.checkIdentity();try WorkflowGroupOwnership.requireLocalAPFS(owner)
        guard owner.parentIdentity == metadata.directoryIdentity,activity.parentIdentity == metadata.directoryIdentity,let current=try owner.readOwnershipMetadata(),current == bytes,WorkflowGroupOwnership.digest(current) == digest else {throw WorkflowFileLeaseError.unsafePath}
    }
    func validateDirectory(_ url:URL)throws {
        guard url.isFileURL,url.path == directory.path else {throw WorkflowFileLeaseError.unsafePath}
        try check()
    }
    /// Both exclusive leases remain retained throughout anchored reads and writes.
    /// Never acquire another activity description while recovery holds its lease.
    private func withAuthority<T>(_ operation:()throws->T)throws->T {
        try check()
        let result=try operation()
        try check()
        return result
    }
    func readGroupRecord()throws->Data? {
        try withAuthority {try owner.readGroupRecord()}
    }
    func replaceGroupRecord(expected:Data,data:Data)throws {
        try withAuthority {try owner.replaceGroupRecord(expected:expected,data:data)}
    }
    func readPhaseSlot(runID:String,phase:String)throws->Data? {
        try withAuthority {try owner.readPhaseSlot(runID:runID,phase:phase)}
    }
    func readLaunch(nonce:String)throws->Data? {
        try withAuthority {try owner.readLaunch(nonce:nonce)}
    }
    func readLaunchClaim(nonce:String)throws->Data? {
        try withAuthority {try owner.readLaunchClaim(nonce:nonce)}
    }
    /// A crash may leave the immutable phase slot before its nonce intent exists.
    /// Revoke that authorization without manufacturing the missing intent.
    func revokePhase(runID:String,phase:String,intentSHA256:String)throws {
        try withAuthority {
            guard WorkflowArchiveCodec.isHash(intentSHA256),
                  let slot=try owner.readPhaseSlot(runID:runID,phase:phase),
                  WorkflowPhaseArtifacts.hash(slot) == intentSHA256 else {throw WorkflowFileLeaseError.unsafePath}
            let intent=try WorkflowLaunchIntent.decode(slot)
            try WorkflowLaunchIntent.validate(intent,groupDirectory:directory)
            guard intent.runID == runID,intent.phase.rawValue == phase,
                  intent.groupID == metadata.groupID,intent.generation == metadata.generation,
                  intent.ownershipSHA256 == digest else {throw WorkflowFileLeaseError.unsafePath}
            if let launch=try owner.readLaunch(nonce:intent.nonce),launch != slot {throw WorkflowFileLeaseError.unsafePath}
            let claim=WorkflowLaunchClaim(schemaVersion:1,nonce:intent.nonce,intentSHA256:intentSHA256,state:"revoked")
            let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
            // Existing claims, including malformed or already revoked claims,
            // remain immutable; the coordinator validates and skips them.
            try owner.publishLaunchClaim(nonce:intent.nonce,data:encoder.encode(claim))
        }
    }
    func revokeLaunch(nonce:String,intentSHA256:String)throws {
        try check()
        guard WorkflowArchiveCodec.isHash(intentSHA256),let intentBytes=try owner.readLaunch(nonce:nonce),WorkflowPhaseArtifacts.hash(intentBytes) == intentSHA256 else {throw WorkflowFileLeaseError.unsafePath}
        let intent=try WorkflowLaunchIntent.decode(intentBytes)
        try WorkflowLaunchIntent.validate(intent,groupDirectory:directory)
        guard intent.nonce == nonce,intent.groupID == metadata.groupID,intent.generation == metadata.generation,intent.ownershipSHA256 == digest,try owner.readPhaseSlot(runID:intent.runID,phase:intent.phase.rawValue) == intentBytes else {throw WorkflowFileLeaseError.unsafePath}
        let claim=WorkflowLaunchClaim(schemaVersion:1,nonce:nonce,intentSHA256:intentSHA256,state:"revoked")
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        try owner.publishLaunchClaim(nonce:nonce,data:encoder.encode(claim))
        try check()
    }
    /// Optional scratch reclamation follows terminal reconciliation and consumes
    /// no retained artifacts. Every recorded authorization must already be closed.
    func cleanupPublicationTemporaries() throws -> WorkflowPublicationCleanupReport {
        try withAuthority {
            guard let raw = try owner.readGroupRecord() else { throw WorkflowFileLeaseError.unsafePath }
            let record = try JSONDecoder().decode(WorkflowExperimentGroupRecord.self, from: raw)
            guard record.schemaVersion == 1, record.groupID == metadata.groupID, record.isTerminal else { throw WorkflowFileLeaseError.unsafePath }
            for variant in record.variants {
                guard let runID = variant.runID else { continue }
                for phase in [WorkflowLaunchIntent.Phase.preparation, .training] {
                    guard let slot = try owner.readPhaseSlot(runID: runID, phase: phase.rawValue) else { continue }
                    let intent = try WorkflowLaunchIntent.decode(slot)
                    try WorkflowLaunchIntent.validate(intent, groupDirectory: directory)
                    guard intent.groupID == metadata.groupID, intent.generation == metadata.generation,
                          intent.ownershipSHA256 == digest, intent.projectID == record.projectID,
                          intent.runID == runID, intent.ordinal == variant.ordinal,
                          intent.directoryID == variant.directoryID, intent.requestIdentity == variant.requestIdentity,
                          intent.phase == phase, let claimBytes = try owner.readLaunchClaim(nonce: intent.nonce),
                          let object = try JSONSerialization.jsonObject(with: claimBytes) as? [String: Any],
                          Set(object.keys) == ["schemaVersion", "nonce", "intentSHA256", "state"] else { throw WorkflowFileLeaseError.unsafePath }
                    let claim = try JSONDecoder().decode(WorkflowLaunchClaim.self, from: claimBytes)
                    let launch = try owner.readLaunch(nonce: intent.nonce)
                    guard claim.schemaVersion == 1, claim.nonce == intent.nonce,
                          claim.intentSHA256 == WorkflowPhaseArtifacts.hash(slot),
                          ["claimed", "revoked"].contains(claim.state),
                          launch == slot || (launch == nil && claim.state == "revoked") else { throw WorkflowFileLeaseError.unsafePath }
                }
            }
            return try owner.cleanupPublicationTemporaries(activity: activity)
        }
    }
    func close() {guard !closed else {return};closed=true;activity.close();owner.close()}
    deinit {close()}
}

struct WorkflowPublishedLaunch {
    let groupDirectory:URL
    let nonce:String
    let intentSHA256:String
    let ownershipSHA256:String
    let intent:WorkflowLaunchIntent
    // Construction requires a complete, byte-bound publication through the owner.
    fileprivate init(groupDirectory:URL,intent:WorkflowLaunchIntent,bytes:Data) {
        self.groupDirectory=groupDirectory;self.intent=intent;nonce=intent.nonce
        intentSHA256=WorkflowPhaseArtifacts.hash(bytes);ownershipSHA256=intent.ownershipSHA256
    }
}
