import Foundation
import KwipMLCore

enum WorkflowExperimentVariantStatus:String,Codable {
    case pending,running,completed,failed,cancelled,interrupted
    var isTerminal:Bool {self != .pending && self != .running}
}
struct WorkflowExperimentGroupRecord:Codable,Equatable {
    struct Variant:Codable,Equatable {
        let ordinal:Int
        let requestIdentity:String
        var status:WorkflowExperimentVariantStatus
        var runID:String?
        var directoryID:String?
        var outcomeSHA256:String?
    }
    let schemaVersion:Int
    var createdAtUnixSeconds:Double? = nil
    let groupID:String
    let projectID:String
    let train:WorkflowNodeAddress
    let sourceRequestIdentity:String
    let inputsManifestSHA256:String
    let authoringSHA256:String
    var revision:UInt64
    var stopRequested:Bool
    var variants:[Variant]
    var recoveryReceipt:WorkflowExperimentRecoveryReceipt? = nil
    var isTerminal:Bool {variants.allSatisfy {$0.status.isTerminal}}
}

/// Owner-authorized serialized group ledger. Reopened history is read-only.
/// Persist before publishing state and reject stale bytes. Hashes are not signatures.
@MainActor final class WorkflowExperimentGroupStore {
    private(set) var record:WorkflowExperimentGroupRecord
    let directory:URL
    private var persisted:Data
    private let groupOwner:WorkflowGroupOwner?
    private init(record:WorkflowExperimentGroupRecord,directory:URL,persisted:Data,groupOwner:WorkflowGroupOwner?=nil) {
        self.record=record;self.directory=directory;self.persisted=persisted;self.groupOwner=groupOwner
    }
    static func create(requests:[WorkflowExecutionRequest],directory:URL,groupOwner:WorkflowGroupOwner)throws->WorkflowExperimentGroupStore {
        try groupOwner.validateDirectory(directory)
        let capture=try prepareNewGroup(requests:requests,directory:directory)
        return try publish(capture,directory:directory,groupOwner:groupOwner)
    }
    /// Background work only validates and encodes immutable inputs. The owner and
    /// anchored publication remain serialized on the main actor after cancellation checks.
    static func createInBackground(requests:[WorkflowExecutionRequest],directory:URL,groupOwner:WorkflowGroupOwner) async throws->WorkflowExperimentGroupStore {
        try Task.checkCancellation()
        try groupOwner.validateDirectory(directory)
        let worker=Task.detached(priority:.userInitiated) {
            try prepareNewGroup(requests:requests,directory:directory)
        }
        let capture=try await withTaskCancellationHandler(operation:{try await worker.value},onCancel:{worker.cancel()})
        try Task.checkCancellation()
        return try publish(capture,directory:directory,groupOwner:groupOwner)
    }
    private static func publish(_ capture:(WorkflowExperimentGroupRecord,Data,Data),directory:URL,groupOwner:WorkflowGroupOwner)throws->WorkflowExperimentGroupStore {
        try Task.checkCancellation()
        try groupOwner.validateDirectory(directory)
        guard groupOwner.metadata.groupID == capture.0.groupID else {
            throw failure("The group owner belongs to a different experiment identity.")
        }
        try groupOwner.publishGroupAuthoring(capture.2)
        try groupOwner.publishGroupRecord(capture.1)
        return .init(record:capture.0,directory:directory,persisted:capture.1,groupOwner:groupOwner)
    }
    nonisolated private static func prepareNewGroup(requests:[WorkflowExecutionRequest],directory:URL)throws->(WorkflowExperimentGroupRecord,Data,Data) {
        try Task.checkCancellation()
        guard (1...32).contains(requests.count),let first=requests.first,let binding=first.experiment,
              directory.lastPathComponent == binding.groupID else {throw failure("A group needs its complete ordered variant requests.")}
        guard let inputs=first.experimentInputs,inputs.directory == directory else {throw failure("Group inputs belong to another directory.")}
        try inputs.verify()
        for (index,request) in requests.enumerated() {
            guard let identity=request.experiment,identity.ordinal == index+1,
                  identity.groupID == binding.groupID,identity.inputsManifestSHA256 == binding.inputsManifestSHA256,
                  identity.sourceRequestIdentity == binding.sourceRequestIdentity,
                  request.authoringDocument == first.authoringDocument,request.train == first.train,
                  request.experimentInputs == inputs else {throw failure("Variant requests do not share one captured source and input group.")}
        }
        // Do not accept a truncated subset of a declared grid.
        // Resolve the captured authority without rewriting it: current Train owns
        // these settings, while retained legacy sources keep their connected setup.
        let authority=try WorkflowTrainRunSetup.resolve(in:first.authoringDocument,train:first.train)
        guard let value=authority.configuration["experiment"] else {throw failure("The captured source has no experiment specification.")}
        let spec=try WorkflowExperimentSpecification.decode(value)
        guard spec.axes.reduce(1,{$0*$1.values.count}) == requests.count else {throw failure("Every declared variant must be recorded before launch.")}
        let source=try WorkflowCodec.encode(first.authoringDocument)
        let record=WorkflowExperimentGroupRecord(schemaVersion:1,createdAtUnixSeconds:Date().timeIntervalSince1970,groupID:binding.groupID,projectID:first.lease.projectID,train:first.train,
            sourceRequestIdentity:binding.sourceRequestIdentity,inputsManifestSHA256:binding.inputsManifestSHA256,
            authoringSHA256:WorkflowPhaseArtifacts.hash(source),revision:0,stopRequested:false,
            variants:requests.enumerated().map {.init(ordinal:$0.offset+1,requestIdentity:$0.element.requestIdentity,status:.pending)})
        try validate(record,directory:directory)
        let bytes=try WorkflowArchiveCodec.encode(record)
        try Task.checkCancellation()
        return (record,bytes,source)
    }
    static func open(directory:URL)throws->WorkflowExperimentGroupStore {
        let record=try verifiedRecord(directory:directory)
        let bytes=try DiagnosticFileIO.read(root:directory,parts:["group.json"],limit:256_000)
        guard try JSONDecoder().decode(WorkflowExperimentGroupRecord.self,from:bytes) == record else {throw failure("Group changed while reopening.")}
        return .init(record:record,directory:directory,persisted:bytes)
    }
    nonisolated static func readRecord(directory:URL)throws->WorkflowExperimentGroupRecord {
        let bytes=try DiagnosticFileIO.read(root:directory,parts:["group.json"],limit:256_000)
        let record=try JSONDecoder().decode(WorkflowExperimentGroupRecord.self,from:bytes)
        try validate(record,directory:directory)
        return record
    }
    nonisolated static func verifiedRecord(directory:URL)throws->WorkflowExperimentGroupRecord {
        let record=try readRecord(directory:directory)
        let source=try DiagnosticFileIO.read(root:directory,parts:["authoring.json"],limit:2_000_000)
        guard WorkflowPhaseArtifacts.hash(source) == record.authoringSHA256,
              try WorkflowCodec.decode(source).projectID == record.projectID else {throw failure("The group's captured authoring source changed.")}
        _=try WorkflowExperimentInputs.open(directory:directory,expectedManifestSHA256:record.inputsManifestSHA256)
        return record
    }
    func begin(ordinal:Int,runID:UUID,directoryID:UUID)throws {
        guard !record.stopRequested,!record.variants.contains(where:{$0.status == .running}),
              record.variants.first(where:{$0.status == .pending})?.ordinal == ordinal else {throw Self.failure("Start only the next pending variant in an unstopped group.")}
        var next=record
        next.variants[ordinal-1].status = .running
        next.variants[ordinal-1].runID=runID.uuidString
        next.variants[ordinal-1].directoryID=directoryID.uuidString
        try commit(next)
    }
    func finish(ordinal:Int,runID:UUID,directoryID:UUID,requestIdentity:String,
                status:WorkflowExperimentVariantStatus,outcomeSHA256:String?=nil)throws {
        guard record.variants.indices.contains(ordinal-1),[.completed,.failed,.cancelled].contains(status) else {throw Self.failure("Invalid terminal transition.")}
        let current=record.variants[ordinal-1]
        guard current.status == .running,current.runID == runID.uuidString,current.directoryID == directoryID.uuidString,
              current.requestIdentity == requestIdentity else {throw Self.failure("Terminal evidence belongs to another or already finished variant.")}
        if status == .completed && outcomeSHA256 == nil {throw Self.failure("Completion requires a recorded outcome hash.")}
        var next=record;next.variants[ordinal-1].status=status;next.variants[ordinal-1].outcomeSHA256=outcomeSHA256
        try commit(next)
    }
    /// Stops future starts. The active worker must be cancelled and acknowledged separately.
    func cancelPending()throws {
        var next=record;next.stopRequested=true
        for index in next.variants.indices where next.variants[index].status == .pending {next.variants[index].status = .cancelled}
        try commit(next)
    }
    /// Recovery requires a separate exclusive recovery capability and outcome reconciliation.
    /// A ledger state or an ordinary owner is not proof that an old worker has exited.
    func recoverInterrupted()throws {
        throw Self.failure("Interrupted recovery requires explicit recovery authority; read-only history cannot recover execution.")
    }
    private func commit(_ proposed:WorkflowExperimentGroupRecord)throws {
        guard let groupOwner else {throw Self.failure("This experiment history is read-only; a live group owner is required.")}
        guard proposed.recoveryReceipt == record.recoveryReceipt else {throw Self.failure("Only locked recovery may publish a recovery receipt.")}
        try groupOwner.validateDirectory(directory)
        guard record.revision<UInt64.max else {throw Self.failure("Group revision limit reached.")}
        var next=proposed;next.revision=record.revision+1
        try Self.validate(next,directory:directory)
        let bytes=try WorkflowArchiveCodec.encode(next)
        try groupOwner.replaceGroupRecord(expected:persisted,data:bytes)
        persisted=bytes;record=next
    }
    /// One recovery transaction, using bytes read while both exclusive leases are held.
    /// Terminal siblings and captured identities are immutable during reconciliation.
    nonisolated static func commitRecovery(expected:Data,proposed:WorkflowExperimentGroupRecord,
                                          directory:URL,recovery:WorkflowGroupRecovery)throws->WorkflowExperimentGroupRecord {
        try Task.checkCancellation()
        try recovery.validateDirectory(directory)
        guard try recovery.readGroupRecord() == expected else {throw failure("Experiment changed during recovery.")}
        let prior=try JSONDecoder().decode(WorkflowExperimentGroupRecord.self,from:expected)
        try validate(prior,directory:directory)
        guard recovery.metadata.groupID == prior.groupID else {throw failure("Recovery belongs to another group.")}
        var permitted=prior
        permitted.stopRequested=proposed.stopRequested
        permitted.variants=proposed.variants
        permitted.recoveryReceipt=proposed.recoveryReceipt
        guard permitted == proposed,(proposed.stopRequested || prior.isTerminal),proposed.variants.count == prior.variants.count else {
            throw failure("Recovery cannot change captured experiment identities.")
        }
        for (old,new) in zip(prior.variants,proposed.variants) {
            guard old.ordinal == new.ordinal,old.requestIdentity == new.requestIdentity,
                  old.runID == new.runID,old.directoryID == new.directoryID else {throw failure("Recovery cannot replace variant identity.")}
            if old.status.isTerminal {guard old == new else {throw failure("Recovery cannot rewrite terminal siblings.")}}
            else if old.status == .pending {
                guard new.status == .cancelled,new.outcomeSHA256 == nil else {throw failure("Pending variants may only be cancelled.")}
            } else {
                guard new.status.isTerminal,(new.status == .interrupted ? new.outcomeSHA256 == nil : new.outcomeSHA256 != nil) else {
                    throw failure("Recovered terminal results require an outcome pin; interrupted variants cannot claim one.")
                }
            }
        }
        if proposed == prior {try recovery.check();return prior}
        if proposed.variants != prior.variants {
            guard let receipt=proposed.recoveryReceipt else {throw failure("Recovery transitions require a receipt.")}
            try receipt.validateTransition(prior:prior,next:proposed)
        } else {
            guard proposed.recoveryReceipt == prior.recoveryReceipt else {throw failure("Recovery cannot rewrite an existing receipt without a transition.")}
        }
        guard prior.revision < UInt64.max else {throw failure("Group revision limit reached.")}
        var next=proposed;next.revision=prior.revision+1
        try validate(next,directory:directory)
        let bytes=try WorkflowArchiveCodec.encode(next)
        try Task.checkCancellation()
        try recovery.replaceGroupRecord(expected:expected,data:bytes)
        return next
    }
    nonisolated private static func validate(_ record:WorkflowExperimentGroupRecord,directory:URL)throws {
        try record.recoveryReceipt?.validate(record:record)
        guard record.createdAtUnixSeconds.map({$0.isFinite && (0...253402300799).contains($0)}) ?? true else {
            throw failure("Invalid experiment creation time.")
        }
        guard record.schemaVersion == 1,UUID(uuidString:record.groupID) != nil,record.groupID == directory.lastPathComponent,
              UUID(uuidString:record.projectID) != nil,record.train.domain == .control,!record.train.localID.isEmpty,WorkflowArchiveCodec.isHash(record.sourceRequestIdentity),
              WorkflowArchiveCodec.isHash(record.inputsManifestSHA256),WorkflowArchiveCodec.isHash(record.authoringSHA256),
              (1...32).contains(record.variants.count),Set(record.variants.map(\.requestIdentity)).count == record.variants.count,
              record.variants.filter({$0.status == .running}).count<=1 else {throw failure("Invalid experiment group record.")}
        for (index,variant) in record.variants.enumerated() {
            let hasRun=variant.runID != nil || variant.directoryID != nil
            guard variant.ordinal == index+1,WorkflowArchiveCodec.isHash(variant.requestIdentity),
                  !hasRun || (UUID(uuidString:variant.runID ?? "") != nil && UUID(uuidString:variant.directoryID ?? "") != nil),
                  variant.status != .pending || !hasRun,
                  ![.running,.completed,.failed,.interrupted].contains(variant.status) || hasRun,
                  variant.status != .completed || variant.outcomeSHA256 != nil,
                  variant.outcomeSHA256.map(WorkflowArchiveCodec.isHash) ?? true,
                  !record.stopRequested || variant.status != .pending else {throw failure("Invalid experiment variant record.")}
        }
    }
    nonisolated private static func failure(_ message:String)->DataFlowContractError {.init(message)}
}
