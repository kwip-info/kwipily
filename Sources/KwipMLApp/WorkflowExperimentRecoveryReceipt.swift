import Foundation
import KwipMLCore

/// Bounded explanation of one locked reconciliation, preserved with its ledger.
struct WorkflowExperimentRecoveryReceipt:Codable,Equatable {
    enum Reason:String,Codable {
        case pendingCancelled,verifiedOutcomeRecovered,terminalEvidenceUnavailable
    }
    struct Variant:Codable,Equatable {
        let ordinal:Int
        let priorStatus:WorkflowExperimentVariantStatus
        let status:WorkflowExperimentVariantStatus
        let reason:Reason
        let detail:String
    }
    let schemaVersion:Int
    let timestampUnixSeconds:Double
    let priorRevision:UInt64
    let variants:[Variant]

    static func boundedDetail(_ text:String)->String {
        var result="",count=0
        for scalar in text.unicodeScalars {
            let width=String(scalar).utf8.count
            guard count+width<=2048 else {break}
            result.unicodeScalars.append(scalar);count+=width
        }
        return result
    }
    func validate(record:WorkflowExperimentGroupRecord)throws {
        guard schemaVersion == 1,timestampUnixSeconds.isFinite,(0...253402300799).contains(timestampUnixSeconds),
              priorRevision<record.revision,(1...32).contains(variants.count),record.stopRequested,
              variants.map(\.ordinal) == variants.map(\.ordinal).sorted(),
              Set(variants.map(\.ordinal)).count == variants.count else {throw failure()}
        for note in variants {
            guard record.variants.indices.contains(note.ordinal-1),note.detail.utf8.count<=2048,
                  record.variants[note.ordinal-1].ordinal == note.ordinal,
                  record.variants[note.ordinal-1].status == note.status else {throw failure()}
            let entry=record.variants[note.ordinal-1]
            switch note.reason {
            case .pendingCancelled:
                guard note.priorStatus == .pending,note.status == .cancelled,entry.outcomeSHA256 == nil else {throw failure()}
            case .verifiedOutcomeRecovered:
                guard note.priorStatus == .running,[.completed,.failed,.cancelled].contains(note.status),
                      entry.outcomeSHA256.map(WorkflowArchiveCodec.isHash) == true else {throw failure()}
            case .terminalEvidenceUnavailable:
                guard note.priorStatus == .running,note.status == .interrupted,entry.outcomeSHA256 == nil else {throw failure()}
            }
        }
    }
    func validateTransition(prior:WorkflowExperimentGroupRecord,next:WorkflowExperimentGroupRecord)throws {
        guard priorRevision == prior.revision,prior.revision<UInt64.max,
              prior.variants.count == next.variants.count else {throw failure()}
        var persisted=next;persisted.revision=prior.revision+1
        try validate(record:persisted)
        let changed=zip(prior.variants,next.variants).filter {pair in pair.0 != pair.1}
        guard changed.map({$0.0.ordinal}) == variants.map(\.ordinal) else {throw failure()}
        for (change,note) in zip(changed,variants) {
            guard change.0.status == note.priorStatus,change.1.status == note.status else {throw failure()}
        }
    }
    private func failure()->DataFlowContractError {.init("Invalid experiment recovery receipt.")}
}
