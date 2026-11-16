import Foundation
import KwipMLCore

/// Projection of the integrity-verified training manifest. Row identities stay
/// private: only counts and membership digests are retained for comparison.
struct WorkflowPartitionEvidence: Equatable {
    let sourceSHA256: String
    let trainingCount: Int
    let validationCount: Int
    let trainingMembershipSHA256: String
    let validationMembershipSHA256: String
    let trainingDataVersion: String
    let validationDataVersion: String

    enum Comparison: Equatable {
        case unavailable, differentRowsOrSource, sameRowsDifferentValues, sameRowsAndValues
        var message: String {
            switch self {
            case .unavailable: return "Held-out comparison unavailable: one or both archives lack recorded partition evidence."
            case .differentRowsOrSource: return "Different recorded held-out rows or source content. Score changes include data or split sensitivity."
            case .sameRowsDifferentValues: return "Same recorded held-out observations, different prepared values or order. Review data transformations and mappings before comparing scores."
            case .sameRowsAndValues: return "Same recorded held-out observations and prepared table values. Review mappings, model and training changes; this alone does not establish statistical significance."
            }
        }
    }
    static func comparison(_ a: Self?, _ b: Self?) -> Comparison {
        guard let a, let b else { return .unavailable }
        guard a.sourceSHA256 == b.sourceSHA256, a.validationCount == b.validationCount,
              a.validationMembershipSHA256 == b.validationMembershipSHA256 else { return .differentRowsOrSource }
        return a.validationDataVersion == b.validationDataVersion ? .sameRowsAndValues : .sameRowsDifferentValues
    }
    static func decode(_ value: JSONValue) throws -> Self? {
        guard case .object(let fields) = value else { throw WorkflowArchiveError("Invalid partition provenance.") }
        let keys: Set<String> = ["sourceHash", "trainingSamples", "validationSamples", "trainingRowIDs", "validationRowIDs"]
        guard !keys.isDisjoint(with: fields.keys) else { return nil }
        func hash(_ raw: JSONValue?) throws -> String {
            guard case .string(let value) = raw, WorkflowArchiveCodec.isHash(value) else { throw WorkflowArchiveError("Invalid recorded partition digest.") }
            return value
        }
        func rows(_ name: String, count: String) throws -> [String] {
            guard case .integer(let n) = fields[count], (1...100_000).contains(n),
                  case .array(let raw) = fields[name], raw.count == Int(n) else { throw WorkflowArchiveError("Recorded partition count does not match its observations.") }
            let ids = try raw.map { item -> String in
                guard case .string(let id) = item, !id.isEmpty, id.utf8.count <= 1024 else { throw WorkflowArchiveError("Invalid recorded observation identity.") }
                return id
            }
            guard Set(ids).count == ids.count else { throw WorkflowArchiveError("Duplicate recorded observation identity.") }
            return ids
        }
        let training = try rows("trainingRowIDs", count: "trainingSamples")
        let validation = try rows("validationRowIDs", count: "validationSamples")
        guard Set(training).isDisjoint(with: validation), case .object(let prepared) = fields["preparation"] else {
            throw WorkflowArchiveError("Recorded partitions overlap or lack preparation evidence.")
        }
        return try Self(sourceSHA256: hash(fields["sourceHash"]), trainingCount: training.count, validationCount: validation.count,
            trainingMembershipSHA256: WorkflowArchiveCodec.hash(WorkflowArchiveCodec.encode(training.sorted())),
            validationMembershipSHA256: WorkflowArchiveCodec.hash(WorkflowArchiveCodec.encode(validation.sorted())),
            trainingDataVersion: hash(prepared["trainingDataVersion"]), validationDataVersion: hash(prepared["validationDataVersion"]))
    }
}
