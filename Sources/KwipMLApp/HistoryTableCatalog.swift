import Foundation
import KwipMLCore

/// Presentation-only catalog. Receipt and ledger claims are never promoted to
/// verified artifacts, and constructing/filtering rows performs no archive IO.
enum HistoryTableCatalog {
    enum Kind: String, CaseIterable {
        case all = "All records", training = "Training", experiments = "Experiments", evaluation = "Evaluation", preparation = "Preparation", backend = "Backend", generation = "Generation"
    }
    enum Status: String, CaseIterable {
        case all = "All statuses", completed = "Completed", failed = "Failed", cancelled = "Cancelled"
        case incomplete = "Incomplete", unavailable = "Unavailable", recorded = "Recorded"
    }
    enum Sort: String, CaseIterable { case newest = "Newest first", oldest = "Oldest first" }
    enum Target: Hashable {
        case training(WorkflowHistoryReference)
        case evaluation(WorkflowEvaluationHistoryReference)
        case experiment(String)
        case backend(String)
        case preparation(String)
        case generation(String)
        case selfPlay(String)
        case gameSession(String)
        case fineTuning(String)
        case causalEvaluation(String)
    }
    struct Row: Identifiable, Equatable {
        let id: String
        let target: Target
        let title: String
        let recordedAt: Date?
        let timeLabel: String
        let kind: Kind
        let typeLabel: String
        let status: Status
        let statusLabel: String
        let evidenceLabel: String
        let message: String?
        /// Full recorded identity remains searchable even when a title is shortened.
        let searchIdentity: String
    }

    @MainActor static func rows(training: [WorkflowHistoryEntry] = [], evaluation: [WorkflowEvaluationHistoryEntry] = [], experiments: [WorkflowExperimentGroupRecord] = [], backends: [BackendFitHistoryStore.Entry] = [], preparations: [WorkflowDataPreparationHistory.Entry] = [], generations:[WorkflowGenerationRecord] = [], fineTuning:[ModelFineTuningRecord]=[], causalEvaluations:[ModelCausalEvaluationRecord]=[], games:[GameEpisodeRecord]=[], selfPlay:[SelfPlayRecord]=[]) -> [Row] {
        let trainingRows = training.map { entry -> Row in
            let identity = entry.runID ?? entry.reference.directoryID.uuidString
            let status: Status
            let evidence: String
            switch entry.verification {
            case .incomplete: status = .incomplete; evidence = "Incomplete receipt"
            case .unavailable: status = .unavailable; evidence = "Unavailable receipt"
            case .legacyV1: status = trainingStatus(entry.status); evidence = "Legacy receipt · not independently verifiable"
            case .recordedV2: status = trainingStatus(entry.status); evidence = "Recorded receipt · select to verify"
            }
            return row(id: "training:\(entry.reference.projectID):\(entry.reference.directoryID.uuidString)", target: .training(entry.reference), title: "Training run · \(identity.prefix(12))", date: parseDate(entry.finishedAt), kind: .training, type: "Training", status: status, evidence: evidence, message: entry.message, identity: identity)
        }
        let evaluationRows = evaluation.map { entry -> Row in
            let status: Status
            let evidence: String
            var label: String?
            switch entry.verification {
            case .incomplete: status = .incomplete; evidence = "Incomplete receipt"
            case .unavailable: status = .unavailable; evidence = "Unavailable receipt"
            case .inUse: status = .recorded; evidence = "Receipt in use · verification unavailable"; label = "In use"
            case .recordedUnverified:
                switch entry.status {
                case .completed: status = .completed
                case .failed: status = .failed
                case .cancelled: status = .cancelled
                case nil: status = .recorded
                }
                evidence = "Recorded receipt · select to verify"
            }
            let identity = entry.reference.evaluationID.uuidString
            return row(id: "evaluation:\(entry.reference.projectID):\(identity)", target: .evaluation(entry.reference), title: "\(entry.partitionRole.map {$0.capitalized + " "} ?? "")Evaluation \(identity.prefix(8))", date: entry.finishedAt, kind: .evaluation, type: "Evaluation", status: status, statusLabel: label, evidence: evidence, message: entry.message, identity: identity)
        }
        let experimentRows = experiments.map { entry -> Row in
            let date = entry.createdAtUnixSeconds.flatMap { value -> Date? in
                guard value.isFinite, value >= 0, value <= 253402300799 else { return nil }
                return Date(timeIntervalSince1970: value)
            }
            return row(id: "experiment:\(entry.projectID):\(entry.groupID)", target: .experiment(entry.groupID), title: "Experiment group · \(entry.groupID.prefix(12))", date: date, kind: .experiments, type: "Experiment group", status: experimentStatus(entry.variants.map(\.status)), evidence: "Recorded ledger · select to verify", message: "\(entry.variants.count) recorded variants", identity: entry.groupID)
        }
        let backendRows = backends.map { entry -> Row in
            let capturedTitle = entry.document.name.trimmingCharacters(in: .whitespacesAndNewlines)
            let type: String
            switch entry.backendID { case "sklearn": type = "scikit-learn"; case "tensorflow": type = "TensorFlow"; default: type = entry.backendID }
            return row(id: "backend:\(entry.document.projectID):\(entry.id)", target: .backend(entry.id), title: capturedTitle.isEmpty ? "Backend fit \(entry.id)" : capturedTitle, date: entry.completedAt, kind: .backend, type: type, status: .completed, evidence: "Verified at scan · recheck on access", message: nil, identity: entry.id)
        }
        let preparationRows = preparations.map { entry -> Row in
            let name = entry.document.name.trimmingCharacters(in: .whitespacesAndNewlines)
            return row(id: "preparation:\(entry.document.projectID):\(entry.id)", target: .preparation(entry.id),
                title: name.isEmpty ? "Preparation · \(entry.id.prefix(12))" : name, date: entry.completedAt,
                kind: .preparation, type: "Data preparation", status: .completed,
                statusLabel: entry.coverage?.isSampled == true ? "Sampled" : nil,
                evidence: entry.coverage?.isSampled == true ? "Verified sample coverage · recheck on access" : "Verified at scan · recheck on access",
                message: "\(entry.savedOutputs.count) recorded output files", identity: entry.id + " " + entry.requestIdentity)
        }
        let generationRows=generations.map {entry in row(id:"generation:"+entry.id,target:.generation(entry.id),title:"Generate text · "+String(entry.id.prefix(8)),date:entry.finishedAt,kind:.generation,type:"Generation",status:entry.status=="completed" ? .completed:entry.status=="cancelled" ? .cancelled:.failed,evidence:"Captured receipt · select to verify",message:nil,identity:entry.id)}
        let fineTuningRows=fineTuning.map {entry in row(id:"fineTuning:"+entry.id,target:.fineTuning(entry.id),title:"Fine-tuning · "+String(entry.id.prefix(8)),date:entry.finishedAt,kind:.training,type:"Fine-tuning",status:.completed,evidence:"Captured receipt · select to verify",message:nil,identity:entry.id)}
        let gameRows=games.map {entry in row(id:"gameSession:"+entry.id,target:.gameSession(entry.id),title:"Game session · "+String(entry.id.prefix(8)),date:entry.finishedAt,kind:.evaluation,type:"Frozen-policy episodes",status:.completed,evidence:"Captured receipt · select to verify",message:nil,identity:entry.id)}
        let selfPlayRows=selfPlay.map {entry in row(id:"selfPlay:"+entry.id,target:.selfPlay(entry.id),title:"Self-play · "+String(entry.id.prefix(8)),date:entry.finishedAt,kind:.training,type:"Policy/value learning",status:entry.completed ? .completed:.incomplete,evidence:"Captured receipt · select to verify",message:nil,identity:entry.id)}
        let causalRows=causalEvaluations.map{entry in row(id:"causalEvaluation:"+entry.id,target:.causalEvaluation(entry.id),title:"Causal evaluation · "+String(entry.id.prefix(8)),date:entry.finishedAt,kind:.evaluation,type:"Token NLL / perplexity",status:.completed,evidence:"Captured receipt · select to verify and account for exposure",message:nil,identity:entry.id)}
        return causalRows + selfPlayRows + gameRows + fineTuningRows + trainingRows + evaluationRows + experimentRows + backendRows + preparationRows + generationRows
    }

    static func filtered(_ rows: [Row], kind: Kind = .all, status: Status = .all, query: String = "", sort: Sort = .newest) -> [Row] {
        let needle = query.trimmingCharacters(in: .whitespacesAndNewlines)
        return rows.filter { row in
            guard kind == .all || row.kind == kind, status == .all || row.status == status else { return false }
            if needle.isEmpty { return true }
            let text = [row.title, row.id, row.searchIdentity, row.kind.rawValue, row.typeLabel, row.statusLabel, row.evidenceLabel].joined(separator: " ")
            return text.localizedCaseInsensitiveContains(needle)
        }.sorted { lhs, rhs in
            switch (lhs.recordedAt, rhs.recordedAt) {
            case let (left?, right?) where left != right: return sort == .newest ? left > right : left < right
            case (_?, nil): return true
            case (nil, _?): return false
            default: return lhs.id < rhs.id
            }
        }
    }

    private static func trainingStatus(_ phase: WorkflowExecutionPhase?) -> Status {
        switch phase {
        case .completed: return .completed
        case .failed: return .failed
        case .cancelled: return .cancelled
        default: return .recorded
        }
    }
    private static func experimentStatus(_ variants: [WorkflowExperimentVariantStatus]) -> Status {
        guard !variants.isEmpty else { return .incomplete }
        if variants.contains(.failed) { return .failed }
        if variants.contains(.interrupted) { return .incomplete }
        if variants.contains(.pending) || variants.contains(.running) { return .recorded }
        if variants.contains(.cancelled) { return .cancelled }
        return .completed
    }
    private static func parseDate(_ value: String?) -> Date? {
        guard let value else { return nil }
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = formatter.date(from: value) { return date }
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.date(from: value)
    }
    private static func row(id: String, target: Target, title: String, date: Date?, kind: Kind, type: String, status: Status, statusLabel: String? = nil, evidence: String, message: String?, identity: String) -> Row {
        let validDate = date.flatMap { $0.timeIntervalSince1970.isFinite ? $0 : nil }
        return Row(id: id, target: target, title: title, recordedAt: validDate, timeLabel: validDate?.formatted(date: .abbreviated, time: .shortened) ?? "Time not recorded", kind: kind, typeLabel: type, status: status, statusLabel: statusLabel ?? status.rawValue, evidenceLabel: evidence, message: message, searchIdentity: identity)
    }
}
