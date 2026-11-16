import Foundation

/// Training lifecycle belongs to the Train component, separate from preparation caching.
public enum WorkflowTrainingMode: String, Codable, CaseIterable {
    case fresh, exactResume, fineTune
    public var title: String {
        switch self {
        case .fresh: return "Start fresh"
        case .exactResume: return "Exact resume"
        case .fineTune: return "Fine-tune weights"
        }
    }
    public var epochExplanation: String {
        switch self {
        case .fresh: return "Epochs to run from newly initialized weights."
        case .exactResume: return "Target total epochs, including completed parent epochs; must exceed the parent's completed count."
        case .fineTune: return "Epochs in this new fine-tuning run, starting from epoch one."
        }
    }
    public var stateExplanation: String {
        switch self {
        case .fresh: return "Initialize weights, optimizer and random state from the configured seed."
        case .exactResume: return "Restore weights, optimizer, CPU random state, completed epochs and history. Requires matching architecture, runtime and data."
        case .fineTune: return "Reuse weights; reset optimizer, random state, epochs and history. Configure the new data and seed explicitly."
        }
    }
}
extension WorkflowControlContracts {
    public static func trainingMode(_ config: [String: JSONValue], schemaVersion: Int) throws -> WorkflowTrainingMode {
        guard [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23].contains(schemaVersion) else { throw DataFlowContractError("Unsupported workflow training-mode schema.") }
        guard let value = config["trainingMode"] else { return .fresh }
        guard schemaVersion >= 3 else { throw DataFlowContractError("Training mode requires explicit workflow schema 3 migration.") }
        guard case .string(let raw) = value, let mode = WorkflowTrainingMode(rawValue: raw) else {
            throw DataFlowContractError("Training mode must be fresh, exactResume or fineTune.")
        }
        return mode
    }
}
