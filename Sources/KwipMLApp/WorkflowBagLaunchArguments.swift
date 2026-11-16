import Foundation
import KwipMLCore

extension WorkflowPhaseArtifacts {
    /// Partition identity is shared; packet column roles remain independently typed.
    static func preparedArguments(_ selection: WorkflowPreparedSelection, bagInput: BagInputMapping,
                                  model: GraphDocument, registry: ComponentRegistry) throws -> [String] {
        try selection.validate()
        let objective: LossTargetMapping?
        let target: String?
        switch selection {
        case .legacy(let selected):
            guard selected.featureColumns == bagInput.indicesColumns else { throw WorkflowPhaseError("Prepared item columns differ from the captured bag mapping.") }
            objective = nil; target = selected.targetColumn
        case .objective(let selected):
            objective = selected.targetMapping; target = nil
        }
        let requirement = try WorkflowMappingRequirements.bagInputRequirement(model, registry: registry, context: objective == nil ? .preparedWorkflow : .explicitObjective)
        try bagInput.validate(sourceConfiguration: requirement.configuration, targetColumn: target, objectiveMapping: objective)
        let bytes = try encode(bagInput.value)
        guard bytes.count <= 32768 else { throw WorkflowPhaseError("Bag mapping exceeds its 32 KiB launch allowance.") }
        let ordinary = try preparedArguments(selection)
        var result: [String] = []
        for offset in stride(from: 0, to: ordinary.count, by: 2) where ordinary[offset] != "--feature-columns-json" {
            result += [ordinary[offset], ordinary[offset + 1]]
        }
        return result + ["--bag-input-mapping-json", String(decoding: bytes, as: UTF8.self)]
    }
}
