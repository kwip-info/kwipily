import Foundation
import KwipMLCore

/// Only independently verified preparation counts supply coordinate geometry.
struct WorkflowBagDerivativeContext {
    let requirement: BagWorkflowInputRequirement
    let preparation: VerifiedBagPreparation
    init(requirement: BagWorkflowInputRequirement, preparation: VerifiedBagPreparation) throws {
        try preparation.selection.validate()
        let config = requirement.configuration
        for (role, partition) in [("training", preparation.training), ("validation", preparation.validation)] {
            guard (1...4096).contains(partition.count), partition.rowIDs.count == partition.count,
                  Set(partition.rowIDs).count == partition.count,
                  partition.itemCounts.count == partition.count,
                  partition.itemCounts.allSatisfy({ (0...config.maximumItemsPerBag).contains($0) && (config.layout != "fixed" || $0 == config.maximumItemsPerBag) }),
                  partition.prefix.count == min(200, partition.count),
                  partition.prefix.enumerated().allSatisfy({ $0.element.input.indices.count == partition.itemCounts[$0.offset] }),
                  partition.sourceIdentity["sourceConfiguration"] == .object(config.values),
                  partition.sourceIdentity["role"] == .string(role) else {
                throw WorkflowPhaseError("Bag derivative counts differ from the actual verified source observations.")
            }
        }
        guard Set(preparation.training.rowIDs).isDisjoint(with: preparation.validation.rowIDs) else {
            throw WorkflowPhaseError("Bag derivative partitions overlap.")
        }
        self.requirement = requirement; self.preparation = preparation
    }
    var modelInput: JSONValue {
        let config = requirement.configuration
        let ports = Dictionary(uniqueKeysWithValues: config.outputPorts.map { role in
            (role, JSONValue.object(["role": .string(role), "dtype": .string(role == "per_sample_weights" ? "float32" : "int64")]))
        })
        return .object(["schemaVersion": 1, "kind": .string("bagPacket"), "sourceID": .string(requirement.sourceNodeID),
                        "configuration": .object(config.values), "ports": .object(ports)])
    }
    func selectedCounts(_ request: JSONValue) throws -> [Int] {
        guard case .object(let fields) = request, let role = fields["partition"]?.stringValue,
              ["training", "validation"].contains(role), let start = fields["row_start"]?.integerValue,
              let count = fields["row_count"]?.integerValue, start >= 0, (1...32).contains(count) else {
            throw WorkflowPhaseError("Choose a bounded whole-bag derivative row range.")
        }
        let partition = role == "training" ? preparation.training : preparation.validation
        guard start <= partition.count, count <= partition.count - start else {
            throw WorkflowPhaseError("Bag derivative rows exceed the verified partition.")
        }
        return Array(partition.itemCounts[start..<start + count])
    }
    func coordinateShape(node: String, port: String, counts: [Int]) throws -> [Int] {
        let config = requirement.configuration
        guard node == requirement.sourceNodeID, config.outputPorts.contains(port), (1...32).contains(counts.count),
              counts.allSatisfy({ (0...config.maximumItemsPerBag).contains($0) && (config.layout != "fixed" || $0 == config.maximumItemsPerBag) }) else {
            throw WorkflowPhaseError("Derivative target is not an actual enabled bag source coordinate.")
        }
        if port == "offsets" { return [counts.count + (config.includeLastOffset ? 1 : 0)] }
        return config.layout == "fixed" ? [counts.count, config.maximumItemsPerBag] : [counts.reduce(0, +)]
    }
}
