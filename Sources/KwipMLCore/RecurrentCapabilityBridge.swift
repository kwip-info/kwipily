import Foundation

/// Observation-first state tensors are explicit graph connections, never mutable hidden carry.
public enum RecurrentCapabilityBridge {
    static func rnnDeclaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor { try declaration(definition) }
    static func lstmDeclaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor { try declaration(definition) }
    static func gruDeclaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor { try declaration(definition) }
    static func rnnCellDeclaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor { try declaration(definition) }
    static func lstmCellDeclaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor { try declaration(definition) }
    static func gruCellDeclaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor { try declaration(definition) }

    private static func declaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor {
        guard RecurrentRuntimeCapabilities.operations.contains(definition.id),
              let entry = BundledRecurrentComponents.entries[definition.id], entry.published,
              entry.definition == definition else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("Recurrence differs from its reviewed sequence, explicit-state, parameter or backend contract.")
        }
        let cell = definition.id.hasSuffix("_cell")
        func port(_ id: String, _ direction: PortDirection) -> PortContract {
            let dimensions: [DimensionContract]
            if id == "in" {
                dimensions = cell ? [.parameter("input_size")] : [.symbol("time"), .parameter("input_size")]
            } else if cell {
                dimensions = [.parameter("hidden_size")]
            } else if id == "out" {
                dimensions = [.symbol("time"), .symbol("directionsTimesOutputWidth")]
            } else {
                dimensions = [.symbol("layersTimesDirections"), id == "cell" ? .parameter("hidden_size") : .symbol("hiddenOrProjectedWidth")]
            }
            // This descriptor is the declared ceiling. Configured ports separately enforce supplied state.
            return PortContract(id, direction: direction,
                schema: .tensor(TensorContract(dimensions, dtype: .float32, leadingAxes: .init(minimum: 1, maximum: 1))),
                phases: [.forward], required: direction == .output || id == "in", maximumConnections: direction == .input ? 1 : nil)
        }
        var state = [StateDeclaration("parameters", kind: .parameters)]
        if !cell { state.append(StateDeclaration("runRandomGenerator", kind: .randomGenerator, sharing: .explicitShared)) }
        return CapabilityDescriptor(id: definition.id, version: ContractVersion(1, 0, 0),
            packageID: BuiltInCapabilityBridge.packageID, backend: .pytorch,
            ports: definition.inputs.map { port($0, .input) } + definition.outputs.map { port($0, .output) },
            fields: definition.fields,
            lifecycle: LifecycleContract(phases: [.initialize, .forward, .backward, .train, .evaluate, .checkpoint, .resume, .export],
                fitBoundary: .trainingPartitionOnly, state: state, resume: .weightsOnly),
            devices: [.cpu, .mps], availability: .existingPOCAdapter,
            adapterID: "kwip.runner.graph-v2.\(definition.id)",
            learning: LearningContract(conceptIDs: ["sequences.explicit-state", "sequences.ordered-context"],
                lessonIDs: entry.learning?.lessonIDs ?? [], exampleIDs: entry.learning.map { [$0.exampleID] } ?? []),
            shapeRuleID: "kwip.shape.\(definition.id).v1")
    }
}
