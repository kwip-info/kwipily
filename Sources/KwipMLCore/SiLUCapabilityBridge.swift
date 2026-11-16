import Foundation

extension BuiltInCapabilityBridge {
    static func siluDeclaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor {
        guard definition == SiLUComponent.definition else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("SiLU differs from its reviewed shape-preserving configuration.")
        }
        let ports = [PortContract("in", direction: .input, schema: .tensor(TensorContract([], dtype: .float32, leadingAxes: .init(minimum: 2, maximum: 8))), phases: [.forward], maximumConnections: 1),
                     PortContract("out", direction: .output, schema: .tensor(TensorContract([], dtype: .float32, leadingAxes: .init(minimum: 2, maximum: 8))), phases: [.forward])]
        return CapabilityDescriptor(id: definition.id, version: ContractVersion(), packageID: packageID, backend: .pytorch,
            ports: ports, fields: definition.fields, lifecycle: LifecycleContract(phases: [.initialize, .forward, .export], state: []),
            devices: [.cpu, .mps], availability: .existingPOCAdapter, adapterID: "kwip.runner.graph-v2.silu",
            learning: LearningContract(conceptIDs: ["neural-networks.elementwise-nonlinearity"], lessonIDs: SiLUComponent.metadata?.learning?.lessonIDs ?? [],
                exampleIDs: SiLUComponent.metadata?.learning.map { [$0.exampleID] } ?? []), shapeRuleID: "kwip.shape.silu.v1")
    }
}
