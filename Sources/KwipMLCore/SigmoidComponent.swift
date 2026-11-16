import Foundation

public enum SigmoidComponent {
    public static let definition = BundledActivationComponents.definition("sigmoid")
}
extension BuiltInCapabilityBridge {
    static func sigmoidDeclaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor {
        guard definition == SigmoidComponent.definition else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("Sigmoid differs from its reviewed parameterless, shape-preserving contract.")
        }
        let ports = [PortContract("in", direction: .input, schema: .tensor(TensorContract([], leadingAxes: .init(minimum: 2, maximum: 8))), phases: [.forward], maximumConnections: 1),
                     PortContract("out", direction: .output, schema: .tensor(TensorContract([], leadingAxes: .init(minimum: 2, maximum: 8))), phases: [.forward])]
        return CapabilityDescriptor(id: definition.id, version: ContractVersion(), packageID: packageID, backend: .pytorch,
            ports: ports, fields: [], lifecycle: LifecycleContract(phases: [.initialize, .forward, .export], state: []),
            devices: [.cpu, .mps], availability: .existingPOCAdapter, adapterID: "kwip.runner.graph-v2.sigmoid",
            learning: LearningContract(conceptIDs: ["neural-networks.elementwise-nonlinearity", "neural-networks.sigmoid"],
                lessonIDs: ["assessment.activation-sigmoid"], exampleIDs: ["prepared-sigmoid-classifier"]),
            shapeRuleID: "kwip.shape.sigmoid.v1")
    }
}
