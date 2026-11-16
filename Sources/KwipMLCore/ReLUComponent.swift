import Foundation

public enum ReLUComponent {
    public static let definition = BundledActivationComponents.definition("relu")
}
extension BuiltInCapabilityBridge {
    static func reluDeclaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor {
        guard definition == ReLUComponent.definition else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("ReLU differs from its reviewed parameterless, shape-preserving contract.")
        }
        let ports = [PortContract("in", direction: .input, schema: .tensor(TensorContract([], leadingAxes: .init(minimum: 2, maximum: 8))), phases: [.forward], maximumConnections: 1),
                     PortContract("out", direction: .output, schema: .tensor(TensorContract([], leadingAxes: .init(minimum: 2, maximum: 8))), phases: [.forward])]
        return CapabilityDescriptor(id: definition.id, version: ContractVersion(1,1,0), packageID: packageID, backend: .pytorch,
            ports: ports, fields: [], lifecycle: LifecycleContract(phases: [.initialize, .forward, .export], state: []),
            devices: [.cpu, .mps], availability: .existingPOCAdapter, adapterID: "kwip.runner.graph-v2.relu",
            learning: LearningContract(conceptIDs: ["neural-networks.elementwise-nonlinearity", "neural-networks.relu"],
                lessonIDs: ["kwip.course.parallel-classifier.v1", "assessment.activation-relu"], exampleIDs: ["kwip.example.parallel-classifier.v1", "prepared-relu-classifier"]),
            shapeRuleID: "kwip.shape.relu.v1")
    }
}
