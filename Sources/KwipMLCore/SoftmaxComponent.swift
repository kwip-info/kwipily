import Foundation

public enum SoftmaxComponent {
    public static let definition = BundledActivationComponents.definition("softmax")
}
extension BuiltInCapabilityBridge {
    static func softmaxDeclaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor {
        guard definition == SoftmaxComponent.definition else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("Softmax differs from its reviewed integer normalization-axis and shape-preserving contract.")
        }
        let ports = [PortContract("in", direction: .input, schema: .tensor(TensorContract([], leadingAxes: .init(minimum: 2, maximum: 8))), phases: [.forward], maximumConnections: 1),
                     PortContract("out", direction: .output, schema: .tensor(TensorContract([], leadingAxes: .init(minimum: 2, maximum: 8))), phases: [.forward])]
        return CapabilityDescriptor(id: definition.id, version: ContractVersion(), packageID: packageID, backend: .pytorch,
            ports: ports, fields: definition.fields, lifecycle: LifecycleContract(phases: [.initialize, .forward, .export], state: []),
            devices: [.cpu, .mps], availability: .existingPOCAdapter, adapterID: "kwip.runner.graph-v2.softmax",
            learning: LearningContract(conceptIDs: ["linear-algebra.normalization-axes", "linear-algebra.softmax-jacobian"],
                lessonIDs: ["assessment.softmax-last-axis", "assessment.softmax-first-sample-axis"],
                exampleIDs: ["kwip.example.softmax-classifier.v1"]), shapeRuleID: "kwip.shape.softmax.v1")
    }
}
