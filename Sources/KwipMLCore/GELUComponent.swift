import Foundation

public enum GELUComponent {
    public static let definition = BundledActivationComponents.definition("gelu")
}
extension BuiltInCapabilityBridge {
    static func geluDeclaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor {
        guard definition == GELUComponent.definition else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("GELU differs from its reviewed exact/tanh approximation Boolean and shape-preserving contract.")
        }
        let ports = [PortContract("in", direction: .input, schema: .tensor(TensorContract([], leadingAxes: .init(minimum: 2, maximum: 8))), phases: [.forward], maximumConnections: 1),
                     PortContract("out", direction: .output, schema: .tensor(TensorContract([], leadingAxes: .init(minimum: 2, maximum: 8))), phases: [.forward])]
        return CapabilityDescriptor(id: definition.id, version: ContractVersion(1,1,0), packageID: packageID, backend: .pytorch,
            ports: ports, fields: definition.fields, lifecycle: LifecycleContract(phases: [.initialize, .forward, .export], state: []),
            devices: [.cpu, .mps], availability: .existingPOCAdapter, adapterID: "kwip.runner.graph-v2.gelu",
            learning: LearningContract(conceptIDs: ["neural-networks.elementwise-nonlinearity", "neural-networks.gelu"],
                lessonIDs: [ModelConstructionLessonIdentity.geluExact.assessmentID, ModelConstructionLessonIdentity.geluTanh.assessmentID],
                exampleIDs: ["kwip.example.gelu-exact-classifier.v1", "kwip.example.gelu-tanh-classifier.v1"]),
            shapeRuleID: "kwip.shape.gelu.v1")
    }
}
