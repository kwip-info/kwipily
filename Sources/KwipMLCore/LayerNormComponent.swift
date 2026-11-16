import Foundation

/// The authored bundle owns the unchanged, shared inspector field contract.
public enum LayerNormComponent {
    public static let definition = BundledNormalizationComponents.definition("layer_norm")
}
extension BuiltInCapabilityBridge {
    static func layerNormDeclaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor {
        guard definition == LayerNormComponent.definition else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("Layer normalization differs from its reviewed trailing shape, epsilon, scale and bias contract.")
        }
        let ports = [PortContract("in", direction: .input, schema: .tensor(TensorContract([], leadingAxes: .init(minimum: 2, maximum: 8))), phases: [.forward], maximumConnections: 1),
                     PortContract("out", direction: .output, schema: .tensor(TensorContract([], leadingAxes: .init(minimum: 2, maximum: 8))), phases: [.forward])]
        return CapabilityDescriptor(id: definition.id, version: ContractVersion(), packageID: packageID, backend: .pytorch,
            ports: ports, fields: definition.fields,
            lifecycle: LifecycleContract(phases: [.initialize, .forward, .export], state: [StateDeclaration("parameters", kind: .parameters)]),
            devices: [.cpu, .mps], availability: .existingPOCAdapter, adapterID: "kwip.runner.graph-v2.layer_norm",
            learning: LearningContract(conceptIDs: ["linear-algebra.normalization-axes", "linear-algebra.population-variance", "neural-networks.learned-scale-and-bias"],
                lessonIDs: LayerNormLearningTask.allCases.map { "assessment." + $0.rawValue },
                exampleIDs: ["kwip.example.layer-norm-rows-classifier.v1", "kwip.example.layer-norm-sample-classifier.v1"]),
            shapeRuleID: "kwip.shape.layer_norm.v1")
    }
}
