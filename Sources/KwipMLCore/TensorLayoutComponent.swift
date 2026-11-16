import Foundation

/// Production and fixture-bound lesson editors share authored fields and defaults.
public enum TensorLayoutComponent {
    public static func definition(_ id: String) -> ComponentDefinition {
        BundledTensorLayoutComponents.definition(id)
    }
}
extension BuiltInCapabilityBridge {
    static func reshapeDeclaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor {
        try tensorLayoutDeclaration(definition, operation: "reshape")
    }
    static func transposeDeclaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor {
        try tensorLayoutDeclaration(definition, operation: "transpose")
    }
    static func flattenDeclaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor {
        try tensorLayoutDeclaration(definition, operation: "flatten")
    }
    private static func tensorLayoutDeclaration(_ definition: ComponentDefinition, operation: String) throws -> CapabilityDescriptor {
        guard definition == TensorLayoutComponent.definition(operation) else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("Tensor layout differs from its reviewed sample dimensions and axis contract.")
        }
        let ports = [PortContract("in", direction: .input, schema: .tensor(TensorContract([], leadingAxes: .init(minimum: 2, maximum: 8))), phases: [.forward], maximumConnections: 1),
                     PortContract("out", direction: .output, schema: .tensor(TensorContract([], leadingAxes: .init(minimum: 2, maximum: 8))), phases: [.forward])]
        return CapabilityDescriptor(id: operation, version: ContractVersion(), packageID: packageID, backend: .pytorch,
            ports: ports, fields: definition.fields, lifecycle: LifecycleContract(phases: [.initialize, .forward, .export], state: []),
            devices: [.cpu, .mps], availability: .existingPOCAdapter, adapterID: "kwip.runner.graph-v2." + operation,
            learning: LearningContract(conceptIDs: ["linear-algebra.tensor-coordinates", "linear-algebra.permutation-gradients"],
                lessonIDs: ["assessment.tensor-" + operation], exampleIDs: ["kwip.example.tensor-layout-classifier.v1"]),
            shapeRuleID: "kwip.shape." + operation + ".v1")
    }
}
