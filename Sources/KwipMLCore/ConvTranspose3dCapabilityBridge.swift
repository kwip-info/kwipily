import Foundation

extension BuiltInCapabilityBridge {
    static func convTranspose3dDeclaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor {
        let reviewed = ConvTranspose3dComponent.definition
        guard definition == reviewed else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("ConvTranspose3d differs from its reviewed channels-first configuration and padding contract.")
        }
        func port(_ id: String, _ direction: PortDirection, _ channels: String, _ depth: String, _ height: String, _ width: String) -> PortContract {
            PortContract(id, direction: direction, schema: .tensor(TensorContract([.parameter(channels), .symbol(depth), .symbol(height), .symbol(width)], dtype: .float32,
                leadingAxes: .init(minimum: 1, maximum: 1))), phases: [.forward], maximumConnections: direction == .input ? 1 : nil)
        }
        return CapabilityDescriptor(id: definition.id, version: ContractVersion(), packageID: packageID, backend: .pytorch,
            ports: [port("in", .input, "in_channels", "inputDepth", "inputHeight", "inputWidth"), port("out", .output, "out_channels", "outputDepth", "outputHeight", "outputWidth")], fields: definition.fields,
            lifecycle: LifecycleContract(phases: [.initialize, .forward, .export], state: [StateDeclaration("parameters", kind: .parameters)]),
            devices: [.cpu, .mps], availability: .existingPOCAdapter, adapterID: "kwip.runner.graph-v2.conv_transpose3d",
            learning: LearningContract(conceptIDs: ["linear-algebra.transposed-convolution", "neural-networks.shared-kernels"], lessonIDs: ["assessment.conv-transpose3d-shared-kernel"], exampleIDs: ["prepared-conv-transpose3d-classifier"]),
            shapeRuleID: "kwip.shape.conv_transpose3d.v1")
    }
}
