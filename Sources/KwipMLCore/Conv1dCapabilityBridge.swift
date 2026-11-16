import Foundation

extension BuiltInCapabilityBridge {
    static func conv1dDeclaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor {
        let reviewed = Conv1dComponent.definition
        guard definition == reviewed else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("Conv1d differs from its reviewed channels-first configuration and padding contract.")
        }
        func port(_ id: String, _ direction: PortDirection, _ channels: String, _ length: String) -> PortContract {
            PortContract(id, direction: direction, schema: .tensor(TensorContract([.parameter(channels), .symbol(length)], dtype: .float32,
                leadingAxes: .init(minimum: 1, maximum: 1))), phases: [.forward], maximumConnections: direction == .input ? 1 : nil)
        }
        return CapabilityDescriptor(id: definition.id, version: ContractVersion(), packageID: packageID, backend: .pytorch,
            ports: [port("in", .input, "in_channels", "inputLength"), port("out", .output, "out_channels", "outputLength")], fields: definition.fields,
            lifecycle: LifecycleContract(phases: [.initialize, .forward, .export], state: [StateDeclaration("parameters", kind: .parameters)]),
            devices: [.cpu, .mps], availability: .existingPOCAdapter, adapterID: "kwip.runner.graph-v2.conv1d",
            learning: LearningContract(conceptIDs: ["linear-algebra.local-cross-correlation", "neural-networks.shared-kernels"], lessonIDs: ["assessment.conv1d-shared-kernel"], exampleIDs: ["prepared-conv1d-classifier"]),
            shapeRuleID: "kwip.shape.conv1d.v1")
    }
}
