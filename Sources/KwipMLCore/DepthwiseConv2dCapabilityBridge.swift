import Foundation

extension BuiltInCapabilityBridge {
    static func depthwiseConv2dDeclaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor {
        let reviewed = DepthwiseConv2dComponent.definition
        guard definition.inputs == reviewed.inputs, definition.outputs == reviewed.outputs, definition.fields == reviewed.fields else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("DepthwiseConv2d differs from its reviewed channels-first configuration and padding contract.")
        }
        func port(_ id: String, _ direction: PortDirection, _ channels: DimensionContract, _ height: String, _ width: String) -> PortContract {
            PortContract(id, direction: direction, schema: .tensor(TensorContract([channels, .symbol(height), .symbol(width)], dtype: .float32,
                leadingAxes: .init(minimum: 1, maximum: 1))), phases: [.forward], maximumConnections: direction == .input ? 1 : nil)
        }
        return CapabilityDescriptor(id: definition.id, version: ContractVersion(), packageID: packageID, backend: .pytorch,
            ports: [port("in", .input, .parameter("in_channels"), "inputHeight", "inputWidth"), port("out", .output, .symbol("derivedOutputChannels"), "outputHeight", "outputWidth")], fields: definition.fields,
            lifecycle: LifecycleContract(phases: [.initialize, .forward, .export], state: [StateDeclaration("parameters", kind: .parameters)]),
            devices: [.cpu, .mps], availability: .existingPOCAdapter, adapterID: "kwip.runner.graph-v2.depthwise_conv2d",
            learning: LearningContract(conceptIDs: ["linear-algebra.local-cross-correlation", "neural-networks.channelwise-filters"], lessonIDs: [], exampleIDs: ["prepared-depthwise-conv2d-classifier"]),
            shapeRuleID: "kwip.shape.depthwise_conv2d.v1")
    }
}
