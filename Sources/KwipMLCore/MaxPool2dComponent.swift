import Foundation

public enum MaxPool2dComponent {
    public static let definition = BundledPoolingComponents.definition("max_pool2d")
    public static var metadata: BundledPoolingComponents.Entry? { BundledPoolingComponents.metadata("max_pool2d") }
}

extension BuiltInCapabilityBridge {
    static func maxPool2dDeclaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor {
        let reviewed = MaxPool2dComponent.definition
        guard definition == reviewed else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("MaxPool2d differs from its reviewed window and output contract.")
        }
        func port(_ id: String, _ direction: PortDirection, _ dtype: ContractDType, _ length: String) -> PortContract {
            PortContract(id, direction: direction, schema: .tensor(TensorContract([.symbol("channels"), .symbol(length+"Height"), .symbol(length+"Width")], dtype: dtype,
                leadingAxes: .init(minimum: 1, maximum: 1))), phases: [.forward], maximumConnections: direction == .input ? 1 : nil)
        }
        return CapabilityDescriptor(id: definition.id, version: ContractVersion(), packageID: packageID, backend: .pytorch,
            ports: [port("in", .input, .float32, "input"), port("out", .output, .float32, "output"), port("indices", .output, .int64, "output")], fields: definition.fields,
            lifecycle: LifecycleContract(phases: [.initialize, .forward, .export], state: []),
            devices: [.cpu, .mps], availability: .existingPOCAdapter, adapterID: "kwip.runner.graph-v2.max_pool2d",
            learning: LearningContract(conceptIDs: ["linear-algebra.local-selection", "neural-networks.pooling"], lessonIDs: MaxPool2dComponent.metadata?.learning?.lessonIDs ?? [], exampleIDs: BundledPoolingComponents.entries["max_pool2d"]?.learning.map { [$0.exampleID] } ?? []),
            shapeRuleID: "kwip.shape.max_pool2d.v1")
    }
}
