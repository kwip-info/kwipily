import Foundation

public enum AvgPool1dComponent {
    public static let definition = BundledPoolingComponents.definition("avg_pool1d")
    public static var metadata: BundledPoolingComponents.Entry? { BundledPoolingComponents.metadata("avg_pool1d") }
}

extension BuiltInCapabilityBridge {
    static func avgPool1dDeclaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor {
        let reviewed = AvgPool1dComponent.definition
        guard definition == reviewed else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("AvgPool1d differs from its reviewed window and output contract.")
        }
        func port(_ id: String, _ direction: PortDirection, _ dtype: ContractDType, _ length: String) -> PortContract {
            PortContract(id, direction: direction, schema: .tensor(TensorContract([.symbol("channels"), .symbol(length)], dtype: dtype,
                leadingAxes: .init(minimum: 1, maximum: 1))), phases: [.forward], maximumConnections: direction == .input ? 1 : nil)
        }
        return CapabilityDescriptor(id: definition.id, version: ContractVersion(), packageID: packageID, backend: .pytorch,
            ports: [port("in", .input, .float32, "inputLength"), port("out", .output, .float32, "outputLength")], fields: definition.fields,
            lifecycle: LifecycleContract(phases: [.initialize, .forward, .export], state: []),
            devices: [.cpu, .mps], availability: .existingPOCAdapter, adapterID: "kwip.runner.graph-v2.avg_pool1d",
            learning: LearningContract(conceptIDs: ["linear-algebra.local-averaging", "neural-networks.pooling"], lessonIDs: AvgPool1dComponent.metadata?.learning?.lessonIDs ?? [], exampleIDs: BundledPoolingComponents.entries["avg_pool1d"]?.learning.map { [$0.exampleID] } ?? []),
            shapeRuleID: "kwip.shape.avg_pool1d.v1")
    }
}
