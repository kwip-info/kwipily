import Foundation

extension BuiltInCapabilityBridge {
    static func lazyLinearDeclaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor {
        let reviewed = LazyLinearComponent.definition
        guard definition == reviewed else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("LazyLinear differs from its reviewed inferred-input, output-width and bias contract.")
        }
        func port(_ id: String, _ direction: PortDirection, _ width: DimensionContract) -> PortContract {
            PortContract(id, direction: direction,
                schema: .tensor(TensorContract([width], dtype: .float32, leadingAxes: .init(minimum: 1, maximum: 7))),
                phases: [.forward], maximumConnections: direction == .input ? 1 : nil)
        }
        return CapabilityDescriptor(id: definition.id, version: ContractVersion(), packageID: packageID, backend: .pytorch,
            ports: [port("in", .input, .symbol("inferredFeatures")), port("out", .output, .parameter("out_features"))],
            fields: definition.fields,
            lifecycle: LifecycleContract(phases: [.initialize, .forward, .export], state: [StateDeclaration("parameters", kind: .parameters)]),
            devices: [.cpu, .mps], availability: .existingPOCAdapter, adapterID: "kwip.runner.graph-v2.lazy-linear",
            learning: LearningContract(conceptIDs: ["linear-algebra.affine-map", "neural-networks.lazy-parameter-initialization"], lessonIDs: ["assessment.lazy-linear-inferred-width"], exampleIDs: ["prepared-lazy-linear-classifier"]),
            shapeRuleID: "kwip.shape.lazy-linear.v1")
    }
}
