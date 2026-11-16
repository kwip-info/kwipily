import Foundation

extension BuiltInCapabilityBridge {
    static func avgPool2dDeclaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor {
        try adaptiveFamilyDeclaration(definition)
    }
    static func adaptiveAvgPool1dDeclaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor {
        try adaptiveFamilyDeclaration(definition)
    }
    static func adaptiveAvgPool2dDeclaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor {
        try adaptiveFamilyDeclaration(definition)
    }
    static func adaptiveMaxPool2dDeclaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor {
        try adaptiveFamilyDeclaration(definition)
    }
    private static func adaptiveFamilyDeclaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor {
        guard let entry = BundledPoolingComponents.entries[definition.id], entry.published,
              entry.definition == definition,
              AdaptivePoolingConfiguration.componentIDs.contains(definition.id) || definition.id == "avg_pool2d" else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("Pooling component differs from its reviewed output-size and value contract.")
        }
        let axes = definition.id == "adaptive_avg_pool1d" ? ["length"] : ["height", "width"]
        func port(_ id: String, _ direction: PortDirection, _ dtype: ContractDType) -> PortContract {
            let prefix = direction == .input ? "input" : "output"
            return PortContract(id, direction: direction, schema: .tensor(TensorContract(
                [.symbol("channels")] + axes.map { .symbol(prefix + $0.capitalized) }, dtype: dtype,
                leadingAxes: .init(minimum: 1, maximum: 1))), phases: [.forward], maximumConnections: direction == .input ? 1 : nil)
        }
        var ports = [port("in", .input, .float32), port("out", .output, .float32)]
        if definition.outputs.contains("indices") { ports.append(port("indices", .output, .int64)) }
        return CapabilityDescriptor(id: definition.id, version: ContractVersion(), packageID: packageID, backend: .pytorch,
            ports: ports, fields: definition.fields,
            lifecycle: LifecycleContract(phases: [.initialize, .forward, .export], state: []),
            devices: [.cpu, .mps], availability: .existingPOCAdapter, adapterID: "kwip.runner.graph-v2.\(definition.id)",
            learning: LearningContract(conceptIDs: ["neural-networks.pooling", "linear-algebra.local-summary"],
                lessonIDs: entry.learning?.lessonIDs ?? [], exampleIDs: entry.learning.map { [$0.exampleID] } ?? []),
            shapeRuleID: "kwip.shape.\(definition.id).v1")
    }
}
