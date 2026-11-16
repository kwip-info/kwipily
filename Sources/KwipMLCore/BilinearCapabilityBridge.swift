import Foundation

extension BuiltInCapabilityBridge {
    static func bilinearDeclaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor {
        let reviewed = BilinearComponent.definition
        guard definition == reviewed else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("Bilinear differs from its reviewed two-input width and bias contract.")
        }
        func port(_ id: String, _ direction: PortDirection, _ width: String) -> PortContract {
            PortContract(id, direction: direction,
                schema: .tensor(TensorContract([.parameter(width)], dtype: .float32, leadingAxes: .init(minimum: 1, maximum: 7))),
                phases: [.forward], maximumConnections: direction == .input ? 1 : nil)
        }
        return CapabilityDescriptor(id: definition.id, version: ContractVersion(), packageID: packageID, backend: .pytorch,
            ports: [port("a", .input, "in1_features"), port("b", .input, "in2_features"), port("out", .output, "out_features")],
            fields: definition.fields,
            lifecycle: LifecycleContract(phases: [.initialize, .forward, .export], state: [StateDeclaration("parameters", kind: .parameters)]),
            devices: [.cpu, .mps], availability: .existingPOCAdapter, adapterID: "kwip.runner.graph-v2.bilinear",
            learning: LearningContract(conceptIDs: ["linear-algebra.bilinear-contraction", "linear-algebra.parameter-tensor"], lessonIDs: BilinearLearningTask.allCases.map { "assessment." + $0.rawValue }, exampleIDs: ["prepared-bilinear-classifier"]),
            shapeRuleID: "kwip.shape.bilinear.v1")
    }
}
