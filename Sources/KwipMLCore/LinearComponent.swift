import Foundation

/// Bundle-owned definition and presentation metadata; instance parameters remain
/// owned by the existing graph/training and explicit lesson submission contracts.
public enum LinearComponent {
    public static let definition = BundledLayerComponents.definition("linear")
    public static var metadata: BundledLayerComponents.Entry? { BundledLayerComponents.entries["linear"] }
}
extension BuiltInCapabilityBridge {
    static func linearDeclaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor {
        guard definition == LinearComponent.definition else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("Linear differs from its reviewed input/output feature widths and optional bias contract.")
        }
        let ports = [
            PortContract("in", direction: .input, schema: .tensor(TensorContract([.parameter("in_features")], leadingAxes: .init(minimum: 1, maximum: 7))), phases: [.forward], maximumConnections: 1),
            PortContract("out", direction: .output, schema: .tensor(TensorContract([.parameter("out_features")], leadingAxes: .init(minimum: 1, maximum: 7))), phases: [.forward])
        ]
        return CapabilityDescriptor(id: definition.id, version: ContractVersion(1,2,0), packageID: packageID, backend: .pytorch,
            ports: ports, fields: definition.fields,
            lifecycle: LifecycleContract(phases: [.initialize, .forward, .export], state: [StateDeclaration("parameters", kind: .parameters)]),
            devices: [.cpu, .mps], availability: .existingPOCAdapter, adapterID: "kwip.runner.graph-v2.linear",
            learning: LearningContract(conceptIDs: ["linear-algebra.affine-map", "linear-algebra.matrix-dimensions"],
                lessonIDs: ["kwip.course.parallel-classifier.v1", ModelConstructionLessonIdentity.composition.assessmentID,
                    ModelConstructionLessonIdentity.recovery(taskSeed: 0).assessmentID, ModelConstructionLessonIdentity.backwardChain.assessmentID,
                    ModelConstructionLessonIdentity.frozenStandardization.assessmentID],
                exampleIDs: ["kwip.example.parallel-classifier.v1"]), shapeRuleID: "kwip.shape.linear.v1")
    }
}
