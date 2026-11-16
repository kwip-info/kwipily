import Foundation

extension BuiltInCapabilityBridge {
    static func concatDeclaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor { try compositionDeclaration(definition) }
    static func addDeclaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor { try compositionDeclaration(definition) }
    static func stackDeclaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor { try compositionDeclaration(definition) }
    static func permuteDeclaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor { try compositionDeclaration(definition) }
    private static func compositionDeclaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor {
        guard let entry = BundledTensorCompositionComponents.entries[definition.id], entry.published, entry.definition == definition else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("Composition differs from its reviewed ordered-source contract.")
        }
        func port(_ id: String, _ direction: PortDirection) -> PortContract {
            PortContract(id,direction:direction,schema:.tensor(TensorContract([],dtype:.float32,leadingAxes:.init(minimum:2,maximum:8))),phases:[.forward],maximumConnections:direction == .input ? 1:nil)
        }
        return CapabilityDescriptor(id:definition.id,version:ContractVersion(1,2,0),packageID:packageID,backend:.pytorch,
            ports:definition.inputs.map {port($0,.input)} + definition.outputs.map {port($0,.output)},fields:definition.fields,
            lifecycle:LifecycleContract(phases:[.initialize,.forward,.export],state:[]),devices:[.cpu,.mps],availability:.existingPOCAdapter,
            adapterID:"kwip.runner.graph-v2.\(definition.id)",learning:LearningContract(conceptIDs:["linear-algebra.tensor-coordinates","linear-algebra.source-influence"],lessonIDs:entry.learning?.lessonIDs ?? [],exampleIDs:entry.learning.map {[$0.exampleID]} ?? []),shapeRuleID:"kwip.shape.\(definition.id).v1")
    }
}
