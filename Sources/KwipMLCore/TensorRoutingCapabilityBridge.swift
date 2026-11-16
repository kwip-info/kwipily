import Foundation

/// Declarations retain ordered port ceilings; configured tensor contracts resolve availability.
public enum TensorRoutingCapabilityBridge {
    static func splitDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {try declaration(definition)}
    static func chunkDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {try declaration(definition)}
    static func gatherDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {try declaration(definition)}
    static func einsumDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {try declaration(definition)}
    private static func declaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {
        guard TensorRoutingConfiguration.componentIDs.contains(definition.id),
              let entry=BundledTensorRoutingComponents.entries[definition.id],entry.published,entry.definition == definition else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("Routing differs from its reviewed ordered-part, selector or named-direction contract.")
        }
        func port(_ id:String,_ direction:PortDirection)->PortContract {
            let dtype:ContractDType=definition.id == "gather" && id == "indices" ? .int64:.float32
            return PortContract(id,direction:direction,schema:.tensor(TensorContract([],dtype:dtype,leadingAxes:.init(minimum:2,maximum:8))),phases:[.forward],maximumConnections:direction == .input ? 1:nil)
        }
        return CapabilityDescriptor(id:definition.id,version:ContractVersion(1,0,0),packageID:BuiltInCapabilityBridge.packageID,backend:.pytorch,
            ports:definition.inputs.map {port($0,.input)}+definition.outputs.map {port($0,.output)},fields:definition.fields,
            lifecycle:LifecycleContract(phases:[.initialize,.forward,.backward,.export],state:[]),devices:[.cpu,.mps],availability:.existingPOCAdapter,
            adapterID:"kwip.runner.graph-v2.\(definition.id)",learning:LearningContract(conceptIDs:["linear-algebra.tensor-coordinates","linear-algebra.source-influence"],lessonIDs:entry.learning?.lessonIDs ?? [],exampleIDs:entry.learning.map {[$0.exampleID]} ?? []),shapeRuleID:"kwip.shape.\(definition.id).v1")
    }
}
