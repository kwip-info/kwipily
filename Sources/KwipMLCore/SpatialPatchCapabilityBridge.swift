import Foundation

/// Patch extraction and addition have distinct ranks, paired geometry and no learned state.
public enum SpatialPatchCapabilityBridge {
    static func unfoldDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {try declaration(definition)}
    static func foldDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {try declaration(definition)}
    private static func declaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {
        guard SpatialPatchConfiguration.componentIDs.contains(definition.id),
              let entry=BundledSpatialPatchComponents.entries[definition.id],entry.published,entry.definition == definition else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("Spatial patches differ from their reviewed paired geometry, column order or overlap-addition contract.")
        }
        let image=[DimensionContract.symbol("channels"),.symbol("height"),.symbol("width")]
        let columns=[DimensionContract.symbol("packedPatchRows"),.symbol("windowColumns")]
        func port(_ id:String,_ direction:PortDirection,_ dimensions:[DimensionContract])->PortContract {
            PortContract(id,direction:direction,schema:.tensor(TensorContract(dimensions,dtype:.float32,leadingAxes:.init(minimum:1,maximum:1))),phases:[.forward],maximumConnections:direction == .input ? 1:nil)
        }
        let extracting=definition.id == "unfold"
        return CapabilityDescriptor(id:definition.id,version:ContractVersion(1,0,0),packageID:BuiltInCapabilityBridge.packageID,backend:.pytorch,
            ports:[port("in",.input,extracting ? image:columns),port("out",.output,extracting ? columns:image)],fields:definition.fields,
            lifecycle:LifecycleContract(phases:[.initialize,.forward,.backward,.export],state:[]),devices:[.cpu,.mps],availability:.existingPOCAdapter,
            adapterID:"kwip.runner.graph-v2.\(definition.id)",learning:LearningContract(conceptIDs:["linear-algebra.patch-coordinates","linear-algebra.overlapping-contributions"],lessonIDs:entry.learning?.lessonIDs ?? [],exampleIDs:entry.learning.map {[$0.exampleID]} ?? []),shapeRuleID:"kwip.shape.\(definition.id).v1")
    }
}
