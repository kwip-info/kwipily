import Foundation

/// Learned channel adjustments and remembered statistics have separate ownership.
public enum StatefulNormalizationCapabilityBridge {
    static func batchNorm1dDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {try declaration(definition)}
    static func batchNorm2dDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {try declaration(definition)}
    static func instanceNorm1dDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {try declaration(definition)}
    static func instanceNorm2dDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {try declaration(definition)}
    static func groupNormDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {try declaration(definition)}
    private static func declaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {
        guard StatefulNormalizationRuntimeCapabilities.operations.contains(definition.id),
              let entry=BundledNormalizationComponents.entries[definition.id],entry.published,entry.definition == definition else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("Normalization differs from its reviewed channels, grouping, scale, shift or saved-statistics controls.")
        }
        let ranks:[String:(Int,Int)]=["batch_norm1d":(2,3),"batch_norm2d":(4,4),"instance_norm1d":(3,3),"instance_norm2d":(4,4),"group_norm":(2,8)]
        let range=ranks[definition.id]!
        func port(_ id:String,_ direction:PortDirection)->PortContract {
            PortContract(id,direction:direction,schema:.tensor(TensorContract([],dtype:.float32,leadingAxes:.init(minimum:range.0,maximum:range.1))),phases:[.forward],maximumConnections:direction == .input ? 1:nil)
        }
        // These are potential configured state groups. Runtime state contracts decide
        // the exact enabled keys; neither absent parameters nor buffers are invented.
        var state=[StateDeclaration("parameters",kind:.parameters)]
        if definition.id != "group_norm" {state.append(StateDeclaration("runningStatistics",kind:.buffers))}
        return CapabilityDescriptor(id:definition.id,version:ContractVersion(1,0,0),packageID:BuiltInCapabilityBridge.packageID,backend:.pytorch,
            ports:[port("in",.input),port("out",.output)],fields:definition.fields,
            lifecycle:LifecycleContract(phases:[.initialize,.forward,.backward,.export],state:state),devices:[.cpu,.mps],availability:.existingPOCAdapter,
            adapterID:"kwip.runner.graph-v2.\(definition.id)",learning:LearningContract(conceptIDs:["linear-algebra.normalization-groups","neural-networks.channel-scale-and-shift","neural-networks.current-and-remembered-statistics"],lessonIDs:StatefulNormalizationLearningTask.allCases.filter {$0.componentID == definition.id}.map {"assessment."+$0.rawValue},exampleIDs:entry.learning.map {[$0.exampleID]} ?? []),shapeRuleID:"kwip.shape.\(definition.id).v1")
    }
}
