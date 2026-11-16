import Foundation

/// A control recipe selects training adjustments; it is never a model-forward layer.
public enum SchedulerCapabilityBridge {
    static func stepLRDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {try declaration(definition)}
    static func cosineAnnealingLRDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {try declaration(definition)}
    static func reduceOnPlateauDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {try declaration(definition)}
    static func oneCycleLRDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {try declaration(definition)}
    static func linearLRDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {try declaration(definition)}
    private static func declaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {
        guard SchedulerConfiguration.componentIDs.contains(definition.id),
              let entry=BundledSchedulerComponents.entries[definition.id],entry.published,entry.definition == definition else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("Scheduler recipe differs from its reviewed controls, parameter ownership or state policy.")
        }
        return CapabilityDescriptor(id:definition.id,version:ContractVersion(1,0,0),packageID:BuiltInCapabilityBridge.packageID,backend:.pytorch,
            ports:[PortContract("out",direction:.output,schema:.state(StatePortContract(key:"schedulerRecipe",kind:.scheduler)),phases:[.train])],fields:definition.fields,
            lifecycle:LifecycleContract(phases:[.initialize,.train,.checkpoint,.resume,.export,.reset],fitBoundary:.trainingPartitionOnly,state:[StateDeclaration("schedulerRecipe",kind:.scheduler)]),
            devices:[.cpu,.mps],availability:.existingPOCAdapter,adapterID:"kwip.runner.scheduler.\(definition.id)",
            learning:LearningContract(conceptIDs:["training.rate-timing","training.scheduler-memory"],lessonIDs:entry.learning?.lessonIDs ?? [],exampleIDs:entry.learning.map {[$0.exampleID]} ?? []),shapeRuleID:"kwip.recipe.\(definition.id).v1")
    }
}
