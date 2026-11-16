import Foundation

/// A control recipe selects training adjustments; it is never a model-forward layer.
public enum OptimizerCapabilityBridge {
    static func sgdDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {try declaration(definition)}
    static func adamDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {try declaration(definition)}
    static func adamwDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {try declaration(definition)}
    static func rmspropDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {try declaration(definition)}
    static func adagradDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {try declaration(definition)}
    private static func declaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {
        guard OptimizerConfiguration.componentIDs.contains(definition.id),
              let entry=BundledOptimizerComponents.entries[definition.id],entry.published,entry.definition == definition else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("Optimizer recipe differs from its reviewed controls, parameter ownership or state policy.")
        }
        return CapabilityDescriptor(id:definition.id,version:ContractVersion(1,0,0),packageID:BuiltInCapabilityBridge.packageID,backend:.pytorch,
            ports:[PortContract("out",direction:.output,schema:.state(StatePortContract(key:"optimizerRecipe",kind:.optimizer)),phases:[.train])],fields:definition.fields,
            lifecycle:LifecycleContract(phases:[.initialize,.train,.checkpoint,.resume,.export,.reset],fitBoundary:.trainingPartitionOnly,state:[StateDeclaration("optimizerRecipe",kind:.optimizer)]),
            devices:[.cpu,.mps],availability:.existingPOCAdapter,adapterID:"kwip.runner.optimizer.\(definition.id)",
            learning:LearningContract(conceptIDs:["training.measured-adjustments","training.optimizer-memory"],lessonIDs:entry.learning?.lessonIDs ?? [],exampleIDs:entry.learning.map {[$0.exampleID]} ?? []),shapeRuleID:"kwip.recipe.\(definition.id).v1")
    }
}
