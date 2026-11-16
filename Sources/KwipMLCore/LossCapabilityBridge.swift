import Foundation

/// An objective recipe describes how to score predictions; it never transforms a model tensor.
public enum LossCapabilityBridge {
    static func crossEntropyDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {try declaration(definition)}
    static func bceWithLogitsDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {try declaration(definition)}
    static func mseDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {try declaration(definition)}
    static func l1Declaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {try declaration(definition)}
    static func nllDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {try declaration(definition)}
    static func klDivDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {try declaration(definition)}
    static func ctcDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {try declaration(definition)}
    static func tripletMarginDeclaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {try declaration(definition)}
    private static func declaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {
        guard BundledLossComponents.entries[definition.id]?.published == true else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("This objective recipe has not completed its publication requirements.")
        }
        return try reviewedDescriptor(definition)
    }
    /// Internal metadata review only; this does not register or admit an unpublished capability.
    static func reviewedDescriptor(_ definition:ComponentDefinition)throws->CapabilityDescriptor {
        guard LossConfiguration.componentIDs.contains(definition.id),let entry=BundledLossComponents.entries[definition.id],entry.definition == definition else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("Objective recipe differs from its reviewed controls, target roles or fixed-weight policy.")
        }
        return CapabilityDescriptor(id:definition.id,version:ContractVersion(1,0,0),packageID:BuiltInCapabilityBridge.packageID,backend:.pytorch,
            ports:[PortContract("out",direction:.output,schema:.state(StatePortContract(key:"objectiveRecipe",kind:.objective)),phases:[.train,.evaluate])],fields:definition.fields,
            lifecycle:LifecycleContract(phases:[.initialize,.train,.evaluate,.checkpoint,.resume,.export,.reset],fitBoundary:.notApplicable,state:[StateDeclaration("objectiveRecipe",kind:.objective)]),
            devices:[.cpu,.mps],availability:.existingPOCAdapter,adapterID:"kwip.runner.objective.\(definition.id)",
            learning:LearningContract(conceptIDs:["training.prediction-scores","training.target-roles"],lessonIDs:entry.learning?.lessonIDs ?? [],exampleIDs:entry.learning.map {[$0.exampleID]} ?? []),shapeRuleID:"kwip.recipe.\(definition.id).v1")
    }
}
