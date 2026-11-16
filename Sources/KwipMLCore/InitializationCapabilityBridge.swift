import Foundation

/// Starting-value recipes are training controls, never model-forward tensor operators.
public enum InitializationCapabilityBridge {
    static func declaration(_ definition:ComponentDefinition)throws->CapabilityDescriptor {
        guard BundledInitializationComponents.metadata(definition.id)?.published == true else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("This starting-settings recipe has not completed its publication requirements.")
        }
        return try reviewedDescriptor(definition)
    }
    static func reviewedDescriptor(_ definition:ComponentDefinition)throws->CapabilityDescriptor {
        guard definition.id == "initialization",let entry=BundledInitializationComponents.metadata(definition.id),entry.definition == definition else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("Starting-settings recipe differs from its reviewed owner editor, ports or configuration contract.")
        }
        return CapabilityDescriptor(id:definition.id,version:ContractVersion(1,0,0),packageID:BuiltInCapabilityBridge.packageID,backend:.pytorch,
            ports:[PortContract("out",direction:.output,schema:.state(StatePortContract(key:"initializationRecipe",kind:.initialization)),phases:[.initialize,.train])],fields:definition.fields,
            lifecycle:LifecycleContract(phases:[.initialize,.train,.checkpoint,.export,.reset],fitBoundary:.notApplicable,state:[StateDeclaration("initializationRecipe",kind:.initialization)]),
            devices:[.cpu,.mps],availability:.existingPOCAdapter,adapterID:entry.adapterID,
            learning:LearningContract(conceptIDs:["training.starting-values"],lessonIDs:entry.learning?.lessonIDs ?? [],exampleIDs:entry.learning?.exampleIDs ?? []),shapeRuleID:"kwip.recipe.initialization.v1")
    }
}
