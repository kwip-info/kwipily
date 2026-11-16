import Foundation

/// Control recipes describe training policy; they never act as tensor-forward operators.
public enum GradientControlCapabilityBridge {
    static func declaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor {
        guard BundledGradientControlComponents.metadata(definition.id)?.published == true else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("This training policy has not completed its publication requirements.")
        }
        return try reviewedDescriptor(definition)
    }

    static func reviewedDescriptor(_ definition: ComponentDefinition) throws -> CapabilityDescriptor {
        guard let entry = BundledGradientControlComponents.metadata(definition.id), entry.definition == definition else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("Training policy differs from its reviewed fields, editor or control ports.")
        }
        let gradient = definition.id == "gradient_policy"
        let key = gradient ? "gradientPolicyRecipe" : "trainabilityRecipe"
        let kind: StateKind = gradient ? .gradientPolicy : .trainability
        return CapabilityDescriptor(
            id: definition.id, version: ContractVersion(1, 0, 0), packageID: BuiltInCapabilityBridge.packageID, backend: .pytorch,
            ports: [PortContract("out", direction: .output, schema: .state(StatePortContract(key: key, kind: kind)), phases: [.initialize, .train, .evaluate])],
            fields: definition.fields,
            lifecycle: LifecycleContract(phases: [.initialize, .train, .evaluate, .checkpoint, .resume, .export, .reset], fitBoundary: .notApplicable, state: [StateDeclaration(key, kind: kind)], resume: .none),
            devices: [.cpu, .mps], availability: entry.published ? .existingPOCAdapter : .contractOnly, adapterID: entry.adapterID,
            learning: LearningContract(conceptIDs: [gradient ? "training.requested-gradients" : "training.trainability"], lessonIDs: entry.learning?.lessonIDs ?? [], exampleIDs: entry.learning?.exampleIDs ?? []),
            shapeRuleID: gradient ? "kwip.recipe.gradient-policy.v1" : "kwip.recipe.trainability.v1")
    }
}
