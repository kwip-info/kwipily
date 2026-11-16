import Foundation

public enum DataLoadingCapabilityBridge {
    static func declaration(_ definition: ComponentDefinition) throws -> CapabilityDescriptor {
        guard BundledDataLoadingComponents.entries[definition.id]?.published == true else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("This reader has not completed its publication requirements.")
        }
        return try reviewedDescriptor(definition)
    }
    static func reviewedDescriptor(_ definition: ComponentDefinition) throws -> CapabilityDescriptor {
        guard let entry = BundledDataLoadingComponents.entries[definition.id], entry.definition == definition else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry("Reader differs from its reviewed fields or control ports.")
        }
        let key = definition.id == "dataset_source" ? "datasetRecipe" : "loaderRecipe"
        return CapabilityDescriptor(
            id: definition.id, version: ContractVersion(1, 0, 0), packageID: BuiltInCapabilityBridge.packageID, backend: .pytorch,
            ports: [PortContract("out", direction: .output, schema: .state(StatePortContract(key: key, kind: .dataCursor)), phases: [.initialize, .train])],
            fields: definition.fields,
            lifecycle: LifecycleContract(phases: [.initialize, .train, .checkpoint, .resume, .export, .reset], fitBoundary: .trainingPartitionOnly, state: [StateDeclaration(key, kind: .dataCursor)], resume: .none),
            devices: [.cpu, .mps], availability: entry.published ? .existingPOCAdapter : .contractOnly, adapterID: entry.adapterID,
            learning: LearningContract(conceptIDs: ["training.data-loading"], lessonIDs: entry.learning?.lessonIDs ?? [], exampleIDs: entry.learning?.exampleIDs ?? []),
            shapeRuleID: "kwip.recipe.\(definition.id).v1")
    }
}
