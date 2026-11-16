import Foundation

/// Exact authored-project upgrade. Captured History bytes are never migrated.
public enum RetiredTrainingRecipeMigration {
    public static let componentIDs = OptimizerConfiguration.componentIDs
        .union(SchedulerConfiguration.componentIDs).union(LossConfiguration.componentIDs)
        .union(InitializationConfiguration.componentIDs).union(GradientPolicyConfiguration.componentIDs)
        .union(TrainabilityConfiguration.componentIDs)
        .union([DatasetConfiguration.componentID, DataLoaderConfiguration.componentID])
    public static let edgeKinds: Set<WorkflowEdgeKind> = [.datasetRecipe, .batchRecipe, .optimizerRecipe,
        .schedulerRecipe, .objectiveRecipe, .initializationRecipe, .gradientPolicyRecipe, .trainabilityRecipe]

    public static func isRetired(_ node: WorkflowNode) -> Bool {
        switch node.payload {
        case .datasetRecipe, .batchRecipe, .optimizerRecipe, .schedulerRecipe, .lossRecipe,
             .initializationRecipe, .gradientPolicyRecipe, .trainabilityRecipe: return true
        case .modelPrimitive(let type,_,_): return componentIDs.contains(type)
        default: return false
        }
    }
    public static func requiresUpgrade(_ document: WorkflowDocument) -> Bool {
        document.nodes.contains(where: isRetired) || document.edges.contains { edgeKinds.contains($0.kind) }
    }
    public static func migrate(_ document: WorkflowDocument) throws -> WorkflowDocument {
        // The existing transaction validates every role and consumer before publishing
        // any changes. Independent initialization seeds and per-Train overrides retain
        // their exact typed envelope rather than guessing at a layer-wide equivalent.
        let result = try WorkflowTrainDataLoading.migrateLegacyRecipes(document)
        try requireCurrent(result)
        return result
    }
    public static func requireCurrent(_ document: WorkflowDocument) throws {
        guard !requiresUpgrade(document) else {
            throw DataFlowContractError("This project contains retired training recipe tools. Upgrade their settings into Train before running or exporting code. If the exact upgrade is unavailable, repair the highlighted original tools; their settings have been preserved.")
        }
    }
}
