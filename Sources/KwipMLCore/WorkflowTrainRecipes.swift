import Foundation

/// Optional typed Train-owned recipes. Uses the existing version-one worker
/// envelopes and validators; absence never writes or selects a default recipe.
public struct WorkflowTrainRecipes:Equatable {
    public static let roleKeys:Set<String>=["optimizer","scheduler","objective","initialization","gradientPolicy","trainability"]
    public let optimizer:OptimizerConfiguration?
    public let scheduler:SchedulerConfiguration?
    public let objective:LossConfiguration?
    public let initialization:InitializationConfiguration?
    public let gradientPolicy:GradientPolicyConfiguration?
    public let trainability:TrainabilityConfiguration?
    public init(configuration:[String:JSONValue],schemaVersion:Int)throws {
        guard (1...24).contains(schemaVersion),schemaVersion>=13 || Self.roleKeys.isDisjoint(with:configuration.keys) else {throw DataFlowContractError("Inline training recipes require workflow schema13 or later.")}
        optimizer=try configuration["optimizer"].map {try OptimizerConfiguration.decodeRecipe($0)}
        scheduler=try configuration["scheduler"].map {try SchedulerConfiguration.decodeRecipe($0)}
        objective=try configuration["objective"].map {try LossConfiguration.decodeRecipe($0)}
        initialization=try configuration["initialization"].map {try InitializationConfiguration.decodeRecipe($0)}
        gradientPolicy=try configuration["gradientPolicy"].map {try GradientPolicyConfiguration.decodeRecipe($0)}
        trainability=try configuration["trainability"].map {try TrainabilityConfiguration.decodeRecipe($0)}
    }
    public var envelopes:[String:JSONValue] {
        var result:[String:JSONValue]=[:]
        result["optimizer"]=optimizer?.recipeValue;result["scheduler"]=scheduler?.recipeValue
        result["objective"]=objective?.recipeValue;result["initialization"]=initialization?.recipeValue
        result["gradientPolicy"]=gradientPolicy?.recipeValue;result["trainability"]=trainability?.recipeValue
        return result
    }
    public var roles:Set<String> {Set(envelopes.keys)}
    public static func canonicalize(_ configuration:[String:JSONValue])throws->[String:JSONValue] {
        let recipes=try Self(configuration:configuration,schemaVersion:13)
        return configuration.merging(recipes.envelopes) {_,canonical in canonical}
    }
    /// Inline fields need schema13, while older composite models must retain
    /// their source2 execution semantics when the surrounding document advances.
    public static func migrateAuthoringSchema(_ document:WorkflowDocument)throws->WorkflowDocument {
        try WorkflowCodec.validate(document)
        var migrated=document
        if document.schemaVersion<11 {WorkflowModelInput.migrateOwnershipMetadata(&migrated)}
        else {migrated.schemaVersion=max(13,document.schemaVersion)}
        try WorkflowCodec.validate(migrated)
        return migrated
    }

    /// One atomic pure transaction for all eight former training-recipe nodes.
    public static func migrateLegacyRecipes(_ document:WorkflowDocument)throws->WorkflowDocument {
        try WorkflowTrainDataLoading.migrateLegacyRecipes(document)
    }

}
