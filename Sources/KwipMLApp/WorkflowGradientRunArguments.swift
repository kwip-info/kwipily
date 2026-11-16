import Foundation
import KwipMLCore

/// Only selected run controls are transported; unrelated global settings stay omitted.
enum WorkflowGradientRunArguments {
    static func arguments(for request:WorkflowExecutionRequest)throws->[String] {
        var arguments:[String]=[]
        if let selected=request.plan.gradientPolicyRecipe {
            guard request.executionBudget != nil,let capture=request.derivativeCapture else {
                throw WorkflowPhaseError("Selected gradient policy requires the run’s captured execution budget and derivative permission.")
            }
            try capture.validate(gradientPolicy:selected.configuration)
            let recipe=try WorkflowPhaseArtifacts.encode(selected.configuration.recipeValue)
            guard recipe.count<=GradientPolicyConfiguration.maximumRecipeBytes else {
                throw WorkflowPhaseError("Gradient policy exceeds its launch limit.")
            }
            arguments += ["--gradient-policy-json",String(decoding:recipe,as:UTF8.self),
                          "--derivative-capture-json",String(decoding:try capture.wireData(),as:UTF8.self)]
        } else if request.derivativeCapture != nil {
            throw WorkflowPhaseError("Derivative permission cannot be transported without a selected gradient policy.")
        }
        if let selected=request.plan.trainabilityRecipe {
            let recipe=try WorkflowPhaseArtifacts.encode(selected.configuration.recipeValue)
            guard recipe.count<=TrainabilityConfiguration.maximumRecipeBytes else {
                throw WorkflowPhaseError("Trainability controls exceed their launch limit.")
            }
            arguments += ["--trainability-json",String(decoding:recipe,as:UTF8.self)]
        }
        arguments += try WorkflowDataLoadingRunArguments.arguments(dataset:request.plan.datasetRecipe,loader:request.plan.loaderRecipe)
        return arguments
    }
}
