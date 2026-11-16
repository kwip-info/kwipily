import Foundation
import KwipMLCore

/// Objective arithmetic and epoch identity do not depend on the model-input domain.
struct WorkflowObjectiveEvaluationContext {
    let recipe: LossRecipe
    let runID: String
    let outputSampleShape: [Int]
    let trainingCount: Int
    let evaluationCount: Int
    let expectedEpochs: Int
    let scalarization: WorkflowObjectiveScalarization
    let device: ContractDevice
    let policy: TrainingSampleCapturePolicy
    let explicitSampleCapture: Bool
    init(recipe: LossRecipe, runID: String, outputSampleShape: [Int], trainingCount: Int,
         evaluationCount: Int, expectedEpochs: Int, scalarization: WorkflowObjectiveScalarization,
         device: ContractDevice, policy: TrainingSampleCapturePolicy, explicitSampleCapture: Bool = false) throws {
        try recipe.configuration.validate(targetMapping: recipe.targetMapping)
        guard !runID.isEmpty, (1...100_000_000).contains(trainingCount),
              (1...100_000_000).contains(evaluationCount), (1...10_000).contains(expectedEpochs),
              [ContractDevice.cpu,.mps].contains(device), (1...7).contains(outputSampleShape.count) else {
            throw WorkflowPhaseError("Objective result context has invalid source, partition, device or geometry.")
        }
        var elements=1
        for dimension in outputSampleShape {
            guard (1...4096).contains(dimension), elements<=100_000_000/dimension else {
                throw WorkflowPhaseError("Objective produced shape exceeds its element budget.")
            }
            elements *= dimension
        }
        try WorkflowMappingRequirements.validateObjective(prediction: TypedTensorContract(sample: outputSampleShape, dtype: .float32), recipe: recipe)
        self.recipe=recipe;self.runID=runID;self.outputSampleShape=outputSampleShape
        self.trainingCount=trainingCount;self.evaluationCount=evaluationCount;self.expectedEpochs=expectedEpochs
        self.scalarization=scalarization;self.device=device;self.policy=policy;self.explicitSampleCapture=explicitSampleCapture
    }
}
