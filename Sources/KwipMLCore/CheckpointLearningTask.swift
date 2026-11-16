import Foundation

public enum CheckpointLearningTask:String,CaseIterable,Codable {
    case restoreWeights="checkpoint-restore-weights",continueTraining="checkpoint-continue-training"
    public var componentID:String {"checkpoint_lifecycle"}
    public var title:String {self == .restoreWeights ? "Keep and reuse learned settings":"Continue from where training stopped"}
    public var configuration:CheckpointLearningConfiguration {try! .init(trainingMode:self == .restoreWeights ? .fineTune:.exactResume,parent:.savedLessonRun,epochs:self == .restoreWeights ? 1:4,batchSize:4)}
    public var controlInstructions:String {"Choose \(self == .restoreWeights ? "Fine-tune weights":"Exact resume") and Saved lesson run. Set \(self == .restoreWeights ? "Epochs to 1":"Target total epochs to 4") and Batch size to 4."}
    public var explanation:String {"Weights are the settings a model learns. A checkpoint saves those settings and training memory. An epoch is one pass through the training rows; a batch is the group of rows handled together. Optimizer memory carries information between adjustments. A seed controls the starting random choices. "+(self == .restoreWeights ? "Fine-tuning reuses learned settings and starts new training memory. The checker saves two completed passes, reopens the checkpoint, then measures your new pass on supplied rows.":"Exact resume continues the same job. The checker saves after two passes and compares the resumed job with four uninterrupted passes, including learned settings, training memory and random state. Exact continuation here uses CPU.")}
    public var measuredShapes:[String:[Int]] {
        let result:[String:[Int]]=self == .restoreWeights ? ["restoredParameterDifference":[2,3],"restoredPredictionDifference":[4,2],"childUpdateDifference":[2,3],"freshMemoryDifference":[2,3],"parentByteDifference":[1],"completedEpochs":[2],"savedParameters":[2,3],"restoredPredictions":[4,2],"updatedParameters":[2,3]]:["continuationStateDifference":[1],"continuationHistoryDifference":[1],"continuationRandomDifference":[1],"parentByteDifference":[1],"continuationEpochs":[3],"completedEpochs":[3],"savedParameters":[2,3],"resumedParameters":[2,3]]
        return result
    }
    public var instructions:String {explanation+" Configure: "+controlInstructions+" Check: choose Check construction. Inspect the saved settings and measured comparison tables. The checker supplies the architecture, rows and saved parent; you do not enter code or file paths."}
}
public enum CheckpointAssessment {
    public static func plan(task:CheckpointLearningTask,runtimeVersion:String,seed:UInt64)throws->LearningAssessmentPlan {
        var state=seed
        func draw()->Double {state=state &* 6364136223846793005 &+ 1442695040888963407;return Double(Int((state >> 32)%33)-16)/16}
        let fixtures=try (0..<4).map {i->LearningFixture in
            var x=(0..<24).map {_ in draw()}
            // Draws are multiples of 1/16. Distinct row offsets stay below that
            // spacing and are exactly representable in production float32.
            for row in 0..<12 {x[row*2] += Double(row)/1024}
            var y:[Double]=[]
            for r in 0..<12 {let a=x[r*2],b=x[r*2+1];y += [0.5*a-0.25*b+0.125,-0.25*a+0.75*b-0.125]}
            state=state &* 6364136223846793005 &+ 1442695040888963407
            let inputs:[String:LearningTensor]=["trainingInputs":.init(shape:[8,2],values:Array(x.prefix(16))),"trainingTargets":.init(shape:[8,2],values:Array(y.prefix(16))),"heldoutInputs":.init(shape:[4,2],values:Array(x.suffix(8))),"heldoutTargets":.init(shape:[4,2],values:Array(y.suffix(8))),"seed":.init(shape:[1],values:[Double((state >> 32)%4_294_967_296)])]
            return .init(id:"\(task.rawValue)-\(seed)-\(i)",inputs:inputs,expected:try CheckpointLessonReference(task:task,inputs:inputs).expected)
        }
        let criteria=fixtures[0].expected.keys.sorted().map {key in LearningCriterion(id:key,description:CheckpointLessonReference.label(key),nextStep:task.controlInstructions,rule:.tensor(key:key,absoluteTolerance:["restoredPredictionDifference","childUpdateDifference","freshMemoryDifference"].contains(key) ? 2e-6:0,relativeTolerance:0))}
        let plan=LearningAssessmentPlan(id:"assessment."+task.rawValue,objectiveID:"checkpoint."+task.rawValue,task:task.instructions,runtime:.init(backend:"pytorch",version:runtimeVersion,capabilities:[task.componentID:ContractVersion()]),budget:.init(milliseconds:60_000,maximumTensorElements:1_000_000),fixtureSeed:seed,fixtures:fixtures,criteria:criteria)
        try plan.validate();return plan
    }
}
