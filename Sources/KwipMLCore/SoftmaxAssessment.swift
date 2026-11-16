import Foundation

/// Private expected tensors. Public lesson views receive the task, never this grading plan.
public enum SoftmaxAssessment {
    public static func plan(axis:Int,runtimeVersion:String,seed:UInt64)throws->LearningAssessmentPlan {
        guard [1,-1].contains(axis) else {throw LearningContractError("Choose the first sample axis or the last axis for this Softmax lesson.")}
        let name=axis == 1 ? "first-sample-axis":"last-axis"
        var state=seed
        func next()->Double {
            state=state &* 6364136223846793005 &+ 1442695040888963407
            return Double((state>>32)%129)/16-4
        }
        let fixtures=try (0..<4).map {index -> LearningFixture in
            let batch=index%2+1,count=batch*6
            // This asymmetric anchor exposes axis mistakes independent of seed.
            // A shifted row also exercises stable normalization without overflow.
            let values=index == 0 ? [1000.0,1001,1002,-1,2,0]:(0..<count).map {_ in next()}
            let input=LearningTensor(shape:[batch,2,3],values:values)
            let reference=try SoftmaxLessonReference(input:input,axis:axis)
            let upstream=LearningTensor(shape:input.shape,values:(0..<count).map {Double(($0*5)%11)-4})
            return LearningFixture(id:"softmax-\(name)-\(seed)-\(index)",inputs:["x":input,"upstream":upstream],expected:["output":reference.output,"inputGradient":try reference.inputGradient(for:upstream)])
        }
        let task=axis == 1
            ? "For each observation [2,3], normalize the two values at each fixed last-axis coordinate. Every column must sum to one. Preserve the shape and keep observations separate."
            : "For each observation [2,3], normalize each row of three values. Every row must sum to one. Preserve the shape and keep observations separate."
        let result=LearningAssessmentPlan(id:"assessment.softmax-\(name)",objectiveID:"normalization.softmax-\(name)",task:task+" Given upstream coefficients g, compute the input gradient of L = sum(y * g). Predict which coordinates influence each other before running. Equivalent constructions are judged by measured behavior.",runtime:.init(backend:"pytorch",version:runtimeVersion,capabilities:["softmax":ContractVersion()]),fixtureSeed:seed,fixtures:fixtures,criteria:[
            .init(id:"axis-probabilities",description:"Normalize the intended slices while preserving shape and independent observations.",nextStep:"Hold every coordinate except the selected axis fixed. Subtract that slice's maximum, exponentiate, and divide by its exponential sum. Check which values compete.",rule:.tensor(key:"output",absoluteTolerance:1e-6,relativeTolerance:1e-6)),
            .init(id:"coupled-gradient",description:"Include interactions between all probabilities within each normalized slice.",nextStep:"Use J = diag(p) − ppᵀ. For upstream g, dL/dx = p ⊙ (g − sum(p ⊙ g)) along the normalized axis. An elementwise p(1−p) derivative omits the cross-coordinate terms.",rule:.tensor(key:"inputGradient",absoluteTolerance:1e-6,relativeTolerance:1e-6))
        ])
        try result.validate();return result
    }
}
