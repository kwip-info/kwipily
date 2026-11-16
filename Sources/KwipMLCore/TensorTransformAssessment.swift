import Foundation

/// Private grading plans. Public lesson views must not expose these expected tensors.
/// Runtime capabilities here are requirements, not claims about the installed worker.
public enum TensorTransformAssessment {
    public static func plan(operation:String,runtimeVersion:String,seed:UInt64)throws->LearningAssessmentPlan {
        let configuration:[String:JSONValue]
        let task:String
        switch operation {
        case "reshape":
            configuration=["shape":.array([3,2])]
            task="For each input observation of shape [2,3], produce [3,2] without changing logical row-major element order. Preserve the batch axis."
        case "transpose":
            configuration=["dim0":1,"dim1":2]
            task="For each input observation of shape [2,3], swap its two sample axes to produce [3,2]. Preserve the batch axis and the coordinate correspondence y[n,j,i] = x[n,i,j]."
        case "flatten":
            configuration=["start_dim":1,"end_dim":-1]
            task="For each input observation of shape [2,3], combine its sample axes into [6] in logical row-major order. Preserve independent observations."
        default:throw LearningContractError("Unknown transform assessment operation.")
        }
        var state=seed
        func next()->Double {
            state=state &* 6364136223846793005 &+ 1442695040888963407
            return Double((state>>32)%129)/16-4
        }
        let fixtures=try (0..<4).map {index -> LearningFixture in
            let batch=index%2+1,count=batch*6
            // A unique monotonic anchor guarantees reshape/transpose separation,
            // even when a seeded random fixture happens to be symmetric.
            let values=index == 0 ? (0..<count).map {Double($0)-2}: (0..<count).map {_ in next()}
            let input=LearningTensor(shape:[batch,2,3],values:values)
            let reference=try TensorTransformLessonReference(input:input,operation:operation,configuration:configuration)
            let upstream=LearningTensor(shape:reference.output.shape,values:(0..<count).map {Double($0+1)/8})
            return LearningFixture(id:"tensor-\(operation)-\(seed)-\(index)",inputs:["x":input,"upstream":upstream],expected:["output":reference.output,"inputGradient":try reference.inputGradient(for:upstream)])
        }
        let result=LearningAssessmentPlan(id:"assessment.tensor-\(operation)",objectiveID:"tensors.\(operation)",task:task+" Given upstream coefficients g, also compute the input gradient of L = sum(y * g). Predict values and derivatives before running. Equivalent constructions are accepted by behavior, not node names.",runtime:.init(backend:"pytorch",version:runtimeVersion,capabilities:[operation:ContractVersion()]),fixtureSeed:seed,fixtures:fixtures,criteria:[
            .init(id:"element-order",description:"Preserve the requested dimensions and coordinate-to-value mapping for every observation.",nextStep:"Track a distinct value through its input and output coordinates. Reshaping changes index grouping; transposing swaps axes. Keep batch separate.",rule:.tensor(key:"output",absoluteTolerance:1e-6,relativeTolerance:1e-6)),
            .init(id:"gradient-routing",description:"Route each upstream derivative back to the input coordinate that produced that output.",nextStep:"Write y = P x. Then dL/dx = Pᵀ g. Use the inverse coordinate mapping; these operations have no trainable parameters.",rule:.tensor(key:"inputGradient",absoluteTolerance:1e-6,relativeTolerance:1e-6))
        ])
        try result.validate();return result
    }
}
