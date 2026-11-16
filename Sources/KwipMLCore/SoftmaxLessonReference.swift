import Foundation

/// Independent bounded teaching oracle. This does not register a runtime component.
public struct SoftmaxLessonReference {
    public let output:LearningTensor
    private let groups:[[Int]]
    public init(input:LearningTensor,axis:Int)throws {
        guard input.wellFormed,input.shape.count>=2 else {
            throw LearningContractError("Softmax reference requires a finite batched tensor with at most 4096 values.")
        }
        let rank=input.shape.count
        guard axis >= -rank,axis < rank else {throw LearningContractError("Softmax axis is outside the tensor rank.")}
        let resolved=axis < 0 ? axis+rank:axis
        guard resolved != 0 else {throw LearningContractError("Choose a sample axis; this lesson preserves independent observations.")}
        let width=input.shape[resolved]
        let inner=input.shape.dropFirst(resolved+1).reduce(1,*)
        let outer=input.values.count/(width*inner)
        var groups:[[Int]]=[]
        var probabilities=Array(repeating:0.0,count:input.values.count)
        for prefix in 0..<outer {
            for suffix in 0..<inner {
                let indices=(0..<width).map {prefix*width*inner+$0*inner+suffix}
                let maximum=indices.map {input.values[$0]}.max()!
                let exponentials=indices.map {exp(input.values[$0]-maximum)}
                let sum=exponentials.reduce(0,+)
                for (index,value) in zip(indices,exponentials) {probabilities[index]=value/sum}
                groups.append(indices)
            }
        }
        self.groups=groups
        output=LearningTensor(shape:input.shape,values:probabilities)
    }
    public func inputGradient(for upstream:LearningTensor)throws->LearningTensor {
        guard upstream.wellFormed,upstream.shape == output.shape else {
            throw LearningContractError("Upstream gradient must match the softmax output shape and contain finite values.")
        }
        var gradient=Array(repeating:0.0,count:output.values.count)
        for indices in groups {
            let dot=indices.reduce(0.0) {$0+output.values[$1]*upstream.values[$1]}
            for index in indices {gradient[index]=output.values[index]*(upstream.values[index]-dot)}
        }
        guard gradient.allSatisfy(\.isFinite) else {throw LearningContractError("Softmax reference gradient exceeded finite numeric range.")}
        return LearningTensor(shape:output.shape,values:gradient)
    }
    public static let explanation="Softmax normalizes each slice along a chosen sample axis: pᵢ = exp(xᵢ − max(x)) / Σⱼ exp(xⱼ − max(x)). Subtracting one shared maximum preserves the result and avoids exponential overflow. Its Jacobian is diag(p) − ppᵀ, so an upstream vector g produces p ⊙ (g − Σⱼ pⱼgⱼ). Outputs in the same slice interact; changing one logit changes all their probabilities. Adding a constant to a slice changes neither its probabilities nor its ordering. The batch axis remains separate. Softmax is not a general repair for exploding activations or gradients."
}
