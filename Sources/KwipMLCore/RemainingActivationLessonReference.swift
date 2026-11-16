import Foundation

/// Independent scalar arithmetic, with stochastic choices supplied by the frozen upstream bank.
public struct RemainingActivationLessonReference {
    public let output:LearningTensor
    public let trainingOutput:LearningTensor?
    private let input:LearningTensor,task:RemainingActivationLearningTask,slopes:LearningTensor?,mask:[Int]?
    public init(input:LearningTensor,task:RemainingActivationLearningTask,slopes:LearningTensor?=nil,canonicalMask:[Int]?=nil)throws {
        guard input.wellFormed,input.shape.count == 3,(2...4).contains(input.shape[0]),Array(input.shape.dropFirst()) == task.sampleShape else {throw LearningContractError("Use two channels with three positions and 2 to 4 observations.")}
        self.input=input;self.task=task;self.mask=canonicalMask
        let state=slopes ?? task.slopes
        if task.isPReLU {
            guard let state,state.wellFormed,state.shape == task.slopes!.shape else {throw LearningContractError("Supply one shared slope or one finite slope per channel.")}
        } else if state != nil {throw LearningContractError("This task has no learned slopes.")}
        if task.isDropout {
            guard let canonicalMask,canonicalMask.count == input.values.count,canonicalMask.allSatisfy({$0 == 0 || $0 == 1}) else {throw LearningContractError("Use a reviewed canonical mask matching the observation shape.")}
        } else if canonicalMask != nil {throw LearningContractError("This task uses no random mask.")}
        self.slopes=state
        var values=input.values
        if task.isPReLU {
            for i in values.indices {let slope=state!.values[state!.values.count == 1 ? 0:(i/3)%2];values[i]=input.values[i]>0 ? input.values[i]:input.values[i]*slope}
        } else if task == .logScores {
            for row in 0..<input.shape[0] {
                for column in 0..<3 {
                    let first=row*6+column,second=first+3,m=max(input.values[first],input.values[second])
                    let logTotal=log(exp(input.values[first]-m)+exp(input.values[second]-m))
                    values[first]=input.values[first]-m-logTotal;values[second]=input.values[second]-m-logTotal
                }
            }
        } else if task == .dropoutTraining {values=zip(input.values,canonicalMask!).map {$0*Double($1)*2}}
        output = .init(shape:input.shape,values:values)
        trainingOutput=task == .dropoutEvaluation ? .init(shape:input.shape,values:zip(input.values,canonicalMask!).map {$0*Double($1)*2}):nil
        guard output.wellFormed,trainingOutput?.wellFormed != false else {throw LearningContractError("Nonfinite scalar reference output.")}
    }
    public func gradients(for upstream:LearningTensor)throws->[String:LearningTensor] {
        guard upstream.wellFormed,upstream.shape == input.shape else {throw LearningContractError("Output influence must match the public array.")}
        var values=upstream.values,result:[String:LearningTensor]=[:]
        if task.isPReLU {
            var dw=Array(repeating:0.0,count:slopes!.values.count)
            for i in values.indices {
                let index=dw.count == 1 ? 0:(i/3)%2
                values[i]=upstream.values[i]*(input.values[i]>0 ? 1:slopes!.values[index])
                if input.values[i]<0 {dw[index]+=upstream.values[i]*input.values[i]}
            }
            result["slopeGradient"] = .init(shape:slopes!.shape,values:dw)
        } else if task == .logScores {
            for row in 0..<input.shape[0] {
                for column in 0..<3 {
                    let first=row*6+column,second=first+3,total=upstream.values[first]+upstream.values[second]
                    values[first]=upstream.values[first]-exp(output.values[first])*total
                    values[second]=upstream.values[second]-exp(output.values[second])*total
                }
            }
        } else if task == .dropoutTraining {values=zip(upstream.values,mask!).map {$0*Double($1)*2}}
        result["inputGradient"] = .init(shape:input.shape,values:values)
        if task == .dropoutEvaluation {result["trainingInputGradient"] = .init(shape:input.shape,values:zip(upstream.values,mask!).map {$0*Double($1)*2})}
        guard result.values.allSatisfy(\.wellFormed) else {throw LearningContractError("Nonfinite scalar reference influence.")}
        return result
    }
    public func inputGradient(for upstream:LearningTensor)throws->LearningTensor {try gradients(for:upstream)["inputGradient"]!}
    public static let explanation="A PReLU slope multiplies negative measurements. Positive measurements pass through; at exactly zero the reviewed input influence uses the slope, while slope influence is zero. Each slope receives the sum of contributions from every negative measurement that uses it. LogSoftmax compares scores along one sample direction using a stable maximum subtraction; exponentiating its logarithmic outputs recovers probabilities. Dropout temporarily removes selected measurements in training and doubles retained values when drop probability is one half. Evaluation preserves every measurement. Its randomness is separately frozen from the upstream CPU runtime for these lessons; expected arithmetic never comes from KwipML’s measured output. Gradients mean small-change influence and are measured automatically. Lessons use CPU float64; production keeps its distinct float32 contract."
}
