import Foundation

public enum ConfigurableActivationLearningTask:String,CaseIterable,Codable {
    case silu="silu-values",leakyReLU="leaky-relu-slope",elu="elu-alpha"
    public var componentID:String {self == .silu ? "silu":self == .leakyReLU ? "leaky_relu":"elu"}
    public var title:String {self == .silu ? "SiLU · smooth signed values":self == .leakyReLU ? "LeakyReLU · negative slope":"ELU · negative saturation"}
    public var sampleShape:[Int] {[2,3]}
    public var configuration:[String:JSONValue] {self == .silu ? [:]:self == .leakyReLU ? ["negative_slope":0.2]:["alpha":2]}
    public var instructions:String {
        let base="Connect Input → \(self == .leakyReLU ? "LeakyReLU":self == .silu ? "SiLU":"ELU") → Model Output. Select the activation block. An array is numbers arranged in rows and columns; each observation has two rows of three numbers. An activation transforms each number independently, keeping the array’s shape and observations separate. "
        let setting=self == .silu ? "SiLU has no adjustable setting. It multiplies each number by a smooth factor between 0 and 1; negative outputs can remain negative. It is not a probability. " : self == .leakyReLU ? "Set Negative slope to 0.2. Slope describes how much an output changes when the input changes: positive values stay unchanged, while negative values are multiplied by 0.2. At zero, the measured influence follows the negative branch. " : "Set Alpha to 2. Alpha controls the size of the negative branch: very negative values approach −2, while positive values stay unchanged. At zero, the measured influence follows the negative branch, using Alpha 2. "
        return base+setting+"Choose Check construction. Output and gradient tables are measured automatically. A gradient describes how a small change affects the result; no code or handwritten derivatives are required. These activations have no learned weights."
    }
}
public enum ConfigurableActivationAssessment {
    public static func plan(task:ConfigurableActivationLearningTask,runtimeVersion:String,seed:UInt64)throws->LearningAssessmentPlan {
        var state=seed
        func next()->Double {state=state &* 6364136223846793005 &+ 1442695040888963407;return Double(Int((state >> 32)%33)-16)/4}
        let fixtures=try (0..<4).map {i -> LearningFixture in
            let shape=[i%2+1]+task.sampleShape
            let anchor=[-1000.0,-1,0,1,2,1000]
            let input=LearningTensor(shape:shape,values:(0..<shape.reduce(1,*)).map {j in i == 0 ? anchor[j%6]:j%6 == 2 ? 0:next()})
            let ref=try ConfigurableActivationLessonReference(input:input,task:task)
            let upstream=LearningTensor(shape:shape,values:(0..<input.values.count).map {_ in next()})
            return .init(id:"\(task.rawValue)-\(i)",inputs:["x":input,"upstream":upstream],expected:["output":ref.output,"inputGradient":try ref.inputGradient(for:upstream)])
        }
        let criteria=[LearningCriterion(id:"output",description:"Transform each signed value with the requested activation setting while preserving the array.",nextStep:"Select the requested activation. Use SiLU’s empty configuration, LeakyReLU negative slope 0.2 or ELU Alpha 2 as specified; connect its output to Model Output.",rule:.tensor(key:"output",absoluteTolerance:1e-6,relativeTolerance:1e-6)),LearningCriterion(id:"inputGradient",description:"Measure how each input influences its corresponding output, including the defined zero convention.",nextStep:"The checker measures influence automatically. Preserve each value’s position; check the negative-branch setting and use the requested activation.",rule:.tensor(key:"inputGradient",absoluteTolerance:1e-6,relativeTolerance:1e-6))]
        let plan=LearningAssessmentPlan(id:"assessment."+task.rawValue,objectiveID:"activation."+task.rawValue,task:task.instructions,runtime:.init(backend:"pytorch",version:runtimeVersion,capabilities:[task.componentID:ContractVersion()]),fixtureSeed:seed,fixtures:fixtures,criteria:criteria)
        try plan.validate();return plan
    }
}
