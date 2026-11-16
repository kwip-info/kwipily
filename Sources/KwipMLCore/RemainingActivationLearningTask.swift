import Foundation

public enum RemainingActivationLearningTask:String,CaseIterable,Codable {
    case sharedSlope="prelu-shared-slope",channelSlopes="prelu-channel-slopes",logScores="logsoftmax-sample-scores",dropoutTraining="dropout-training-pattern",dropoutEvaluation="dropout-evaluation-pattern"
    public var componentID:String {isPReLU ? "prelu":self == .logScores ? "log_softmax":"dropout"}
    public var isPReLU:Bool {self == .sharedSlope || self == .channelSlopes}
    public var isDropout:Bool {self == .dropoutTraining || self == .dropoutEvaluation}
    public var sampleShape:[Int] {[2,3]}
    public var configuration:[String:JSONValue] {isPReLU ? ["num_parameters":self == .sharedSlope ? 1:2,"init":0.25]:self == .logScores ? ["dim":1]:["p":0.5]}
    public var slopes:LearningTensor? {isPReLU ? .init(shape:[self == .sharedSlope ? 1:2],values:self == .sharedSlope ? [0.25]:[0.1,0.5]):nil}
    public var title:String {switch self {case .sharedSlope:return "PReLU · one learned negative multiplier";case .channelSlopes:return "PReLU · a multiplier per channel";case .logScores:return "LogSoftmax · compare relative scores";case .dropoutTraining:return "Dropout · temporarily remove measurements";case .dropoutEvaluation:return "Dropout · switch from training to evaluation"}}
    public var controlInstructions:String {switch self {
    case .sharedSlope:return "Select PReLU. Set Slope count to 1 and Initial slope to 0.25. In learned parameters set the one slope to 0.25; select this block for slope influence."
    case .channelSlopes:return "Select PReLU. Set Slope count to 2 and Initial slope to 0.25. In learned parameters set channel slopes to 0.1 and 0.5; select this block for slope influence."
    case .logScores:return "Select LogSoftmax. Set Normalization direction to 1. Compare the two channel scores at each column independently within each observation."
    case .dropoutTraining:return "Select Dropout. Set Drop probability to 0.5. This task measures training behavior: some cells become zero and retained cells double. The checker supplies a repeatable random draw."
    case .dropoutEvaluation:return "Select Dropout. Set Drop probability to 0.5. This task checks evaluation identity and also verifies the same block’s training behavior; setting probability to zero is not a substitute."
    }}
    public var instructions:String {"**Build**\nConnect Input → the requested block → Model Output. Each observation has two channels with three measurements each.\n\n**Configure**\n"+controlInstructions+"\n\n**Check**\nChoose Check construction. Values and small-change influence are measured automatically. An array is numbers arranged in directions; a channel is one measurement field (here each row), and an observation is one independent example. A gradient describes how a small change affects the result; no code or handwritten derivatives are required. "+(self == .logScores ? "Relative scores express preferences. Softmax compares them as probabilities; LogSoftmax stores those probabilities on a logarithmic, compressed scale. Its outputs are not probabilities themselves.":isDropout ? "Training adjusts a model; evaluation uses it to assess new examples. A mask records which cells a random draw keeps. A seed makes the same runtime and call schedule repeatable, while successive training calls normally draw fresh masks.":"A learned slope is an adjustable multiplier for negative values. Positive values stay unchanged. A shared slope receives influence from every channel that uses it.")}
}

public enum RemainingActivationMaskBank {
    public struct Entry:Decodable {public let seed:Int;public let shape:[Int];public let mask:[Int]}
    private struct Bank:Decodable {let schemaVersion:Int;let maskVersion:String;let torchVersion:String;let device:String;let dtype:String;let p:Double;let entries:[Entry]}
    public static let sha256="5a96044ae7f6286b354d6cb43cd6bec4e040a15b55bffc01ecaa57bbc1056538"
    public static func entry(index:Int,runtimeVersion:String)throws->Entry {
        guard runtimeVersion == "2.14.1" else {throw LearningContractError("Canonical dropout masks require Torch 2.14.1; reviewed lesson evidence is stale for this runtime.")}
        guard let url=Bundle.module.url(forResource:"remaining_activation_dropout_masks_v1",withExtension:"json") else {throw LearningContractError("Canonical dropout mask resource is unavailable.")}
        let data=try Data(contentsOf:url)
        guard learningHash(data) == sha256 else {throw LearningContractError("Canonical dropout mask integrity differs.")}
        let bank=try JSONDecoder().decode(Bank.self,from:data)
        guard bank.schemaVersion == 1,bank.maskVersion == "kwip.dropout.cpu-float64.v1",bank.torchVersion == runtimeVersion,bank.device == "cpu",bank.dtype == "float64",bank.p == 0.5,bank.entries.count == 12,bank.entries.indices.contains(index) else {throw LearningContractError("Canonical dropout mask version or selection differs.")}
        let result=bank.entries[index]
        guard result.mask.count == result.shape.reduce(1,*),result.mask.allSatisfy({$0 == 0 || $0 == 1}) else {throw LearningContractError("Malformed canonical dropout mask.")}
        return result
    }
}

public enum RemainingActivationAssessment {
    public static func plan(task:RemainingActivationLearningTask,runtimeVersion:String,seed:UInt64)throws->LearningAssessmentPlan {
        var state=seed
        func next()->Double {state=state &* 6364136223846793005 &+ 1442695040888963407;return Double(Int((state >> 32)%33)-16)/4}
        let fixtures=try (0..<4).map {i -> LearningFixture in
            let rows=i%3+2,shape=[rows]+task.sampleShape
            let input=LearningTensor(shape:shape,values:(0..<(rows*6)).map {j in j%6 == 2 ? 0:next()})
            let upstream=LearningTensor(shape:shape,values:(0..<(rows*6)).map {_ in next()})
            let bankIndex=(Int(seed%4)+i)%4*3+(rows-2)
            let entry=task.isDropout ? try RemainingActivationMaskBank.entry(index:bankIndex,runtimeVersion:runtimeVersion):nil
            let reference=try RemainingActivationLessonReference(input:input,task:task,canonicalMask:entry?.mask)
            var inputs=["x":input,"upstream":upstream]
            if task.isDropout {inputs["maskBankIndex"] = .init(shape:[1],values:[Double(bankIndex)])}
            var expected=try reference.gradients(for:upstream);expected["output"]=reference.output
            if let trainingOutput=reference.trainingOutput {expected["trainingOutput"]=trainingOutput}
            return .init(id:"\(task.rawValue)-\(seed)-\(i)",inputs:inputs,expected:expected)
        }
        let keys=fixtures[0].expected.keys.sorted()
        let criteria=keys.map {key in LearningCriterion(id:key,description:key == "output" ? "Preserve the requested values, directions and separate observations.":key == "slopeGradient" ? "Measure influence on each learned negative multiplier, adding all its uses.":key.hasPrefix("training") ? "Verify configured training behavior as well as evaluation identity.":"Measure how each input affects the result automatically.",nextStep:task.controlInstructions,rule:.tensor(key:key,absoluteTolerance:1e-6,relativeTolerance:1e-6))}
        let plan=LearningAssessmentPlan(id:"assessment."+task.rawValue,objectiveID:"remaining-activation."+task.rawValue,task:task.instructions,runtime:.init(backend:"pytorch",version:runtimeVersion,capabilities:[task.componentID:ContractVersion(1,2,0)]),fixtureSeed:seed,fixtures:fixtures,criteria:criteria)
        try plan.validate();return plan
    }
}
