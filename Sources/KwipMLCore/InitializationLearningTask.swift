import Foundation

public enum InitializationLearningTask:String,CaseIterable,Codable {
    case startingValues="initialization-starting-values",startingSpread="initialization-starting-spread"
    public var componentID:String {"initialization"}
    public var title:String {self == .startingValues ? "Choose starting values":"Match the starting spread"}
    public static let widths:[[Int]]=[[2,3,2],[3,4,2],[4,3,2],[2,5,3]]
    public static let suppliedOptimizerValues:[String:JSONValue]=["lr":.number(0.05),"momentum":.number(0),"dampening":.number(0),"weight_decay":.number(0),"nesterov":false,"maximize":false]
    public var configuration:[String:JSONValue] {
        func rule(_ node:String,_ parameter:String,_ algorithm:String,_ settings:[String:JSONValue])->JSONValue {
            .object(["owner":.object(["node":.string(node),"parameter":.string(parameter)]),"algorithm":.string(algorithm),"settings":.object(settings)])
        }
        return ["seed":42,"rules":.array(self == .startingValues ? [rule("first","bias","zeros",[:]),rule("first","weight","normal",["mean":.number(0.15),"std":.number(0.2)])]:[rule("first","weight","xavier_uniform",["gain":.number(0.7)]),rule("second","weight","kaiming_uniform",["a":.number(0.2),"mode":.string("fan_out"),"nonlinearity":.string("leaky_relu")])])]
    }
    public var controlInstructions:String {self == .startingValues ? "Set Seed to 42. Choose Normal for first.weight with Center 0.15 and Spread 0.2. Choose Zeros for first.bias.":"Set Seed to 42. Choose Xavier uniform for first.weight with Gain 0.7. Choose Kaiming uniform for second.weight with Slope 0.2, Outgoing fan and Leaky ReLU."}
    public var instructions:String {"Learned settings are the numbers a model adjusts during training. Starting values choose where those adjustments begin. A seed is a repeatable starting point for random values; changing it changes the draw without changing the training random stream. Connect Starting settings out to Train initialization and Supplied SGD out to Train optimizer. "+controlInstructions+" Choose Check construction. Compare the starting arrays, six predictions and first training adjustment. Zero difference means a match to the independent reference. The checker also repeats the draw and saves and reloads the learned arrays. This saved reuse checks weights, not a full training continuation. Normal uses a center and spread; Zeros fills an array with zero. Xavier and Kaiming choose a spread from the number of incoming or outgoing connections (the fan). Gain and slope adjust that spread. No code is required."}
}
public enum InitializationAssessment {
    public static func plan(task:InitializationLearningTask,runtimeVersion:String,seed:UInt64)throws->LearningAssessmentPlan {
        var state=seed
        func draw()->Double {state=state &* 6364136223846793005 &+ 1442695040888963407;return Double(Int((state >> 32)%37)-18)/16}
        let fixtures=try InitializationLearningTask.widths.enumerated().map {index,w->LearningFixture in
            func tensor(_ shape:[Int])->LearningTensor {.init(shape:shape,values:(0..<shape.reduce(1,*)).map {_ in draw()})}
            let inputs=["x":tensor([6,w[0]]),"target":tensor([4,w[2]]),"firstWeight":tensor([w[1],w[0]]),"firstBias":tensor([w[1]]),"secondWeight":tensor([w[2],w[1]]),"secondBias":tensor([w[2]])]
            return .init(id:"\(task.rawValue)-\(seed)-\(index)",inputs:inputs,expected:try InitializationLessonReference(inputs:inputs).expected)
        }
        let criteria=fixtures[0].expected.keys.sorted().map {key in LearningCriterion(id:key,description:InitializationLessonReference.label(key)+": match the independent measured reference.",nextStep:task.controlInstructions,rule:.tensor(key:key,absoluteTolerance:["startingSettings","repeatIsolation"].contains(key) ? 0:2e-6,relativeTolerance:0))}
        let plan=LearningAssessmentPlan(id:"assessment."+task.rawValue,objectiveID:"initialization."+task.rawValue,task:task.instructions,runtime:.init(backend:"pytorch",version:runtimeVersion,capabilities:["initialization":ContractVersion(),"optimizer_sgd":ContractVersion()]),fixtureSeed:seed,fixtures:fixtures,criteria:criteria)
        try plan.validate();return plan
    }
}
