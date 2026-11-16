import Foundation

public enum OptimizerLearningTask:String,CaseIterable,Codable {
    case sgd="sgd-momentum-steps",adam="adam-running-averages",adamW="adamw-separate-decay",rmsprop="rmsprop-squared-memory",adagrad="adagrad-accumulated-steps"
    public var componentID:String {switch self {case .sgd:return "optimizer_sgd";case .adam:return "optimizer_adam";case .adamW:return "optimizer_adamw";case .rmsprop:return "optimizer_rmsprop";case .adagrad:return "optimizer_adagrad"}}
    public var componentTitle:String {switch self {case .sgd:return "SGD";case .adam:return "Adam";case .adamW:return "AdamW";case .rmsprop:return "RMSprop";case .adagrad:return "Adagrad"}}
    public var title:String {componentTitle+" · "+(self == .sgd ? "Remember previous changes":self == .adam ? "Use running averages":self == .adamW ? "Shrink settings separately":self == .rmsprop ? "Track recent squared changes":"Accumulate squared changes")}
    public var configuration:[String:JSONValue] {
        var c:[String:JSONValue]=["lr":.number(0.1),"weight_decay":.number(0.1)]
        switch self {
        case .sgd:c["momentum"] = .number(0.5);c["dampening"] = .number(0.25)
        case .adam,.adamW:c["beta1"] = .number(0.5);c["beta2"] = .number(0.75);c["eps"] = .number(0.01);c["amsgrad"] = .bool(true)
        case .rmsprop:c["alpha"] = .number(0.5);c["eps"] = .number(0.01);c["momentum"] = .number(0.25);c["centered"] = .bool(true)
        case .adagrad:c["lr_decay"] = .number(0.2);c["initial_accumulator_value"] = .number(0.5);c["eps"] = .number(0.01)
        }
        return c
    }
    public var controlInstructions:String {
        let base="Set Learning rate to 0.1 and Weight decay to 0.1. Keep Maximize off. "
        switch self {
        case .sgd:return base+"Set Momentum to 0.5 and Momentum dampening to 0.25. Keep Look ahead off."
        case .adam,.adamW:return base+"Set First running average to 0.5, Second running average to 0.75 and Stability amount to 0.01. Enable Keep largest squared average."+(self == .adam ? " Keep Separate weight decay off.":"")
        case .rmsprop:return base+"Set Squared-average memory to 0.5, Momentum to 0.25 and Stability amount to 0.01. Enable Center the squared average."
        case .adagrad:return base+"Set Learning-rate decay to 0.2, Starting accumulated squares to 0.5 and Stability amount to 0.01."
        }
    }
    public var explanation:String {
        let common="Learning rate controls the size of an adjustment. Weight decay discourages large learned settings. A square is a number multiplied by itself; averaging squared gradients helps measure the size of recent changes. Remembered quantities start fresh for every supplied example sequence. "
        switch self {
        case .sgd:return common+"Momentum carries some of the previous change into the next step. Dampening reduces the new gradient's contribution after the first step. Here weight decay contributes to the gradient before momentum is updated."
        case .adam:return common+"Adam remembers an average of signed gradients and an average of their squares. It corrects these averages for their short starting history, then uses their sizes to adjust each setting. Keep largest squared average prevents the remembered squared amount from decreasing. Stability amount is a small cushion in the divisor. Here weight decay contributes to the gradient."
        case .adamW:return common+"AdamW uses the same two running averages as Adam, but shrinks settings separately from the gradient calculation. The shrink does not become part of the remembered gradient averages. Keep largest squared average prevents the remembered squared amount from decreasing. Stability amount is a small cushion in the divisor."
        case .rmsprop:return common+"RMSprop remembers recent squared gradients. Squared-average memory chooses how much earlier information is retained. Center the squared average also remembers the average gradient and removes its squared value when measuring spread. Momentum remembers previous adjustments; Stability amount prevents an unsafe divisor."
        case .adagrad:return common+"Adagrad adds each gradient's square to its accumulated total. Settings with larger totals receive smaller adjustments. Starting accumulated squares is the initial cushion in that total. Learning-rate decay gradually reduces the rate with each actual update; a missing gradient does not advance that setting's count. Stability amount is a small cushion in the divisor."
        }
    }
    public var instructions:String {"A weight is a learned setting. A gradient measures how changing a setting affects the model's error; the lesson supplies it. A training step adjusts settings after one group of examples. An optimizer recipe tells Train how to make that adjustment; it does not transform model input. "+explanation+" Build: connect \(componentTitle) out to Train optimizer. Configure: "+controlInstructions+" Check: choose Check construction. Compare starting settings, supplied gradients, updated settings and remembered quantities over four steps. When a setting has no gradient, it receives no update; zero learning rate leaves settings unchanged. The checker owns starting settings and remembered state. No code or hidden-state editing is required."}
}
public enum OptimizerAssessment {
    public static func plan(task:OptimizerLearningTask,runtimeVersion:String,seed:UInt64)throws->LearningAssessmentPlan {
        var state=seed
        func draw()->Double {state=state &* 6364136223846793005 &+ 1442695040888963407;return Double(Int((state >> 32)%33)-16)/8}
        let fixtures=try (0..<4).map {i->LearningFixture in
            let initial=LearningTensor(shape:[3],values:(0..<3).map {_ in draw()})
            let gradients=LearningTensor(shape:[4,3],values:(0..<12).map {_ in draw()})
            let active=LearningTensor(shape:[4,2],values:[1,1,1,0,1,1,0,1])
            let expected=try OptimizerLessonReference(initial:initial,gradients:gradients,active:active,task:task).expected
            return .init(id:"\(task.rawValue)-\(seed)-\(i)",inputs:["initial":initial,"gradients":gradients,"active":active],expected:expected)
        }
        let criteria=fixtures[0].expected.keys.sorted().map {key in LearningCriterion(id:key,description:key == "updatedParameters" ? "Adjust learned settings through the supplied update sequence.":"Remember the declared update quantities and respect settings without gradients.",nextStep:task.controlInstructions,rule:.tensor(key:key,absoluteTolerance:1e-8,relativeTolerance:1e-8))}
        let plan=LearningAssessmentPlan(id:"assessment."+task.rawValue,objectiveID:"optimizer."+task.rawValue,task:task.instructions,runtime:.init(backend:"pytorch",version:runtimeVersion,capabilities:[task.componentID:ContractVersion()]),fixtureSeed:seed,fixtures:fixtures,criteria:criteria)
        try plan.validate();return plan
    }
}
