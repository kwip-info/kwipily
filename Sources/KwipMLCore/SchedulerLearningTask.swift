import Foundation

public enum SchedulerLearningTask:String,CaseIterable,Codable {
    case step="step-lr-intervals",cosine="cosine-lr-curve",linear="linear-lr-warmup",plateau="plateau-lr-improvement",oneCycle="one-cycle-lr-batches"
    public var componentID:String {switch self {case .step:return "scheduler_step_lr";case .cosine:return "scheduler_cosine_annealing_lr";case .linear:return "scheduler_linear_lr";case .plateau:return "scheduler_reduce_on_plateau";case .oneCycle:return "scheduler_one_cycle_lr"}}
    public var componentTitle:String {switch self {case .step:return "StepLR";case .cosine:return "CosineAnnealingLR";case .linear:return "LinearLR";case .plateau:return "ReduceLROnPlateau";case .oneCycle:return "OneCycleLR"}}
    public var title:String {componentTitle+" · "+(self == .step ? "Reduce after a fixed interval":self == .cosine ? "Ease toward a smaller rate":self == .linear ? "Start with smaller adjustments":self == .plateau ? "Wait for improvement":"Rise and fall across batches")}
    public static let suppliedOptimizerValues:[String:JSONValue]=["lr":.number(0.1),"momentum":.number(0.5),"dampening":.number(0),"weight_decay":.number(0),"nesterov":.bool(false),"maximize":.bool(false)]
    public var configuration:[String:JSONValue] {
        switch self {
        case .step:return ["step_size":.integer(2),"gamma":.number(0.5)]
        case .cosine:return ["T_max":.integer(4),"eta_min":.number(0.02)]
        case .linear:return ["start_factor":.number(0.25),"end_factor":.number(1),"total_iters":.integer(3)]
        case .plateau:return ["mode":.string("min"),"factor":.number(0.5),"patience":.integer(0),"threshold":.number(0.05),"threshold_mode":.string("abs"),"cooldown":.integer(1),"min_lr":.number(0.01),"eps":.number(1e-8),"metric":.string("validation_loss")]
        case .oneCycle:return ["max_lr":.number(0.1),"planned_total_batches":.integer(8),"pct_start":.number(0.5),"anneal_strategy":.string("cos"),"cycle_momentum":.bool(false),"base_momentum":.number(0.85),"max_momentum":.number(0.95),"div_factor":.number(10),"final_div_factor":.number(10),"three_phase":.bool(false)]
        }
    }
    public var controlInstructions:String {
        switch self {
        case .step:return "Set Passes between reductions to 2 and Rate multiplier to 0.5."
        case .cosine:return "Set Passes to minimum to 4 and Minimum learning rate to 0.02."
        case .linear:return "Set Starting rate fraction to 0.25, Ending rate fraction to 1 and Passes to ending fraction to 3."
        case .plateau:return "Choose validation loss for Measure to watch and min for Improvement direction. Set Reduction multiplier to 0.5, Passes to wait to 0, Required improvement to 0.05, Improvement comparison to abs, Recovery passes to 1, Minimum learning rate to 0.01 and Smallest rate change to 0.00000001. Min means smaller is better; abs means an absolute improvement amount."
        case .oneCycle:return "Set Peak learning rate to 0.1, Planned total batches to 8, Rising portion to 0.5 and Curve shape to cos. Disable Vary remembered movement and Use three phases. Set Starting rate divisor and Final rate divisor to 10. Cos means the smooth cosine curve."
        }
    }
    public var explanation:String {
        let base="Learning rate is the size of an adjustment. A batch is one group of examples. An epoch is one pass through all training examples. These examples have two batches per epoch. The supplied SGD optimizer starts with learning rate 0.1 and momentum 0.5; momentum remembers part of the previous change. "
        switch self {
        case .step:return base+"StepLR keeps the same rate until its epoch interval ends, then multiplies it by the chosen fraction. A multiplier of 0.5 halves the rate for the next epoch."
        case .cosine:return base+"CosineAnnealingLR follows a smooth falling curve toward the minimum. You do not need trigonometry to read its table. The rate changes after each completed epoch."
        case .linear:return base+"LinearLR begins at a fraction of the supplied rate, then changes that fraction by equal amounts each epoch until it reaches the ending fraction. Its smaller starting rate applies before the first update."
        case .plateau:return base+"Validation loss measures error on held-out examples; smaller loss is better here. An absolute improvement threshold is the amount the loss must fall to count as improvement. Patience is how many unimproved epochs may pass before a reduction. Cooldown pauses this wait after a reduction. The minimum reduction ignores rate changes too small to matter."
        case .oneCycle:return base+"OneCycleLR starts below the peak, rises and then falls across individual batches. A cosine curve changes smoothly. Division factors set how much smaller the start and finish are than the peak. This lesson keeps momentum constant. Its last next rate is recorded but unused because the planned sequence is finished."
        }
    }
    public var instructions:String {"A scheduler recipe tells Train when to change its learning rate. It does not transform model data. "+explanation+" Build: connect the supplied SGD out to Train optimizer, and \(componentTitle) out to Train scheduler. Configure: "+controlInstructions+" Check: choose Check construction. Compare the rate used, next rate, learned settings and remembered momentum across eight batches. The checker also saves after two epochs and measures a resumed sequence; it should continue with the same rates and settings. Starting settings, gradients, metrics and saved state are supplied by the checker. No code or state editing is required."}
}
public enum SchedulerAssessment {
    public static func plan(task:SchedulerLearningTask,runtimeVersion:String,seed:UInt64)throws->LearningAssessmentPlan {
        var state=seed
        func draw()->Double {state=state &* 6364136223846793005 &+ 1442695040888963407;return Double(Int((state >> 32)%33)-16)/8}
        let fixtures=try (0..<4).map {i->LearningFixture in
            let initial=LearningTensor(shape:[3],values:(0..<3).map {_ in draw()}),gradients=LearningTensor(shape:[8,3],values:(0..<24).map {_ in draw()})
            let active=LearningTensor(shape:[8,2],values:[1,1,1,0,1,1,0,1,1,1,1,0,1,1,1,1]),epochs=LearningTensor(shape:[8],values:[1,1,2,2,3,3,4,4])
            let offset=abs(draw())/8,metrics=LearningTensor(shape:[4],values:[1+offset,1.1+offset,1.2+offset,0.8+offset])
            return .init(id:"\(task.rawValue)-\(seed)-\(i)",inputs:["initial":initial,"gradients":gradients,"active":active,"epochOfBatch":epochs,"metrics":metrics],expected:try SchedulerLessonReference(initial:initial,gradients:gradients,active:active,epochOfBatch:epochs,metrics:metrics,task:task).expected)
        }
        let criteria=fixtures[0].expected.keys.sorted().map {LearningCriterion(id:$0,description:"Use the declared rate at the correct update and continue the same saved sequence.",nextStep:task.controlInstructions,rule:.tensor(key:$0,absoluteTolerance:1e-8,relativeTolerance:1e-8))}
        let plan=LearningAssessmentPlan(id:"assessment."+task.rawValue,objectiveID:"scheduler."+task.rawValue,task:task.instructions,runtime:.init(backend:"pytorch",version:runtimeVersion,capabilities:[task.componentID:ContractVersion(),"optimizer_sgd":ContractVersion()]),fixtureSeed:seed,fixtures:fixtures,criteria:criteria)
        try plan.validate();return plan
    }
}
