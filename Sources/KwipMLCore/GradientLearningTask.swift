import Foundation

public enum GradientLearningTask:String,CaseIterable,Codable {
    case selectedInfluence="gradient-selected-influence",combineSmallGroups="gradient-combine-small-groups",keepSettings="trainability-keep-settings",nextRun="trainability-next-run"
    public static let widths=[[2,2,1],[3,2,1],[2,3,2],[3,3,2]]
    public var componentIDs:[String] {["gradient_policy","trainability","optimizer_sgd"]}
    public var title:String {switch self {case .selectedInfluence:return "Ask what changes a result";case .combineSmallGroups:return "Combine small groups before adjusting settings";case .keepSettings:return "Keep one layer's settings fixed";case .nextRun:return "Choose what can learn next"}}
    public var controlInstructions:String {switch self {
    case .selectedInfluence:return "Add a request for the Average of second.out on validation rows 0–2. Select the input and first.weight as the two influence targets."
    case .combineSmallGroups:return "Set Small groups per adjustment to 2. Keep Mean and no influence requests."
    case .keepSettings:return "Keep first.weight and first.bias Frozen. Leave the second layer able to learn."
    case .nextRun:return "Set first.weight and first.bias to Trainable for the supplied fine-tune run. Keep the second layer able to learn."
    }}
    public var instructions:String {"Learned settings are the weights and offsets a model adjusts. A gradient describes how a small change affects a result. A batch is a small group of observations; an optimizer uses gradients to adjust the settings. SGD is the supplied optimizer. Connect the three recipe controls to Train. "+controlInstructions+" Choose Check construction. Compare the measured differences with the independent reference; zero means match within the stated tolerance. Evaluation chooses predictable layer behavior. Freezing a setting prevents its adjustment; it does not turn off all calculations or stop input influence. Recording keeps calculations needed to ask about influence. "+(self == .combineSmallGroups ? "An accumulation window combines several small groups before one adjustment. The last short group still counts, with each observation weighted fairly.":self == .nextRun ? "Fine-tuning starts from a real saved parent and uses fresh optimizer memory. It keeps that saved parent unchanged; it is not an exact continuation.":"")}
    public var gradientValues:[String:JSONValue] {
        var requests:[JSONValue]=[]
        if self == .selectedInfluence || self == .keepSettings {
            var targets:[JSONValue]=[.object(["kind":.string("input"),"node":.string("input"),"port":.string("out")])]
            if self == .selectedInfluence {targets.append(.object(["kind":.string("parameter"),"node":.string("first"),"parameter":.string("weight")]))}
            requests=[.object(["id":.string(self == .selectedInfluence ? "selected-effect":"input-effect"),"output":.object(["node":.string("second"),"port":.string("out")]),"partition":.string("validation"),"row_start":0,"row_count":3,"reduction":.string("mean"),"upstream":.null,"targets":.array(targets),"unavailable":.string("error")])]
        }
        return ["accumulation_steps":.integer(self == .combineSmallGroups ? 2:1),"gradient_scale":.string("mean"),"requests":.array(requests)]
    }
    public var trainabilityValues:[String:JSONValue] {
        let owners:[JSONValue]=(self == .keepSettings || self == .nextRun) ? ["weight","bias"].map {.object(["node":.string("first"),"parameter":.string($0),"mode":.string(self == .keepSettings ? "frozen":"trainable")])}:[]
        return ["owners":.array(owners)]
    }
    public var optimizerValues:[String:JSONValue] {["lr":.number(0.05),"momentum":.number(self == .nextRun ? 0.5:0),"dampening":.number(0),"weight_decay":.number(0),"nesterov":false,"maximize":false]}
    public var derivativeValuesAllowed:Bool {self == .selectedInfluence || self == .keepSettings}
    public func measuredShapes(widths w:[Int])throws->[String:[Int]] {
        guard Self.widths.contains(w) else {throw LearningContractError("Use one of the four supplied layer shapes.")}
        let i=w[0],h=w[1],o=w[2],p=h*i+h+o*h+o
        switch self {
        case .selectedInfluence:return ["selectedResultDifference":[3,o],"inputInfluenceDifference":[3,i],"parameterInfluenceDifference":[h,i],"gradientBufferDifference":[p],"analysisStateDifference":[3]]
        case .combineSmallGroups:return ["windowSchedule":[4,4],"windowObjectiveDifference":[4],"windowGradientDifference":[4,p],"windowResetDifference":[4,p],"updatedSettingsDifference":[4,p]]
        case .keepSettings:return ["fixedSettingsDifference":[h*i+h],"learningSettingsDifference":[o*h+o],"inputInfluenceDifference":[3,i],"ownerInventoryDifference":[p,3],"recordingAvailability":[3]]
        case .nextRun:return ["restoredStartingDifference":[p],"childOwnerInventoryDifference":[p,3],"freshMemoryDifference":[p],"childSettingsDifference":[p+3*o],"parentByteDifference":[1]]
        }
    }
}
public enum GradientLearningAssessment {
    public static func plan(task:GradientLearningTask,runtimeVersion:String,seed:UInt64)throws->LearningAssessmentPlan {
        var state=seed
        func draw()->Double {state=state &* 6364136223846793005 &+ 1442695040888963407;return Double(Int((state >> 32)%33)-16)/16}
        let fixtures=try GradientLearningTask.widths.enumerated().map {index,w->LearningFixture in
            let i=w[0],o=w[2];var x=(0..<(8*i)).map {_ in draw()}
            for row in 0..<8 {x[row*i] += Double(row)/1024}
            let coefficients=(0..<(i*o)).map {_ in draw()},bias=(0..<o).map {_ in draw()}
            let y=(0..<(8*o)).map {k->Double in let row=k/o,out=k%o;return (0..<i).reduce(bias[out]) {$0+coefficients[out*i+$1]*x[row*i+$1]}}
            state=state &* 6364136223846793005 &+ 1442695040888963407
            let inputs:[String:LearningTensor]=["trainingInputs":.init(shape:[5,i],values:Array(x.prefix(5*i))),"trainingTargets":.init(shape:[5,o],values:Array(y.prefix(5*o))),"heldoutInputs":.init(shape:[3,i],values:Array(x.suffix(3*i))),"heldoutTargets":.init(shape:[3,o],values:Array(y.suffix(3*o))),"seed":.init(shape:[1],values:[Double((state >> 32)%4_294_967_296)])]
            return .init(id:"\(task.rawValue)-\(seed)-\(index)",inputs:inputs,expected:try GradientLessonReference(task:task,inputs:inputs).expected)
        }
        let exact:Set<String>=["windowSchedule","windowResetDifference","gradientBufferDifference","analysisStateDifference","fixedSettingsDifference","ownerInventoryDifference","recordingAvailability","childOwnerInventoryDifference","restoredStartingDifference","parentByteDifference"]
        let criteria=fixtures[0].expected.keys.sorted().map {key in LearningCriterion(id:key,description:GradientLessonReference.label(key),nextStep:task.controlInstructions,rule:.tensor(key:key,absoluteTolerance:exact.contains(key) ? 0:2e-6,relativeTolerance:0))}
        let plan=LearningAssessmentPlan(id:"assessment."+task.rawValue,objectiveID:"gradient."+task.rawValue,task:task.instructions,runtime:.init(backend:"pytorch",version:runtimeVersion,capabilities:Dictionary(uniqueKeysWithValues:task.componentIDs.map {($0,ContractVersion())})),budget:.init(milliseconds:60_000,maximumTensorElements:1_000_000),fixtureSeed:seed,fixtures:fixtures,criteria:criteria)
        try validate(plan);return plan
    }
    public static func validate(_ plan:LearningAssessmentPlan)throws {
        guard let task=GradientLearningTask.allCases.first(where:{plan.id=="assessment."+$0.rawValue}),plan.objectiveID=="gradient."+task.rawValue,plan.contentVersion==ContractVersion(),plan.assessmentVersion==ContractVersion(),plan.runtime.backend=="pytorch",plan.runtime.capabilities==Dictionary(uniqueKeysWithValues:task.componentIDs.map {($0,ContractVersion())}),plan.fixtures.count==4,plan.criteria.count==5 else {throw LearningContractError("Use the current four-fixture, five-check gradient assessment.")}
        try plan.validate()
        for (index,fixture) in plan.fixtures.enumerated() {
            guard fixture.expected==((try GradientLessonReference(task:task,inputs:fixture.inputs)).expected),fixture.expected.mapValues(\.shape)==(try task.measuredShapes(widths:GradientLearningTask.widths[index])),Set(plan.criteria.map(\.id))==Set(fixture.expected.keys) else {throw LearningContractError("Gradient criteria and fixture geometry must match every reviewed measured role.")}
            for criterion in plan.criteria {
                let exact:Set<String>=["windowSchedule","windowResetDifference","gradientBufferDifference","analysisStateDifference","fixedSettingsDifference","ownerInventoryDifference","recordingAvailability","childOwnerInventoryDifference","restoredStartingDifference","parentByteDifference"]
                guard case .tensor(let key,let absolute,let relative)=criterion.rule,key==criterion.id,absolute==(exact.contains(key) ? 0:2e-6),relative==0 else {throw LearningContractError("Each gradient check needs its exact current measured role and tolerance.")}
            }
        }
    }

}
