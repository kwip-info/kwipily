import Foundation

public enum StatefulNormalizationLearningTask:String,CaseIterable,Codable {
    case bn1Current="batch-norm1d-current-group",bn1Remembered="batch-norm1d-remembered-summary",bn2Current="batch-norm2d-current-group",bn2Remembered="batch-norm2d-remembered-summary",in1Current="instance-norm1d-individual-patterns",in1Remembered="instance-norm1d-remembered-summary",in2Current="instance-norm2d-individual-grids",in2Remembered="instance-norm2d-remembered-summary",groupTeams="group-norm-channel-teams",groupAdjustments="group-norm-channel-adjustments"
    public var componentID:String {switch self {case .bn1Current,.bn1Remembered:return "batch_norm1d";case .bn2Current,.bn2Remembered:return "batch_norm2d";case .in1Current,.in1Remembered:return "instance_norm1d";case .in2Current,.in2Remembered:return "instance_norm2d";case .groupTeams,.groupAdjustments:return "group_norm"}}
    public var componentTitle:String {switch componentID {case "batch_norm1d":return "BatchNorm1d";case "batch_norm2d":return "BatchNorm2d";case "instance_norm1d":return "InstanceNorm1d";case "instance_norm2d":return "InstanceNorm2d";default:return "GroupNorm"}}
    public var remembered:Bool {[Self.bn1Remembered,.bn2Remembered,.in1Remembered,.in2Remembered].contains(self)}
    public var affine:Bool {remembered || self == .groupAdjustments}
    public var sampleShape:[Int] {componentID == "group_norm" ? [4,3]:componentID.hasSuffix("2d") ? [3,2,3]:[3,4]}
    public var scale:LearningTensor? {affine ? .init(shape:[sampleShape[0]],values:Array([0.5,-1,2,0.75].prefix(sampleShape[0]))):nil}
    public var bias:LearningTensor? {affine ? .init(shape:[sampleShape[0]],values:Array([1,-0.5,0.25,-1].prefix(sampleShape[0]))):nil}
    public var configuration:[String:JSONValue] {
        var c:[String:JSONValue]=["eps":.number(0.125),"affine":.bool(affine),"bias":.bool(true)]
        if componentID == "group_norm" {c["num_channels"] = .integer(4);c["num_groups"] = .integer(2)}
        else {c["num_features"] = .integer(3);c["momentum"] = .number(0.25);c["track_running_stats"] = .bool(remembered);if componentID.hasPrefix("batch") {c["averaging"] = .string("moving")}}
        return c
    }
    public var title:String {
        if componentID == "group_norm" {return componentTitle+" · "+(affine ? "Adjust each channel after grouping":"Normalize teams of channels")}
        let name=componentID.hasSuffix("2d") ? "grids":"patterns"
        return componentTitle+" · "+(remembered ? "Use remembered summaries for new \(name)":componentID.hasPrefix("batch") ? "Compare channels across the current group":"Normalize each observation's \(name) separately")
    }
    public var controlInstructions:String {
        let adjustments=sampleShape[0]==4 ? "Use channel scale [0.5, -1, 2, 0.75] and offset [1, -0.5, 0.25, -1]. ":"Use channel scale [0.5, -1, 2] and offset [1, -0.5, 0.25]. "
        let base="Set Channels to \(sampleShape[0]) and Safety value to 0.125. "
        if componentID == "group_norm" {return base+"Set Channel groups to 2 and Learn scale and shift to \(affine ? "on":"off"). "+(affine ? "Keep Learn shift on. "+adjustments+"Inspect this component's parameter influence.":"Learn shift is inactive when learned adjustments are off.")}
        return base+"Set Track running statistics to \(remembered ? "on":"off"), Running-stat update weight to 0.25 and Learn scale and shift to \(affine ? "on":"off"). "+(componentID.hasPrefix("batch") ? "Use Moving for Averaging method. ":"")+(affine ? "Keep Learn shift on. "+adjustments+"Inspect this component's parameter influence.":"Learn shift is inactive when learned adjustments are off.")
    }
    public var instructions:String {
        let terms="An observation is one example. A batch is the current group of examples. Channels are parallel measurements or grids. Normalization centers numbers and adjusts their spread. Input influence, also called a gradient, measures how a small change affects the result; KwipML measures it for you. "
        let grouping=componentID.hasPrefix("batch") ? "Each channel compares all observations and spatial positions in the current batch. ":componentID.hasPrefix("instance") ? "Each observation's channel compares only its own positions; other observations remain separate. ":"Two equal teams of channels share a center and spread within each observation. "
        let memory=remembered ? "Running statistics are remembered summaries. KwipML first processes two supplied calibration groups, then evaluates new examples without updating the summaries. The update weight chooses how much of each new summary is retained; it is unrelated to optimizer momentum. ":"Current input values provide the summaries in both training and evaluation. "
        return terms+grouping+memory+"Safety value is a small positive cushion in the spread calculation. Learned scale and shift adjust each channel after centering. Build: connect Input → \(componentTitle) → Model Output. Configure: "+controlInstructions+" Check: choose Check construction and inspect output, measured influence and any remembered-summary tables. No code or handwritten derivatives are required."
    }
    public var task:String {instructions}
}
public enum StatefulNormalizationAssessment {
    public static func plan(task:StatefulNormalizationLearningTask,runtimeVersion:String,seed:UInt64)throws->LearningAssessmentPlan {
        var state=seed
        func tensor(_ shape:[Int],shift:Double=0)->LearningTensor {.init(shape:shape,values:(0..<shape.reduce(1,*)).map {_ in state=state &* 6364136223846793005 &+ 1442695040888963407;return Double(Int((state >> 32)%65)-32)/8+shift})}
        let fixtures=try (0..<4).map {index->LearningFixture in
            var x=tensor([index%3+1]+task.sampleShape,shift:Double(index)*3)
            if index == 0 {
                let spatial=task.sampleShape.dropFirst().reduce(1,*)
                for j in 0..<spatial {x.values[j]=10;x.values[spatial+j]=2+Double(j%3-1)/100}
            }
            let upstream=tensor(x.shape)
            var inputs=["x":x,"upstream":upstream]
            if task.remembered {inputs["calibrationA"] = tensor([2]+task.sampleShape,shift:-4);inputs["calibrationB"] = tensor([3]+task.sampleShape,shift:7)}
            let reference=try StatefulNormalizationLessonReference(input:x,task:task,calibrationA:inputs["calibrationA"],calibrationB:inputs["calibrationB"])
            return LearningFixture(id:"\(task.rawValue)-\(seed)-\(index)",inputs:inputs,expected:try reference.expected(for:upstream))
        }
        let keys=fixtures[0].expected.keys.sorted();let criteria=keys.map {key in LearningCriterion(id:key,description:key.contains("running") || key.contains("Running") || key.contains("Tracked") ? "Retain the declared summaries and preserve them during evaluation.":key.contains("Gradient") ? "Trace measured influence through the intended comparison group and any channel adjustments.":"Normalize the intended comparison group using the declared summaries and channel adjustments.",nextStep:task.controlInstructions,rule:.tensor(key:key,absoluteTolerance:key.contains("Tracked") ? 0:1e-6,relativeTolerance:key.contains("Tracked") ? 0:1e-6))}
        let plan=LearningAssessmentPlan(id:"assessment."+task.rawValue,objectiveID:"stateful-normalization."+task.rawValue,task:task.instructions,runtime:.init(backend:"pytorch",version:runtimeVersion,capabilities:[task.componentID:ContractVersion()]),fixtureSeed:seed,fixtures:fixtures,criteria:criteria)
        try plan.validate();return plan
    }
}
