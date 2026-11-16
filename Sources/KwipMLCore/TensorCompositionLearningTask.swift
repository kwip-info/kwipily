import Foundation

public enum TensorCompositionLearningTask:String,CaseIterable,Codable {
    case concat="concat-sample-axis",add="add-shared-pattern",stack="stack-new-direction",permute="permute-sample-order"
    public var componentID:String {switch self {case .concat:return "concat";case .add:return "add";case .stack:return "stack";case .permute:return "permute"}}
    public var isBinary:Bool {self != .permute}
    public var leftSampleShape:[Int] {self == .permute ? [2,3,4]:[2,3]}
    public var rightSampleShape:[Int]? {switch self {case .concat:return [1,3];case .add:return [3];case .stack:return [2,3];case .permute:return nil}}
    public var outputSampleShape:[Int] {switch self {case .concat:return [3,3];case .add:return [2,3];case .stack:return [2,2,3];case .permute:return [3,4,2]}}
    public var configuration:[String:JSONValue] {switch self {case .concat:return ["dim":1];case .add:return ["alpha":.number(-0.5),"sample_broadcast":true];case .stack:return ["dim":2];case .permute:return ["dims":.array([0,2,3,1])]}}
    public var title:String {switch self {case .concat:return "Combine along an existing direction";case .add:return "Add a shared pattern";case .stack:return "Keep two sources in a new direction";case .permute:return "Reorder the sample directions"}}
    /// Concise public controls for configuration panels; definitions and wiring live elsewhere.
    public var controlInstructions:String {
        switch self {
        case .concat:return "Select Concatenate. Set Join direction to 1. Append the second source’s row after the first source’s two rows; each observation becomes a 3-by-3 table."
        case .add:return "Select Add. Set Second source multiplier to -0.5 and enable Reuse sample directions of size 1. Reuse the three-value second source in each row of the same observation; keep the 2-by-3 result."
        case .stack:return "Select Stack. Set New axis position to 2. Keep both 2-by-3 sources in a new size-2 direction between rows and columns; each observation becomes [2, 2, 3]. Position 0 comes from the first source and position 1 from the second."
        case .permute:return "Select Transpose and choose Method → Reorder axes. Set Direction order · observations first to [0, 2, 3, 1]. Keep observation direction 0 first; reorder each sample from [2, 3, 4] to [3, 4, 2] without changing its values."
        }
    }
    public var instructions:String {
        let intro="An array is a table of numbers that may have more than two directions. An axis names one direction. The first axis, 0, holds independent observations; keep it first. KwipML measures output and input influence automatically. Input influence describes how a small change affects the result; no code or handwritten derivatives are required. "
        let wiring=isBinary ? "Add the left and right Inputs, the requested block and Model Output. Connect left to a, right to b, then the block to Model Output. " : "Connect the left Input → Permute → Model Output. "
        switch self {
        case .concat:return intro+wiring+"Select Concatenate and set Join direction to 1. Each observation has a left 2-by-3 table and a right 1-by-3 table. Append the right row after the left rows, producing a 3-by-3 table. The column count stays 3. Concatenate extends an existing direction; it does not create one. Choose Check construction and compare the order of the measured values."
        case .add:return intro+wiring+"Select Add, set Second source multiplier to -0.5 and enable Reuse sample directions of size 1. Each observation has a 2-by-3 left table and a three-value right pattern. Broadcasting reuses that pattern in each row of the same observation. Multiply the pattern by -0.5, then add it to each row. Patterns never pass between observations. Choose Check construction; reused right values collect influence from both rows."
        case .stack:return intro+wiring+"Select Stack and set New axis position to 2. Each observation has two 2-by-3 tables. Stack keeps both sources by inserting a new size-2 direction between rows and columns, producing [2, 2, 3]. Position 0 in that new direction comes from left; position 1 comes from right. Unlike Concatenate, Stack creates a new direction. Choose Check construction and inspect both sources."
        case .permute:return intro+wiring+"Select Transpose, choose Method → Reorder axes, then set Direction order · observations first to [0, 2, 3, 1]. Each observation starts with sample directions [2, 3, 4]; the result has [3, 4, 2]. List every full-array direction once, keeping observation direction 0 first. The values stay attached to their original coordinates. This is different from Reshape, which reads numbers in order and changes their grouping. Choose Check construction to inspect measured order and input influence."
        }
    }
    public var task:String {instructions}
}

public enum TensorCompositionAssessment {
    public static func plan(task:TensorCompositionLearningTask,runtimeVersion:String,seed:UInt64)throws->LearningAssessmentPlan {
        var state=seed
        func tensor(_ shape:[Int])->LearningTensor {
            .init(shape:shape,values:(0..<shape.reduce(1,*)).map {_ in
                state=state &* 6364136223846793005 &+ 1442695040888963407
                return Double(Int((state >> 32)%33)-16)/4
            })
        }
        let fixtures=try (0..<4).map {index -> LearningFixture in
            let rows=index%3+1,left=tensor([rows]+task.leftSampleShape),right=task.rightSampleShape.map {tensor([rows]+$0)}
            let reference=try TensorCompositionLessonReference(left:left,right:right,task:task),upstream=tensor(reference.output.shape)
            let gradients=try reference.gradients(for:upstream)
            var inputs=["left":left,"upstream":upstream],expected=["output":reference.output,"leftGradient":gradients.left]
            if let right {inputs["right"]=right}
            if let value=gradients.right {expected["rightGradient"]=value}
            return .init(id:"\(task.rawValue)-\(seed)-\(index)",inputs:inputs,expected:expected)
        }
        func criterion(_ key:String,_ description:String)->LearningCriterion {
            .init(id:key,description:description,nextStep:task.instructions,rule:.tensor(key:key,absoluteTolerance:1e-6,relativeTolerance:1e-6))
        }
        var criteria=[criterion("output","Keep values in the requested directions and source order, separately for each observation."),criterion("leftGradient","Trace measured influence to the original left input positions.")]
        if task.isBinary {criteria.append(criterion("rightGradient","Trace measured influence to the right source, adding its repeated uses."))}
        let plan=LearningAssessmentPlan(id:"assessment."+task.rawValue,objectiveID:"tensor-composition."+task.rawValue,task:task.instructions,runtime:.init(backend:"pytorch",version:runtimeVersion,capabilities:[task.componentID:ContractVersion()]),fixtureSeed:seed,fixtures:fixtures,criteria:criteria)
        try plan.validate();return plan
    }
}
