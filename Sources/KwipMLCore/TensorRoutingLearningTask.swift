import Foundation

/// Public controls use familiar table/position language; private fixture values remain assessment-owned.
public enum TensorRoutingLearningTask:String,CaseIterable,Codable {
    case splitSize="split-size-order",splitSections="split-sections-order",chunk="chunk-uneven-parts",gather="gather-repeated-positions",einsumCombine="einsum-combine-tables",einsumDiagonal="einsum-diagonal-view"
    public var kind:String {switch self {case .splitSize,.splitSections:return "split";case .chunk:return "chunk";case .gather:return "gather";case .einsumCombine,.einsumDiagonal:return "einsum"}}
    public var componentID:String {kind}
    public var sourceRoles:[String] {self == .einsumCombine ? ["left","right"] : self == .gather ? ["x","indices"] : ["x"]}
    public var sampleShapes:[String:[Int]] {switch self {case .splitSize,.splitSections:return ["x":[2,5]];case .chunk:return ["x":[2,13]];case .gather:return ["x":[2,4],"indices":[2,3]];case .einsumCombine:return ["left":[2,3],"right":[3,2]];case .einsumDiagonal:return ["x":[3,3]]}}
    public var outputSampleShape:[Int] {switch self {case .splitSize,.splitSections:return [2,5];case .chunk:return [2,1];case .gather:return [2,3];case .einsumCombine:return [2,2];case .einsumDiagonal:return [3]}}
    public var configuration:[String:JSONValue] {switch self {
    case .splitSize:return ["mode":.string("size"),"dim":2,"split_size":2,"sections":.array([1,1])]
    case .splitSections:return ["mode":.string("sections"),"dim":2,"split_size":2,"sections":.array([1,3,1])]
    case .chunk:return ["chunks":6,"dim":2]
    case .gather:return ["dim":2]
    case .einsumCombine:return ["pattern":.string("multiply"),"equation":.string("nab,nbc->nac")]
    case .einsumDiagonal:return ["pattern":.string("diagonal"),"equation":.string("naa->na")]
    }}
    public var title:String {switch self {case .splitSize:return "Split and reorder table sections";case .splitSections:return "Choose unequal table sections";case .chunk:return "Find the final uneven part";case .gather:return "Pick repeated table positions";case .einsumCombine:return "Combine two tables";case .einsumDiagonal:return "Read a table’s diagonal"}}
    public var controlInstructions:String {switch self {
    case .splitSize:return "Add Split tensor, choose Method → Split sizes, then set Split mode to size, Sample direction to 2 and Split size to 2. Join part2, part0, then part1 with two Concatenate blocks; set each Join direction to 2."
    case .splitSections:return "Add Split tensor, choose Method → Split sizes, then set Split mode to sections, Sample direction to 2 and Section sizes to [1, 3, 1]. Join part2, part1, then part0 with two Concatenate blocks; set each Join direction to 2."
    case .chunk:return "Add Split tensor and choose Method → Request chunks. Set Requested parts to 6 and Sample direction to 2. Connect part4 to Model Output. The thirteen columns form five actual parts: [3, 3, 3, 3, 1]; there is no part5."
    case .gather:return "Set Sample direction to 2. Connect the number source to source and the provided whole-number positions to indices. Repeated positions reuse a source value."
    case .einsumCombine:return "Set Arrangement preset to multiply. Connect the 2-by-3 left table to a and the 3-by-2 right table to b; each observation produces a 2-by-2 table."
    case .einsumDiagonal:return "Set Arrangement preset to diagonal. Connect the 3-by-3 table to a; each observation produces the three values from its top-left to bottom-right diagonal."
    }}
    public var instructions:String {
        let terms="An array is a table of numbers that may have more than two directions. Direction 0 holds independent observations; keep it first. A port is a named connection point. Parts are numbered from part0. Input influence shows how a small change affects the result; KwipML measures it for you. "
        let build:String
        switch self {case .splitSize,.splitSections:build="Build: connect the x Input to Split tensor using Split sizes. Add two Concatenate blocks to rejoin the named parts, then connect the final result to Model Output. "
        case .chunk:build="Build: place Split tensor, choose Request chunks and connect x Input → Chunk, then connect the selected part → Model Output. "
        case .gather:build="Build: connect the x and indices Inputs to Gather’s matching ports, then Gather → Model Output. Whole-number positions start at 0; they select values rather than numbers to train. "
        case .einsumCombine:build="Build: connect left and right Inputs to Einsum, then Einsum → Model Output. Multiplying tables combines each left row with each right column by multiplying paired values and adding the products. "
        case .einsumDiagonal:build="Build: connect x Input → Einsum → Model Output. A diagonal joins positions with the same row and column number. "}
        return terms+build+"Configure: "+controlInstructions+" Check: choose Check construction to inspect the measured values and which original source positions influenced them. No code or handwritten derivatives are required."
    }
    public var task:String {instructions}
}

public enum TensorRoutingAssessment {
    public static func plan(task:TensorRoutingLearningTask,runtimeVersion:String,seed:UInt64)throws->LearningAssessmentPlan {
        var state=seed
        func next()->UInt64 {state=state &* 6364136223846793005 &+ 1442695040888963407;return state >> 32}
        func tensor(_ shape:[Int])->LearningTensor {.init(shape:shape,values:(0..<shape.reduce(1,*)).map {_ in Double(Int(next()%33)-16)/4})}
        let fixtures=try (0..<4).map {index->LearningFixture in
            let rows=index%3+1
            var inputs:[String:LearningTensor]=[:]
            for role in task.sourceRoles {
                let shape=[rows]+task.sampleShapes[role]!
                if role == "indices" {
                    var values:[Double]=[]
                    for _ in 0..<(rows*2) {let position=Double(next()%4),other=Double(next()%4);values += [position,position,other]}
                    inputs[role] = .init(shape:shape,values:values)
                } else {inputs[role]=tensor(shape)}
            }
            let reference=try TensorRoutingLessonReference(inputs:inputs,task:task),upstream=tensor(reference.output.shape)
            var expected=try reference.gradients(for:upstream);expected["output"]=reference.output;inputs["upstream"]=upstream
            return .init(id:"\(task.rawValue)-\(seed)-\(index)",inputs:inputs,expected:expected)
        }
        let keys=task == .einsumCombine ? ["output","leftGradient","rightGradient"] : ["output","inputGradient"]
        let criteria=keys.map {key in LearningCriterion(id:key,description:key == "output" ? "Produce the requested table values and order independently for each observation." : "Trace influence to the original number source, adding repeated uses.",nextStep:task.controlInstructions,rule:.tensor(key:key,absoluteTolerance:1e-6,relativeTolerance:1e-6))}
        let plan=LearningAssessmentPlan(id:"assessment."+task.rawValue,objectiveID:"tensor-routing."+task.rawValue,task:task.instructions,runtime:.init(backend:"pytorch",version:runtimeVersion,capabilities:[task.componentID:ContractVersion()]),fixtureSeed:seed,fixtures:fixtures,criteria:criteria)
        try plan.validate();return plan
    }
}
