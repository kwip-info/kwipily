import Foundation

public enum SpatialPatchLearningTask:String,CaseIterable,Codable {
    case unfoldOverlap="unfold-overlapping-windows",unfoldSpaced="unfold-padded-spaced-windows",foldOverlap="fold-overlap-addition",foldHoles="fold-uncovered-positions"
    public var componentID:String {isUnfold ? "unfold":"fold"}
    public var isUnfold:Bool {self == .unfoldOverlap || self == .unfoldSpaced}
    public var sampleShape:[Int] {isUnfold ? [2,3,4]:self == .foldOverlap ? [8,20]:[8,4]}
    public var outputSampleShape:[Int] {isUnfold ? [8,6]:self == .foldOverlap ? [2,3,4]:[2,5,5]}
    public var configuration:[String:JSONValue] {
        var values:[String:JSONValue]=["kernel_size":.array([2,2]),"stride":.array([1,1]),"dilation":.array([1,1]),"padding":.array([0,0])]
        switch self {
        case .unfoldOverlap:break
        case .unfoldSpaced:values["stride"] = .array([2,2]);values["dilation"] = .array([2,1]);values["padding"] = .array([1,1])
        case .foldOverlap:values["output_size"] = .array([3,4]);values["padding"] = .array([1,1])
        case .foldHoles:values["output_size"] = .array([5,5]);values["stride"] = .array([3,3])
        }
        return values
    }
    public var title:String {switch self {case .unfoldOverlap:return "Turn nearby cells into patch columns";case .unfoldSpaced:return "Extract spaced patches with zero borders";case .foldOverlap:return "Rebuild a grid by adding overlapping parts";case .foldHoles:return "See where patch parts leave gaps"}}
    public var controlInstructions:String {switch self {
    case .unfoldOverlap:return "Set Patch size to [2, 2], Spacing between patches to [1, 1], Spacing within a patch to [1, 1] and Zero padding to [0, 0]. Each observation becomes eight packed rows and six patch columns."
    case .unfoldSpaced:return "Set Patch size to [2, 2], Spacing between patches to [2, 2], Spacing within a patch to [2, 1] and Zero padding to [1, 1]. The result still has eight packed rows and six columns; its values and visited cells differ."
    case .foldOverlap:return "Set Output grid size to [3, 4], Patch size to [2, 2], both spacing settings to [1, 1] and Zero padding to [1, 1]. Add overlapping contributions; do not average them. Border-only entries make no contribution."
    case .foldHoles:return "Set Output grid size to [5, 5], Patch size to [2, 2], Spacing between patches to [3, 3], Spacing within a patch to [1, 1] and Zero padding to [0, 0]. The middle row and column are uncovered and stay zero."
    }}
    public var instructions:String {
        let terms="An array is a table of numbers with named directions. A channel is a separate grid processed in parallel. An observation is one independent example; direction 0 keeps examples separate. A patch is a small rectangle selected from a grid. Stride is the spacing between patch starts; dilation is the spacing between cells inside a patch. Padding adds declared border zeros. Input influence, also called a gradient, describes how a small change affects the result; KwipML measures it for you. "
        let build="Build: connect Input → \(isUnfold ? "Unfold":"Fold") → Model Output. "
        let order=isUnfold ? "Unfold packs each patch into one column: channel first, then patch row and patch column. Patch starts are read row by row. " : "Fold reads the same packed order and adds entries that land on the same grid position. It does not guess uncovered values or undo overlap by averaging. "
        return terms+build+"Configure: "+controlInstructions+" "+order+"Check: choose Check construction and compare output and input-influence tables. No code or handwritten derivatives are required."
    }
    public var task:String {instructions}
}

public enum SpatialPatchAssessment {
    public static func plan(task:SpatialPatchLearningTask,runtimeVersion:String,seed:UInt64)throws->LearningAssessmentPlan {
        var state=seed
        func tensor(_ shape:[Int])->LearningTensor {.init(shape:shape,values:(0..<shape.reduce(1,*)).map {_ in state=state &* 6364136223846793005 &+ 1442695040888963407;return Double(Int((state >> 32)%33)-16)/4})}
        let fixtures=try (0..<4).map {index->LearningFixture in
            let rows=index%3+1,input=tensor([rows]+task.sampleShape),reference=try SpatialPatchLessonReference(input:input,task:task),upstream=tensor(reference.output.shape)
            return .init(id:"\(task.rawValue)-\(seed)-\(index)",inputs:["x":input,"upstream":upstream],expected:["output":reference.output,"inputGradient":try reference.inputGradient(for:upstream)])
        }
        let criteria=[LearningCriterion(id:"output",description:task.isUnfold ? "Pack visited cells and border zeros into the requested patch columns." : "Add overlapping entries into the requested grid and leave uncovered cells zero.",nextStep:task.controlInstructions,rule:.tensor(key:"output",absoluteTolerance:1e-6,relativeTolerance:1e-6)),LearningCriterion(id:"inputGradient",description:"Trace measured influence to the original input positions, adding repeated uses and ignoring border-only entries.",nextStep:task.controlInstructions,rule:.tensor(key:"inputGradient",absoluteTolerance:1e-6,relativeTolerance:1e-6))]
        let plan=LearningAssessmentPlan(id:"assessment."+task.rawValue,objectiveID:"spatial-patches."+task.rawValue,task:task.instructions,runtime:.init(backend:"pytorch",version:runtimeVersion,capabilities:[task.componentID:ContractVersion()]),fixtureSeed:seed,fixtures:fixtures,criteria:criteria)
        try plan.validate();return plan
    }
}
