import Foundation

public enum AdaptivePoolingLearningTask:String,CaseIterable,Codable {
    case averageTwo="avg-pool2d-windows",adaptiveAverageOne="adaptive-avg-pool1d-bins",adaptiveAverageTwo="adaptive-avg-pool2d-bins",adaptiveMaximumTwo="adaptive-max-pool2d-bins"
    public var isAverage:Bool {self != .adaptiveMaximumTwo}
    public var dimensions:Int {self == .adaptiveAverageOne ? 1:2}
    public var componentID:String {switch self {case .averageTwo:return "avg_pool2d";case .adaptiveAverageOne:return "adaptive_avg_pool1d";case .adaptiveAverageTwo:return "adaptive_avg_pool2d";case .adaptiveMaximumTwo:return "adaptive_max_pool2d"}}
    public var title:String {switch self {case .averageTwo:return "Average a map · boundary windows";case .adaptiveAverageOne:return "Choose a length · shared average bins";case .adaptiveAverageTwo:return "Choose a map height · keep its width";case .adaptiveMaximumTwo:return "Choose a map height · winning positions"}}
    public var sampleShape:[Int] {self == .averageTwo ? [2,6,6]:dimensions == 1 ? [2,5]:[2,5,4]}
    public var outputSampleShape:[Int] {self == .averageTwo ? [2,4,3]:dimensions == 1 ? [2,3]:[2,3,4]}
    public var configuration:[String:JSONValue] {
        switch self {
        case .averageTwo:return ["kernel_size":.array([3,3]),"stride":.array([2,3]),"padding":.array([1,1]),"ceil_mode":true,"count_include_pad":true,"divisor_override":.null]
        case .adaptiveAverageOne:return ["output_size":3]
        case .adaptiveAverageTwo:return ["output_height":3,"output_width":.null]
        case .adaptiveMaximumTwo:return ["output_height":3,"output_width":.null,"return_indices":true]
        }
    }
    public var instructions:String {
        let block=self == .averageTwo ? "AvgPool2d":self == .adaptiveAverageOne ? "AdaptiveAvgPool1d":self == .adaptiveAverageTwo ? "AdaptiveAvgPool2d":"AdaptiveMaxPool2d"
        let base="Connect Input → \(block) → Model Output. Each observation has two channels, meaning separate measurement fields. Pooling combines nearby positions within each channel, keeping channels and observations separate. "
        let detail:String
        switch self {
        case .averageTwo:detail="Each channel is a map with six rows and six columns. Set Window [height, width] to 3, 3; Stride [height, width] to 2, 3; and Padding [height, width] to 1, 1. A window is a group of nearby cells. Stride is how far it moves down and across; padding declares extra zero cells at the edges. Enable Include partial final window and Include padding in average divisor. Keep Custom divisor empty to use the window count. A divisor is the number the total is divided by. The output has four rows and three columns. A partial window counts declared padding, but never extra cells beyond it. "
        case .adaptiveAverageOne:detail="Each channel has five positions. Set Output length to 3. Adaptive pooling chooses groups, called bins, to produce the requested number of positions. Each bin averages its included values. When five positions become three, neighboring bins share boundary positions; values can influence more than one output. It is not simple row sampling. "
        case .adaptiveAverageTwo:detail="Each channel is a map with five rows and four columns. Set Output height to 3 and leave Output width empty to keep its input size. Each output cell averages a group of input cells, called a bin. Keeping the width preserves four columns; neighboring row bins can share boundary rows. The output has three rows and four columns. "
        case .adaptiveMaximumTwo:detail="Each channel is a map with five rows and four columns. Set Output height to 3, leave Output width empty to keep its input size, and enable Expose winning indices. Each output cell selects the largest real value in its input group, called a bin. Bins can share boundary rows. Equal winners use the first position in reading order. An index is a zero-based position within one channel’s input map; it is separate from the value. "
        }
        return base+detail+"Choose Check construction to inspect measured values and input influence. A gradient records how a small input change affects the result. KwipML measures this automatically; no code or handwritten derivatives are required. These blocks have no learned weights."
    }
}

public enum AdaptivePoolingAssessment {
    public static func plan(task:AdaptivePoolingLearningTask,runtimeVersion:String,seed:UInt64)throws->LearningAssessmentPlan {
        var state=seed
        func next()->Double {state=state &* 6364136223846793005 &+ 1442695040888963407;return Double(Int((state >> 32)%19)-11)/4}
        let fixtures=try (0..<4).map {i -> LearningFixture in
            let shape=[i%2+1]+task.sampleShape
            let input=LearningTensor(shape:shape,values:(0..<shape.reduce(1,*)).map {_ in i == 0 ? -2:next()})
            let reference=try AdaptivePoolingLessonReference(input:input,task:task)
            let upstream=LearningTensor(shape:reference.output.shape,values:(0..<reference.output.values.count).map {_ in next()})
            var expected:[String:LearningTensor]=["output":reference.output,"inputGradient":try reference.inputGradient(for:upstream)]
            if let indices=reference.indices {expected["indices"]=indices}
            return .init(id:"\(task.rawValue)-\(i)",inputs:["x":input,"upstream":upstream],expected:expected)
        }
        var criteria=[LearningCriterion(id:"output",description:"Combine the requested input groups, keeping channels and observations separate.",nextStep:"Check the requested pooling block and output sizes. For map windows use the specified padding, partial-window and divisor settings; for adaptive maps keep input width.",rule:.tensor(key:"output",absoluteTolerance:1e-6,relativeTolerance:1e-6)),LearningCriterion(id:"inputGradient",description:"Measure input influence, adding contributions when groups share positions.",nextStep:"The checker measures influence automatically. Average shares it among contributing values using the divisor; maximum routes it to the first winning position.",rule:.tensor(key:"inputGradient",absoluteTolerance:1e-6,relativeTolerance:1e-6))]
        if !task.isAverage {criteria.append(.init(id:"indices",description:"Report the first winning input position within each channel’s map.",nextStep:"Enable Expose winning indices. Positions start at zero in reading order; equal values choose the first position. Indices are positions, not averages or learned values.",rule:.tensor(key:"indices",absoluteTolerance:0,relativeTolerance:0)))}
        let plan=LearningAssessmentPlan(id:"assessment."+task.rawValue,objectiveID:"pooling."+task.rawValue,task:task.instructions,runtime:.init(backend:"pytorch",version:runtimeVersion,capabilities:[task.componentID:ContractVersion()]),fixtureSeed:seed,fixtures:fixtures,criteria:criteria)
        try plan.validate();return plan
    }
}
