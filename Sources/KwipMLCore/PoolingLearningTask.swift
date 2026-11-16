import Foundation

public enum PoolingLearningTask:String,CaseIterable,Codable {
    case maxOne="max-pool1d-windows",maxTwo="max-pool2d-windows",maxThree="max-pool3d-windows",averageOne="avg-pool1d-windows"
    public var isAverage:Bool {self == .averageOne}
    public var dimensions:Int {self == .maxTwo ? 2 : self == .maxThree ? 3 : 1}
    public var componentID:String {isAverage ? "avg_pool1d":"max_pool\(dimensions)d"}
    public var title:String {isAverage ? "Average windows · partial boundaries":"MaxPool\(dimensions)d · winning positions"}
    public var sampleShape:[Int] {[2]+Array(repeating:isAverage ? 6:4,count:dimensions)}
    public var configuration:[String:JSONValue] {
        func axes(_ value:Int)->JSONValue {dimensions == 1 ? .integer(Int64(value)):.array(Array(repeating:.integer(Int64(value)),count:dimensions))}
        var result:[String:JSONValue]=["kernel_size":axes(3),"stride":axes(2),"padding":axes(1),"ceil_mode":true]
        if isAverage {result["count_include_pad"]=true} else {result["dilation"]=axes(1);result["return_indices"]=true}
        return result
    }
    public var instructions:String {
        let base="Connect Input → \(isAverage ? "AvgPool1d":"MaxPool\(dimensions)d") → Model Output. Select the pooling block and set Window length 3 (or Window dimensions all 3), Stride 2 and Padding 1 in every spatial direction. Enable Include partial final window to keep the final partial window. A channel is a separate measurement field; pooling treats the two channels and each observation independently. A window is the nearby positions considered together. Stride moves the window; padding declares extra boundary positions. Choose Check construction to inspect output and input influence automatically; no code or handwritten derivatives are required."
        if isAverage {return base+" Enable Include padding in average divisor. Each channel has six input positions and four output positions. Average adds the real values and divides by the window’s count, including declared zero padding. The first window divides by 3; the last partial window divides by 2 because the further position lies beyond declared padding. Disabling Include padding in average divisor would instead divide only by actual input positions."}
        return base+" Set Dilation 1 and enable Expose winning indices. Dilation spaces the positions within a window. Each spatial direction has four input positions and three output positions. Max pooling chooses the largest real value; boundary padding cannot win, even when all inputs are negative. Equal winners use the first position in spatial reading order. Indices report zero-based positions within each channel’s input, not learned values. Input influence returns to the winning positions; repeated wins add their contributions."
    }
}
public enum PoolingAssessment {
    public static func plan(task:PoolingLearningTask,runtimeVersion:String,seed:UInt64)throws->LearningAssessmentPlan {
        var state=seed
        func next()->Double {state=state &* 6364136223846793005 &+ 1442695040888963407;return Double(Int((state >> 32)%17)-12)/4}
        let fixtures=try (0..<4).map {i -> LearningFixture in
            let shape=[i%2+1]+task.sampleShape
            let input=LearningTensor(shape:shape,values:(0..<shape.reduce(1,*)).map {_ in i == 0 ? -2:next()})
            let reference=try PoolingLessonReference(input:input,task:task)
            let upstream=LearningTensor(shape:reference.output.shape,values:(0..<reference.output.values.count).map {_ in next()})
            var expected:[String:LearningTensor]=["output":reference.output,"inputGradient":try reference.inputGradient(for:upstream)]
            if let indices=reference.indices {expected["indices"]=indices}
            return .init(id:"\(task.rawValue)-\(i)",inputs:["x":input,"upstream":upstream],expected:expected)
        }
        var criteria=[LearningCriterion(id:"output",description:task.isAverage ? "Average windows using the declared padding count, including the final partial window.":"Select each window’s largest real value, keeping channels and observations separate.",nextStep:"Select the requested pooling block. Check Window length 3 (or Window dimensions all 3), Stride 2, Padding 1 and enabled Include partial final window; use the requested averaging or winner-index setting.",rule:.tensor(key:"output",absoluteTolerance:1e-6,relativeTolerance:1e-6)),LearningCriterion(id:"inputGradient",description:"Trace measured influence to the contributing input positions, adding repeated uses.",nextStep:"Overlapping windows reuse input positions. Max sends influence to its winner; Average shares influence using the window divisor. KwipML measures these effects automatically.",rule:.tensor(key:"inputGradient",absoluteTolerance:1e-6,relativeTolerance:1e-6))]
        if !task.isAverage {criteria.append(.init(id:"indices",description:"Record the first winning spatial input position, using zero-based indices.",nextStep:"Enable Expose winning indices. Padding cannot win; equal real values choose the first spatial position. Each index is within one input channel, separate from observation and channel numbers.",rule:.tensor(key:"indices",absoluteTolerance:0,relativeTolerance:0)))}
        let plan=LearningAssessmentPlan(id:"assessment."+task.rawValue,objectiveID:"pooling."+task.rawValue,task:task.instructions,runtime:.init(backend:"pytorch",version:runtimeVersion,capabilities:[task.componentID:ContractVersion()]),fixtureSeed:seed,fixtures:fixtures,criteria:criteria)
        try plan.validate();return plan
    }
}
