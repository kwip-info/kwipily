import Foundation

/// Independent coordinate arithmetic, never a call to the measured routing operator.
public struct TensorRoutingLessonReference {
    public let output:LearningTensor
    private let inputs:[String:LearningTensor]
    private let task:TensorRoutingLearningTask
    public init(inputs:[String:LearningTensor],task:TensorRoutingLearningTask)throws {
        guard Set(inputs.keys) == Set(task.sourceRoles),let first=inputs[task.sourceRoles[0]],let rows=first.shape.first,(1...3).contains(rows) else {throw LearningContractError("Use the declared source fixtures and one to three observations.")}
        for role in task.sourceRoles {let t=inputs[role]!;guard t.wellFormed, t.shape == [rows]+task.sampleShapes[role]! else {throw LearningContractError("Source fixture shape differs from the public task.")}}
        if let selectors=inputs["indices"] {guard selectors.values.allSatisfy({$0.rounded() == $0 && $0>=0 && $0<4}) else {throw LearningContractError("Selectors must be whole positions from 0 through 3.")}}
        self.inputs=inputs;self.task=task
        var values:[Double]=[]
        if task == .einsumCombine {
            let a=inputs["left"]!.values,b=inputs["right"]!.values
            for n in 0..<rows {for row in 0..<2 {for col in 0..<2 {var value=0.0;for k in 0..<3 {value += a[n*6+row*3+k]*b[n*6+k*2+col]};values.append(value)}}}
        } else {
            let x=inputs["x"]!.values
            for index in 0..<(rows*task.outputSampleShape.reduce(1,*)) {values.append(x[Self.sourceIndex(task:task,index:index,selectors:inputs["indices"]?.values)])}
        }
        output = .init(shape:[rows]+task.outputSampleShape,values:values)
    }
    private static func sourceIndex(task:TensorRoutingLearningTask,index:Int,selectors:[Double]?)->Int {
        switch task {
        case .splitSize:return (index/5)*5+[4,0,1,2,3][index%5]
        case .splitSections:return (index/5)*5+[4,1,2,3,0][index%5]
        case .chunk:return index*13+12
        case .gather:return (index/3)*4+Int(selectors![index])
        case .einsumDiagonal:return (index/3)*9+(index%3)*4
        case .einsumCombine:preconditionFailure("Binary table product has two source coordinates.")
        }
    }
    public func gradients(for upstream:LearningTensor)throws->[String:LearningTensor] {
        guard upstream.wellFormed, upstream.shape == output.shape else {throw LearningContractError("Output influence must match the requested result shape.")}
        if task == .einsumCombine {
            let a=inputs["left"]!,b=inputs["right"]!;var ga=Array(repeating:0.0,count:a.values.count),gb=Array(repeating:0.0,count:b.values.count)
            for n in 0..<output.shape[0] {for row in 0..<2 {for col in 0..<2 {let weight=upstream.values[n*4+row*2+col];for k in 0..<3 {let ai=n*6+row*3+k,bi=n*6+k*2+col;ga[ai]+=weight*b.values[bi];gb[bi]+=weight*a.values[ai]}}}}
            return ["leftGradient":.init(shape:a.shape,values:ga),"rightGradient":.init(shape:b.shape,values:gb)]
        }
        let x=inputs["x"]!;var gradient=Array(repeating:0.0,count:x.values.count)
        for index in upstream.values.indices {gradient[Self.sourceIndex(task:task,index:index,selectors:inputs["indices"]?.values)]+=upstream.values[index]}
        return ["inputGradient":.init(shape:x.shape,values:gradient)]
    }
}
