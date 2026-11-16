import Foundation

/// Scalar coordinate mapping; no production adapter supplies expected values.
public struct TensorCompositionLessonReference {
    public let output:LearningTensor
    private let left:LearningTensor,right:LearningTensor?
    private let leftPositions:[Int?],rightPositions:[Int?],rightCoefficient:Double
    public init(left:LearningTensor,right:LearningTensor?=nil,task:TensorCompositionLearningTask)throws {
        guard left.wellFormed,(1...3).contains(left.shape[0]),left.shape == [left.shape[0]]+task.leftSampleShape else {throw LearningContractError("Left lesson input must match the public task shape.")}
        if let shape=task.rightSampleShape {
            guard let right,right.wellFormed,right.shape == [left.shape[0]]+shape else {throw LearningContractError("Right lesson input must match its public shape and observation count.")}
        } else if right != nil {throw LearningContractError("Permute has only one input.")}
        self.left=left;self.right=right;let coefficient=task == .add ? -0.5:1;rightCoefficient=coefficient
        let shape=[left.shape[0]]+task.outputSampleShape
        func coordinates(_ flat:Int)->[Int] {var i=flat,result=Array(repeating:0,count:shape.count);for axis in shape.indices.reversed() {result[axis]=i%shape[axis];i/=shape[axis]};return result}
        func index(_ coords:[Int],_ dimensions:[Int])->Int {zip(coords,dimensions).reduce(0) {$0*$1.1+$1.0}}
        var values:[Double]=[],aPositions:[Int?]=[],bPositions:[Int?]=[]
        for i in 0..<shape.reduce(1,*) {
            var p=coordinates(i);let a:Int?,b:Int?
            switch task {
            case .concat:
                if p[1]<2 {a=index(p,left.shape);b=nil} else {p[1]-=2;a=nil;b=index(p,right!.shape)}
            case .add:a=index(p,left.shape);b=index([p[0],p[2]],right!.shape)
            case .stack:
                let source=p.remove(at:2)
                if source == 0 {a=index(p,left.shape);b=nil} else {a=nil;b=index(p,right!.shape)}
            case .permute:a=index([p[0],p[3],p[1],p[2]],left.shape);b=nil
            }
            aPositions.append(a);bPositions.append(b)
            values.append((a.map {left.values[$0]} ?? 0)+(b.map {right!.values[$0]*coefficient} ?? 0))
        }
        output = .init(shape:shape,values:values);leftPositions=aPositions;rightPositions=bPositions
    }
    public func gradients(for upstream:LearningTensor)throws->(left:LearningTensor,right:LearningTensor?) {
        guard upstream.wellFormed,upstream.shape == output.shape else {throw LearningContractError("Output influence must match the task result shape.")}
        var a=Array(repeating:0.0,count:left.values.count),b=right.map {Array(repeating:0.0,count:$0.values.count)}
        for i in upstream.values.indices {
            if let position=leftPositions[i] {a[position]+=upstream.values[i]}
            if let position=rightPositions[i] {b![position]+=upstream.values[i]*rightCoefficient}
        }
        return (.init(shape:left.shape,values:a),right.map {.init(shape:$0.shape,values:b!)})
    }
    public func leftGradient(for upstream:LearningTensor)throws->LearningTensor {try gradients(for:upstream).left}
    public func rightGradient(for upstream:LearningTensor)throws->LearningTensor? {try gradients(for:upstream).right}
    public static let explanation="Concatenate appends two sources along an existing sample direction. Stack adds a direction to keep two equal-shaped sources separate. Add combines corresponding numbers, optionally multiplying the second source first. Sample broadcasting reuses size-one directions only inside each observation; input influence adds across repeated uses. Permute lists every direction once and reorders coordinates without changing which numbers belong together. Observations stay first and never mix. These blocks own no learned weights. The lesson uses independently calculated scalar coordinates and float64 arithmetic for its checker; production execution keeps its reviewed dtype and device boundaries."
}
