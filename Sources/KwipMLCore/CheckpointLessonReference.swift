import Foundation

public struct CheckpointLessonReference {
    public let expected:[String:LearningTensor]
    public init(task:CheckpointLearningTask,inputs:[String:LearningTensor])throws {
        let shapes=["trainingInputs":[8,2],"trainingTargets":[8,2],"heldoutInputs":[4,2],"heldoutTargets":[4,2],"seed":[1]]
        guard Set(inputs.keys)==Set(shapes.keys),inputs.allSatisfy({$0.value.wellFormed && $0.value.shape == shapes[$0.key]}),let seed=inputs["seed"]?.values.first,seed.rounded()==seed,(0...4_294_967_295).contains(seed) else {throw LearningContractError("Use the supplied bounded checkpoint rows and seed.")}
        func zero(_ shape:[Int])->LearningTensor {.init(shape:shape,values:Array(repeating:0,count:shape.reduce(1,*)))}
        if task == .restoreWeights {expected=["restoredParameterDifference":zero([2,3]),"restoredPredictionDifference":zero([4,2]),"childUpdateDifference":zero([2,3]),"freshMemoryDifference":zero([2,3]),"parentByteDifference":zero([1]),"completedEpochs":.init(shape:[2],values:[2,1])]}
        else {expected=["continuationStateDifference":zero([1]),"continuationHistoryDifference":zero([1]),"continuationRandomDifference":zero([1]),"parentByteDifference":zero([1]),"continuationEpochs":.init(shape:[3],values:[2,3,4]),"completedEpochs":.init(shape:[3],values:[2,4,4])]}
    }
    public static func label(_ key:String)->String {
        switch key {
        case "restoredParameterDifference":return "Reopen the saved learned settings"
        case "restoredPredictionDifference":return "Use those settings on different rows"
        case "childUpdateDifference":return "Start the new adjustments from saved settings"
        case "freshMemoryDifference":return "Start new optimizer memory for fine-tuning"
        case "continuationStateDifference":return "Continue learned settings and training memory exactly"
        case "continuationHistoryDifference":return "Keep the complete training history"
        case "continuationRandomDifference":return "Continue the saved random choices"
        case "continuationEpochs":return "Execute only the remaining passes"
        case "parentByteDifference":return "Keep the saved parent unchanged"
        default:return "Complete the requested number of passes"
        }
    }
    /// Independent scalar MSE derivative and SGD update, starting with zero optimizer memory.
    public static func affineUpdate(parameters:[Double],rows:[[Double]],targets:[[Double]],batchSize:Int = 4,epochs:Int = 1)throws->(parameters:[Double],memory:[Double]) {
        guard parameters.count==6,parameters.allSatisfy(\.isFinite),rows.count==targets.count,!rows.isEmpty,rows.count<=8,rows.allSatisfy({$0.count==2 && $0.allSatisfy(\.isFinite)}),targets.allSatisfy({$0.count==2 && $0.allSatisfy(\.isFinite)}),(1...8).contains(batchSize),(1...8).contains(epochs) else {throw LearningContractError("Use bounded affine settings and paired rows.")}
        var p=parameters,m=Array(repeating:0.0,count:6),first=true
        for _ in 0..<epochs {for begin in stride(from:0,to:rows.count,by:batchSize) {
            let end=min(begin+batchSize,rows.count);var g=Array(repeating:0.0,count:6)
            for r in begin..<end {for o in 0..<2 {let j=o*3;let d=(p[j]*rows[r][0]+p[j+1]*rows[r][1]+p[j+2]-targets[r][o])/Double(end-begin);g[j]+=d*rows[r][0];g[j+1]+=d*rows[r][1];g[j+2]+=d}}
            for i in 0..<6 {m[i]=first ? g[i]:0.5*m[i]+g[i];p[i]-=0.1*m[i]};first=false
        }}
        guard p.allSatisfy(\.isFinite),m.allSatisfy(\.isFinite) else {throw LearningContractError("Reference adjustments exceeded finite bounds.")}
        return (p,m)
    }
}
