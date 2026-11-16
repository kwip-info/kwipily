import Foundation

/// Scalar reference arithmetic never runs autograd, an optimizer or a second trainer.
public struct GradientLessonReference {
    public let expected:[String:LearningTensor]
    public init(task:GradientLearningTask,inputs:[String:LearningTensor])throws {
        guard Set(inputs.keys)==["trainingInputs","trainingTargets","heldoutInputs","heldoutTargets","seed"],inputs.allSatisfy({$0.value.wellFormed && ($0.key == "seed" || $0.value.values.allSatisfy {abs($0)<=1_000_000})}),let x=inputs["trainingInputs"],let y=inputs["trainingTargets"],x.shape.count==2,y.shape.count==2 else {throw LearningContractError("Use the supplied finite training and held-out rows and seed.")}
        let i=x.shape[1],o=y.shape[1]
        // Each reviewed I/O pair uniquely fixes the hidden width.
        guard let w=GradientLearningTask.widths.first(where:{$0[0]==i && $0[2]==o}),x.shape == [5,i],y.shape == [5,o],inputs["heldoutInputs"]?.shape == [3,i],inputs["heldoutTargets"]?.shape == [3,o],inputs["seed"]?.shape == [1],let seed=inputs["seed"]?.values.first,seed.rounded()==seed,(0...4_294_967_295).contains(seed) else {throw LearningContractError("Keep the reviewed model shape and five training / three held-out rows.")}
        let rows=x.values+inputs["heldoutInputs"]!.values
        guard Set(stride(from:0,to:rows.count,by:i).map {Float(rows[$0]).bitPattern}).count==8 else {throw LearningContractError("Training and held-out feature rows must stay distinct at float32 precision.")}
        var expected=try task.measuredShapes(widths:w).mapValues {shape in LearningTensor(shape:shape,values:Array(repeating:0,count:shape.reduce(1,*)))}
        if task == .combineSmallGroups {expected["windowSchedule"] = .init(shape:[4,4],values:[1,2,4,1,1,1,1,2,2,2,4,3,2,1,1,4])}
        if task == .keepSettings {expected["recordingAvailability"] = .init(shape:[3],values:[1,0,-1])}
        self.expected=expected
    }
    public struct Derivatives {public let output:[Double];public let input:[Double];public let firstWeight:[Double]}
    public static func meanDerivatives(widths w:[Int],parameters p:[Double],rows x:[Double])throws->Derivatives {
        guard w.count==3,w.allSatisfy({(1...3).contains($0)}),!x.isEmpty,x.count%w[0]==0,x.count/w[0]<=8,p.count==w[1]*w[0]+w[1]+w[2]*w[1]+w[2],(p+x).allSatisfy({$0.isFinite && abs($0)<=1_000_000}) else {throw LearningContractError("Use bounded complete affine arrays and observation rows.")}
        let i=w[0],h=w[1],o=w[2],n=x.count/i,b1=h*i,w2=b1+h,b2=w2+o*h,denominator=Double(n*o)
        var output=[Double](),dx=Array(repeating:0.0,count:x.count),dw=Array(repeating:0.0,count:h*i)
        for r in 0..<n {
            let hidden=(0..<h).map {j in (0..<i).reduce(p[b1+j]) {$0+p[j*i+$1]*x[r*i+$1]}}
            for k in 0..<o {output.append((0..<h).reduce(p[b2+k]) {$0+p[w2+k*h+$1]*hidden[$1]})}
            for j in 0..<h {
                let influence=(0..<o).reduce(0.0) {$0+p[w2+$1*h+j]}/denominator
                for k in 0..<i {dx[r*i+k]+=influence*p[j*i+k];dw[j*i+k]+=influence*x[r*i+k]}
            }
        }
        guard (output+dx+dw).allSatisfy(\.isFinite) else {throw LearningContractError("Independent influence exceeded finite bounds.")}
        return .init(output:output,input:dx,firstWeight:dw)
    }
    public struct Adjustment {public let parameters:[Double];public let gradients:[Double];public let memory:[Double]?;public let objective:Double}
    /// One independent mean-MSE/SGD adjustment. This computes expected arithmetic only.
    public static func adjustment(widths w:[Int],parameters p:[Double],rows x:[Double],targets:[Double],learningRate:Double=0.05,momentum:Double=0,memory:[Double]?=nil,freezeFirst:Bool=false)throws->Adjustment {
        let predictions=try meanDerivatives(widths:w,parameters:p,rows:x).output
        guard targets.count==predictions.count,targets.allSatisfy({$0.isFinite && abs($0)<=1_000_000}),learningRate.isFinite,(0...1).contains(learningRate),momentum.isFinite,(0...1).contains(momentum),memory.map({$0.count==p.count && $0.allSatisfy(\.isFinite)}) ?? true else {throw LearningContractError("Use finite paired targets and bounded SGD controls/memory.")}
        let i=w[0],h=w[1],o=w[2],n=x.count/i,b1=h*i,w2=b1+h,b2=w2+o*h
        var gradient=Array(repeating:0.0,count:p.count),objective=0.0
        for r in 0..<n {
            let hidden=(0..<h).map {j in (0..<i).reduce(p[b1+j]) {$0+p[j*i+$1]*x[r*i+$1]}}
            var hiddenInfluence=Array(repeating:0.0,count:h)
            for k in 0..<o {
                let error=predictions[r*o+k]-targets[r*o+k],d=2*error/Double(n*o)
                objective+=error*error/Double(n*o);gradient[b2+k]+=d
                for j in 0..<h {gradient[w2+k*h+j]+=d*hidden[j];hiddenInfluence[j]+=d*p[w2+k*h+j]}
            }
            for j in 0..<h {gradient[b1+j]+=hiddenInfluence[j];for k in 0..<i {gradient[j*i+k]+=hiddenInfluence[j]*x[r*i+k]}}
        }
        var updated=p,remembered=memory ?? Array(repeating:0.0,count:p.count)
        for j in p.indices where !freezeFirst || j>=w2 {
            remembered[j]=momentum==0 ? gradient[j]:(memory==nil ? gradient[j]:momentum*remembered[j]+gradient[j])
            updated[j]-=learningRate*remembered[j]
        }
        guard (updated+gradient+remembered+[objective]).allSatisfy(\.isFinite) else {throw LearningContractError("Independent adjustment exceeded finite bounds.")}
        return .init(parameters:updated,gradients:gradient,memory:momentum==0 ? nil:remembered,objective:objective)
    }
    public static func label(_ key:String)->String {switch key {
    case "selectedResultDifference":return "Ask about the selected final result"
    case "inputInfluenceDifference":return "Measure how the input affects the average result"
    case "parameterInfluenceDifference":return "Measure the selected learned array's influence"
    case "gradientBufferDifference":return "Leave training gradients unchanged while asking"
    case "analysisStateDifference":return "Keep settings, random choices and optimizer memory unchanged"
    case "windowSchedule":return "Finish every group and the short last window"
    case "windowObjectiveDifference":return "Give each observation its fair share"
    case "windowGradientDifference":return "Combine the actual group contributions"
    case "windowResetDifference":return "Clear old gradients before the next adjustment"
    case "updatedSettingsDifference":return "Make one correct adjustment per window"
    case "fixedSettingsDifference":return "Keep the selected first-layer settings fixed"
    case "learningSettingsDifference":return "Let the second layer keep learning"
    case "ownerInventoryDifference","childOwnerInventoryDifference":return "Give the optimizer exactly the settings allowed to learn"
    case "recordingAvailability":return "Tell a measured zero from a calculation without recording"
    case "restoredStartingDifference":return "Start the child from the saved learned settings"
    case "freshMemoryDifference":return "Start fresh optimizer memory for fine-tuning"
    case "childSettingsDifference":return "Adjust the newly enabled settings and use the result"
    case "parentByteDifference":return "Leave the saved parent unchanged"
    default:return key
    }}
}
