import Foundation

/// Independent tiny scalar objectives. No backend constructor is used to obtain answers.
public struct LossLessonReference {
    public let raw:LearningTensor
    public let objective:LearningTensor
    public let inputInfluence:LearningTensor
    public init(task:LossLearningTask,fixture:LossLessonFixture,configuration:[String:JSONValue]?=nil,scalarization:String="mean")throws {
        let config=try LossConfiguration(componentID:task.componentID,values:configuration ?? task.configuration)
        guard ["mean","sum"].contains(scalarization),let input=fixture.inputs["prediction"],input.wellFormed,(1...3).contains(input.shape.first ?? 0),input.values.count<=72 else {throw LearningContractError("Use one to three bounded supplied observations and an explicit Mean or Sum combination.")}
        let c=config.values,n=input.shape[0],sample=Array(input.shape.dropFirst())
        var roles:Set<String> = task == .triplet ? ["prediction"]:["prediction","target"]
        if task == .ctc {roles.formUnion(["input_lengths","target_lengths"])}
        if task == .bce {for key in ["weight","pos_weight"] where c[key+"_mode"] == .string("mapped") {roles.insert(key)}}
        guard Set(fixture.inputs.keys)==roles,fixture.inputs.values.allSatisfy(\.wellFormed) else {throw LearningContractError("Use exactly the supplied prediction and typed target roles.")}
        func item(_ role:String)throws->LearningTensor {guard let tensor=fixture.inputs[role],tensor.shape.first==n else {throw LearningContractError("Every target role must preserve the same observations.")};return tensor}
        let target=task == .triplet ? nil:try item("target")
        var rawShape=[n]+sample
        var denominator=Double(input.values.count)
        if [.crossEntropy,.nll].contains(task) {
            guard sample.count==1,(2...4).contains(sample[0]),c["class_axis"] == -1 || c["class_axis"] == 1 else {throw LearningContractError("Class exercise predictions use one direction with two to four classes.")}
            let probability=task == .crossEntropy && c["target_mode"] == .string("probabilities")
            guard target!.shape == (probability ? input.shape:[n]) else {throw LearningContractError("Match desired class format to the prediction class direction.")}
            if !probability {guard target!.values.allSatisfy({$0.rounded()==$0 && ($0==Double(c["ignore_index"]!.integerValue!) || (0...Double(sample[0]-1)).contains($0))}) else {throw LearningContractError("Class targets must be integer choices or the configured ignored class.")}}
            else {for row in 0..<n {let values=Array(target!.values[row*sample[0]..<(row+1)*sample[0]]);guard values.allSatisfy({(0...1).contains($0)}),abs(values.reduce(0,+)-1)<=1e-4 else {throw LearningContractError("Desired class shares must add to one.")}}}
            let weights=Self.numbers(c["class_weights"]!)
            guard weights.isEmpty || weights.count==sample[0] else {throw LearningContractError("Class importance must match the number of choices.")}
            denominator=probability ? Double(n):target!.values.reduce(0) {sum,value in Int(value)==c["ignore_index"]!.integerValue! ? sum:sum+(weights.isEmpty ? 1:weights[Int(value)])}
            rawShape=[n]
        } else if [.mse,.l1].contains(task) {
            guard sample.count==1,target!.shape == input.shape || (c["target_broadcast"] == true && target!.shape == [n]) else {throw LearningContractError("Match desired values exactly or explicitly reuse a scalar inside each observation.")}
        } else if task == .bce {
            guard sample.count==1,target!.shape==input.shape,target!.values.allSatisfy({(0...1).contains($0)}) else {throw LearningContractError("Independent labels require matching desired shares between zero and one.")}
            for prefix in ["weight","pos_weight"] where c[prefix+"_mode"] == .string("mapped") {
                let t=try item(prefix);guard [input.shape,[n],[n,1]].contains(t.shape),t.values.allSatisfy({$0>=0}) else {throw LearningContractError("Mapped importance must align with each observation and be nonnegative.")}
            }
        } else if task == .kl {
            guard sample.count==1,(2...4).contains(sample[0]),target!.shape==input.shape else {throw LearningContractError("Compare matching class distributions.")}
            for row in 0..<n {let rowValues=Array(target!.values[row*sample[0]..<(row+1)*sample[0]]);let probabilities=c["log_target"] == true ? rowValues.map(exp):rowValues;guard probabilities.allSatisfy({(0...1).contains($0)}),abs(probabilities.reduce(0,+)-1)<=1e-4 else {throw LearningContractError("Desired shares must form a normalized distribution.")}}
        } else if task == .ctc {
            guard sample.count==2,(1...6).contains(sample[0]),(2...4).contains(sample[1]),target!.shape.count==2,(1...3).contains(target!.shape[1]),c["blank"]!.integerValue!<sample[1] else {throw LearningContractError("Use a bounded time-by-class array, padded desired labels and an available blank class.")}
            for role in ["input_lengths","target_lengths"] {guard try item(role).shape==[n] else {throw LearningContractError("Sequence lengths must be one integer per observation.")}}
            for row in 0..<n {
                let time=fixture.inputs["input_lengths"]!.values[row],length=fixture.inputs["target_lengths"]!.values[row]
                guard time.rounded()==time,length.rounded()==length,(1...Double(sample[0])).contains(time),(0...Double(target!.shape[1])).contains(length) else {throw LearningContractError("Active lengths must fit each supplied observation.")}
                let active=target!.values[row*target!.shape[1]..<row*target!.shape[1]+Int(length)]
                guard active.allSatisfy({$0.rounded()==$0 && (0...Double(sample[1]-1)).contains($0) && $0 != Double(c["blank"]!.integerValue!)}) else {throw LearningContractError("Active sequence labels must be integers and cannot equal blank.")}
            }
            rawShape=[n];denominator=Double(n)
        } else {
            guard sample.count==2,sample[0]==3,(1...4).contains(sample[1]) else {throw LearningContractError("Provide anchor, match and nonmatch in three ordered sample positions.")}
            rawShape=[n];denominator=Double(n)
        }
        func compute(_ scores:[Double])throws->([Double],Double) {
            var values:[Double]=[]
            let d=sample.last!
            func logProb(_ start:Int)->[Double] {Self.logSoftmax(Array(scores[start..<start+d]))}
            func importance(_ prefix:String,_ row:Int,_ col:Int)throws->Double {
                let mode=c[prefix+"_mode"]!.stringValue!
                if mode=="none" {return 1}
                if mode=="mapped" {let t=fixture.inputs[prefix]!;return t.values[t.values.count==n ? row:row*d+col]}
                let shape=Self.numbers(c[prefix+"_shape"]!).map(Int.init),v=Self.numbers(c[prefix+"_values"]!)
                guard shape.isEmpty || shape == [1] || shape == [d] else {throw LearningContractError("Configured importance must be a scalar or match this exercise's value direction.")}
                return v[v.count==1 ? 0:col]
            }
            switch task {
            case .crossEntropy,.nll:
                let weights=Self.numbers(c["class_weights"]!),smoothing=c["label_smoothing"]?.doubleValue ?? 0,probability=c["target_mode"] == .string("probabilities")
                for row in 0..<n {
                    if !probability && Int(target!.values[row])==c["ignore_index"]!.integerValue! {values.append(0);continue}
                    let logs=logProb(row*d)
                    var value=0.0
                    for col in 0..<d {
                        let share=probability ? target!.values[row*d+col]:(Int(target!.values[row])==col ? 1.0:0)
                        let wanted=(1-smoothing)*share+smoothing/Double(d)
                        value -= wanted*(weights.isEmpty ? 1:weights[col])*logs[col]
                    }
                    values.append(value)
                }
            case .bce:
                for row in 0..<n {for col in 0..<d {
                    let x=scores[row*d+col],y=target!.values[row*d+col],positive=try importance("pos_weight",row,col),weight=try importance("weight",row,col)
                    let stable=log1p(exp(-abs(x)))
                    values.append(weight*((1-y)*(max(x,0)+stable)+positive*y*(max(-x,0)+stable)))
                }}
            case .mse,.l1:
                for row in 0..<n {for col in 0..<d {let wanted=target!.values[target!.shape==[n] ? row:row*d+col],diff=scores[row*d+col]-wanted;values.append(task == .mse ? diff*diff:abs(diff))}}
            case .kl:
                for row in 0..<n {let logs=logProb(row*d);for col in 0..<d {let supplied=target!.values[row*d+col],p=c["log_target"] == true ? exp(supplied):supplied;values.append(p==0 ? 0:p*((c["log_target"] == true ? supplied:log(p))-logs[col]))}}
            case .ctc:
                for row in 0..<n {
                    let time=Int(fixture.inputs["input_lengths"]!.values[row]),length=Int(fixture.inputs["target_lengths"]!.values[row]),blank=c["blank"]!.integerValue!
                    let wanted=target!.values[row*target!.shape[1]..<row*target!.shape[1]+length].map(Int.init)
                    let logs=(0..<time).map {logProb((row*sample[0]+$0)*d)}
                    var total=0.0
                    func paths(_ step:Int,_ previous:Int?,_ collapsed:[Int],_ probability:Double) {
                        if step==time {if collapsed==wanted {total+=probability};return}
                        for label in 0..<d {
                            var next=collapsed;if label != blank && previous != label {next.append(label)}
                            if next.count<=wanted.count,Array(wanted.prefix(next.count))==next {paths(step+1,label,next,probability*exp(logs[step][label]))}
                        }
                    }
                    paths(0,nil,[],1)
                    if total==0 {guard c["zero_infinity"] == true else {throw LearningContractError("These repeated labels cannot fit the active time positions. Inspect lengths or explicitly mask impossible alignments.")};values.append(0)} else {values.append(-log(total))}
                }
            case .triplet:
                let p=c["p"]!.doubleValue!,eps=c["eps"]!.doubleValue!,margin=c["margin"]!.doubleValue!
                for row in 0..<n {
                    func distance(_ a:Int,_ b:Int)->Double {pow((0..<d).reduce(0) {sum,col in sum+pow(abs(scores[(row*3+a)*d+col]-scores[(row*3+b)*d+col]+eps),p)},1/p)}
                    let negative=c["swap"] == true ? min(distance(0,2),distance(1,2)):distance(0,2)
                    values.append(max(0,distance(0,1)-negative+margin))
                }
            }
            let reduction=c["reduction"]!.stringValue!,sum=values.reduce(0,+)
            let result:Double
            if reduction=="none" {result=scalarization=="sum" ? sum:sum/Double(values.count)}
            else if reduction=="sum" {result=sum}
            else if reduction=="batchmean" {result=sum/Double(n)}
            else if task == .ctc {result=values.enumerated().reduce(0) {$0+$1.element/max(1,fixture.inputs["target_lengths"]!.values[$1.offset])}/Double(n)}
            else {guard denominator>0 else {throw LearningContractError("No active importance remains for a defined mean score.")};result=sum/denominator}
            guard values.allSatisfy(\.isFinite),result.isFinite else {throw LearningContractError("The configured objective produced a nonfinite score.")}
            return (values,result)
        }
        let measured=try compute(input.values)
        raw = .init(shape:rawShape,values:measured.0);objective = .init(shape:[1],values:[measured.1])
        let step=1e-5
        var influence:[Double]=[]
        for i in input.values.indices {var plus=input.values,minus=input.values;plus[i]+=step;minus[i]-=step;influence.append((try compute(plus).1-compute(minus).1)/(2*step))}
        guard influence.allSatisfy(\.isFinite) else {throw LearningContractError("Inspect the sharp or undefined input influence for this configuration.")}
        inputInfluence = .init(shape:input.shape,values:influence)
    }
    private static func numbers(_ value:JSONValue)->[Double] {if case .array(let items)=value {return items.map {$0.doubleValue!}};return []}
    private static func logSoftmax(_ values:[Double])->[Double] {let m=values.max()!,normalizer=m+log(values.reduce(0) {$0+exp($1-m)});return values.map {$0-normalizer}}
}
