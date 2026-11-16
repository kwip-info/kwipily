import Foundation

/// Independent scalar update arithmetic. No native optimizer or production adapter.
public struct OptimizerLessonReference {
    public let expected:[String:LearningTensor]
    public init(initial:LearningTensor,gradients:LearningTensor,active:LearningTensor,task:OptimizerLearningTask)throws {
        guard initial.wellFormed,initial.shape == [3],gradients.wellFormed,gradients.shape == [4,3],active.wellFormed,active.shape == [4,2],active.values.allSatisfy({$0 == 0 || $0 == 1}) else {throw LearningContractError("Use three supplied learned settings, four gradient rows and the supplied parameter activity schedule.")}
        var w=initial.values,m=Array(repeating:0.0,count:3),v=m,mx=m,ga=m,acc=Array(repeating:0.5,count:3),counts=[0,0],present=[task == .adagrad,task == .adagrad],tables:[String:[Double]]=[:]
        func record(_ key:String,_ values:[Double]) {tables[key,default:[]].append(contentsOf:values)}
        for step in 0..<4 {
            for p in 0..<2 where active.values[step*2+p] == 1 {
                counts[p]+=1;let first = !present[p];present[p]=true
                for i in (p == 0 ? [0,1]:[2]) {
                    var g=gradients.values[step*3+i]
                    if task == .adamW {w[i] *= 0.99} else {g += 0.1*w[i]}
                    switch task {
                    case .sgd:m[i]=first ? g:0.5*m[i]+0.75*g;w[i] -= 0.1*m[i]
                    case .adam,.adamW:
                        m[i]=0.5*m[i]+0.5*g;v[i]=0.75*v[i]+0.25*g*g;mx[i]=max(mx[i],v[i]);w[i] -= 0.1*m[i]/(1-pow(0.5,Double(counts[p])))/(sqrt(mx[i]/(1-pow(0.75,Double(counts[p]))))+0.01)
                    case .rmsprop:v[i]=0.5*v[i]+0.5*g*g;ga[i]=0.5*ga[i]+0.5*g;m[i]=0.25*m[i]+g/(sqrt(max(0,v[i]-ga[i]*ga[i]))+0.01);w[i] -= 0.1*m[i]
                    case .adagrad:acc[i] += g*g;w[i] -= 0.1/(1+Double(counts[p]-1)*0.2)*g/(sqrt(acc[i])+0.01)
                    }
                }
            }
            record("updatedParameters",w);record("statePresence",present.map {$0 ? 1:0});record("updateCounts",counts.map(Double.init))
            switch task {
            case .sgd:record("momentumBuffer",m)
            case .adam,.adamW:record("firstAverage",m);record("squaredAverage",v);record("largestSquaredAverage",mx);record("stepCounts",counts.map(Double.init))
            case .rmsprop:record("squaredAverage",v);record("gradientAverage",ga);record("momentumBuffer",m);record("stepCounts",counts.map(Double.init))
            case .adagrad:record("accumulatedSquares",acc);record("stepCounts",counts.map(Double.init))
            }
        }
        let result=tables.mapValues {LearningTensor(shape:[4,$0.count/4],values:$0)}
        guard result.values.allSatisfy(\.wellFormed) else {throw LearningContractError("Independent update arithmetic exceeded finite bounds.")}
        expected=result
    }
}
