import Foundation

/// Independent scalar schedule/update reference; never native scheduler output.
public struct SchedulerLessonReference {
    public let expected:[String:LearningTensor]
    public init(initial:LearningTensor,gradients:LearningTensor,active:LearningTensor,epochOfBatch:LearningTensor,metrics:LearningTensor,task:SchedulerLearningTask)throws {
        guard initial.wellFormed,initial.shape == [3],gradients.wellFormed,gradients.shape == [8,3],active.wellFormed,active.shape == [8,2],active.values.allSatisfy({$0 == 0 || $0 == 1}),epochOfBatch.wellFormed,epochOfBatch.shape == [8],epochOfBatch.values == [1,1,2,2,3,3,4,4],metrics.wellFormed,metrics.shape == [4] else {throw LearningContractError("Use the supplied eight batches, four epochs, starting settings, gradients and validation measurements.")}
        var w=initial.values,m=Array(repeating:0.0,count:3),present=[false,false],rate=task == .linear ? 0.025:0.1,ticks=0,best:Double?,bad=0,cooldown=0,tables:[String:[Double]]=[:]
        func cycle(_ tick:Int)->Double {
            let start=0.01,peak=0.1,end=0.001,phaseEnd=3.0,finish=7.0,t=Double(tick)
            let a=t<=phaseEnd ? start:peak,b=t<=phaseEnd ? peak:end,p=t<=phaseEnd ? t/phaseEnd:(t-phaseEnd)/(finish-phaseEnd)
            return b+(a-b)*(1+cos(Double.pi*p))/2
        }
        if task == .oneCycle {rate=cycle(0)}
        func record(_ key:String,_ values:[Double]) {tables[key,default:[]].append(contentsOf:values)}
        for batch in 0..<8 {
            let used=rate
            for parameter in 0..<2 where active.values[batch*2+parameter] == 1 {
                let first = !present[parameter];present[parameter]=true
                for index in (parameter == 0 ? [0,1]:[2]) {
                    let g=gradients.values[batch*3+index]
                    m[index]=first ? g:0.5*m[index]+g;w[index] -= used*m[index]
                }
            }
            if task == .oneCycle {ticks+=1;rate=cycle(ticks)}
            else if batch%2 == 1 {
                ticks+=1
                switch task {
                case .step:rate=0.1*pow(0.5,Double(ticks/2))
                case .cosine:rate=0.02+0.08*(1+cos(Double.pi*Double(ticks)/4))/2
                case .linear:rate=0.1*(0.25+0.75*Double(min(ticks,3))/3)
                case .plateau:
                    let value=metrics.values[ticks-1]
                    if best == nil || value<best!-0.05 {best=value;bad=0} else {bad+=1}
                    if cooldown>0 {cooldown-=1;bad=0}
                    if bad>0 {let next=max(rate*0.5,0.01);if rate-next>1e-8 {rate=next};cooldown=1;bad=0}
                case .oneCycle:break
                }
            }
            record("usedRates",[used]);record("nextRates",[rate]);record("scheduleTicks",[Double(ticks)]);record("updatedParameters",w);record("momentumBuffer",m)
        }
        for (original,resumed) in [("usedRates","resumedUsedRates"),("nextRates","resumedNextRates"),("scheduleTicks","resumedTicks"),("updatedParameters","resumedParameters"),("momentumBuffer","resumedMomentum")] {
            let values=tables[original]!;tables[resumed]=Array(values.suffix(values.count/2))
        }
        let result=tables.mapValues {LearningTensor(shape:[$0.count == 4 || $0.count == 12 ? 4:8,$0.count == 24 || $0.count == 12 ? 3:1],values:$0)}
        guard result.values.allSatisfy(\.wellFormed) else {throw LearningContractError("Schedule reference arithmetic exceeded finite bounds.")}
        expected=result
    }
}
