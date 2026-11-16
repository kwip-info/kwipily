import Foundation

enum TrainingChartAxis:String,CaseIterable {case updates="Updates",batches="Batches",epoch="Epoch",time="Time"}
enum TrainingChartMetric:String,CaseIterable {case loss="Loss",accuracy="Accuracy",routing="MoE objective"}
struct TrainingChartValue:Identifiable,Equatable {
    let id:String
    let x:Double
    let value:Double
    let series:String
    let epoch:Int
    let batches:Int?
    let updates:Int?
    let elapsed:Double?
}
enum TrainingChartModel {
    static func values(metrics:[EpochMetric],progress:[TrainingProgressPoint],axis:TrainingChartAxis,metric:TrainingChartMetric,smoothed:Bool)->[TrainingChartValue] {
        var result:[TrainingChartValue]=[]
        if progress.isEmpty {
            guard axis == .epoch else {return []}
            for row in metrics {
                if metric == .loss {result.append(.init(id:"t\(row.epoch)",x:Double(row.epoch),value:row.trainLoss,series:"Training",epoch:row.epoch,batches:nil,updates:nil,elapsed:nil))}
                if let value=metric == .loss ? row.validationLoss:row.accuracy {result.append(.init(id:"v\(row.epoch)",x:Double(row.epoch),value:value,series:"Validation",epoch:row.epoch,batches:nil,updates:nil,elapsed:nil))}
            }
        } else {
            for row in progress {
                let x:Double
                switch axis {case .updates:x=Double(row.updates);case .batches:x=Double(row.batches);case .time:x=row.elapsedSeconds;case .epoch:x=Double(row.epoch-1)+Double(row.epochBatch)/Double(row.epochBatches)}
                if metric == .routing {
                    if let moe=row.moe,let loss=row.trainingLoss {
                        for (name,value) in [("Task loss",loss),("Balance penalty",moe.balancePenalty),("Router z penalty",moe.routerZPenalty),("Training objective",moe.trainingObjective)] {result.append(.init(id:"moe\(row.sequence)"+name,x:x,value:value,series:name,epoch:row.epoch,batches:row.batches,updates:row.updates,elapsed:row.elapsedSeconds))}
                    }
                    continue
                }
                let validation=row.kind=="validation"
                let value=metric == .accuracy ? (validation ? row.accuracy:nil):(validation ? row.validationLoss:row.trainingLoss)
                if let value {result.append(.init(id:"p\(row.sequence)",x:x,value:value,series:validation ? "Validation":"Training",epoch:row.epoch,batches:row.batches,updates:row.updates,elapsed:row.elapsedSeconds))}
            }
        }
        result=result.filter {$0.x.isFinite && $0.value.isFinite}
        guard smoothed,metric == .loss else {return result}
        var ema:Double?
        return result.map {row in
            guard row.series=="Training" else {return row}
            let value=ema.map {$0*0.8+row.value*0.2} ?? row.value;ema=value
            return .init(id:row.id,x:row.x,value:value,series:row.series,epoch:row.epoch,batches:row.batches,updates:row.updates,elapsed:row.elapsed)
        }
    }
    static func domain(_ rows:[TrainingChartValue])->ClosedRange<Double> {
        let low=rows.map(\.x).min() ?? 0,high=rows.map(\.x).max() ?? 1
        return low...(high>low ? high:low+1)
    }
    static func window(_ domain:ClosedRange<Double>,lower:Double,upper:Double)->ClosedRange<Double> {
        let span=domain.upperBound-domain.lowerBound
        let l=min(0.999,max(0,lower)),u=min(1,max(l+0.001,upper))
        return (domain.lowerBound+span*l)...(domain.lowerBound+span*u)
    }
    /// Preserve local extrema independently for each series. Canonical values are
    /// never replaced; smoothing and this bounded render projection are display only.
    static func render(_ rows:[TrainingChartValue],limit:Int=400)->[TrainingChartValue] {
        guard rows.count>limit,limit>=8 else {return rows}
        var result:[TrainingChartValue]=[]
        for series in Set(rows.map(\.series)).sorted() {
            let values=rows.filter {$0.series==series}
            let buckets=max(1,limit/max(4,Set(rows.map(\.series)).count*2)-1),width=max(1,Int(ceil(Double(values.count)/Double(buckets))))
            var indices:Set<Int>=values.isEmpty ? []:[0,values.count-1]
            for start in stride(from:0,to:values.count,by:width) {
                let range=start..<min(start+width,values.count)
                if let low=range.min(by:{values[$0].value<values[$1].value}),let high=range.max(by:{values[$0].value<values[$1].value}) {indices.insert(low);indices.insert(high)}
            }
            result += indices.sorted().map {values[$0]}
        }
        return result.sorted {$0.x==$1.x ? $0.id<$1.id:$0.x<$1.x}
    }
}
