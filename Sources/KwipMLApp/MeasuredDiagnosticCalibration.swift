import Foundation

/// Identified deterministic controls. These confusion counts measure agreement
/// on this small designed corpus, never real-world predictive performance.
enum MeasuredDiagnosticCalibration {
    static let corpusID="kwip.measured-controls.v1"
    struct Fixture {
        let id:String
        let ruleID:String
        let positive:Bool
        let input:MeasuredDiagnosticInput
    }
    struct Score:Codable,Equatable {
        let ruleID:String
        let positives:Int
        let negatives:Int
        let falsePositives:Int
        let falseNegatives:Int
        let withheldControls:Int
    }
    static var fixtures:[Fixture] {
        func input()->MeasuredDiagnosticInput {.init(runID:"988C4317-220E-436D-810B-1078477D60D9",requestSHA256:String(repeating:"c",count:64),metrics:[])}
        func gradient(_ norm:Double=1,zero:Int=0,missing:Int=0,nonfinite:Int=0,epochs:Int=1)->MeasuredDiagnosticInput {
            var value=input();value.gradients=(1...epochs).map{.init(epoch:$0,batch:0,presentElements:10,nonfiniteElements:nonfinite,zeroElements:zero,missingParameters:missing,l2Norm:nonfinite>0 ? nil:norm)};return value
        }
        func activation(_ component:String,_ values:[Double])->MeasuredDiagnosticInput {var value=input();value.activations=[.init(nodeID:"control",component:component,values:values)];return value}
        func loss(_ start:Double,_ end:Double)->MeasuredDiagnosticInput {var value=input();value.metrics=(1...20).map{MeasuredMetric(.init(epoch:$0,trainLoss:$0<=5 ? start:end,validationLoss:0.7,accuracy:0.5))};return value}
        func rows(_ count:Int)->MeasuredDiagnosticInput {var value=input();value.validationRows=count;return value}
        func error(_ text:String)->MeasuredDiagnosticInput {var value=input();value.rawError=text;return value}
        var result:[Fixture]=[]
        func add(_ id:String,_ positive:MeasuredDiagnosticInput,_ negatives:[MeasuredDiagnosticInput]) {
            result.append(.init(id:id+"/positive",ruleID:id,positive:true,input:positive))
            for (i,value) in negatives.enumerated(){result.append(.init(id:id+"/healthy-\(i+1)",ruleID:id,positive:false,input:value))}
        }
        add("gradient.nonfinite.v1",gradient(nonfinite:1),[gradient(),gradient(0,zero:10)])
        add("gradient.missing.v1",gradient(missing:1),[gradient(),gradient(0,zero:10)])
        var partlyZero=gradient(0,zero:10,epochs:3);partlyZero.gradients[2] = .init(epoch:3,batch:0,presentElements:10,nonfiniteElements:0,zeroElements:0,missingParameters:0,l2Norm:1)
        add("gradient.zero.v1",gradient(0,zero:10,epochs:3),[gradient(epochs:3),partlyZero])
        var growth=gradient(epochs:10);growth.gradients=(1...10).map{.init(epoch:$0,batch:0,presentElements:10,nonfiniteElements:0,zeroElements:0,missingParameters:0,l2Norm:$0<=5 ? 0.01:2)}
        add("gradient.growth.v1",growth,[gradient(epochs:10),gradient(500,epochs:10)])
        add("activation.zero-sample.v1",activation("relu",Array(repeating:0,count:8)),[activation("relu",Array(repeating:1,count:8)),activation("relu",[0,0,0,0,0,0,0,1])])
        add("activation.saturation-sample.v1",activation("tanh",Array(repeating:0.999,count:8)),[activation("tanh",Array(repeating:0.5,count:8)),activation("tanh",[1,1,1,1,1,1,0.5,0.5])])
        add("optimization.plateau.v1",loss(0.7,0.7),[loss(0.9,0.5),loss(0.1,0.1)])
        add("data.coarse-holdout.v1",rows(19),[rows(20),rows(100)])
        add("runtime.memory.v1",error("MPS backend out of memory"),[error("Memory policy loaded; invalid CSV column"),error("CPU shape mismatch")])
        return result
    }
    static func evaluate()throws->[Score] {
        try MeasuredDiagnostics.rules.map {rule in
            let cases=fixtures.filter{$0.ruleID==rule.id}
            var fp=0,fn=0,withheld=0
            for fixture in cases {
                let report=try MeasuredDiagnostics.evaluate(fixture.input)
                let emitted=report.findings.contains{$0.ruleID==rule.id}
                if report.checks.first(where:{$0.ruleID==rule.id})?.status == .withheld {withheld+=1}
                if emitted && !fixture.positive {fp+=1}
                if !emitted && fixture.positive {fn+=1}
            }
            return .init(ruleID:rule.id,positives:cases.filter(\.positive).count,negatives:cases.filter{!$0.positive}.count,falsePositives:fp,falseNegatives:fn,withheldControls:withheld)
        }
    }
}
