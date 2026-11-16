import Foundation
import CryptoKit

/// Reviewed local rules; evaluation records distinguish missing evidence from a negative trigger.
enum RunDiagnostics {
    static let limitations=[
        "Trend thresholds are uncalibrated product heuristics, not statistical confidence or proof of a cause.",
        "Input observations may be sampled; missing epochs are never interpolated. Complete/sampled is caller-declared, not independently verified.",
        "Run identity, unchanged partitions, continuation boundaries and evaluation settings are not independently verified.",
        "Gradient norms, activation distributions and hardware memory are not measured by this rule set.",
        "An empty finding list does not establish model health. Input SHA256 is local provenance, not authenticity or a security guarantee."
    ]
    /// Compatibility entry point for the existing UI. New callers can inspect report.evaluations.
    static func evaluate(_ metrics:[EpochMetric],error:String?=nil)->[RunDiagnostic] {
        (try? report(metrics,error:error).findings) ?? []
    }
    static func report(_ metrics:[EpochMetric],error:String?=nil,context:DiagnosticInputContext=DiagnosticInputContext(),registry:DiagnosticRuleRegistry = .builtIn)throws->DiagnosticReport {
        try registry.validate();try context.validate()
        let collected=DiagnosticMeasurements(metrics)
        let hash=try fingerprint(metrics,error:error,context:context)
        var findings:[RunDiagnostic]=[],evaluations:[DiagnosticEvaluation]=[]
        func record(_ rule:DiagnosticRule,_ status:DiagnosticEvaluationStatus,_ reason:DiagnosticEvaluationReason) {
            evaluations.append(DiagnosticEvaluation(ruleID:rule.id,status:status,reason:reason))
        }
        func finding(_ rule:DiagnosticRule,evidence:String,observations:[DiagnosticObservationReference]=[],derived:[String:Double]=[:],rawError:Bool=false) {
            findings.append(RunDiagnostic(id:rule.id,title:rule.title,evidence:evidence,suggestion:rule.suggestion,provenance:DiagnosticProvenance(registryID:registry.id,registryVersion:registry.version,ruleVersion:rule.version,inputSHA256:hash,context:context,observedInputCount:metrics.count,observations:observations,usesRawError:rawError,derivedSignals:derived)))
            record(rule,.emitted,.matched)
        }
        // Error precedence remains device, shape, data, then generic, before measurement findings.
        var classified=false
        for rule in registry.rules where [.deviceError,.shapeError,.dataError,.workerError].contains(rule.evaluator) {
            guard let error,!error.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty else {record(rule,.withheld,.missingError);continue}
            if classified {record(rule,.withheld,.earlierErrorClassifier);continue}
            if rule.evaluator == .workerError || rule.terms.contains(where:{error.lowercased().contains($0)}) {
                finding(rule,evidence:error,rawError:true);classified=true
            } else {record(rule,.notTriggered,.thresholdNotMet)}
        }
        for rule in registry.rules where [.nonfinite,.generalizationGap,.lossGrowth].contains(rule.evaluator) {
            if rule.evaluator == .nonfinite {
                if metrics.isEmpty {record(rule,.withheld,.noObservations)}
                else if collected.invalid.isEmpty {record(rule,.notTriggered,.thresholdNotMet)}
                else {
                    finding(rule,evidence:"Non-finite loss or accuracy at epoch(s): \(collected.invalid.prefix(8).map {String($0.epoch)}.joined(separator:", ")). \(collected.invalid.count) affected observations.",observations:Array(collected.invalid.prefix(8)),derived:["affectedObservationCount":Double(collected.invalid.count)])
                }
                continue
            }
            guard collected.invalid.isEmpty else {record(rule,.withheld,.nonfiniteMeasurements);continue}
            let threshold=rule.thresholds
            let window=Array(collected.distinct.suffix(Int(threshold["windowObservations"]!)))
            guard window.count>=Int(threshold["minimumObservations"]!) else {record(rule,.withheld,.insufficientObservations);continue}
            guard window.allSatisfy({$0.metric.trainLoss>=0 && $0.metric.validationLoss>=0 && $0.metric.accuracy.map {(0...1).contains($0)} == true}) else {record(rule,.withheld,.invalidMetricDomain);continue}
            let signals=DiagnosticTrendSignals(window:window,meanCount:Int(threshold["meanObservations"]!))
            guard signals.values.values.allSatisfy(\.isFinite) else {record(rule,.withheld,.nonfiniteDerivedSignals);continue}
            let refs=window.map {DiagnosticObservationReference(inputIndex:$0.index,epoch:$0.metric.epoch)}
            switch rule.evaluator {
            case .generalizationGap:
                if signals.trainStart-signals.trainEnd >= max(threshold["minimumAbsoluteChange"]!,signals.trainStart*threshold["relativeChange"]!),signals.valEnd-signals.valStart >= max(threshold["minimumAbsoluteChange"]!,signals.valStart*threshold["relativeChange"]!),signals.gapIncrease>=threshold["minimumGapIncrease"]! {
                    finding(rule,evidence:signals.summary+" The validation–training gap widened by \(DiagnosticTrendSignals.number(signals.gapIncrease)).",observations:refs,derived:signals.values)
                } else {record(rule,.notTriggered,.thresholdNotMet)}
            case .lossGrowth:
                let train=signals.trainEnd>=max(signals.trainStart,threshold["baselineFloor"]!)*threshold["growthFactor"]! && signals.trainEnd-signals.trainStart>=threshold["minimumAbsoluteGrowth"]!
                let validation=signals.valEnd>=max(signals.valStart,threshold["baselineFloor"]!)*threshold["growthFactor"]! && signals.valEnd-signals.valStart>=threshold["minimumAbsoluteGrowth"]!
                if train || validation {finding(rule,evidence:signals.summary,observations:refs,derived:signals.values)}
                else {record(rule,.notTriggered,.thresholdNotMet)}
            default: break
            }
        }
        let ordered=registry.rules.compactMap {rule in evaluations.first {$0.ruleID==rule.id}}
        let result=DiagnosticReport(registryID:registry.id,registryVersion:registry.version,inputSHA256:hash,context:context,observedInputCount:metrics.count,findings:findings,evaluations:ordered,limitations:limitations)
        try result.validate(registry:registry)
        return result
    }
    private static func fingerprint(_ metrics:[EpochMetric],error:String?,context:DiagnosticInputContext)throws->String {
        struct Observation:Encodable {let epoch:Int;let trainBits:String;let validationBits:String;let accuracyBits:String?}
        struct Snapshot:Encodable {let schemaVersion:Int;let context:DiagnosticInputContext;let observations:[Observation];let rawError:String?}
        let rows=metrics.map {Observation(epoch:$0.epoch,trainBits:String($0.trainLoss.bitPattern,radix:16),validationBits:String($0.validationLoss.bitPattern,radix:16),accuracyBits:$0.accuracy.map {String($0.bitPattern,radix:16)})}
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes]
        return SHA256.hash(data:try encoder.encode(Snapshot(schemaVersion:1,context:context,observations:rows,rawError:error))).map {String(format:"%02x",$0)}.joined()
    }
}

/// Collection preserves source indices, then derives the old last-record-wins epoch view.
private struct DiagnosticMeasurements {
    struct Indexed {let index:Int;let metric:EpochMetric}
    let invalid:[DiagnosticObservationReference]
    let distinct:[Indexed]
    init(_ metrics:[EpochMetric]) {
        invalid=metrics.enumerated().filter {!$0.element.trainLoss.isFinite || !$0.element.validationLoss.isFinite || $0.element.accuracy.map {!$0.isFinite} == true}.map {DiagnosticObservationReference(inputIndex:$0.offset,epoch:$0.element.epoch)}
        var unique:[Int:Indexed]=[:]
        for (index,metric) in metrics.enumerated() where metric.epoch>0 {unique[metric.epoch]=Indexed(index:index,metric:metric)}
        distinct=unique.values.sorted {$0.metric.epoch<$1.metric.epoch}
    }
}
private struct DiagnosticTrendSignals {
    let trainStart:Double,trainEnd:Double,valStart:Double,valEnd:Double
    let count:Int,firstEpoch:Int,lastEpoch:Int,meanCount:Int
    var gapIncrease:Double {(valEnd-trainEnd)-(valStart-trainStart)}
    var values:[String:Double] { ["trainingStartMean":trainStart,"trainingEndMean":trainEnd,"validationStartMean":valStart,"validationEndMean":valEnd,"gapIncrease":gapIncrease] }
    init(window:[DiagnosticMeasurements.Indexed],meanCount:Int) {
        func mean(_ rows:[DiagnosticMeasurements.Indexed],_ key:KeyPath<EpochMetric,Double>)->Double {rows.reduce(0) {$0+$1.metric[keyPath:key]/Double(rows.count)}}
        let first=Array(window.prefix(meanCount)),last=Array(window.suffix(meanCount))
        trainStart=mean(first,\.trainLoss);trainEnd=mean(last,\.trainLoss);valStart=mean(first,\.validationLoss);valEnd=mean(last,\.validationLoss)
        count=window.count;firstEpoch=window.first!.metric.epoch;lastEpoch=window.last!.metric.epoch;self.meanCount=meanCount
    }
    static func number(_ value:Double)->String {String(format:"%.4f",locale:Locale(identifier:"en_US_POSIX"),value)}
    var summary:String {"\(count) observations across epochs \(firstEpoch)–\(lastEpoch); first/last \(meanCount)-observation means: training \(Self.number(trainStart)) → \(Self.number(trainEnd)), validation \(Self.number(valStart)) → \(Self.number(valEnd))."}
}
