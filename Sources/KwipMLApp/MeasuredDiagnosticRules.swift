import Foundation
import KwipMLCore

/// Deterministic evidence checks. Thresholds describe this named fixture-tested
/// revision, not statistically calibrated probabilities or universal ML laws.
enum MeasuredDiagnostics {
    static let rules:[MeasuredDiagnosticRule]=[
        .init(id:"gradient.nonfinite.v1",family:"numerical",title:"Non-finite captured gradients",prerequisites:["Captured post-backward gradient element counts"],thresholds:["minimumNonfiniteElements":1],confounders:["Only scheduled minibatches were observed","No offending parameter or cause is localized"],experiment:"Inspect the failing operation and input ranges; compare a lower learning rate with the same split and seed.",lesson:"A gradient is a derivative of loss with respect to a parameter. A NaN or infinity in a captured derivative is a numerical observation, not proof that learning rate caused it."),
        .init(id:"gradient.missing.v1",family:"gradients",title:"Parameters without captured gradients",prerequisites:["Captured missing-parameter counts","No non-finite captured gradients"],thresholds:["minimumMissingParameters":1],confounders:["Conditional branches and unused parameters may be intentional","No per-parameter absence evidence is available"],experiment:"Inspect which branches contribute to the loss and whether trainable parameters are intentionally unused.",lesson:"Reverse-mode differentiation traverses operations connecting loss to a parameter. No gradient tensor differs from a gradient tensor whose entries are zero."),
        .init(id:"gradient.zero.v1",family:"gradients",title:"Repeated all-zero gradient samples",prerequisites:["At least three distinct captured epochs","Every observed parameter present and finite"],thresholds:["minimumEpochs":3,"zeroFraction":1],confounders:["A solution may already be stationary","Scheduled batches are not the full dataset","Clipping, underflow and architecture can differ"],experiment:"Inspect the sampled batches and activations; compare loss and a known nonconstant input before changing the model.",lesson:"An optimizer update is proportional to a gradient or a transformed gradient. All-zero sampled derivatives may produce no update, but do not by themselves establish a broken model."),
        .init(id:"gradient.growth.v1",family:"gradients",title:"Large relative gradient growth",prerequisites:["At least ten distinct captured epochs","Finite present gradients","One uninterrupted run"],thresholds:["minimumEpochs":10,"windowEpochs":20,"meanEpochs":5,"factor":100,"baselineFloor":0.000001,"minimumAbsoluteGrowth":1],confounders:["Loss scaling and sample difficulty affect norms","Global norms cannot locate a layer","This threshold is an explicit product heuristic"],experiment:"Compare a shorter run or lower learning rate with the same split and seed. Inspect finite loss and activations before considering gradient clipping.",lesson:"The global L2 norm is sqrt(sum(g_i²)). A ratio compares it with the same run's earlier observations; a large constant norm is not evidence of growth."),
        .init(id:"activation.zero-sample.v1",family:"activations",title:"Zero-valued ReLU sample",prerequisites:["Eight captured scalar values from a ReLU output"],thresholds:["minimumScalars":8,"zeroFraction":1],confounders:["Only eight first-row scalars are retained","ReLU legitimately maps negative inputs to zero","No whole-layer or neuron conclusion is justified"],experiment:"Inspect more inputs and the preceding affine transform; compare a nonzero input before changing the activation.",lesson:"ReLU(x)=max(0,x), with derivative zero on negative inputs. Eight zero outputs describe only the saved sample, not a dead neuron or the whole dataset."),
        .init(id:"activation.saturation-sample.v1",family:"activations",title:"Tanh sample near its limits",prerequisites:["Eight captured scalar values from a Tanh output"],thresholds:["minimumScalars":8,"absoluteThreshold":0.99,"minimumFraction":0.875],confounders:["Tiny deterministic sample is not a distribution estimate","Saturation can be intentional","No gradients are inferred from activation samples"],experiment:"Inspect preceding values and feature scale; compare a controlled normalization change only after reviewing the data contract.",lesson:"d tanh(x)/dx = 1 − tanh(x)². Outputs near ±1 have small local derivatives, but the saved sample cannot establish the network's overall gradient behavior."),
        .init(id:"optimization.plateau.v1",family:"optimization",title:"Little recent training-loss change",prerequisites:["Twenty consecutive finite epochs","Fresh uninterrupted run","Nonnegative losses and bounded accuracy"],thresholds:["observations":20,"meanObservations":5,"maximumRelativeChange":0.01,"minimumEndLoss":0.5],confounders:["Stable loss can reflect convergence or label noise","Absolute threshold applies to this binary cross-entropy task","Different loss reductions are unsupported"],experiment:"Compare with a simple baseline and verify targets. Change one optimizer setting in a new run, retaining the same split and seed.",lesson:"Compare the first and last five-epoch means of a twenty-epoch window. A small difference describes an observed plateau; it does not prove that a larger learning rate or more layers will help."),
        .init(id:"data.coarse-holdout.v1",family:"data",title:"Coarse held-out accuracy resolution",prerequisites:["Verified held-out row count"],thresholds:["minimumRecommendedRows":20],confounders:["More rows do not establish independent subjects or absence of leakage","No confidence interval or population accuracy is inferred"],experiment:"Inspect the split component and independent held-out coverage. Keep the original partition for comparisons; create a separately named split experiment if more data becomes available.",lesson:"For n equally weighted classification rows, accuracy changes in steps of 1/n. Ten rows mean ten percentage points per correct prediction, without implying statistical confidence."),
        .init(id:"runtime.memory.v1",family:"runtime",title:"Runtime reported allocation failure",prerequisites:["Raw runtime error contains an explicit allocation-failure phrase"],thresholds:["minimumExplicitPhraseMatches":1],confounders:["This is error classification, not a memory measurement","Allocator fragmentation, device limits and exhaustion may differ"],experiment:"Inspect the exact failing allocation. Try a smaller batch or tensor shape as an explicit experiment; changing devices is never automatic.",lesson:"Tensor storage scales with element count and bytes per element; retained activations and optimizer states add more. An allocation failure identifies a symptom, not the available physical memory or its cause.")
    ]
    /// Keep `.rules` and `evaluate` pinned to the nine-rule v1 archive contract.
    static let currentRules = rules + [MeasuredGeneralizationCheck.rule]
    static func rules(forVersion version: Int) throws -> [MeasuredDiagnosticRule] {
        switch version {
        case 1: return rules
        case 2: return currentRules
        default: throw DiagnosticContractError("Unsupported measured diagnostic registry version.")
        }
    }

    /// Explicit current evaluation. The retained v1 path is evaluated unchanged,
    /// then the new check is appended under a distinct registry and input identity.
    static func evaluateCurrent(_ input: MeasuredDiagnosticInput,
                                context: MeasuredGeneralizationCheck.VerifiedContext? = nil) throws -> MeasuredDiagnosticReport {
        let legacy = try evaluate(input)
        let divergence = try MeasuredGeneralizationCheck.evaluate(input, context: context)
        let fingerprint = CurrentInputFingerprint(input: input, context: context.map(ContextFingerprint.init))
        let report = MeasuredDiagnosticReport(
            registryVersion: 2, runID: input.runID, requestSHA256: input.requestSHA256,
            inputSHA256: try WorkflowArchiveCodec.hash(WorkflowArchiveCodec.encode(fingerprint)),
            findings: legacy.findings + (divergence.finding.map { [$0] } ?? []),
            checks: legacy.checks + [divergence.check]
        )
        try report.validate()
        return report
    }

    /// Copy all context fields exactly, including unavailable or mismatched
    /// attestations; withholding decisions also need distinct provenance.
    private struct ContextFingerprint: Encodable {
        let runID: String
        let requestSHA256: String
        let partitionSHA256: String
        let evaluationSHA256: String
        let completeEpochCapture: Bool
        let unchangedPartitions: Bool
        let unchangedEvaluation: Bool
        let binaryCrossEntropy: Bool
        init(_ context: MeasuredGeneralizationCheck.VerifiedContext) {
            runID = context.runID; requestSHA256 = context.requestSHA256
            partitionSHA256 = context.partitionSHA256; evaluationSHA256 = context.evaluationSHA256
            completeEpochCapture = context.completeEpochCapture
            unchangedPartitions = context.unchangedPartitions; unchangedEvaluation = context.unchangedEvaluation
            binaryCrossEntropy = context.binaryCrossEntropy
        }
    }
    private struct CurrentInputFingerprint: Encodable {
        let input: MeasuredDiagnosticInput
        let context: ContextFingerprint?
        private enum CodingKeys: String, CodingKey { case schemaVersion, registryID, registryVersion, input, context }
        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(1, forKey: .schemaVersion)
            try container.encode("kwip.measured-diagnostics", forKey: .registryID)
            try container.encode(2, forKey: .registryVersion)
            try container.encode(input, forKey: .input)
            if let context { try container.encode(context, forKey: .context) }
            else { try container.encodeNil(forKey: .context) }
        }
    }
    static func evaluate(_ input:MeasuredDiagnosticInput)throws->MeasuredDiagnosticReport {
        try input.validate()
        var findings:[MeasuredDiagnosticFinding]=[],checks:[MeasuredDiagnosticCheck]=[]
        let batches=input.gradients.sorted{($0.epoch,$0.batch)<($1.epoch,$1.batch)}
        let nonfinite=batches.contains{$0.nonfiniteElements>0},missing=batches.contains{$0.missingParameters>0}
        func mark(_ rule:MeasuredDiagnosticRule,_ status:DiagnosticEvaluationStatus,_ reason:String){checks.append(.init(ruleID:rule.id,status:status,reason:reason))}
        func emit(_ rule:MeasuredDiagnosticRule,_ observation:String,_ interpretation:String,refs:[String],values:[String:Double],node:String?=nil) {
            findings.append(.init(ruleID:rule.id,nodeID:node,observation:observation,interpretation:interpretation,references:refs,values:values))
        }
        for rule in rules {
            let start=findings.count,t=rule.thresholds
            switch rule.id {
            case "gradient.nonfinite.v1","gradient.missing.v1","gradient.zero.v1","gradient.growth.v1":
                guard !batches.isEmpty else {mark(rule,.withheld,"No gradient batches were captured.");continue}
                if rule.id != "gradient.nonfinite.v1",nonfinite {mark(rule,.withheld,"Non-finite captured gradients take precedence; lower-specificity advice is withheld.");continue}
                if ["gradient.zero.v1","gradient.growth.v1"].contains(rule.id),missing {mark(rule,.withheld,"Missing parameter gradients prevent a complete global-gradient comparison.");continue}
                if rule.id == "gradient.nonfinite.v1" {
                    let rows=batches.filter{$0.nonfiniteElements>0},count=rows.reduce(0){$0+$1.nonfiniteElements}
                    if count>=Int(t["minimumNonfiniteElements"]!) {emit(rule,"\(count) non-finite elements across \(rows.count) captured batches.","Observed numerical failure in the sampled derivatives; the cause is not established.",refs:rows.map(\.reference),values:["nonfiniteElements":Double(count)])}
                } else if rule.id == "gradient.missing.v1" {
                    let rows=batches.filter{$0.missingParameters>0}
                    if !rows.isEmpty {emit(rule,"\(rows.count) captured batches contain parameters with no gradient tensor.","Check intentional branch use before treating absent gradients as a defect.",refs:rows.map(\.reference),values:["affectedBatches":Double(rows.count)])}
                } else if rule.id == "gradient.zero.v1" {
                    guard Set(batches.map(\.epoch)).count>=Int(t["minimumEpochs"]!) else {mark(rule,.withheld,"Fewer than three distinct captured epochs.");continue}
                    if batches.allSatisfy({$0.presentElements>0 && $0.zeroElements==$0.presentElements}) {emit(rule,"Every present gradient element is zero across \(batches.count) captured batches.","Repeated zero derivatives in the observed samples; convergence and unrepresentative batches remain possible.",refs:batches.map(\.reference),values:["capturedBatches":Double(batches.count)])}
                } else {
                    guard !input.continuation else {mark(rule,.withheld,"Continuation boundary prevents the fresh-run norm comparison.");continue}
                    let grouped=Dictionary(grouping:batches,by:\.epoch)
                    let epochs=grouped.keys.sorted().suffix(Int(t["windowEpochs"]!))
                    guard epochs.count>=Int(t["minimumEpochs"]!),batches.allSatisfy({$0.presentElements>0 && $0.l2Norm != nil}) else {mark(rule,.withheld,"Ten captured epochs with present finite norms are required.");continue}
                    let norms=epochs.map{epoch in let values=grouped[epoch]!.compactMap(\.l2Norm);return values.reduce(0){$0+$1/Double(values.count)}}
                    let n=Int(t["meanEpochs"]!),first=mean(Array(norms.prefix(n))),last=mean(Array(norms.suffix(n)))
                    guard first.isFinite,last.isFinite else {mark(rule,.withheld,"Derived norm means are not finite.");continue}
                    if last>=max(first,t["baselineFloor"]!)*t["factor"]!,last-first>=t["minimumAbsoluteGrowth"]! {emit(rule,"First/last \(n)-epoch norm means: \(number(first)) → \(number(last)).","Large relative growth in the measured global gradient norm; this does not identify a failing layer or establish a cause.",refs:batches.filter{epochs.contains($0.epoch)}.map(\.reference),values:["firstMean":first,"lastMean":last])}
                }
            case "activation.zero-sample.v1","activation.saturation-sample.v1":
                let component=rule.id == "activation.zero-sample.v1" ? "relu":"tanh"
                let samples=input.activations.filter{$0.component==component && $0.values.count>=Int(t["minimumScalars"]!)}
                guard !samples.isEmpty else {mark(rule,.withheld,"No eight-scalar \(component) sample is available; capture may be omitted or unsupported.");continue}
                for sample in samples {
                    let fraction=Double(sample.values.filter{component == "relu" ? $0==0:abs($0)>=t["absoluteThreshold"]!}.count)/Double(sample.values.count)
                    if fraction>=(t["zeroFraction"] ?? t["minimumFraction"]!) {emit(rule,"\(Int(fraction*Double(sample.values.count)))/\(sample.values.count) saved \(component) scalars meet the disclosed threshold.","This is a small activation sample, not evidence about the whole layer or dataset.",refs:["activations.json:node=\(sample.nodeID)"],values:["sampleScalars":Double(sample.values.count),"observedFraction":fraction],node:sample.nodeID)}
                }
            case "optimization.plateau.v1":
                let metrics=input.metrics.compactMap(\.metric),n=Int(t["observations"]!)
                guard !input.continuation,metrics.count>=n else {mark(rule,.withheld,"Twenty epochs from a fresh uninterrupted run are required.");continue}
                let window=Array(metrics.suffix(n))
                guard window.allSatisfy({$0.epoch>0 && $0.trainLoss.isFinite && $0.validationLoss.isFinite && $0.trainLoss>=0 && $0.validationLoss>=0 && $0.accuracy.map {$0.isFinite && (0...1).contains($0)} == true}),zip(window,window.dropFirst()).allSatisfy({$0.epoch<Int.max && $1.epoch==$0.epoch+1}) else {mark(rule,.withheld,"Sampled, duplicated, invalid or discontinuous epochs cannot establish this plateau.");continue}
                let m=Int(t["meanObservations"]!),first=mean(window.prefix(m).map(\.trainLoss)),last=mean(window.suffix(m).map(\.trainLoss))
                if first.isFinite,last.isFinite,last>=t["minimumEndLoss"]!,abs(last-first)<=max(first,0.000001)*t["maximumRelativeChange"]! {emit(rule,"First/last \(m)-epoch training means: \(number(first)) → \(number(last)).","Little observed progress in this binary cross-entropy window; convergence and data limits remain alternative explanations.",refs:window.map{"metrics.json:epoch=\($0.epoch)"},values:["firstMean":first,"lastMean":last])}
            case "data.coarse-holdout.v1":
                guard input.metrics.allSatisfy({$0.metric?.accuracy != nil}) else {mark(rule,.withheld,"Actual observations omit classification accuracy; a classification resolution claim is unavailable.");continue}
                guard let count=input.validationRows else {mark(rule,.withheld,"No verified held-out row count is available.");continue}
                if count<Int(t["minimumRecommendedRows"]!) {emit(rule,"\(count) held-out rows; each equally weighted prediction changes accuracy by \(number(100/Double(count))) percentage points.","The measured score has coarse resolution; this is not an independence or leakage diagnosis.",refs:["manifest.json:provenance.validationSamples"],values:["validationRows":Double(count),"accuracyStep":1/Double(count)])}
            case "runtime.memory.v1":
                guard let error=input.rawError,!error.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty else {mark(rule,.withheld,"No runtime error was supplied.");continue}
                let text=error.lowercased(),terms=["out of memory","cannot allocate memory","can't allocate memory","oom when allocating tensor","failed to allocate memory"]
                if terms.contains(where:{text.contains($0)}) {emit(rule,"The runtime error contains an explicit allocation-failure phrase.","An allocation failed; device capacity, fragmentation and simultaneous workloads remain different hypotheses. Inspect the original error locally.",refs:["outcome.json:error"],values:["matchedError":1])}
            default:throw DiagnosticContractError("Unsupported measured evaluator.")
            }
            mark(rule,findings.count>start ? .emitted:.notTriggered,findings.count>start ? "Finding emitted from the supplied measurements.":"Measurements evaluated; the disclosed threshold was not met.")
        }
        let report=MeasuredDiagnosticReport(runID:input.runID,requestSHA256:input.requestSHA256,inputSHA256:try WorkflowArchiveCodec.hash(WorkflowArchiveCodec.encode(input)),findings:findings,checks:checks)
        try report.validate();return report
    }
    private static func mean(_ values:[Double])->Double {values.reduce(0){$0+$1/Double(values.count)}}
    private static func number(_ value:Double)->String {String(format:"%.6g",locale:Locale(identifier:"en_US_POSIX"),value)}
}
