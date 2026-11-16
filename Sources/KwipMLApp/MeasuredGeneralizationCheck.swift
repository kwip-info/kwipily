import Foundation
import KwipMLCore

/// Version-two check. It does not alter the pinned legacy or measured
/// v1 registries. The History bridge derives context from verified retained
/// artifacts; matching hashes here bind that context, not authenticate it.
enum MeasuredGeneralizationCheck {
    struct VerifiedContext: Equatable {
        var runID: String
        var requestSHA256: String
        var partitionSHA256: String
        var evaluationSHA256: String
        var completeEpochCapture: Bool
        var unchangedPartitions: Bool
        var unchangedEvaluation: Bool
        var binaryCrossEntropy: Bool
    }
    struct Result: Equatable {
        let check: MeasuredDiagnosticCheck
        let finding: MeasuredDiagnosticFinding?
    }
    static let rule = MeasuredDiagnosticRule(
        id: "generalization.divergence.v1", family: "generalization",
        title: "Training and held-out loss moved apart",
        prerequisites: ["Verified run/request, partition and evaluation identity", "Complete consecutive finite epoch capture", "Twenty epochs from a fresh PyTorch binary cross-entropy run"],
        thresholds: ["observations": 20, "meanObservations": 5, "minimumAbsoluteChange": 0.02, "relativeChange": 0.1, "minimumGapIncrease": 0.05],
        confounders: ["Thresholds are a fixture-reviewed product heuristic, not field-calibrated confidence", "Small or unrepresentative held-out data and label noise can affect the trend", "Unchanged settings do not prove representative partitions or absence of data leakage", "Loss reductions and different tasks are not comparable under this rule"],
        experiment: "Inspect the training and validation rows and compare an earlier checkpoint on the same validation split. Try a separately named shorter run with the same seed; keep final test data reserved.",
        lesson: "Loss is the error score the model tries to reduce. Validation rows are held aside from learning. Falling training loss while validation loss rises is an observed disagreement; it does not prove that the model memorized the training rows."
    )

    static func evaluate(_ input: MeasuredDiagnosticInput, context: VerifiedContext? = nil) throws -> Result {
        try input.validate()
        func withheld(_ reason: String) -> Result {
            .init(check: .init(ruleID: rule.id, status: .withheld, reason: reason), finding: nil)
        }
        guard input.backend == "pytorch", !input.continuation else {
            return withheld("This check requires a fresh PyTorch run; continuation and other task adapters are outside this revision.")
        }
        guard let context, context.runID == input.runID, context.requestSHA256 == input.requestSHA256,
              WorkflowArchiveCodec.isHash(context.partitionSHA256), WorkflowArchiveCodec.isHash(context.evaluationSHA256),
              context.completeEpochCapture, context.unchangedPartitions, context.unchangedEvaluation,
              context.binaryCrossEntropy else {
            return withheld("Verified complete capture, unchanged partition/evaluation identity and binary cross-entropy task context are unavailable or do not match this run.")
        }
        let metrics = input.metrics.compactMap(\.metric)
        let count = Int(rule.thresholds["observations"]!)
        guard metrics.count >= count else {
            return withheld("Twenty verified epochs are required; no missing observations are interpolated.")
        }
        guard metrics.allSatisfy({ $0.epoch > 0 && $0.trainLoss.isFinite && $0.validationLoss.isFinite && $0.trainLoss >= 0 && $0.validationLoss >= 0 && $0.accuracy.map {$0.isFinite && (0...1).contains($0)} == true }),
              metrics.first?.epoch == 1,
              zip(metrics, metrics.dropFirst()).allSatisfy({ $0.epoch < Int.max && $1.epoch == $0.epoch + 1 }) else {
            return withheld("Fresh complete capture must start at epoch one and preserve consecutive finite valid observations; duplicate, reordered or missing epochs are not evidence of divergence.")
        }
        let window = Array(metrics.suffix(count))
        let meanCount = Int(rule.thresholds["meanObservations"]!)
        // Divide before adding to keep a finite nonnegative mean bounded by its
        // inputs. Derived differences are checked independently below.
        func mean(_ rows: ArraySlice<EpochMetric>, _ key: KeyPath<EpochMetric, Double>) -> Double {
            rows.reduce(0) { $0 + $1[keyPath: key] / Double(rows.count) }
        }
        let first = window.prefix(meanCount), last = window.suffix(meanCount)
        let trainStart = mean(first, \.trainLoss), trainEnd = mean(last, \.trainLoss)
        let validationStart = mean(first, \.validationLoss), validationEnd = mean(last, \.validationLoss)
        let trainingDrop = trainStart - trainEnd, validationRise = validationEnd - validationStart
        let gapIncrease = (validationEnd - trainEnd) - (validationStart - trainStart)
        let values = ["trainingStartMean": trainStart, "trainingEndMean": trainEnd,
                      "validationStartMean": validationStart, "validationEndMean": validationEnd,
                      "trainingDrop": trainingDrop, "validationRise": validationRise, "gapIncrease": gapIncrease]
        guard values.values.allSatisfy(\.isFinite) else {
            return withheld("Derived loss comparisons are not finite; no trend claim is made.")
        }
        let absolute = rule.thresholds["minimumAbsoluteChange"]!, relative = rule.thresholds["relativeChange"]!
        guard trainingDrop >= max(absolute, trainStart * relative),
              validationRise >= max(absolute, validationStart * relative),
              gapIncrease >= rule.thresholds["minimumGapIncrease"]! else {
            return .init(check: .init(ruleID: rule.id, status: .notTriggered,
                                     reason: "Verified observations evaluated; both disclosed loss changes and gap widening were not all met."), finding: nil)
        }
        func number(_ value: Double) -> String {
            String(format: "%.6g", locale: Locale(identifier: "en_US_POSIX"), value)
        }
        let finding = MeasuredDiagnosticFinding(
            ruleID: rule.id, nodeID: nil,
            observation: "Epochs \(window.first!.epoch)–\(window.last!.epoch), first/last five-epoch means: training \(number(trainStart)) → \(number(trainEnd)); validation \(number(validationStart)) → \(number(validationEnd)); gap widened by \(number(gapIncrease)).",
            interpretation: "Measured training/validation divergence in this verified window. It does not prove overfitting or its cause; held-out composition and label noise remain possible explanations.",
            references: window.map { "metrics.json:epoch=\($0.epoch)" }, values: values
        )
        return .init(check: .init(ruleID: rule.id, status: .emitted,
                                 reason: "Verified observations meet the disclosed fixture-reviewed divergence thresholds."), finding: finding)
    }
}
