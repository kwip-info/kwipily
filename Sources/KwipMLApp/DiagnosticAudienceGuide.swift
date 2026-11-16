import Foundation

/// Presentation only. Original observations, pinned registries and reports retain
/// their identities; this guide neither evaluates evidence nor applies a repair.
struct DiagnosticAudienceGuide {
    let observedFact: String
    let meaning: String
    let action: String
    let alternatives: String
    let limits: String
    let isFallback: Bool

    /// Matches the existing unscored iteration/transfer entry's searchable content.
    static let lessonSearchTerm = "diagnostics"
    static var knownRuleIDs: Set<String> { Set(explanations.keys) }
    static func make(ruleID: String, observedFact: String) -> Self {
        let explanation = explanations[ruleID]
        return Self(observedFact: observedFact,
            meaning: explanation?.meaning ?? "This is an observation supplied by the diagnostic report. This reader has no reviewed beginner explanation for this rule.",
            action: explanation?.action ?? "Review the original observation, rule details and evidence before making any change. Keep the run record for comparison.",
            alternatives: explanation?.alternatives ?? "A finding can have more than one explanation. No cause is inferred by this presentation.",
            limits: explanation?.limits ?? "This guide cannot explain this rule's prerequisites or thresholds. Missing guidance does not mean the check passed or the model is healthy.",
            isFallback: explanation == nil)
    }
    private struct Explanation {
        let meaning: String
        let action: String
        let alternatives: String
        let limits: String
    }
    private static let explanations: [String: Explanation] = [
        "generalization.divergence.v1": .init(
            meaning: "The error score improved on training rows while it worsened on held-out checking rows. The two groups disagree about the benefit of continued learning.",
            action: "In History, compare the training and validation curves. Inspect the captured rows, labels and Split settings. Keep the same checking set and seed; compare an earlier checkpoint or a separately named shorter fresh run. Keep final test rows reserved.",
            alternatives: "A small or different checking group, changed label patterns and noisy labels can create this pattern. Learning training-specific details is another possible explanation.",
            limits: "This is not proof of overfitting or its cause. The check requires twenty consecutive epochs from a verified fresh run with fixed prepared rows and evaluation settings. Its thresholds are disclosed product heuristics, not confidence estimates. Missing context or continuation withholds the check."),
        "measurement.nonfinite.v1": .init(
            meaning: "A recorded measurement is not an ordinary finite number. NaN means a calculation did not produce a usable number; infinity is also unsuitable for the comparison. Trend advice cannot be trusted for these measurements.",
            action: "Read the original run message. On Build, inspect the named component and its Configuration. Select the relevant component and open Results → Output to check captured source values and scaling; use Results → Ports & connections → Inspect for a specific port when evidence is available, then retry an explicit controlled change.",
            alternatives: "Invalid source values, very large calculations or damaged measurements can produce this symptom. The finding alone does not identify which occurred.",
            limits: "This rule observes measurements, not the exact failing operation. It withholds trend analysis rather than filling in missing or invalid values."),
        "trend.generalization-gap.v1": .init(
            meaning: "Training results and validation results are moving apart. Training rows teach the model; validation rows check examples held out of that training. The pattern may mean the model is becoming too specific to training examples.",
            action: "In History, compare the recorded training and validation curves. Keep the split fixed and compare an earlier checkpoint or a shorter fresh run before choosing a larger model. Review Split on Build and the captured preparation.",
            alternatives: "Different data groups or evaluation procedures can also produce a gap. Repeatedly tuning on validation can make those checking rows less useful.",
            limits: "This heuristic is not proof of overfitting. The legacy rule does not independently verify unchanged partitions, complete sampling or continuation boundaries. Absence of a finding is not a clean bill of health."),
        "trend.loss-growth.v1": .init(
            meaning: "Recorded loss increased sharply compared with earlier observations. Loss measures how wrong predictions are under the selected training rule; a larger loss needs investigation.",
            action: "Compare the same run's earlier values in History. On Build, review feature scaling, model changes and Train’s learning rate. Try one controlled shorter run with the same split and seed.",
            alternatives: "Harder examples, a changed loss calculation or a changed evaluation can also increase loss. A learning-rate change is an experiment, not a proven repair.",
            limits: "This legacy rule uses recorded losses, not measured gradient norms. It cannot diagnose exploding gradients or independently verify sampling and continuation boundaries."),
        "error.device.v1": .init(
            meaning: "The runtime message mentions a compute device or backend. A backend is the library that executes the model; a device is the processor it uses.",
            action: "Read the exact error, then select Train on Build and review Device in Run settings. Choose a verified supported option explicitly. Check the operation named in the message before retrying.",
            alternatives: "The requested processor may be unavailable, an operation may be unsupported, or a matching word may be incidental to another error.",
            limits: "This is a message classifier, not a hardware probe. It does not switch devices automatically or establish support for an operation."),
        "error.shape.v1": .init(
            meaning: "The runtime message may describe incompatible array sizes or connections. A shape lists the number of values in each direction; two connected components must agree on what they send and receive.",
            action: "On Build, select the component named in the original error. Compare its incoming connector metadata with its declared sizes in Configuration. Repair the setting or connection, check validation, then retry.",
            alternatives: "A changed upstream component, reversed input or disconnected branch can cause incompatibility. Matching words can also occur in unrelated messages.",
            limits: "The finding classifies the message; it does not independently measure the current tensors or prove which edit will solve the problem. Do not reshape data simply to hide a mismatch."),
        "error.data.v1": .init(
            meaning: "The runtime message may concern a source table, column or mapping. Features are the columns used as inputs; the target is the answer column.",
            action: "On Build, select the source or data component named in the error. Review its Configuration and selected feature/target columns. Open Results → Output for retained captured tables, or Results → Ports & connections → Inspect for a specific port; read the original message for unavailable rows or failed preparation. Fix the cited issue before retrying.",
            alternatives: "A missing column, invalid value, changed source or mistaken mapping can cause a data error. A matching word alone does not establish which case occurred.",
            limits: "The message classifier does not reread or repair source records. Do not silently drop rows to make an error disappear. A captured table describes that run, not later edits to the external file."),
        "error.worker.v1": .init(
            meaning: "The worker, the separate process that runs the computation, reported an error. The available classifier could not identify a more specific category.",
            action: "Read the original error in Activity or the recorded History outcome. Keep that record, inspect any cited component or input, and change only a setting supported by the evidence before retrying.",
            alternatives: "Many source, runtime, resource or computation problems can reach this fallback. The message alone does not establish a cause.",
            limits: "This finding is an error summary, not a diagnosis or a successful model. Missing evidence must stay unavailable rather than being presented as an evaluated healthy result."),
        "gradient.nonfinite.v1": .init(
            meaning: "Captured update signals include NaN or infinity. A gradient describes how a small change influences the training result; these values are not usable ordinary finite numbers.",
            action: "Review the captured evidence and input ranges. On Build, inspect Train’s run and optimizer settings. If Propose lower learning rate… is offered, review that one change before Apply; otherwise make a supported manual experiment. Preserve split and seed when comparing the next run.",
            alternatives: "Large inputs, an unstable operation or another numerical issue can produce this symptom. The capture does not prove that learning rate caused it.",
            limits: "Only scheduled batches were observed, and global counts do not locate a failing layer. Apply changes configuration and does not run training. Partial failed-run observations remain separate from the completed measured-report contract."),
        "gradient.missing.v1": .init(
            meaning: "Some parameters had no gradient tensor in a captured batch. A missing signal is different from zero: zero is a present value, while missing means no value was recorded for that parameter.",
            action: "On Build, follow the model paths from inputs to Model Output and Train. Inspect whether the branches you intend to train contribute to the result. Review the captured source before altering an intentional unused branch.",
            alternatives: "A conditional branch or intentionally unused parameter can legitimately have no update signal. Not every missing gradient is a broken model.",
            limits: "Captured counts are global and sampled. They do not identify every absent parameter or prove that the whole network is disconnected. More specific comparisons may be withheld."),
        "gradient.zero.v1": .init(
            meaning: "Every present captured gradient value was zero across the required sampled epochs. An epoch is one pass through training rows. These samples may not lead to a weight update.",
            action: "Inspect sampled activations and recorded loss in History or bottom Results. On Build, check the inputs and activation choices. Compare a known changing input or an earlier run before changing the architecture.",
            alternatives: "The model may already have settled, sampled rows may be unrepresentative, or a numerical/activation effect may stop changes.",
            limits: "The rule needs present finite gradients across at least three distinct captured epochs. All-zero samples do not prove that all updates or all data are zero, nor that training is broken."),
        "gradient.growth.v1": .init(
            meaning: "The overall size of captured gradients grew a lot compared with earlier samples from the same run. A norm is one number summarizing the size of a group of values; growth is a comparison, not just a large constant value.",
            action: "Review earlier and later evidence. Keep the split and seed fixed. If Propose lower learning rate… is offered, review and explicitly apply that single change, then run a new comparison yourself. Revert this change is available only while its guarded state still matches.",
            alternatives: "Difficult samples or a different loss scale can also change gradient size. Global growth does not identify which layer contributed most.",
            limits: "This disclosed product threshold is not proof of a cause. It needs sufficient finite samples from an uninterrupted fresh run. Applying a proposal does not run training or guarantee that the next result improves."),
        "activation.zero-sample.v1": .init(
            meaning: "The saved ReLU sample contains zeros. ReLU keeps positive inputs and changes negative inputs to zero, so zeros can be the expected output.",
            action: "Use Show component on canvas, then inspect the ReLU's input and preceding Linear component on Build. Compare captured values from a controlled different input when available before replacing the activation.",
            alternatives: "Negative inputs or a limited sample may explain these zeros. The activation can still produce useful positive values for other observations.",
            limits: "At most eight saved scalar values are available per node. A scalar is one number. These samples do not establish a dead neuron, a whole-layer distribution or behavior across the dataset."),
        "activation.saturation-sample.v1": .init(
            meaning: "Saved Tanh values are close to -1 or 1. Tanh squeezes inputs into that range; near its limits, changes to the input can have a small effect on its output.",
            action: "Use Show component on canvas and inspect the preceding values and feature scaling on Build. Compare one controlled normalization or input change only after checking that it preserves the data contract.",
            alternatives: "Strong inputs or an intentional bounded representation may produce near-limit values. The tiny sample may not represent other observations.",
            limits: "The sample does not describe the whole layer or estimate its distribution. Gradient behavior is not measured by activation values alone, and saturation is not automatically an error."),
        "optimization.plateau.v1": .init(
            meaning: "Recent training loss changed very little. This is a plateau: the recorded error is not moving much within the reviewed window.",
            action: "Compare the recorded curve with a simple baseline and inspect the target mapping on Build. Keep the split and seed fixed; try one supported optimizer setting in a new run rather than adding many layers at once.",
            alternatives: "The model may have settled, the labels may be noisy, or the available features may not predict the target well. More training or a higher learning rate need not help.",
            limits: "The check needs twenty consecutive finite epochs from a fresh uninterrupted run and uses this binary classification loss. It does not establish a cause or generalize the threshold to other loss calculations."),
        "data.coarse-holdout.v1": .init(
            meaning: "The checking set has few rows, so each correct answer can change accuracy by a large step. With ten equally weighted rows, one answer changes accuracy by ten percentage points.",
            action: "Inspect the captured preparation and held-out row count. On Build, review Split and whether rows represent independent observations. Keep the original partition as a baseline; make a separately named experiment if more suitable data becomes available.",
            alternatives: "A small checking set can be a deliberate teaching example or reflect limited available data. More rows may repeat the same people or events rather than add independent evidence.",
            limits: "The count describes score resolution, not statistical confidence. It does not prove rows are independent, identify leakage or estimate population accuracy. Increasing validation percentage also changes the training set."),
        "runtime.memory.v1": .init(
            meaning: "The runtime explicitly reported an allocation failure: it could not reserve memory for a calculation. Memory stores input arrays, intermediate values, weights and update state.",
            action: "Read the exact allocation error. On Build, review the failing tensor shape and Train’s batch size. Try a smaller supported batch or shape as one explicit comparison; select a different verified device only if you intend that change.",
            alternatives: "Device capacity, fragmentation or other simultaneous workloads can produce similar errors. The amount of free physical memory is not established by the message.",
            limits: "This is error classification, not a memory measurement or proof of a leak. The rule does not change devices automatically, and a lower batch size is an experiment rather than a guaranteed fix.")
    ]
}
