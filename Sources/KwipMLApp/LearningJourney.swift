import Foundation

/// Recommendations over existing Learn entries. Navigation carries no completion or execution authority.
struct LearningJourney: Identifiable, Sendable {
    enum Evidence: String, Sendable {
        case unscoredPractice = "Editable practice · unscored"
        case assessedLab = "Lab · guided and assessment modes"
    }
    struct Step: Identifiable, Sendable {
        let entryID: String
        let title: String
        let advice: String
        let evidence: Evidence
        var isOptional = false
        var id: String { entryID }
    }
    let id: String
    let title: String
    let objective: String
    let prerequisites: String
    let runtimeGuidance: String
    let nextAction: String
    let steps: [Step]
    var optionalSteps: [Step] = []
    var requiresCompletionGate: Bool { false }

    private static let checker = "Construction labs use the supported managed PyTorch runtime. Check the lesson’s current runtime status and execution allowance before checking a construction. That allowance limits the checker’s work, not your time to learn."
    private static func lab(_ id: String, _ title: String, _ advice: String, optional: Bool = false) -> Step {
        .init(entryID: id, title: title, advice: advice, evidence: .assessedLab, isOptional: optional)
    }
    private static func practice(_ id: String, _ title: String, _ advice: String) -> Step {
        .init(entryID: id, title: title, advice: advice, evidence: .unscoredPractice)
    }
    static let all: [Self] = [
        .init(id: "first-model", title: "First model",
              objective: "Build a classifier visually, follow its connections and inspect what it produces.",
              prerequisites: "Start here. No Python or calculus is required; a row is one observation and a column is one feature.",
              runtimeGuidance: "The classifier uses the supported managed PyTorch runtime. Train controls device, epochs (passes through training rows), seed and batches. Start with a small run; this is separate from a construction checker’s allowance.",
              nextAction: "Continue with Data and shape safety. Affine combinations is an optional deeper explanation.",
              steps: [practice("guided", "Build your first flow", "Follow the supplied canvas walkthrough. Inspect results, introduce a width mismatch and repair it."),
                      practice("parallel", "Follow two branches", "Trace how both branches contribute to one prediction."),
                      practice("preprocess", "Prepare using training rows", "Use one Preprocess to learn replacements, categories and scaling from training only, then reuse the same state on validation and test populations.")],
              optionalSteps: [lab("affine", "Why a learned layer works", "Explore how weights combine inputs and an offset shifts the result.", optional: true)]),
        .init(id: "data-shapes", title: "Data and shape safety",
              objective: "Clean familiar tables, preserve observation identity and rearrange arrays safely.",
              prerequisites: "Basic row and column familiarity is enough. The table practices below are editable examples, not scored mastery checks.",
              runtimeGuidance: "Table practices use the local managed preparation runner and bundled sample sources. Tensor layout uses the managed PyTorch checker; inspect its runtime status and check budget before execution. Source samples and partial observations are not complete-run evidence.",
              nextAction: "Continue with Layers and branches once you can explain which observation owns each value.",
              steps: [practice("data-preparation-course", "Check and prepare tables", "Inspect source columns, choose names and types, and keep training, validation and final-test rows separate."),
                      practice("columns", "Choose names and types", "Select, order, rename and explicitly convert known columns in one grid."),
                      practice("filter", "Choose rows and find text patterns", "Compare Matching and Nonmatching outputs using conditions, an expression or a text pattern."),
                      practice("formula", "Create and clean useful columns", "Calculate values, extract fields, clean patterned text or parse real date/time values; inspect output types and missing values."),
                      practice("preprocess", "Prepare fixed rules and learned state", "Clean text before Partition, or learn replacements, categories and scaling from training only."),
                      practice("table-output", "Save a prepared table", "Write a complete table to a local file or database and inspect its recorded outcome."),
                      lab("tensor-layout", "Rearrange without losing values", "Try reshape, transpose and flatten. Keep observations distinct while changing the array’s directions.")]),
        .init(id: "layers-branches", title: "Layers and branches",
              objective: "Compare individual responses, combine branches and reuse a named Block.",
              prerequisites: "Try tensor layout first so rows, features and array directions feel familiar.",
              runtimeGuidance: checker,
              nextAction: "Choose Sequence and attention or Training and generalization. Optional spatial labs extend the same ideas.",
              steps: [lab("relu-lesson", "Keep positive responses", "Compare negative, zero and positive inputs through ReLU."),
                      lab("sigmoid-lesson", "Squeeze values into a range", "Explore a smooth response between zero and one."),
                      lab("tanh-lesson", "Keep a signed response", "Compare a smooth response between minus one and one."),
                      lab("configurable-activations", "Adjust an individual response", "Change a slope or shape and compare measured results."),
                      practice("parallel", "Combine branches", "Follow each branch into the combined prediction."),
                      lab("module-composition", "Build and reuse Blocks", "Distinguish stored children from connected calculations; inspect private copies and order."),
                      practice("module-composition-examples", "Inspect connected Blocks", "Open a worked example in guided mode, then check the construction; a guided result is practice evidence.")],
              optionalSteps: [lab("convolution", "Shared windows", "Apply the same learned window across a grid.", optional: true),
                              lab("pooling", "Summarize a window", "Compare selecting a largest value with averaging a region.", optional: true),
                              lab("channel-normalization", "Normalize channel patterns", "Compare current groups with remembered summaries.", optional: true),
                              practice("spatial-practice", "Build a spatial flow", "Extend familiar connections to a grid-based example.")]),
        .init(id: "sequence-attention", title: "Sequence and attention",
              objective: "Represent categories, carry sequence memory and decide which positions can communicate.",
              prerequisites: "Recommend tensor layout and learned-layer basics. This is finite architecture practice, not a complete GPT-2 or Llama training course.",
              runtimeGuidance: checker,
              nextAction: "Continue with Training and generalization. You do not need LogSoftmax for the basic attention exercises.",
              steps: [lab("embedding", "Turn categories into vectors", "Look up a learned row for each category rather than treating its number as a quantity."),
                      lab("matrix", "Combine rows and columns", "Follow which values meet during a matrix product."),
                      lab("softmax", "Turn scores into shares", "Choose which direction competes; the shares in that direction add up to one."),
                      lab("recurrent-memory", "Carry a sequence summary", "Start with RNN, then compare the other memory choices and supplied starting states."),
                      lab("attention-connections", "Choose communicating positions", "Start with Scaled attention or MultiheadAttention. Compare allowed and blocked connections before encoder, decoder and Transformer alternatives.")]),
        .init(id: "training-generalization", title: "Training and generalization",
              objective: "Compare outcomes, control adjustments and protect observations reserved for checking.",
              prerequisites: "Recommend First model and basic per-value responses. Begin with simple errors and updates before the more specialized alternatives.",
              runtimeGuidance: checker + " The protected-test example trains a real classifier under its Train settings budget; keep its final-test branch separate.",
              nextAction: "Continue with Saved state and transfer. Use the optional LogSoftmax exercise before losses that need its compressed probability representation.",
              steps: [practice("train-data-access", "Configure a training run", "Select Train to choose the objective, optimizer, rate schedule, gradients, data access and batches. Select each layer to configure its starting weights, biases and trainability, then compare recorded outcomes. This editable practice is unscored."),
                      practice("holdout", "Protect the final check", "Keep related subjects and future observations separate. Connect Train’s trained model or a Saved checkpoint to Evaluate with Test selected.")],
              optionalSteps: [lab("remaining-activations", "Prepare logarithmic probabilities", "Choose LogSoftmax before NLL, KL or CTC when needed; inspect the objective’s required input representation on Train.", optional: true)]),
        .init(id: "saved-transfer", title: "Saved state and transfer",
              objective: "Restore learned values, preserve grouping and reuse a computation with declared input rules.",
              prerequisites: "Recommend Training and generalization. Configure ordinary data access and batches on Train. Historical assessed grouping evidence remains in Saved lessons.",
              runtimeGuidance: checker + " Exact saved-continuation lessons explicitly require CPU. Backend foundations are CPU introductions with their own runtime checks; they do not imply broad framework coverage.",
              nextAction: "Use the transfer checklist, then save an editable project of your own. Revisit a lab when its current result needs rechecking; LBFGS is not a prerequisite.",
              steps: [lab("checkpoint-lifecycle", "Restore or continue", "Distinguish restoring weights, fresh fine-tuning and an exact continuation."),
                      practice("checkpoint-examples", "Inspect saved-run examples", "Use a connected guided example to compare the saved state and the restored work."),
                      practice("train-data-access", "Choose training observations and batches", "Configure Data access and Batches on Train, then inspect the recorded run. Keep held-out validation separate; this ordinary workflow practice is unscored."),
                      lab("embedding-bag", "Optional: keep whole groups together", "A bag is one observation’s variable-length group of items. Explore it when your model needs grouped category IDs; ordinary table training does not require this detour.", optional: true),
                      lab("program-export", "Save a computation", "Declare supported input sizes and compare the reloaded result; exporting is not a guarantee that every calculation is supported."),
                      practice("iteration-transfer", "Apply the transfer checklist", "Compare controlled runs, saved-state choices, inference exports, reusable files and diagnostic evidence.")],
              optionalSteps: [practice("sklearn", "scikit-learn foundation", "Try the shipped binary logistic-regression example and inspect its saved coefficients."),
                              practice("tensorflow", "TensorFlow foundation", "Try the shipped dense classifier and inspect its input rules and saved weights.")])
    ]
    static func matching(_ query: String) -> [Self] {
        let terms = query.lowercased().split(whereSeparator: \.isWhitespace)
        return all.filter { journey in
            let fields = [journey.title, journey.objective, journey.prerequisites, journey.nextAction]
                + (journey.steps + journey.optionalSteps).flatMap { [$0.entryID, $0.title, $0.advice] }
            let text = fields.joined(separator: " ").lowercased()
            return terms.allSatisfy { text.contains($0) }
        }
    }
}
