import SwiftUI
import KwipMLCore

/// Unscored, bounded practice over the current universal-canvas controls.
enum IterationTransferLessonContent {
    struct Topic: Identifiable {
        let id: String
        let title: String
        let goal: String
        let steps: [String]
        let observation: String
        let limits: String
    }
    static let assessmentDisclosure = "Visual practice and manual review only; no automatically scored completion or mastery receipt. Save the example as your own project to keep your work."
    /// A checked example recipe, not a mutation of the learner’s Train settings.
    static let practiceRecipe = WorkflowExperimentSpecification(
        axes: [.init(field: "learningRate", values: [.number(0.03), .number(0.01)])],
        budget: .init(maximumRuns: 2, maximumTotalEpochs: 6))
    static let topics: [Topic] = [
        .init(id: "run", title: "Control the size of a run",
            goal: "Make a small, repeatable run before spending more time or processing more data.",
            steps: [
                "Choose Open parallel classifier practice, then select Train on its canvas. Review Run settings, optimizer and batch controls in Configuration. Device chooses a supported processor. Epochs counts passes through training rows; Batch size is the group used for one update. Seed helps repeat starting random choices, while Learning rate controls the size of weight adjustments.",
                "Start with CPU, 3 epochs, batch size 32, seed 42 and learning rate 0.03. Choose Run Flow. Clear the canvas selection to watch Results → Activity. Select a data component and open Results → Output to inspect captured data, then open the completed record in History.",
                "For processing limits, open project Configure and its execution settings. Review Time allowance and Worker memory policy. These are separate from the model's training settings. To train on fewer observations, use Train’s Data access controls to select training positions or a finite prefix; validation and protected test observations remain separate."
            ],
            observation: "Compare the configured batch size with the number of rows actually used. Changing batch size changes how many rows are used per update; it does not select a smaller dataset. Partition has its own seed, separate from the model seed.",
            limits: "Device choices depend on the verified runtime. Selected training positions and finite prefixes must fit the prepared training population. Captured preview limits only change what is displayed, not how much trains. Worker memory is sampled resident memory, not a hard whole-app or GPU memory cap. A cancelled or failed run is not a completed trained model."),
        .init(id: "experiments", title: "Compare two controlled runs",
            goal: "Use one fixed preparation and change a single training setting. An experiment variant is one set of choices in the comparison.",
            steps: [
                "Select Train and open Experiments → Edit combinations. Choose Add value twice for Learning rate and enter 0.03 and 0.01 in the separate fields; leave other combination fields empty so they keep their ordinary values. Keep Epochs at 3 in Train’s Run settings. Set Maximum runs to 2 and Total epoch budget to 6: two runs of three epochs each.",
                "Select Preview exact runs in the editor. Review both planned runs, then Save combinations. Select Train on the canvas and choose Run experiment group. Ordinary Run Flow still uses the single-run settings; saving combinations alone does not execute the group.",
                "Open History and inspect a completed variant. Choose Set as baseline. Inspect the other completed variant and choose Compare. A baseline is the saved result you use as a reference. Keep split and seed fixed while comparing the learning-rate change."
            ],
            observation: "Compare validation loss and accuracy alongside configuration differences. Loss measures error under the training rule; accuracy is the fraction of correct answers. A change that improves training alone may fail on the checking rows.",
            limits: "Experiment groups currently require fresh training. Combinations vary Train’s run settings, not arbitrary model or data searches. Runs execute sequentially and stop on the first failure. Preview checks the graph/settings; files and device availability are checked during execution. Saved evidence is not a new check of today's external files. Two results do not establish statistical significance."),
        .init(id: "checkpoint", title: "Resume, fine-tune or start fresh",
            goal: "Choose what state a new training run should inherit. A checkpoint saves learned weights and, for exact resume, the additional training state needed to continue.",
            steps: [
                "Finish a fresh CPU run first. Select Train and find Training mode. Start fresh initializes new weights. Exact resume continues the saved training process. Fine-tune weights starts a new training process using earlier learned weights.",
                "For Exact resume, select Choose checkpoint and choose a completed compatible CPU run. In Train’s Run settings, choose CPU and a target total epoch count greater than the parent's completed count. For example, a parent with 3 completed epochs and a total target of 5 continues for 2 more epochs.",
                "For Fine-tune weights, choose the parent explicitly and configure the new data, seed and epoch budget. Here the epoch count belongs to the new run. For Start fresh, a retained checkpoint selection is ignored; Clear checkpoint removes it. Run the selected mode and inspect the recorded parent and outcome in History."
            ],
            observation: "Exact resume restores weights, the optimizer's update state, random state, completed epochs and history. Fine-tuning keeps weights but resets the other training state. These are different experiments even if their model pictures look alike.",
            limits: "Exact resume requires a CPU parent and matching architecture, runtime and data. Compatibility is checked again when running; choosing a listed checkpoint does not guarantee compatibility with later edits. Fine-tuning still requires compatible weights. Failed or incomplete captures are not offered as completed parents. These contracts describe the PyTorch flow; backend starter examples have their own narrower limits."),
        .init(id: "export", title: "Take a trained model with you",
            goal: "Distinguish a model diagram, its learned weights and the preparation needed before prediction. Inference means using a trained model to predict new inputs.",
            steps: [
                "In History, select a verified completed PyTorch run. Open Saved model source, then Export trained model…. Review Input contract and Column order. The package uses the captured model, so later canvas edits do not rewrite it.",
                "Review the listed files and choose a Package folder name. Select Choose destination… to save the package in a new folder; an existing folder is not replaced. Show in Finder opens the saved location.",
                "Review the package's README and inference helper. The helper expects prepared inputs in the recorded shape, type and column order. Before giving it new raw data, apply the same preparation rules used in training. Running outside KwipML uses Python and the recorded runtime; creating and inspecting the package here requires no code editing."
            ],
            observation: "The captured source describes the architecture; checkpoint.pt holds learned state. Predictions depend on both, plus correctly prepared inputs. The synthetic input example in the export is a usage demonstration, not evidence of accuracy on new data.",
            limits: "The portable trained-model package does not include fitted preprocessing or private source rows. It includes user component names, topology and checkpoint/provenance state. Standalone inference requires Python with the recorded PyTorch version. Exporting a graph alone does not export trained weights. Backend foundations use their own state formats; weights are not automatically interchangeable."),
        .init(id: "blocks", title: "Open and edit a reusable Block",
            goal: "Keep a larger architecture understandable by grouping connected components. A Block is reusable structure; its own canvas can be opened beside the main flow.",
            steps: [
                "Choose Open separable Block practice. Select the Block and choose Open reusable block. It contains a depthwise filter (each channel handled separately) followed by a pointwise filter (channels mixed at each position). Return to the main canvas using its tab.",
                "To practice the file-backed lifecycle, return to the main project and choose File → Convert to file-backed flow. Choose the folder when prompted. The main flow and subflows now have separate files; select the Block and open its reusable canvas again.",
                "Edit and save the subflow, then return to the main tab. A pinned revision is the specific saved version this caller uses. Saving the subflow does not silently change that pin. Select the Block, choose Check for saved update, review the offered change, then choose Adopt revision or Keep current. You can undo adoption."
            ],
            observation: "The caller's pinned revision makes its architecture reproducible. Editing a child file and adopting its new revision are separate actions. The separable example's exposed fields let you configure its two stages without unfolding every control.",
            limits: "Saving a child file does not update the caller’s pinned revision; adopt a reviewed saved revision explicitly. Reusing a Block definition does not share learned weights between instances. Each instance has independent parameters. Adoption updates every instance of that shared definition in the current flow; existing runs remain unchanged. File-backed subflows require retained folder access and available saved definitions. A private draft and a saved reusable revision have different ownership; do not assume a draft has updated callers."),
        .init(id: "diagnostics", title: "Review evidence before applying a repair",
            goal: "Treat a diagnostic as a supported observation and a possible explanation, then test a controlled change. A finding is not an automatic verdict about model quality.",
            steps: [
                "Inspect a verified run's measured diagnostics in History, or clear the canvas selection and open Results → Diagnostics for available run evidence. Read what was observed and which checks were withheld. Withheld means the required evidence was not captured, not that the model passed that check.",
                "When a supported finding offers Show component on canvas or Show Train component, use it to locate the affected part. For eligible gradient-growth or nonfinite-gradient findings, choose Propose lower learning rate… and review the old and proposed values. A gradient measures how a small change influences the training result; nonfinite means a value was not an ordinary finite number.",
                "Choose Apply learning-rate change only if you want that specific proposed edit. Keep the data split and seed fixed, run again yourself and compare outcomes. Revert this change restores the proposal's prior setting when the current state still allows it. If no supported proposal is offered, review the evidence and configure the relevant component manually."
            ],
            observation: "Compare the repaired run with the captured baseline. Check whether the observed issue changed and whether validation improved; a lower learning rate is an experiment, not a guaranteed solution. If training error falls while held-out error rises, inspect the checking rows and labels before concluding that the model memorized training examples. The measured divergence check needs twenty consecutive epochs from a verified fresh run with fixed data and evaluation; missing evidence is withheld.",
            limits: "Findings are observations and hypotheses, not proof of a cause or of model health. A supported repair changes configuration and does not run training. Apply/Revert validates the current revision and data binding; stale proposals can be rejected. No finding can prove safety when the needed evidence is missing, and sampled captures do not describe every batch.")
    ]
}

struct IterationTransferLearningView: View {
    @ObservedObject var store: WorkspaceStore
    @Environment(\.lessonRailPresented) private var inRail
    var body:some View {
        if inRail {
            VStack(alignment:.leading,spacing:10) {
                Button("Open parallel classifier practice") {_ = store.loadPreparedWorkflowExample()}
                Button("Open separable Block practice") {_ = store.loadSeparableConv2dWorkflowExample()}
                Button("Open deliberate label-shift example") {_ = store.loadGeneralizationPracticeExample()}
                Button("Open consistent-label comparison") {_ = store.loadGeneralizationPracticeExample(control:true)}
            }.buttonStyle(FlatButtonStyle()).disabled(store.isBusy)
        } else {referenceBody}
    }
    private var referenceBody:some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Iteration, saved state and reusable flows").font(.title3.weight(.semibold))
            Text(IterationTransferLessonContent.assessmentDisclosure).font(.system(size: 13)).foregroundStyle(MLTheme.muted)
            HStack(spacing: 12) {
                Button("Open parallel classifier practice") { _ = store.loadPreparedWorkflowExample() }
                Button("Open separable Block practice") { _ = store.loadSeparableConv2dWorkflowExample() }
            }.buttonStyle(FlatButtonStyle()).disabled(store.isBusy)
            ForEach(IterationTransferLessonContent.topics) { topic in
                InspectorSection(topic.title, symbol: "square.stack.3d.up", identifier: "learn.iteration.\(topic.id)", initiallyExpanded: topic.id == "run") {
                    LessonRichText(text: topic.goal)
                    if topic.id == "diagnostics" {
                        HStack(spacing: 12) {
                            Button("Open deliberate label-shift example") { _ = store.loadGeneralizationPracticeExample() }
                            Button("Open consistent-label comparison") { _ = store.loadGeneralizationPracticeExample(control: true) }
                        }.buttonStyle(FlatButtonStyle()).disabled(store.isBusy)
                        LessonRichText(text: "**Compare the public examples**\n\nEach uses the same simple model and twenty-epoch budget. In the label-shift example, the answers on validation rows are deliberately reversed. Run both yourself, then compare History. This shows why a warning calls for checking the data; it does not prove memorization. Sixteen future test rows stay reserved and unused.")
                    }
                    LessonRichText(text: "**Try it**\n\n" + topic.steps.enumerated().map { "\($0.offset + 1). \($0.element)" }.joined(separator: "\n\n"))
                    LessonRichText(text: "**What to inspect**\n\n" + topic.observation)
                    LessonRichText(text: "**Current limits**\n\n" + topic.limits).foregroundStyle(MLTheme.muted)
                }
            }
        }
    }
}
