import SwiftUI
import KwipMLCore

struct WorkflowControlInspectorView:View {
    @ObservedObject var context:WorkflowControlNodeContext
    var configurationIssue:String?=nil
    var canCaptureGradients=false
    var executionHost:WorkflowWorkspaceSession?=nil
    var history:WorkflowHistoryStore?=nil
    @Environment(\.colorScheme) private var colorScheme
    @State private var showsExperiments=false
    @State private var usesNamedMapping=false
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            InspectorConfigurationSection(isEnabled:!context.isBusy || context.isGameSession || context.configuration["selfPlay"] != nil || context.configuration["fineTuning"] != nil) {
                if let configurationIssue {Text(configurationIssue).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled)}
                TextField("Component label",text:Binding(get:{context.authoredName},set:{_=context.rename($0)})).flatField().disabled(context.isBusy)
                if context.isModelDefinition {
                    WorkflowModelInputInspector(context:context)
                    if let output=try? ModelDefinitionOutputInspector(context:context) {output}
                } else if context.isEstimator {
                    PrebuiltModelInspector(context:context,host:executionHost)
                } else if context.isMapping && !context.isTrain {
                    if context.hasObjectiveMapping {LossTargetMappingInspector(context:context).id(context.contextIdentity)}
                    else if context.requiresBagMapping {BagTrainingMappingInspector(context:context).id(context.contextIdentity)}
                    else {WorkflowTrainingMappingEditor(context:context).id(context.contextIdentity)}
                } else if context.isRunSetup {
                    ForEach(WorkflowControlContracts.runFields,id:\.key) {field in
                        runField(field).disabled(field.key=="learningRate" && context.learningRateIsInactive)
                        if field.key=="learningRate",context.learningRateIsInactive {Text("Learning rate comes from Train’s optimizer settings. This saved Run setup value is inactive.").font(.caption2).foregroundStyle(MLTheme.muted)}
                    }
                    Button("Experiment combinations…") {showsExperiments=true}.buttonStyle(FlatButtonStyle())
                    cachePolicyEditor
                    gradientEditor
                } else if context.isGameSession {
                    if let host=executionHost {GameSessionInspectorView(context:context,host:host)}
                } else if context.isGenerateText {
                    if let host=executionHost {GenerateTextInspectorView(context:context,host:host)}
                } else if context.isEvaluateTest {
                    evaluationEditor
                } else if context.isFrozenModel {
                    if let host=executionHost,let history,host.owner === context.authority.owner,history.sessionID == host.sessionID {
                        LoadModelInspectorView(context:context,host:host,store:host.modelLoader,history:history)
                    }
                } else if context.isSaveModel {
                    if let host=executionHost,host.owner === context.authority.owner {
                        SaveModelInspectorView(context:context,host:host)
                    }
                } else if context.isTrain {
                    FlatChoiceControl(title:"Training task",selection:context.configuration["selfPlay"] == nil ? "supervised":"selfPlay",options:[.init(id:"supervised",title:"Supervised / fine-tuning"),.init(id:"selfPlay",title:"Self-play")]) {task in
                        let expected=context.configuration
                        _=context.authority.configure(task=="selfPlay" ? ["selfPlay":.object(SelfPlayConfiguration.defaults)]:["runSetup":.object([:]),"dataMapping":.object([:])],expected:expected)
                    }.disabled(context.isBusy)
                    if context.configuration["selfPlay"] != nil,let host=executionHost {SelfPlayInspectorView(context:context,host:host)} else {
                    FlatChoiceControl(title:"Model source",selection:context.configuration["fineTuning"] == nil ? "canvas":"loaded",options:[.init(id:"canvas",title:"Canvas model"),.init(id:"loaded",title:"Loaded model")]) {source in
                        let expected=context.configuration;var next=expected
                        if source=="loaded" {
                            let edge=context.authority.owner.document.edges.first{$0.to.address==context.authority.address && $0.to.port=="model"}
                            let typed=edge.flatMap{executionHost?.modelLoader.reference($0.from.address)}?.metadata["task"] == .string("typedDecisions")
                            next["fineTuning"] = .object(typed ? TypedDecisionConfiguration.trainingDefaults:FineTuningConfiguration.defaults)
                        }else {next.removeValue(forKey:"fineTuning")}
                        _=context.authority.configure(next,expected:expected)
                    }.disabled(context.isBusy)
                    if context.configuration["fineTuning"] != nil,let host=executionHost {
                        FineTuningInspectorView(context:context,host:host,store:host.modelLoader)
                    } else {
                    TrainInspectorSection("Data mapping",symbol:"tablecells.badge.ellipsis",initiallyExpanded:true) {
                        if usesNamedMapping || context.mappingConfiguration["namedInputPlan"] != nil {
                            NamedInputMappingInspector(context:context,host:executionHost).id(context.contextIdentity)
                        } else if context.hasObjectiveMapping {LossTargetMappingInspector(context:context).id(context.contextIdentity)}
                        else if context.requiresBagMapping {BagTrainingMappingInspector(context:context).id(context.contextIdentity)}
                        else {WorkflowTrainingMappingEditor(context:context,host:executionHost).id(context.contextIdentity)}
                        if context.mappingConfiguration["namedInputPlan"] == nil {Button(usesNamedMapping ? "Use column mapping":"Use named tensor mapping") {usesNamedMapping.toggle()}.buttonStyle(FlatButtonStyle())}
                    }
                    }
                    if context.configuration["fineTuning"] == nil {
                    TrainInspectorSection("Run settings",symbol:"slider.horizontal.3",initiallyExpanded:true) {
                        ForEach(WorkflowControlContracts.runFields,id:\.key) {field in
                            if (field.key != "learningRate" || !context.learningRateIsInactive) && (field.key != "batchSize" || context.configuration["batching"] == nil) {runField(field)}
                        }
                        TrainingControlInspector(context:context).id(context.contextIdentity)
                        if context.configuration["fineTuning"] == nil {cachePolicyEditor;gradientEditor}
                        TrainInspectorSection("History",symbol:"clock") {
                            RunHistoryCaptureControls(owner:context.authority.owner,workspace:executionHost?.settingsWorkspace)
                        }
                    }
                    }
                    if context.configuration["fineTuning"] == nil {
                    TrainInspectorSection("Training state",symbol:"arrow.clockwise",summary:context.trainingMode?.title,initiallyExpanded:true) {
                        trainingModeEditor
                        if let host=executionHost,let history,host.owner === context.authority.owner,history.sessionID == host.sessionID {
                            WorkflowCheckpointSelectionView(context:context,host:host,history:history,inlinePresentation:true)
                            WorkflowCacheSelectionView(context:context,host:host,history:history,inlinePresentation:true)
                        }
                    }
                    }
                    if context.configuration["fineTuning"] == nil {TrainInlineRecipesInspectorView(authority:context.authority)}
                    if context.configuration["fineTuning"] == nil {
                    TrainInspectorSection("Data access and batches",symbol:"tablecells",summary:batchSummary) {
                        TrainDataAccessInspector(context:context)
                    }
                    if let host=executionHost,host.owner === context.authority.owner {
                        WorkflowExperimentLaunchView(context:context,host:host)
                    }
                    }
                    }
                } else {
                    Text("Configure this component through its typed connections. No hidden training settings are stored here.").font(.caption).foregroundStyle(MLTheme.muted)
                }
            }.id(context.contextIdentity)
            InspectorInformationSection {if context.isTrain {trainInformation} else if context.isEstimator {PrebuiltModelInformationView()} else if context.isGameSession {GameSessionInformationView()} else {LessonRichText(text:information)}}.id(context.contextIdentity)
            if !context.statusMessage.isEmpty {Text(context.statusMessage).font(.caption).foregroundStyle(MLTheme.muted)}
        }
        .sheet(isPresented:$showsExperiments) {WorkflowExperimentEditor(context:context)}
    }
    private var batchSummary:String {
        if case .object(let batching)=context.configuration["batching"],let size=batching["batch_size"]?.integerValue {return "Batch \(size)"}
        return "Run settings defaults"
    }
    private var trainInformation:some View {
        VStack(alignment:.leading,spacing:12) {
            if context.configuration["selfPlay"] != nil {
                trainInfo("Self-play","Connect a typed Game definition and an explicit paired policy/value model. Train freezes collection weights, searches bounded episodes, learns only terminal targets and commits one resumable iteration at a time. Truncated/failed attempts remain visible but do not invent draw targets.")
                trainInfo("Independent evaluation","The initial model is a pinned opponent. Balanced seat pairs use independent collection/evaluation seeds, no root noise and equal search budgets. Policy-only and search-assisted results stay separate; no automatic champion or strength guarantee is inferred.")
                trainInfo("Campaigns and compute","Run Flow starts a new campaign. Resume preserves settings and state; Extend adds an applied time/iteration budget and schedule phase; Fork reuses weights with fresh optimizer and replay. The app must remain open and the Mac awake. Rules and tree decisions use CPU; one owned queue batches neural inference. MPS must pass actual loss, gradient, optimizer and recovery checks. Automatic remains CPU until a measured speed profile qualifies; there is no silent operator fallback.")
                trainInfo("Retention","Required replay, optimizer, CPU RNG and pins remain separate from optional capture and the 8 MB table allowance. Resume verifies exact settings/rules/model/runtime. CPU reference continuation and numerically qualified MPS recovery have different guarantees. Rolling still needs room for the next commit; the previous recovery boundary stays until commit succeeds. Use the Learn Self-play practice for the complete workflow.")
            } else if case .object(let f)=context.configuration["fineTuning"],f["task"] == .string("typedDecisions") {
                trainInfo("Calibration and reuse","Reserved Training source groups split into policy selection and an independent audit. Temperatures and cutoff use selection only; any accepted wrong answer makes its audit group a failure. Correlated rows do not increase independent support. Soft targets may fit temperature but cannot authorize automatic acceptance. Insufficient support keeps decisions in review. This error objective is not a clinical, security or future-population guarantee. Save model retains inference state and question/calibration provenance; loading it starts a fresh optimizer. CPU float32 is qualified, with at most 16 GB task admission or a lower project limit. GPU, RLCD and exact optimizer resume remain unavailable.")
                trainInfo("Typed decisions","Load model supplies the verified Laya encoder, decision head and tokenizer. State columns describe records; each question maps to a declared option key or complete target distribution. Head-only freezes the encoder. Full training changes encoder/head weights. Source groups separate Training, Validation and optional calibration; Test is evaluated independently. Capture off still retains the required model and question revision. Predictions are standard tables; use Decide for all option scores. RLCD, Apple GPU and exact resume remain unqualified for this task.")
            } else if context.configuration["fineTuning"] != nil {
                trainInfo("Imported models","Load model supplies a verified GPT-2 or Llama causal model and its matching tokenizer. Full model updates all weights; Selected layers updates only chosen groups. Linked input/output embeddings remain linked. LoRA learns adapter changes on actual named projections and optional independent modules. Original model files remain unchanged; loaded weights are never reinitialized.")
                trainInfo("Documents and token positions","Training and Validation are disjoint prepared text tables. Text column must exist in both. The loaded tokenizer retains its vocabulary and special IDs. Context bounds one window; stride advances within each document. Targets shift exactly once. Attention and loss masks exclude padding positions, while a real EOS remains a target even if its ID also supplies padding.")
                trainInfo("Run and inspect","Choose a device, epoch count, batch size and seed. Accumulation combines microbatches using actual valid-token counts. Optimization uses the shared optimizer and scheduler controls. Metrics show completed batches and held-out Validation; neither chart inspection nor reporting changes training. Resource admission precedes token/model materialization. Cancel keeps the original base and does not publish partial weights.")
                trainInfo("Save and reuse","Continue a prompt accepts Train’s selected inference model. Full training exports safe updated weights/configuration/tokenizer. LoRA offers adapter-only with exact base identity or an explicitly merged standalone model. Reopening an adapter requires selecting its exact matching local base. Latest training boundary is separate from selected Best validation weights; CPU continuation state retains optimizer/RNG at the same completed epoch. MPS promises weights reuse, not exact continuation.")
                trainInfo("History and scope","Required model, input tables, tokenizer, epoch metrics and provenance remain when optional capture is off. Optional within-epoch observations and unconnected predictions are not stored when disabled. History verifies pinned files before showing them. Quantized training, arbitrary repository code, Laya-specific heads and imported-task Code export need separate adapters. Use the Fine-tune a loaded model practice in Training & evaluation.")
            } else {
            trainInfo("Model selection and stopping","**Final epoch** retains the last completed model; **Best validation** retains the model with the lowest measured validation loss. Early stopping is separate: patience counts complete validation epochs without a loss decrease greater than Minimum improvement. Predictions and Save model use selected weights; exact continuation retains the final training state. Existing projects keep Final epoch and early stopping off.")
            trainInfo("Inputs","**Model** defines the architecture. **Training** and **Validation** accept prepared tables directly. **Data mapping** assigns their ordered inputs and outcomes. Run settings on Train supply the epoch budget, device and experiment settings.")
            trainInfo("Data mapping","Input columns fill each model observation in saved order, with the last dimension changing fastest. Both Training and Validation use that order. Model definition owns the required shape and number type. Column choices use current local inspection or recorded metadata; execution verifies actual columns and values. Targets must not become input features. Explicit objectives retain their typed outcome roles; grouped inputs retain paired IDs, lengths and weights. Shapes are model declarations, not verified source values. Category IDs select learned table rows; item weights are source values, not learned parameters. Packed used-item counts select each group’s prefix, including empty groups and their targets; offsets come from those counts. Triplet objectives use equal anchor, related and unrelated groups without a target column.")
            trainInfo("Training state","**Start fresh** initializes new parameters. **Fine-tune weights** restores learned values. **Resume exactly** restores compatible training state; its epoch budget is the desired total. Checkpoint compatibility is verified before training. Prior preparation is used only when Train’s preparation cache selects Reuse; source files are checked again.")
            trainInfo("Named tensors","Numeric columns fill each named model input in saved order. Token inputs use exact integer sequences and explicit length, vocabulary and padding settings. All inputs share the same observations. Conversion buffers are bounded by Maximum tensor batch; retained training tensors are currently limited to 100,000 examples and 128 MiB per partition. Image/audio adapters remain later additions; Generate text consumes completed next-token models.")
            trainInfo("Next-token sequences","Connect Training and Validation directly from one partition-aware Sequence windows tool, then choose Use next-token mapping. Integer inputs and shifted targets must match its context, vocabulary, padding, valid-length and mask columns exactly. Loss averages over valid tokens, with no contribution or gradient from padding. Validation perplexity is exp(loss), not training quality certification. The model must output [context, vocabulary] logits; mixing token positions requires an explicitly causal architecture. Optional previews retain at most eight windows within the byte allowance; aggregate Validation metrics cover all windows. Original token populations and vocabulary remain required replay evidence even with optional capture off. Generate text consumes this completed model and captured tokenizer. Frozen sequence evaluation remains unsupported.")
            trainInfo("Optimization","Optimizer controls learning updates; its learning rate overrides the fallback learning rate. A schedule needs an explicit compatible optimizer. Objective outcomes are mapped in Train’s Data mapping. Mean/Sum combines unreduced objective values and must agree with the gradient policy.")
            trainInfo("Weights and biases","Configure starting weights, biases and trainability on each layer. Train’s seed determines fresh starting numbers. Restoring a checkpoint preserves its learned values. Older projects may retain per-Train overrides; those override the corresponding layer defaults and remain editable in Legacy layer overrides.")
            trainInfo("Data and batches","Access settings select training observations without reducing validation. Batches control ordering, collation and reader processes. This path uses prepared tensors and retains its 100,000-row and memory bounds; extra readers do not make training out of core.")
            trainInfo("Experiments","Each value is one explicit option. The displayed count is the Cartesian combination count; budgets bound execution. Experiments run sequentially using one captured source population. Ordinary Run Flow uses the single-run settings.")
            trainInfo("Outputs","**Trained model** connects to Evaluate or Save model. **Predictions** is a complete captured Validation table with observation IDs, targets, outputs and class probabilities where applicable. It never publishes the protected Test population. Inspect it in Output or connect data tools; aggregate loss/accuracy belong to Metrics. Unconnected prediction capture can be disabled in project capture settings; connected prediction tables remain required workflow data; old bounded previews are never presented as complete tables.")
            trainInfo("History capture","Capture controls collection and storage on future runs across this project. Turning it off skips optional tensor snapshots, prediction previews, diagnostic analysis, performance samples and intermediate preparation copies. Epoch metrics, weights, source/replay evidence, final prepared tables, fitted state, explicit View/file outputs, evaluation tables and requested caches remain. Connected predictions are required workflow data. Existing runs are unchanged. Save records canvas versions separately.")
            trainInfo("Training chart","Select Metrics for live completed-batch loss and measured Validation. Updates counts optimizer adjustments; Batches counts processed groups. Reporting settings apply to the next run; elapsed intervals are checked only after a batch completes. Pause freezes the chart, not training or capture. Raw/Smoothed and range controls affect display only. Optional within-epoch History retention follows diagnostic capture; required epoch/checkpoint evidence stays intact. Resuming training still uses completed epoch boundaries.")
            trainInfo("Recorded results","History preserves captured settings and results. Current configuration edits do not rewrite old runs. Export a verified trained package from History.")
            }
        }
    }
    private func trainInfo(_ title:String,_ body:String)->some View {
        VStack(alignment:.leading,spacing:5) {
            Text(title).font(.caption.weight(.semibold)).foregroundStyle(colorScheme == .dark ? ComponentVisualFamily.training.primary:ComponentVisualFamily.training.secondary)
            LessonRichText(text:body)
        }.frame(maxWidth:.infinity,alignment:.leading)
    }
    private var information:String {
        if context.isGenerateText {return "Generate text uses a verified completed next-token model and its saved tokenizer. One prompt creates one row; Table column keeps the complete source rows and appends named outputs. New tokens counts emitted IDs, including EOS. Text contains only the completion. Inspect IDs for exact captured token labels.\n\nGreedy selects the highest admissible logit; Sample applies temperature, top-k then top-p with an independent CPU RNG for each prompt. PAD and BOS are never emitted; unknown tokens require permission. EOS stopping applies only to a tokenizer that includes EOS. CPU replay is exact for the captured runtime; GPU numerics may differ.\n\nShort prompts right-pad; generation uses the last real token position. Long prompts refuse unless Keep latest context is explicit. The model owns causality: generation never adds an attention mask. Limits are enforced before execution; cancellation does not publish a partial table. Required model, tokenizer, prompt and output evidence remains even with optional History capture off. Whole-workflow Code currently refuses this terminal rather than omit it. Open Learn → Generate bounded text for an editable practice."}

        if context.isModelDefinition {return "Model inputs define named observations entering this architecture; the batch dimension is separate. Choose entry ports on the first computational components. Prepared columns are assigned in Train’s Data mapping. Floating values use float32; integer indices retain exact int64 typing. Input settings and internal boundary identities are saved with this model, while recorded runs remain unchanged. Bag Input retains its coordinated named ports on the canvas."}
        if context.isEstimator,PrebuiltModelConfiguration.isPrebuilt(context.configuration) {return "Prebuilt numeric models use explicit ordered features and task-specific targets. Fit learns only the declared population; frozen inference never refits. Inspect complete standard output tables and separate metrics. Cluster IDs, distances and memberships are not churn probabilities. Configure model exposes task-specific options and limits."}
        if context.isEstimator {return BackendEstimatorInspector.information(backend:context.configuration["backendID"]?.stringValue ?? "sklearn")}
        if context.isRunSetup {return "This component captures training settings, random seeds and observation budgets for each run. Configure experiment combinations to compare controlled changes."}
        if context.isMapping,context.requiresBagMapping {return "Map ordered category ID columns to each observation group, with paired item weights when enabled and a used-item count for packed groups. Bag Input owns the layout and capacity controls. Targets belong to whole groups, including empty groups; mapping requirements describe the connected model, not observed source values."}
        if context.isMapping,context.hasObjectiveMapping {return "Map prepared columns to ordered model inputs and the connected objective’s desired outcomes. Shapes describe array directions within each observation; a scalar is one value per observation. These are declared requirements, not verified source values. Training and validation keep their separate partitions."}
        if context.isMapping {return "Map prepared columns to ordered model features and the target. Training and validation inputs retain their separate partitions."}
        if context.isEvaluateTest {return "Evaluate the verified trained model on its archived Validation or Test partition. Validation supports model choices; Test is reserved for final evaluation. Weights and fitted preparation remain unchanged. Output contains the complete selected population; batch size controls execution memory. History sample rows limits only the retained summary sample, never the prediction table."}
        if context.isSaveModel {return "Save a verified trained model as a portable package or checkpoint. Connect Train’s Trained model output or Load model. Imported GPT-2/Llama models use Hugging Face folder export with their matching tokenizer and safe weights. A fine-tuned result offers full model, adapter with exact base identity, or explicitly merged model where supported. Latest training boundary retains consistent terminal state for supported CPU continuation; it is separate from Best validation inference weights. Choose the destination on this Mac, then Save model. Save model reuses a current completed output, or runs its connected Train if none is available. Run Flow starts Train afresh. The portable package expects already-prepared inputs; fitted preparation is not included. A .pt checkpoint contains weights and training state."}
        if context.isFrozenModel {return "Load a verified saved run, a supported local Hugging Face model folder or a pinned Hub snapshot. Saved runs retain their original weights, tokenizer and fitted preparation. External GPT-2 and Llama causal adapters support inspection, tokenization, generation and safe folder export. Connect a reviewed causal model to Train in Loaded model mode for full, selected-layer or LoRA fine-tuning. Other tasks need their own reviewed adapter. Tokenizer IDs and special tokens must match the model. Downloads and previews run only when requested."}
        return "Typed connections define this component's role in the flow. Inspect connected inputs and outputs on the canvas."
    }
    private var evaluationEditor:some View {
        let expected=context.configuration
        let settings=context.evaluationSettings
        return VStack(alignment:.leading,spacing:8) {
            Text("Partition").font(.caption)
            FlatChoiceControl(title:"Evaluation partition",selection:settings?.partitionRole ?? "",options:[
                .init(id:"validation",title:"Validation"),.init(id:"test",title:"Test")]) {role in
                _=context.setEvaluationPartition(role,expected:expected)
            }
            Text("Evaluation device").font(.caption)
            FlatChoiceControl(title:"Evaluation device",selection:settings?.device ?? "",options:[
                .init(id:"auto",title:"Automatic"),.init(id:"cpu",title:"CPU"),.init(id:"mps",title:"Apple GPU")]) {value in
                _=context.setEvaluationValue("device",value:.string(value),expected:expected)
            }
            if expected["device"] == nil {evaluationDefault}
            evaluationNumber("batchSize",title:"Evaluation batch size",value:settings?.batchSize ?? 256,range:1...100000,expected:expected)
            evaluationNumber("previewLimit",title:"History sample rows",value:settings?.previewLimit ?? 8,range:0...100,expected:expected)
            if settings == nil {Text("Evaluation settings are invalid. Reload or repair the workflow before running.").font(.caption).foregroundStyle(MLTheme.rose)}
        }
    }
    private var evaluationDefault:some View {
        Text("Using execution default; no override saved.").font(.caption2).foregroundStyle(MLTheme.muted)
    }
    private func evaluationNumber(_ key:String,title:String,value:Int,range:ClosedRange<Int>,expected:[String:JSONValue])->some View {
        VStack(alignment:.leading,spacing:4) {
            Text(title).font(.caption)
            BoundedNumberField(title:title,value:.wholeDisplay(Double(value)),domain:.integer(minimum:Double(range.lowerBound),maximum:Double(range.upperBound)),origin:context.numericOrigin(key:key),currentOrigin:{context.currentNumericOrigin(key:key)},onDraftChange:{origin,pending in
                context.setNumericPending(origin:origin,key:key,pending:pending)
            }) {candidate in
                guard case .integer(let number)=candidate else {return false}
                return context.setEvaluationValue(key,value:.integer(number),expected:expected)
            }
            if expected[key] == nil {evaluationDefault}
        }
    }
    private var trainingModeEditor:some View {
        let expected=context.configuration
        return VStack(alignment:.leading,spacing:8) {
            Text("Training mode").font(.caption)
            FlatChoiceControl(title:"Training mode",selection:context.trainingMode?.rawValue ?? "",
                options:WorkflowTrainingMode.allCases.map {.init(id:$0.rawValue,title:$0.title)}) {value in
                guard let mode=WorkflowTrainingMode(rawValue:value) else {return}
                _=context.setTrainingMode(mode,expected:expected)
            }
            if context.trainingMode != nil {
                EmptyView()
            } else {
                Text("Choose a supported training mode before running.").font(.caption).foregroundStyle(MLTheme.rose)
            }
        }
    }
    private var cachePolicyEditor:some View {
        let expected=context.configuration
        return VStack(alignment:.leading,spacing:6) {
            Text("Preparation cache").font(.caption)
            FlatChoiceControl(title:"Preparation cache policy",selection:context.preparationCachePolicy?.rawValue ?? "",
                options:[.init(id:"off",title:"Off"),.init(id:"materialize",title:"Store"),.init(id:"reuse",title:"Reuse")]) {value in
                guard let policy=WorkflowPreparationCachePolicy(rawValue:value) else {return}
                _=context.setPreparationCachePolicy(policy,expected:expected)
            }
            Text(cacheExplanation).font(.caption2).foregroundStyle(MLTheme.muted)
        }
    }
    private var cacheExplanation:String {
        switch context.preparationCachePolicy {
        case .off:return "Recompute preparation on each run. Outputs are still available for inspection."
        case .materialize:return "Compute preparation and retain reusable results for a later run."
        case .reuse:return "Select a prior result on the Train component. Sources are read again; changed dependencies are recomputed."
        case nil:return "Cache policy is invalid. Choose a supported policy before running."
        }
    }
    private var gradientEditor:some View {
        let expected=context.configuration
        let origin=context.numericOrigin(key:"gradientCapture")
        return GradientCaptureConfigurationEditor(configuration:context.gradientConfiguration,canCapture:canCaptureGradients,isBusy:context.isBusy,origin:origin,currentOrigin:{context.currentNumericOrigin(key:"gradientCapture")},onDraftChange:{reported,pending in
            // Each gradient cell owns a distinct token; its suffix is a fixed field key.
            let prefix=origin+":"
            guard reported.hasPrefix(prefix) else {return}
            let key="gradientCapture:"+String(reported.dropFirst(prefix.count))
            context.setNumericPending(origin:reported,key:key,pending:pending)
        }) {next in
            guard next == nil || canCaptureGradients else {return false}
            return context.setGradientCapture(next,expected:expected)
        }
    }
    @ViewBuilder private func runField(_ field:WorkflowRunField)->some View {
        let expected=context.configuration
        let value=context.runConfiguration?[field.key] ?? field.defaultValue
        Text(field.title).font(.caption)
        if context.isRunSetup,context.runConfiguration?[field.key] == nil {Text("Using execution default; no override saved.").font(.caption2).foregroundStyle(MLTheme.muted)}
        switch field.kind {
        case .choice(let choices):
            FlatChoiceControl(title:field.title,selection:value.stringValue ?? "",options:choices.map {.init(id:$0,title:$0)}) {selected in
                _=context.setRunValue(field.key,value:.string(selected),expected:expected)
            }
        case .integer(let bounds):
            numeric(field,value:.wholeDisplay(value.doubleValue ?? 0),domain:.integer(minimum:Double(bounds.lowerBound),maximum:Double(bounds.upperBound)),expected:expected)
        case .positiveDecimal(let maximum),.nonnegativeDecimal(let maximum):
            numeric(field,value:.decimal(value.doubleValue ?? 0),domain:.decimal(minimum:0,maximum:maximum),expected:expected)
        }
    }
    private func numeric(_ field:WorkflowRunField,value:NumericValue,domain:NumericDomain,expected:[String:JSONValue])->some View {
        BoundedNumberField(title:field.title,value:value,domain:domain,origin:context.numericOrigin(key:field.key),currentOrigin:{context.currentNumericOrigin(key:field.key)},onDraftChange:{origin,pending in context.setNumericPending(origin:origin,key:field.key,pending:pending)}) {candidate in
            let json:JSONValue
            switch candidate {case .integer(let number):json = .integer(number);case .decimal(let number):json = .number(number)}
            return context.setRunValue(field.key,value:json,expected:expected)
        }
    }
}

private struct WorkflowTrainingMappingEditor:View {
    @ObservedObject var context:WorkflowControlNodeContext
    var host:WorkflowWorkspaceSession?=nil
    @Environment(\.colorScheme) private var colorScheme
    @State private var draft=TrainingMappingDraft(features:["",""],target:"")
    @State private var expected:[String:JSONValue]=[:]
    @State private var initialized=false
    @State private var requirements:[WorkflowMappingRequirement]=[]
    @State private var requirementError:String?
    @State private var rejection:String?
    private let key="trainingMappingDraft"
    var body:some View {
        let columns=knownColumns
        return VStack(alignment:.leading,spacing:8) {
            if let requirementError {
                Text(requirementError).font(.caption).foregroundStyle(MLTheme.rose)
            } else if requirements.isEmpty {
                Text("Connect a Model definition to inspect required input dimensions.").font(.caption).foregroundStyle(MLTheme.muted)
            }
            ForEach(requirements,id:\.train) {requirement in
                VStack(alignment:.leading,spacing:3) {
                    Text(requirement.label).font(.caption.bold())
                    if let contract=requirement.contract {
                        Text("Sample \(contract.layout.sample.dimensions.description) · \(contract.dtype.rawValue) · \(contract.layout.sample.elementCount) columns")
                            .font(.caption).foregroundStyle(MLTheme.muted)
                        if draft.rows.count != contract.layout.sample.elementCount {
                            Text("Mapped \(draft.rows.count); requires \(contract.layout.sample.elementCount).")
                                .font(.caption).foregroundStyle(MLTheme.rose)
                        }
                    } else if let problem=requirement.problem {
                        Text(problem).font(.caption).foregroundStyle(MLTheme.rose)
                    }
                }
            }
            Text("Input columns").font(.caption)
            ForEach(Array(draft.rows.enumerated()),id:\.element.id) {index,row in
                HStack(spacing:4) {
                    Text("\(index+1)").font(.caption.monospacedDigit()).foregroundStyle(MLTheme.muted)
                    columnField(title:"Input column \(index+1)",columns:columns,value:draft.rows.first {$0.id==row.id}?.name ?? "") {draft.setName($0,for:row.id)}
                    Button {draft.move(row.id,by:-1)} label:{Image(systemName:"chevron.up")}
                        .disabled(index==0).accessibilityLabel("Move input column \(index+1) earlier")
                    Button {draft.move(row.id,by:1)} label:{Image(systemName:"chevron.down")}
                        .disabled(index==draft.rows.count-1).accessibilityLabel("Move input column \(index+1) later")
                    Button {draft.remove(row.id)} label:{Image(systemName:"minus")}
                        .disabled(draft.rows.count<=1).accessibilityLabel("Remove input column \(index+1)")
                }.buttonStyle(.borderless)
            }
            Button("Add input column") {draft.add()}.buttonStyle(FlatButtonStyle())
                .disabled(draft.rows.count>=TrainingColumnMapping.maximumColumns)
            Text("Target column").font(.caption)
            columnField(title:"Target column",columns:columns,value:draft.target) {draft.target=$0}
            HStack {
                Button("Apply column mapping",action:apply).buttonStyle(FlatButtonStyle())
                Button("Revert") {reload();report()}.buttonStyle(FlatButtonStyle())
            }
            if let rejection {Text(rejection).font(.caption).foregroundStyle(MLTheme.rose).accessibilityIdentifier("train.mapping.rejection")}
        }.onAppear {
            guard !initialized else {return};expected=context.configuration
            do {requirements=try context.mappingRequirements()} catch {requirementError=error.localizedDescription}
            let features:[String]
            if case .array(let values)=context.mappingConfiguration["featureColumns"] {features=values.map {$0.stringValue ?? ""}}
            else {features=["",""]}
            draft=TrainingMappingDraft(features:features,target:context.mappingConfiguration["targetColumn"]?.stringValue ?? "");initialized=true
        }.onChange(of:draft) {_,_ in report()}
            .onDisappear {context.setNumericPending(origin:context.numericOrigin(key:key),key:key,pending:false)}
    }
    private var knownColumns:[String] {
        guard let host,host.owner === context.authority.owner else {return []}
        return host.sourceProvider.recordedSchema(for:context.authority.address,in:host.owner.document,inputPort:"training").map(\.name)
    }
    @ViewBuilder private func columnField(title:String,columns:[String],value:String,onChange:@escaping(String)->Void)->some View {
        let knownColumns=columns
        if knownColumns.isEmpty {TextField(title,text:Binding(get:{value},set:onChange)).flatField().accessibilityLabel(title).onSubmit(apply)}
        else {
            let names=knownColumns.contains(value) || value.isEmpty ? knownColumns:[value]+knownColumns
            FlatChoiceControl(title:title,selection:value,options:[.init(id:"",title:"Choose column")]+names.map {.init(id:$0,title:$0)},accent:colorScheme == .dark ? ComponentVisualFamily.training.primary:ComponentVisualFamily.training.secondary) {onChange($0)}
        }
    }
    private var candidate:[String:JSONValue] {["featureColumns":.array(draft.features.map(JSONValue.string)),"targetColumn":.string(draft.target)]}
    private func report() {
        guard initialized else {return}
        context.setNumericPending(origin:context.numericOrigin(key:key),key:key,pending:candidate != context.mappingConfiguration)
    }
    private func reload() {
        rejection=nil
        let value=context.mappingConfiguration
        let names={if case .array(let values)=value["featureColumns"] {return values.compactMap(\.stringValue)};return ["",""]}()
        draft=TrainingMappingDraft(features:names,target:value["targetColumn"]?.stringValue ?? "")
    }
    private func apply() {
        guard context.authority.accepts(expected) else {rejection="The selected Train changed. Reopen Data mapping before applying this draft.";report();return}
        do {try TrainingColumnMapping.validate(features:draft.features,target:draft.target)}
        catch {rejection=error.localizedDescription;report();return}
        guard context.applyMapping(features:draft.features,target:draft.target,expected:expected) else {
            rejection=context.statusMessage
            context.authority.message=nil
            report();return
        }
        rejection=nil;expected=context.configuration;report()
    }
}

/// Retains the same output draft and canonical transactions used by the model inspector.
private struct ModelDefinitionOutputInspector:View {
    @StateObject private var output:WorkflowModelNodeContext
    init(context:WorkflowControlNodeContext)throws {
        let owner=context.authority.owner,address=context.authority.address
        guard let node=owner.document.nodes.first(where:{$0.address==address}),case .modelPrimitive("output",_,_)=node.payload else {throw CocoaError(.featureUnsupported)}
        let model=try WorkflowModelNodeContext(authority:context.authority)
        _output=StateObject(wrappedValue:model)
    }
    var body:some View {
        InspectorSection("Expected output",symbol:"checkmark.circle",identifier:"model.expected-output",initiallyExpanded:true) {
            ModelOutputConfigurationView(context:output)
        }
    }
}
