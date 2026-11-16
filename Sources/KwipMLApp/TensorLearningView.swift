import SwiftUI
import KwipMLCore

struct TensorLearningView:View {
    @ObservedObject var lesson:TensorLearningStore
    @ObservedObject var workspace:WorkspaceStore
    private var compositionTask:ModuleCompositionLearningTask? {ModuleCompositionLearningTask(rawValue:lesson.draft?.operation ?? "")}
    private var gradientTask:GradientLearningTask? {GradientLearningTask(rawValue:lesson.draft?.operation ?? "")}
    private var initializationTask:InitializationLearningTask? {InitializationLearningTask(rawValue:lesson.draft?.operation ?? "")}
    private var dataLoadingTask:DataLoadingLearningTask? {DataLoadingLearningTask(rawValue:lesson.draft?.operation ?? "")}
    private var programExportTask:ProgramExportLearningTask? {ProgramExportLearningTask(rawValue:lesson.draft?.operation ?? "")}
    private var checkpointTask:CheckpointLearningTask? {CheckpointLearningTask(rawValue:lesson.draft?.operation ?? "")}
    private var lossTask:LossLearningTask? {LossLearningTask(rawValue:lesson.draft?.operation ?? "")}
    private var attentionTask:AttentionLearningTask? {AttentionLearningTask(rawValue:lesson.draft?.operation ?? "")}
    private var recurrentTask:RecurrentLearningTask? {RecurrentLearningTask(rawValue:lesson.draft?.operation ?? "")}
    private var schedulerTask:SchedulerLearningTask? {SchedulerLearningTask(rawValue:lesson.draft?.operation ?? "")}
    private var optimizerTask:OptimizerLearningTask? {OptimizerLearningTask(rawValue:lesson.draft?.operation ?? "")}
    private var statefulNormalizationTask:StatefulNormalizationLearningTask? {StatefulNormalizationLearningTask(rawValue:lesson.draft?.operation ?? "")}
    private var isSpatialPatch:Bool {SpatialPatchLearningTask(rawValue:lesson.draft?.operation ?? "") != nil}
    private var isTensorRouting:Bool {TensorRoutingLearningTask(rawValue:lesson.draft?.operation ?? "") != nil}
    private var isRemainingActivation:Bool {RemainingActivationLearningTask(rawValue:lesson.draft?.operation ?? "") != nil}
    private var isTensorComposition:Bool {TensorCompositionLearningTask(rawValue:lesson.draft?.operation ?? "") != nil}
    private var isAdaptivePooling:Bool {AdaptivePoolingLearningTask(rawValue:lesson.draft?.operation ?? "") != nil}
    private var isConfigurableActivation:Bool {ConfigurableActivationLearningTask(rawValue:lesson.draft?.operation ?? "") != nil}
    private var isPooling:Bool {PoolingLearningTask(rawValue:lesson.draft?.operation ?? "") != nil}
    private var isConvolution:Bool {ConvolutionLearningTask(rawValue:lesson.draft?.operation ?? "") != nil}
    private var isLazyLinear:Bool {lesson.draft?.operation == LazyLinearAssessment.operation}
    private var isBilinear:Bool {BilinearLearningTask(rawValue:lesson.draft?.operation ?? "") != nil}
    private var isMatrix:Bool {MatrixLearningTask(rawValue:lesson.draft?.operation ?? "") != nil}
    private var isEmbedding:Bool {EmbeddingLearningTask(rawValue:lesson.draft?.operation ?? "") != nil}
    private var isSoftmax:Bool {TensorLearningTask.softmaxAxis(for:lesson.draft?.operation ?? "") != nil}
    private var isLayerNorm:Bool {LayerNormLearningTask(rawValue:lesson.draft?.operation ?? "") != nil}
    // The store owns eligibility, including active-project and busy-state checks.
    var showsRuntimeRefresh:Bool {!RetiredTrainingLessonDestination.contains(lesson.draft?.operation) && lesson.canRefreshLessonRuntime}
    var body:some View {
        if RetiredTrainingLessonDestination.contains(lesson.draft?.operation) {
            ThemedScrollView {
                VStack(alignment:.leading,spacing:16) {
                    LessonTextPanel(title:"Training controls moved into their owners",text:"Your saved lesson draft and progress remain available as historical evidence. Current workflows configure objectives, optimizers, schedules, gradients, data access and batches in Train. Select a model layer for its starting values and trainability. This replacement practice is unscored.")
                    TrainDataAccessLearningView(store:workspace)
                }.padding(16)
            }
        } else {activeBody.onAppear {
            synchronizeDeliveryMode()
        }}
    }
    /// Only current Construction destinations select assessment mode. Worked examples
    /// and reopened historical Guided/Practice drafts keep their recorded delivery.
    func synchronizeDeliveryMode() {
        guard let entryID=workspace.lessonRailEntryID,
              ConstructionLessonCatalog.contains(entryID),lesson.draft?.mode != .assessment else {return}
        _ = lesson.setMode(.assessment)
    }
    private var activeBody:some View {
        VStack(spacing:0) {
            HStack(spacing:10) {
                VStack(alignment: .leading, spacing: 4) {
                    SectionLabel(title: programExportTask != nil ? "Executable computation · artifact action":dataLoadingTask != nil ? "Observations and groups · construction":compositionTask != nil ? (lesson.draft?.mode == .assessment ? "Reusable architecture · construction":"Reusable architecture · worked example"):gradientTask != nil ? "Learning controls · construction":initializationTask != nil ? "Starting learned settings · construction":checkpointTask != nil ? "Saved runs · learning exercise":lossTask != nil ? "Objectives & desired values · construction":attentionTask != nil ? "Sequence comparisons · construction" : recurrentTask != nil ? "Sequence and memory · construction" : (schedulerTask != nil || optimizerTask != nil) ? "Training recipes · construction":"Tensor mathematics · construction")
                    Text(lesson.draft.flatMap {TensorLearningTask.title(for:$0.operation)} ?? "Tensor exercise").font(.system(size: 16, weight: .semibold))
                }
                Spacer()

            }.padding(.horizontal, 16).padding(.vertical, 12).background(MLTheme.panel)
            Divider()
            if let task=compositionTask,let session=lesson.compositionCanvas {
                ModuleCompositionLearningWorkbenchView(session:session,task:task,showInspector:workspace.showInspector,results:results,showsInstructions:!workspace.showLessonRail || workspace.lessonRailEntryID == nil).id(lesson.drafts.identity)
            } else if let task=programExportTask {
                VSplitView {
                    HSplitView {
                        if !workspace.showLessonRail || workspace.lessonRailEntryID == nil {ThemedScrollView {LessonTextPanel(title:task.title,text:task.instructions).padding(16)}.frame(minWidth:360)}
                        if workspace.showInspector {ThemedScrollView {DataProgramLearningControlsView(lesson:lesson).padding(14)}.frame(minWidth:280,idealWidth:340,maxWidth:430)}
                    }.frame(minHeight:220)
                    ThemedScrollView {results.frame(maxWidth:.infinity,alignment:.leading).padding(12)}.frame(minHeight:150,idealHeight:280)
                }.id(lesson.drafts.identity)
            } else if let task=checkpointTask {
                CheckpointLearningWorkbenchView(lesson:lesson,showInspector:workspace.showInspector,task:task,results:results,showsInstructions:!workspace.showLessonRail || workspace.lessonRailEntryID == nil).id(lesson.drafts.identity)
            } else if let canvas=lesson.canvas {
                TensorLearningWorkbenchView(session:lesson.drafts,canvas:canvas,showComponents:workspace.showLibrary,showInspector:workspace.showInspector,
                    task:lesson.publicTask?.instructions ?? "Task unavailable.",results:results,
                    audienceGuide: LessonAudienceGuide.make(operation: lesson.draft?.operation ?? ""),
                    readableRequirements: LessonAudienceGuide.requirements(operation: lesson.draft?.operation ?? "", fallback: lesson.publicTask?.instructions ?? "Task unavailable."))
                    .id(lesson.drafts.identity)
            }
        }
    }
    private var results:some View {
        VStack(alignment:.leading,spacing:10) {
            if let readiness=lesson.runtimeReadinessError {
                if showsRuntimeRefresh {
                    VStack(alignment:.leading,spacing:10) {
                        Text("Lesson runtime").font(.system(size:13,weight:.semibold))
                        Divider()
                        Text(readiness).font(.caption).textSelection(.enabled)
                        Button("Refresh lesson runtime") {_ = lesson.refreshLessonRuntime()}
                            .buttonStyle(FlatButtonStyle()).disabled(!lesson.canRefreshLessonRuntime)
                    }.padding(12).frame(maxWidth:.infinity,alignment:.leading)
                        .background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))
                } else {Text(readiness).font(.caption).foregroundStyle(.orange)}
            }
            if lesson.grade?.status == .executionFailed {
                LessonTextPanel(title: checkpointTask != nil ? "The saved-run check could not complete":"The construction could not be measured", text: checkpointTask != nil
                    ? "Read the execution detail below. Check the training behavior, parent checkpoint, epoch budget and batch size, then try again. A restore needs a compatible saved parent; exact resume must continue beyond the saved epoch. Refresh the lesson runtime if its support is unavailable."
                    : lesson.statusMessage.contains("Output shape differs")
                    ? "The Output shape does not match the shape this task requires. Select the transformation block, compare its dimensions with Required shapes & rules, then correct the setting and choose Check construction again. No successful result has been saved for this attempt."
                    : "Read the execution detail below, then check the indicated block, its configuration and its input connections. Correct one setting at a time and choose Check construction again. A failed run does not provide measured output values.")
            }
            Text(lesson.statusMessage).font(.caption).textSelection(.enabled)
            if let grade=lesson.grade {
                let state=LessonCheckState(grade:grade,running:lesson.isBusy,canSubmit:lesson.canSubmit,unavailable:false,current:lesson.resultIsCurrent)
                Label(state.label,systemImage:state.symbol).font(.system(size:12,weight:.semibold)).foregroundStyle(state.tint)
                Text(lesson.resultSaved ? "Verified result saved on this Mac":"This result has not been saved.").font(.caption).foregroundStyle(MLTheme.muted)
                if !lesson.resultIsCurrent {Text("This result belongs to an earlier construction.").font(.caption).foregroundStyle(.orange)}
                InspectorSection("Assessment criteria", symbol: "checkmark.square", identifier: "lesson.criteria") {
                LessonCriteriaRowsView(items:grade.outcomes.enumerated().map {index,outcome in
                    let feedback=LessonAudienceGuide.criterionFeedback(operation:lesson.draft?.operation ?? "",criterionID:outcome.criterionID)
                    return .project(outcome,title:feedback?.title ?? (compositionTask != nil ? ModuleCompositionLearningTask.label(outcome.criterionID):"Construction check \(index+1)"),publicNextStep:feedback?.nextStep,assessment:lesson.draft?.mode == .assessment)
                },showsTechnicalFeedback:true)
                }
            } else {
                LessonRichText(text: programExportTask != nil ? "Apply the artifact controls, then check actual reloaded computation, requested input-size rules and saved-parent integrity. The tables below show real measurements.":dataLoadingTask != nil ? "Apply Dataset and DataLoader controls and connect both named recipe inputs. Check construction compares actual selected rows and grouped desired outcomes; inputs and targets must stay aligned.":compositionTask != nil ? "Open each Block to connect its parts, then apply your changes. Check construction compares the results with an independent calculation. Zero means a match; a nonzero value shows where to look next. Public modes also show stored children, private settings and the actual order of work.":gradientTask != nil ? "Connect Gradient policy, Trainability and supplied SGD to Train. Choose Check construction to measure the effects of your choices. Compare the public difference tables with their independent reference; zero means match. Schedule and recording tables show actual recorded values, including an unavailable result rather than an invented zero.":initializationTask != nil ? "Connect both training controls and configure Starting settings. The tables show measured differences from an independent reference, not raw weights. Zero means match. The checker measures starting arrays, predictions, a first adjustment, repeated draws and saved-weight reuse.":checkpointTask != nil ? "Apply the requested saved-run controls, then choose Check construction. Compare the saved settings and measured differences below. A zero difference means the two actual results match; the checker also verifies when training starts and ends.":lossTask != nil ? "Connect the objective recipe and supplied produced/desired roles to Objective check. Compare individual scores, the combined score and input influence. A loss describes disagreement with a desired answer; a smaller score usually means a closer match.":attentionTask != nil ? "Connect supplied sequences and Attention mask to the requested block, then every result role to Model Output. Check construction measures position mixtures and source influence; learned tables belong to the checker.":recurrentTask != nil ? "Connect supplied measurement and starting-state roles to one recurrent block, then connect every result/state port to its matching Model Output. Check construction measures ordered values and influence on inputs, starting state and every learned table; no code or weight editing.":schedulerTask != nil ? "Connect the scheduler and supplied SGD to Train, apply the requested schedule controls, then choose Check construction. Compare the rate used and next rate, learned settings and the resumed sequence; no code or saved-state editing.":optimizerTask != nil ? "Connect the optimizer recipe to Train optimizer, apply the requested controls, then choose Check construction. Compare the supplied gradients, updated settings and remembered quantities over four steps; no code or hidden-state editing.":isLazyLinear ? "Connect Input through LazyLinear to Model Output, choose four output features, then choose Check construction. Inspect the inferred width, created parameter shapes and measured repeat/reload checks. This task checks shapes and preserved state, without training or hand-calculated gradients." : "Connect the supplied inputs through your construction to Model Output, then choose Check construction. Inspect the measured table and compare it with your prediction. KwipML also checks how a small change travels back through the connected blocks; no handwritten derivatives are required.").foregroundStyle(MLTheme.muted)
            }
            if let measured=lesson.compositionResults {
                ModuleCompositionLearningResultsView(task:measured.task,projection:measured.projection,metadata:measured.metadata,nodeLabels:measured.nodeLabels)
                    .id(measured.projection.submissionDigest)
            }
            if let task=gradientTask,let mode=lesson.draft?.mode,
               let presentation=GradientLearningResultsPresentation.project(lesson.gradientProjection,task:task,mode:mode) {
                InspectorSection("Measured influence, adjustments and allowed learning",symbol:"tablecells",identifier:"lesson.gradient.results") {
                    GradientLearningResultsView(presentation:presentation).id(lesson.gradientProjection?.submissionDigest ?? "")
                }
            }
            if let projection=lesson.recipeProjection {RecipeFormativeResultsView(projection:projection).id(projection.submissionDigest)}
            if let projection=lesson.initializationProjection {
                InitializationDifferenceResultsView(projection:projection).id(projection.submissionDigest)
            }
            if let projection=lesson.checkpointProjection {
                CheckpointMeasuredResultsView(projection:projection).id(projection.submissionDigest)
            }
            if let rows = lesson.lazyStatePresentation {
                InspectorSection("Measured shape & state", symbol: "tablecells", identifier: "lesson.lazy.state") {
                    Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 10) {
                        GridRow {
                            Text("Check").font(.caption.weight(.semibold))
                            Text("Measured").font(.caption.weight(.semibold))
                            Text("Meaning").font(.caption.weight(.semibold))
                        }
                        Divider().gridCellColumns(3)
                        ForEach(rows) { row in
                            GridRow {
                                Text(row.check).font(.caption.weight(.medium))
                                Text(row.measured).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                                Text(row.meaning).font(.caption).foregroundStyle(MLTheme.muted)
                            }
                        }
                    }.frame(maxWidth: .infinity, alignment: .leading)
                    Text("Shape and preservation measurements across the independent checks; no prediction-quality claim.")
                        .font(.caption).foregroundStyle(MLTheme.muted)
                }
            }
            if let projection=lesson.lossProjection,let task=lossTask {
                InspectorSection("Measured scores, desired roles and influence",symbol:"tablecells",identifier:"lesson.loss.results") {LossResultsView(projection:projection,task:task).id(projection.submissionDigest)}
            }
            if let projection=lesson.attentionProjection,let task=attentionTask {
                InspectorSection("Measured positions, connections and influence",symbol:"tablecells",identifier:"lesson.attention.results") {AttentionResultsView(projection:projection,task:task).id(projection.submissionDigest)}
            }
            if let projection=lesson.recurrentProjection,let task=recurrentTask {
                InspectorSection("Measured sequence, state and influence",symbol:"tablecells",identifier:"lesson.recurrent.results") {RecurrentResultsView(projection:projection,task:task).id(projection.submissionDigest)}
            }
            if let projection=lesson.schedulerProjection {
                InspectorSection("Measured rates and resumed updates",symbol:"tablecells",identifier:"lesson.scheduler.updates") {SchedulerUpdateResultsView(projection:projection).id(projection.submissionDigest)}
            }
            if let projection=lesson.optimizerProjection {
                InspectorSection("Measured training updates",symbol:"tablecells",identifier:"lesson.optimizer.updates") {OptimizerUpdateResultsView(projection:projection).id(projection.submissionDigest)}
            }
            if let projection=lesson.formativeProjection {
                Divider()
                InspectorSection("Measured tensors", symbol: "tablecells", identifier: "lesson.measured") {
                    if isSpatialPatch {
                        LessonRichText(text: "Compare one observation at a time. Unfold reads patches into columns; Fold adds entries landing on the same grid position. Select Input influence to see repeated uses add together and unused positions remain zero.").foregroundStyle(MLTheme.muted)
                    }
                    TensorFormativeResultsView(projection:projection).id(projection.submissionDigest)
                }
            }
            if lesson.draft?.mode == .guided {
                Divider()
                InspectorSection("Optional mathematics & reasoning", symbol: "function", identifier: "lesson.mathematics", initiallyExpanded: false) {
                    LessonRichText(text: dataLoadingTask?.instructions ?? programExportTask?.instructions ?? gradientTask?.instructions ?? initializationTask?.instructions ?? checkpointTask?.explanation ?? attentionTask?.explanation ?? recurrentTask?.explanation ?? schedulerTask?.explanation ?? optimizerTask?.explanation ?? statefulNormalizationTask?.instructions ?? (isSpatialPatch ? SpatialPatchLessonReference.explanation : isTensorRouting ? LessonAudienceGuide.routingExplanation : isRemainingActivation ? RemainingActivationLessonReference.explanation : isTensorComposition ? TensorCompositionLessonReference.explanation : isAdaptivePooling ? AdaptivePoolingLessonReference.explanation : isConfigurableActivation ? ConfigurableActivationLessonReference.explanation : isPooling ? PoolingLessonReference.explanation : isConvolution ? (ConvolutionLearningTask(rawValue:lesson.draft?.operation ?? "")?.isTransposed == true ? TransposedConvolutionLessonReference.explanation : ConvolutionLessonReference.explanation) : isLazyLinear ? LazyLinearAssessment.explanation : isBilinear ? BilinearLessonReference.explanation : isMatrix ? MatrixLessonReference.explanation : isEmbedding ? EmbeddingLessonReference.explanation : isLayerNorm ? LayerNormLessonReference.explanation : isSoftmax ? SoftmaxLessonReference.explanation : TensorTransformLessonReference.explanation))
                }
            }
        }
    }
}

/// Formative update tables only. The receipt projection refuses assessment-origin values.
private struct OptimizerUpdateResultsView:View {
    let projection:OptimizerFormativeProjection
    @State private var fixtureID=""
    @State private var stateKey=""
    @State private var compareTarget=false
    private var fixture:OptimizerFormativeFixture? {projection.fixtures.first {$0.id == fixtureID} ?? projection.fixtures.first}
    private let labels=["updatedParameters":"Updated settings","momentumBuffer":"Remembered adjustments","firstAverage":"Signed-gradient average","squaredAverage":"Squared-gradient average","largestSquaredAverage":"Largest squared average","gradientAverage":"Recent gradient average","accumulatedSquares":"Accumulated squares","stepCounts":"Recorded optimizer steps","updateCounts":"Gradient applications · derived","statePresence":"Remembered state initialized"]
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            FlatChoiceControl(title:"Supplied example",selection:fixture?.id ?? "",options:projection.fixtures.enumerated().map {.init(id:$0.element.id,title:"Example \($0.offset+1)")}) {fixtureID=$0}
            Toggle("Show target comparison",isOn:$compareTarget).toggleStyle(.checkbox).font(.caption)
            Text("Columns keep two learned weights and one learned bias in their declared order. Rows are training steps. A dash in the gradient table means that entire parameter has no gradient and receives no update.").font(.caption).foregroundStyle(MLTheme.muted)
            if let f=fixture {
                ThemedScrollView(.horizontal) {HStack(alignment:.top,spacing:16) {
                    VStack(alignment:.leading,spacing:8) {SectionLabel(title:"Starting settings");table(.init(shape:[1,3],values:f.initial.values),key:"initial",fixture:f)}.frame(maxWidth:.infinity,alignment:.leading)
                    VStack(alignment:.leading,spacing:8) {SectionLabel(title:"Supplied gradients");table(f.gradients,key:"gradients",fixture:f)}.frame(maxWidth:.infinity,alignment:.leading)
                    if let values=f.measured["updatedParameters"] {VStack(alignment:.leading,spacing:8) {SectionLabel(title:"Settings after each step");table(values,key:"updatedParameters",fixture:f)}.frame(maxWidth:.infinity,alignment:.leading)}
                }.frame(minWidth:720)}.frame(height:compareTarget ? 260:180)
                let keys=f.measured.keys.filter {$0 != "updatedParameters"}.sorted()
                let selected=keys.contains(stateKey) ? stateKey:keys.first ?? ""
                FlatChoiceControl(title:"Remembered quantities",selection:selected,options:keys.map {.init(id:$0,title:labels[$0] ?? $0)}) {stateKey=$0}
                if let values=f.measured[selected] {table(values,key:selected,fixture:f)}
                Text("Gradient applications are counted by the lesson from the supplied schedule. Recorded optimizer steps come from actual stored counters; SGD has no stored counter. Not initialized indicates absent state, rather than an invented zero-valued memory.").font(.caption).foregroundStyle(MLTheme.muted)
            }
        }
    }
    private func table(_ values:LearningTensor,key:String,fixture:OptimizerFormativeFixture)->some View {
        let rows=values.shape[0],width=values.shape.last ?? 3
        let names=width == 2 ? ["Weights","Bias"]:["Weight 1","Weight 2","Bias"]
        return Grid(alignment:.leading,horizontalSpacing:14,verticalSpacing:8) {
            GridRow {Text("Step").font(.caption.weight(.semibold));ForEach(0..<width,id:\.self) {c in Text(names[c]).font(.caption.weight(.semibold))}}
            Divider().gridCellColumns(width+1)
            ForEach(0..<rows,id:\.self) {r in
                GridRow {
                    Text(key == "initial" ? "Start":String(r+1)).font(.caption)
                    ForEach(0..<width,id:\.self) {c in
                        let i=r*width+c
                        VStack(alignment:.leading,spacing:3) {
                            Text(display(values.values[i],key:key,row:r,column:c,fixture:fixture)).font(.system(.caption,design:.monospaced)).textSelection(.enabled)
                            if compareTarget,let expected=fixture.expected[key],expected.values.indices.contains(i) {Text("Target "+String(format:"%.6g",expected.values[i])).font(.caption2).foregroundStyle(MLTheme.muted)}
                        }
                    }
                }
            }
        }.frame(maxWidth:.infinity,alignment:.leading).padding(10).background(MLTheme.background).overlay(Rectangle().stroke(MLTheme.rule))
    }
    private func display(_ value:Double,key:String,row:Int,column:Int,fixture:OptimizerFormativeFixture)->String {
        if key == "gradients",fixture.active.values[row*2+(column<2 ? 0:1)] == 0 {return "—"}
        if key == "statePresence" {return value == 0 ? "Not initialized":"Initialized"}
        if !["initial","gradients","updatedParameters","updateCounts","stepCounts"].contains(key),let presence=fixture.measured["statePresence"],presence.values[row*2+(column<2 ? 0:1)] == 0 {return "Not initialized"}
        return String(format:"%.6g",value)
    }
}

private struct SchedulerUpdateResultsView:View {
    let projection:SchedulerFormativeProjection
    @State private var selectedFixture=""
    @State private var selectedState="updatedParameters"
    @State private var compareTarget=false
    private var fixture:SchedulerFormativeFixture? {projection.fixtures.first {$0.id == selectedFixture} ?? projection.fixtures.first}
    private let labels=["updatedParameters":"Learned settings after each batch","momentumBuffer":"Remembered SGD movement","resumedParameters":"Learned settings after restart","resumedMomentum":"Remembered movement after restart","resumedUsedRates":"Rate used after restart","resumedNextRates":"Next rate after restart","resumedTicks":"Completed schedule ticks after restart"]
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            HStack {
                FlatChoiceControl(title:"Supplied sequence",selection:fixture?.id ?? "",options:projection.fixtures.enumerated().map {.init(id:$0.element.id,title:"Sequence \($0.offset+1)")}) {selectedFixture=$0}
                Toggle("Compare with target",isOn:$compareTarget).toggleStyle(.checkbox).font(.caption)
            }
            if let f=fixture {
                ThemedScrollView(.horizontal) {
                    HStack(alignment:.top,spacing:16) {
                        VStack(alignment:.leading,spacing:8) {SectionLabel(title:"Supplied gradients");table(f.gradients,key:"gradients",fixture:f)}.frame(minWidth:260)
                        VStack(alignment:.leading,spacing:8) {SectionLabel(title:"Rates and timing");rateTable(f)}.frame(minWidth:350)
                        VStack(alignment:.leading,spacing:8) {SectionLabel(title:"Validation measurements");table(f.metrics,key:"metrics",fixture:f)}.frame(minWidth:200)
                    }.frame(minWidth:850)
                }.frame(height:compareTarget ? 420:300)
                Text("Starting settings: "+f.initial.values.map {String(format:"%.6g",$0)}.joined(separator:" · ")+". Supplied SGD uses momentum 0.5; the scheduler controls its rate. A pass has two batches here.").font(.caption).foregroundStyle(MLTheme.muted)
                let keys=f.measured.keys.filter {labels[$0] != nil}.sorted()
                let selected=keys.contains(selectedState) ? selectedState:keys.first ?? ""
                FlatChoiceControl(title:"Updates and saved continuation",selection:selected,options:keys.map {.init(id:$0,title:labels[$0] ?? $0)}) {selectedState=$0}
                if let values=f.measured[selected] {table(values,key:selected,fixture:f)}
                Text("Rate used belongs to that batch's actual update. Next rate belongs to the next update; after the final planned OneCycle batch it is unused. Restart tables measure newly constructed objects restored from the saved state after pass 2, rather than copying the first table.").font(.caption).foregroundStyle(MLTheme.muted)
            }
        }
    }
    private func rateTable(_ f:SchedulerFormativeFixture)->some View {
        Grid(alignment:.leading,horizontalSpacing:12,verticalSpacing:8) {
            GridRow {Text("Batch / pass");Text("Used rate");Text("Next rate");Text("Ticks")}.font(.caption.weight(.semibold))
            Divider().gridCellColumns(4)
            ForEach(0..<8,id:\.self) {r in
                GridRow {
                    Text("\(r+1) / \(Int(f.epochOfBatch.values[r]))").font(.caption)
                    ForEach(["usedRates","nextRates","scheduleTicks"],id:\.self) {key in
                        VStack(alignment:.leading,spacing:3) {
                            Text(f.measured[key].map {String(format:"%.6g",$0.values[r])} ?? "Unavailable").font(.system(.caption,design:.monospaced)).textSelection(.enabled)
                            if compareTarget,let target=f.expected[key] {Text("Target "+String(format:"%.6g",target.values[r])).font(.caption2).foregroundStyle(MLTheme.muted)}
                        }
                    }
                }
            }
        }.padding(10).background(MLTheme.background).overlay(Rectangle().stroke(MLTheme.rule))
    }
    private func table(_ values:LearningTensor,key:String,fixture f:SchedulerFormativeFixture)->some View {
        let width=values.shape.count == 1 ? 1:values.shape.last ?? 1,rows=values.values.count/width
        let names=width == 3 ? ["Weight 1","Weight 2","Bias"]:[key == "metrics" ? "Validation value":"Measured value"]
        return Grid(alignment:.leading,horizontalSpacing:14,verticalSpacing:8) {
            GridRow {Text(key == "metrics" ? "Pass":"Batch");ForEach(0..<width,id:\.self) {c in Text(names[c])}}.font(.caption.weight(.semibold))
            Divider().gridCellColumns(width+1)
            ForEach(0..<rows,id:\.self) {r in
                GridRow {
                    Text(String(r+1+(key.hasPrefix("resumed") ? 4:0))).font(.caption)
                    ForEach(0..<width,id:\.self) {c in
                        let i=r*width+c
                        VStack(alignment:.leading,spacing:3) {
                            Text(key == "gradients" && f.active.values[r*2+(c==2 ? 1:0)] == 0 ? "—":String(format:"%.6g",values.values[i])).font(.system(.caption,design:.monospaced)).textSelection(.enabled)
                            if compareTarget,let target=f.expected[key] {Text("Target "+String(format:"%.6g",target.values[i])).font(.caption2).foregroundStyle(MLTheme.muted)}
                        }
                    }
                }
            }
        }.frame(maxWidth:.infinity,alignment:.leading).padding(10).background(MLTheme.background).overlay(Rectangle().stroke(MLTheme.rule))
    }
}


/// Receipt-protected measured tables; all private-mode values are withheld upstream.
private struct RecurrentResultsView:View {
    let projection:RecurrentFormativeProjection
    let task:RecurrentLearningTask
    @State private var fixtureID=""
    @State private var observation="0"
    @State private var resultRole="result"
    @State private var sourceRole="input"
    @State private var parameter=""
    @State private var compareTarget=false
    private var fixture:RecurrentFormativeFixture? {projection.fixtures.first {$0.id==fixtureID} ?? projection.fixtures.first}
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            if let f=fixture,let input=f.inputs["input"] {
                let n=input.shape[0],index=min(max(Int(observation) ?? 0,0),n-1)
                LazyVGrid(columns:[GridItem(.adaptive(minimum:180),alignment:.leading)],alignment:.leading) {
                    FlatChoiceControl(title:"Supplied sequence",selection:f.id,options:projection.fixtures.enumerated().map {.init(id:$0.element.id,title:"Fixture \($0.offset+1)")}) {fixtureID=$0;observation="0"}
                    FlatChoiceControl(title:"Observation",selection:String(index),options:(0..<n).map {.init(id:String($0),title:"Observation \($0+1)")}) {observation=$0}
                    Toggle("Compare with target",isOn:$compareTarget).toggleStyle(.checkbox).font(.caption)
                }
                Text(task.isCell ? "One step uses the supplied starting memory. Result and hidden are the same new hidden state; both named roles are measured.":"Measurements retain their time order. Result contains every time position; hidden and cell are final states. Each observation's memory stays separate.").font(.caption).foregroundStyle(MLTheme.muted)
                InspectorSection("Named results and final states",symbol:"tablecells",identifier:"recurrent.outputs") {
                    FlatChoiceControl(title:"Result role",selection:resultRole,options:task.outputRoles.map {.init(id:$0,title:$0.capitalized)}) {resultRole=$0}
                    LazyVGrid(columns:[GridItem(.adaptive(minimum:240),alignment:.topLeading)],alignment:.leading) {
                        if let table=f.measured[resultRole]?.batchObservation(at:index) {LearningTensorTable(tensor:table,title:"Measured "+resultRole)}
                        if let table=f.inputs["upstream."+resultRole]?.batchObservation(at:index) {LearningTensorTable(tensor:table,title:"Supplied result influence weights")}
                        if compareTarget,let table=f.expected[resultRole]?.batchObservation(at:index) {LearningTensorTable(tensor:table,title:"Target "+resultRole)}
                    }
                }
                InspectorSection("Measurements and starting-state influence",symbol:"tablecells",identifier:"recurrent.sources") {
                    FlatChoiceControl(title:"Supplied role",selection:sourceRole,options:task.sourceRoles.map {.init(id:$0,title:$0.capitalized)}) {sourceRole=$0}
                    LazyVGrid(columns:[GridItem(.adaptive(minimum:240),alignment:.topLeading)],alignment:.leading) {
                        if let table=f.inputs[sourceRole]?.batchObservation(at:index) {LearningTensorTable(tensor:table,title:"Supplied "+sourceRole)}
                        if let table=f.measured[sourceRole+"Gradient"]?.batchObservation(at:index) {LearningTensorTable(tensor:table,title:"Measured influence")}
                        if compareTarget,let table=f.expected[sourceRole+"Gradient"]?.batchObservation(at:index) {LearningTensorTable(tensor:table,title:"Target influence")}
                    }
                }
                InspectorSection("Shared learned tables and their influence",symbol:"tablecells",identifier:"recurrent.parameters") {
                    let keys=task.parameterShapes.keys.sorted(),selected=keys.contains(parameter) ? parameter:keys.first ?? ""
                    FlatChoiceControl(title:"Learned table",selection:selected,options:keys.map {.init(id:$0,title:parameterTitle($0))}) {parameter=$0}
                    LazyVGrid(columns:[GridItem(.adaptive(minimum:240),alignment:.topLeading)],alignment:.leading) {
                        if let table=f.inputs["parameter."+selected] {LearningTensorTable(tensor:table,title:"Supplied "+parameterTitle(selected))}
                        if let table=f.measured["parameter."+selected+"Gradient"] {LearningTensorTable(tensor:table,title:"Measured learned-value influence")}
                        if compareTarget,let table=f.expected["parameter."+selected+"Gradient"] {LearningTensorTable(tensor:table,title:"Target learned-value influence")}
                    }
                    Text("Shared learned tables have no observation direction. Their influence adds contributions from every observation and every connected result. Starting state is supplied anew for each check; it is never silently carried between runs.").font(.caption).foregroundStyle(MLTheme.muted)
                }
            }
        }
    }
    private func parameterTitle(_ key:String)->String {key.contains("weight_ih") ? "Measurement table":key.contains("weight_hh") ? "Earlier-memory table":key.contains("bias_ih") ? "Measurement offset":"Earlier-memory offset"}
}

private struct AttentionResultsView:View {
    let projection:AttentionFormativeProjection
    let task:AttentionLearningTask
    @State private var fixtureID=""
    @State private var observation="0"
    @State private var resultRole="result"
    @State private var sourceRole=""
    @State private var parameter=""
    @State private var compareTarget=false
    private var fixture:AttentionFormativeFixture? {projection.fixtures.first {$0.id==fixtureID} ?? projection.fixtures.first}
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            if let f=fixture,let input=f.inputs[task.sourceRoles[0]] {
                let n=input.shape[0],index=min(max(Int(observation) ?? 0,0),n-1)
                LazyVGrid(columns:[GridItem(.adaptive(minimum:180),alignment:.leading)],alignment:.leading) {
                    FlatChoiceControl(title:"Supplied sequence",selection:f.id,options:projection.fixtures.enumerated().map {.init(id:$0.element.id,title:"Fixture \($0.offset+1)")}) {fixtureID=$0;observation="0"}
                    FlatChoiceControl(title:"Observation",selection:String(index),options:(0..<n).map {.init(id:String($0),title:"Observation \($0+1)")}) {observation=$0}
                    Toggle("Compare with target",isOn:$compareTarget).toggleStyle(.checkbox).font(.caption)
                }
                Text("Each observation stays separate. Result rows are requested positions; connection weights show the share of allowed source positions. Blocked connections have no share.").font(.caption).foregroundStyle(MLTheme.muted)
                InspectorSection("Position mixtures and connection weights",symbol:"tablecells",identifier:"attention.outputs") {
                    FlatChoiceControl(title:"Result role",selection:resultRole,options:task.outputRoles.map {.init(id:$0,title:$0.capitalized)}) {resultRole=$0}
                    LazyVGrid(columns:[GridItem(.adaptive(minimum:240),alignment:.topLeading)],alignment:.leading) {
                        if let table=f.measured[resultRole]?.batchObservation(at:index) {LearningTensorTable(tensor:table,title:"Measured "+resultRole)}
                        if let table=f.inputs["upstream."+resultRole]?.batchObservation(at:index) {LearningTensorTable(tensor:table,title:"Supplied result influence weights")}
                        if compareTarget,let table=f.expected[resultRole]?.batchObservation(at:index) {LearningTensorTable(tensor:table,title:"Target "+resultRole)}
                    }
                }
                InspectorSection("Blocked connections and source positions",symbol:"tablecells",identifier:"attention.checkpoints") {
                    Text("Whole-number mask tables: 1 blocks and 0 allows. Rows are query positions and columns are source positions; the source-position table blocks a source for every query.").font(.caption).foregroundStyle(MLTheme.muted)
                    LazyVGrid(columns:[GridItem(.adaptive(minimum:240),alignment:.topLeading)],alignment:.leading) {
                        ForEach(["blockedConnections","blockedSourcePositions"],id:\.self) {key in
                            if let table=f.measured[key]?.batchObservation(at:index) {LearningTensorTable(tensor:table,title:key == "blockedConnections" ? "Measured blocked connections · int64":"Measured blocked source positions · int64")}
                            if compareTarget,let table=f.expected[key]?.batchObservation(at:index) {LearningTensorTable(tensor:table,title:key == "blockedConnections" ? "Target blocked connections":"Target blocked source positions")}
                        }
                    }
                }
                InspectorSection("Supplied sequences and their influence",symbol:"tablecells",identifier:"attention.sources") {
                    let selectedSource=task.sourceRoles.contains(sourceRole) ? sourceRole:task.sourceRoles[0]
                    FlatChoiceControl(title:"Supplied role",selection:selectedSource,options:task.sourceRoles.map {.init(id:$0,title:$0.capitalized)}) {sourceRole=$0}
                    LazyVGrid(columns:[GridItem(.adaptive(minimum:240),alignment:.topLeading)],alignment:.leading) {
                        if let table=f.inputs[selectedSource]?.batchObservation(at:index) {LearningTensorTable(tensor:table,title:"Supplied "+selectedSource)}
                        if let table=f.measured[selectedSource+"Gradient"]?.batchObservation(at:index) {LearningTensorTable(tensor:table,title:"Measured influence")}
                        if compareTarget,let table=f.expected[selectedSource+"Gradient"]?.batchObservation(at:index) {LearningTensorTable(tensor:table,title:"Target influence")}
                    }
                }
                if !task.parameterShapes.isEmpty {InspectorSection("Shared learned tables and their influence",symbol:"tablecells",identifier:"attention.parameters") {
                    let keys=task.parameterShapes.keys.sorted(),selected=keys.contains(parameter) ? parameter:keys.first ?? ""
                    FlatChoiceControl(title:"Learned table",selection:selected,options:keys.map {.init(id:$0,title:parameterTitle($0))}) {parameter=$0}
                    LazyVGrid(columns:[GridItem(.adaptive(minimum:240),alignment:.topLeading)],alignment:.leading) {
                        if let table=f.inputs["parameter."+selected] {LearningTensorTable(tensor:table,title:"Supplied "+parameterTitle(selected))}
                        if let table=f.measured["parameter."+selected+"Gradient"] {LearningTensorTable(tensor:table,title:"Measured learned-value influence")}
                        if compareTarget,let table=f.expected["parameter."+selected+"Gradient"] {LearningTensorTable(tensor:table,title:"Target learned-value influence")}
                    }
                    Text("Shared learned tables have no observation direction. Their influence adds contributions from every observation and every connected result. The checker changes the supplied learned tables between fixtures.").font(.caption).foregroundStyle(MLTheme.muted)
                }} else {Text("Scaled attention owns no learned tables or saved parameters. Its measured influence belongs to the supplied query, key and value sequences.").font(.caption).foregroundStyle(MLTheme.muted)}
            }
        }
    }
    private func parameterTitle(_ key:String)->String {key.replacingOccurrences(of:".",with:" · ")}
}

private struct LossResultsView:View {
    let projection:LossFormativeProjection
    let task:LossLearningTask
    @State private var fixtureID=""
    @State private var compareTarget=false
    private var fixture:LossFormativeFixture? {projection.fixtures.first {$0.id==fixtureID} ?? projection.fixtures.first}
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            if let f=fixture {
                HStack {
                    FlatChoiceControl(title:"Supplied example",selection:f.id,options:projection.fixtures.enumerated().map {.init(id:$0.element.id,title:"Example \($0.offset+1)")}) {fixtureID=$0}
                    Toggle("Compare with target",isOn:$compareTarget).toggleStyle(.checkbox).font(.caption)
                }
                Text("Individual scores show each comparison before combining. The combined training/evaluation scores follow the selected objective settings; they are teaching measurements, not a claim about model quality.").font(.caption).foregroundStyle(MLTheme.muted)
                InspectorSection("Individual and combined scores",symbol:"tablecells",identifier:"loss.scores") {
                    LazyVGrid(columns:[GridItem(.adaptive(minimum:240),alignment:.topLeading)],alignment:.leading) {
                        ForEach(["reduction","trainingObjective","evaluationObjective"],id:\.self) {key in
                            if let tensor=f.measured[key] {LearningTensorTable(tensor:tensor,title:title(key))}
                            if compareTarget,let tensor=f.expected[key] {LearningTensorTable(tensor:tensor,title:"Target · "+title(key))}
                        }
                    }
                }
                InspectorSection("Supplied values and desired roles",symbol:"tablecells",identifier:"loss.roles") {
                    LazyVGrid(columns:[GridItem(.adaptive(minimum:240),alignment:.topLeading)],alignment:.leading) {
                        ForEach(task.sourceRoles,id:\.self) {role in
                            if let tensor=f.inputs[role] {LearningTensorTable(tensor:tensor,title:role.replacingOccurrences(of:"_",with:" ").capitalized)}
                        }
                    }
                }
                InspectorSection("Influence on the original produced values",symbol:"tablecells",identifier:"loss.influence") {
                    Text(task.usesLogSoftmax ? "Influence is measured before the visible LogSoftmax probability preparation. It shows how a small change in the original scores affects the combined objective.":"Influence shows how a small change in each original produced value affects the combined objective.").font(.caption).foregroundStyle(MLTheme.muted)
                    LazyVGrid(columns:[GridItem(.adaptive(minimum:240),alignment:.topLeading)],alignment:.leading) {
                        if let tensor=f.measured["inputInfluence"] {LearningTensorTable(tensor:tensor,title:"Measured input influence")}
                        if compareTarget,let tensor=f.expected["inputInfluence"] {LearningTensorTable(tensor:tensor,title:"Target input influence")}
                    }
                }
            }
        }
    }
    private func title(_ key:String)->String {key == "reduction" ? "Individual scores":key == "trainingObjective" ? "Combined training score":"Combined evaluation score"}
}

/// Difference-only evidence is labelled explicitly; private fixtures never reach this view.
private struct InitializationDifferenceResultsView:View {
    let projection:TensorLearningStore.InitializationMeasuredProjection
    @State private var fixtureIndex=0
    @State private var selected="startingSettings"
    private let keys=["startingSettings","predictions","firstAdjustment","repeatIsolation","savedReuse"]
    var body:some View {
        InspectorSection("Measured differences · zero means match",symbol:"tablecells",identifier:"lesson.initialization.results") {
            Text("Each number is the actual measured value minus an independent reference. These are differences, not the model’s learned weights. Saved reuse checks weights and predictions, without claiming full training continuation.").font(.caption).foregroundStyle(MLTheme.muted)
            if !projection.fixtures.isEmpty {
                FlatChoiceControl(title:"Check",selection:String(fixtureIndex),options:projection.fixtures.enumerated().map {.init(id:String($0.offset),title:"Check \($0.offset+1)")}) {fixtureIndex=Int($0) ?? 0}
                FlatChoiceControl(title:"Measurement",selection:selected,options:keys.map {.init(id:$0,title:InitializationLessonReference.label($0))}) {selected=$0}
                Text(selected == "firstAdjustment" ? "First row: sensitivity (gradient) differences. Second row: differences after one training adjustment. Array order is first weights, first bias, second weights, second bias." : selected == "repeatIsolation" ? "Compare repeated draws and reordered rules, followed by the number of changed training-random-state bytes. Zero means unchanged." : selected == "savedReuse" ? "Compare reloaded learned arrays and predictions, followed by the number of changed saved-checkpoint bytes. This checks weight reuse, not a full continuation." : selected == "predictions" ? "Six observation rows: four used for the first adjustment and two kept aside. These differences check the calculation, not prediction quality." : "Learned-array order: first weights, first bias, second weights, second bias. Both selected and unchanged arrays are checked.").font(.caption).foregroundStyle(MLTheme.muted)
                let fixture=projection.fixtures[min(fixtureIndex,projection.fixtures.count-1)]
                if let table=fixture.measured[selected] {LearningTensorTable(tensor:table,title:InitializationLessonReference.label(selected)+" · difference from reference",signed:true)}
            }
        }
    }
}
