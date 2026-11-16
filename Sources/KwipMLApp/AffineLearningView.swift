import SwiftUI
import KwipMLCore

struct AffineLearningView: View {
    @ObservedObject var lesson: AffineLearningStore
    @ObservedObject var workspace: WorkspaceStore
    @Environment(\.lessonRailPresented) private var railPresented
    private var geluMode: Bool? { lesson.lessonTask?.identity.geluApproximateTanh }
    private var isReLU: Bool { lesson.lessonTask?.identity == .relu }
    private var isTanh: Bool { lesson.lessonTask?.identity == .tanh }
    private var isSigmoid: Bool { lesson.lessonTask?.identity == .sigmoid }
    private var isFrozen: Bool { lesson.lessonTask?.identity == .frozenStandardization }
    private var isBackward: Bool { lesson.lessonTask?.identity == .backwardChain }
    private var isComposition:Bool {lesson.lessonTask?.identity == .composition}
    private var isRecovery:Bool {lesson.lessonTask?.identity.assessmentID == AffineLessonIdentity.recovery(taskSeed:0).assessmentID}
    private var activationCopy:AffineRemainingActivationCopy.Packet? {AffineRemainingActivationCopy.packet(lesson.lessonTask?.identity)}
    private var hasReviewedActivation:Bool {isReLU || activationCopy != nil}
    private var hasReviewedCopy:Bool {isComposition || isRecovery || isBackward || isFrozen || hasReviewedActivation}
    private func reviewedTitle(_ id:String)->String {if isReLU {return AffineReLULessonCopy.title(id)}; if let value=activationCopy?.reference.first(where:{$0.id==id}) {return value.title}; if isFrozen {return AffineFrozenLessonCopy.title(id)}; return isBackward ? AffineBackwardLessonCopy.title(id):(isRecovery ? AffineRecoveryLessonCopy.title(id):AffineCompositionLessonCopy.title(id))}
    private func reviewedText(_ id:String)->String {if isReLU {return AffineReLULessonCopy.text(id)}; if let value=activationCopy?.reference.first(where:{$0.id==id}) {return value.text}; if isFrozen {return AffineFrozenLessonCopy.text(id)}; return isBackward ? AffineBackwardLessonCopy.text(id):(isRecovery ? AffineRecoveryLessonCopy.text(id):AffineCompositionLessonCopy.text(id))}
    private enum ContextPanel:String,CaseIterable {case task="Task",parameters="Parameters",results="Results"}
    @State private var contextPanel = ContextPanel.task

    var body: some View {
        VStack(spacing: 0) {
            if !railPresented {
            ViewThatFits(in: .horizontal) {
                HStack(alignment: .center, spacing: 16) { lessonTitle; Spacer(minLength: 16); lessonActions }
                VStack(alignment: .leading, spacing: 12) { lessonTitle; HStack(spacing: 16) { Spacer(); lessonActions } }
            }.padding(.horizontal, 14).padding(.vertical, 12).background(MLTheme.panel)
                .overlay(alignment: .bottom) { Rectangle().fill(MLTheme.rule).frame(height: 1) }
            }
            if workspace.showLibrary { ComponentPaletteView(store:workspace,allowedTypes:lesson.lessonTask?.allowedNodeTypes).disabled(lesson.isBusy) }
            HSplitView {
                if workspace.showInspector {
                    VStack(spacing:0) {
                        HStack(spacing:6) {
                            ForEach(railPresented ? [ContextPanel.parameters,.results]:ContextPanel.allCases,id:\.self) { panel in
                                Button(panel.rawValue) {contextPanel=panel}
                                    .buttonStyle(FlatButtonStyle(prominent:contextPanel == panel))
                            }
                        }.padding(10)
                        if contextPanel == .task {taskPanel}
                        else {
                            ThemedScrollView {
                                VStack(alignment:.leading,spacing:16) {
                                    if contextPanel == .results {resultPanel} else {parameterPanel}
                                }.padding(14)
                            }
                        }
                    }.frame(minWidth:300,idealWidth:360,maxWidth:520).background(MLTheme.panel).workspacePane(.inspector)
                        .workspaceSplitLayout(store:workspace,scope:.affineRibbonDock)
                }
                VStack(spacing: 0) {
                    HStack {
                        Button("Undo") { workspace.undo() }.disabled(!workspace.history.canUndo || lesson.isBusy)
                        Button("Redo") { workspace.redo() }.disabled(!workspace.history.canRedo || lesson.isBusy)
                        Spacer()
                        Text("Build and connect your construction").font(.caption).foregroundStyle(MLTheme.muted)
                    }.buttonStyle(.borderless).padding(10)
                    GraphCanvasView(store: workspace)
                }.frame(minWidth: 350)
            }
            if !railPresented {
            HStack {
                Text(lesson.statusMessage).font(.system(size: 11)).lineLimit(3).textSelection(.enabled)
                Spacer()
                if lesson.isBusy { Button("Cancel assessment") { lesson.cancel() }.buttonStyle(FlatButtonStyle()) }
                else {
                    Button(lesson.mode == .assessment ? "Submit assessment" : "Check construction") { contextPanel = .results; workspace.showInspector = true; lesson.submit() }
                        .buttonStyle(FlatButtonStyle(prominent: true)).disabled(!lesson.canSubmit)
                }
            }.padding(10).background(MLTheme.panel)
            }
        }.background(MLTheme.background)
            .onChange(of:railPresented) {_,shown in if shown && contextPanel == .task {contextPanel = .parameters}}
            .onAppear {if lesson.mode != .assessment {lesson.setMode(.assessment)}; if railPresented && contextPanel == .task {contextPanel = .parameters}}
    }
    private var lessonTitle: some View {
        VStack(alignment: .leading, spacing: 4) {
            SectionLabel(title: "Model mathematics · construction")
            Text(lesson.lessonTask?.title ?? "Affine construction").font(.system(size: 16, weight: .semibold)).fixedSize(horizontal: false, vertical: true)
        }
    }
    private var lessonActions: some View {
        HStack(spacing: 8) {
            Button("Save") { lesson.save() }.buttonStyle(FlatButtonStyle()).disabled(lesson.isBusy)
            Button(lesson.leaveActionTitle) { lesson.leave(returnToPreviousProject: true) }.buttonStyle(FlatButtonStyle()).disabled(lesson.isBusy)
        }
    }
    private var taskPanel: some View { ThemedScrollView {taskContents} }
    var taskContents: some View {
            VStack(alignment: .leading, spacing: 16) {
                let notesProjectID = lesson.draft?.projectID
                LessonTextPanel(title: lesson.mode.rawValue.capitalized + " mode", text: modeExplanation)
                if let task = lesson.publicTask {
                    InspectorSection(hasReviewedCopy ? reviewedTitle("reference.goal"):"Construction task", symbol: "doc.text", identifier: "lesson.affine.task") {
                    LessonRichText(text: practicalObjective)
                    if hasReviewedCopy {
                        LessonTextPanel(title:reviewedTitle("reference.workflow"),text:reviewedText("reference.workflow"))
                    } else {
                    LessonRichText(text: "**Canvas workflow:** drag allowed blocks from **All** onto the canvas. Connect an output port to the next input port; a port is a block’s connection point. Select a block and open **Parameters** to set its sizes and numbers. **Check construction** shows measured results in Guided or Practice; **Submit assessment** checks additional private examples.")
                    }
                    if let reference = lesson.lessonTask {
                        if isComposition {
                            InspectorSection(AffineCompositionLessonCopy.title("reference.maps"),symbol:"tablecells",identifier:"lesson.affine.composition.maps") {
                                AffineTaskReferenceView(reference:LearningExamples.affineReference,instructions:AffineCompositionLessonCopy.mapIntroduction,conclusion:AffineCompositionLessonCopy.mapConclusion,showsCombinedEquation:false)
                            }
                        } else if hasReviewedActivation {
                            LessonContentBlockView(block:.init(id:"affine.activation.toolbar",kind:.callout,title:"Find the activation",text:"Use Activation in the toolbar. This assessment restricts its Function selector to the required function. Select the placed block to inspect its Parameters."+(geluMode == nil ? "":" GELU uses the Approximation selector to choose Exact or Tanh approximation.")))
                            ForEach(["reference.examples","reference.configure","reference.inspect"],id:\.self) {id in
                                LessonContentBlockView(block:.init(id:"affine.activation."+id,kind:id == "reference.examples" ? .visual:(id == "reference.configure" ? .canvasAction:.observation),title:reviewedTitle(id),text:reviewedText(id)))
                            }
                        } else if reference.identity.assessmentID == AffineLessonIdentity.recovery(taskSeed: 0).assessmentID {
                            AffineRecoveryReferenceView(lesson: lesson, task: reference)
                        } else if isFrozen {
                            LessonTextPanel(title:AffineFrozenLessonCopy.title("reference.statistics"),text:AffineFrozenLessonCopy.text("reference.statistics"))
                            LessonContentBlockView(block:.init(id:"affine.frozen.scope",kind:.callout,title:AffineFrozenLessonCopy.title("reference.scope"),text:AffineFrozenLessonCopy.text("reference.scope")))
                        } else if isBackward {
                            InspectorSection(AffineBackwardLessonCopy.title("reference.maps"),symbol:"tablecells",identifier:"lesson.affine.backward.maps") {
                                AffineTaskReferenceView(reference:LearningExamples.affineReference,instructions:AffineBackwardLessonCopy.mapIntroduction,conclusion:AffineBackwardLessonCopy.mapConclusion,showsCombinedEquation:false)
                            }
                            LessonTextPanel(title:AffineBackwardLessonCopy.title("reference.meaning"),text:AffineBackwardLessonCopy.text("reference.meaning"))
                        } else { AffineTaskReferenceView(reference: LearningExamples.affineReference) }
                    }
                    }
                    InspectorSection(hasReviewedCopy ? reviewedTitle("reference.criteria"):"Success criteria", symbol: "checklist", identifier: "lesson.affine.criteria") {
                    if hasReviewedCopy {LessonRichText(text:reviewedText("reference.criteria"))}
                    else {
                    LessonRichText(text: "Match the requested transformation for every observation, keeping feature order and the requested input/output sizes. Different valid block arrangements are accepted. A saved passing Assessment is required for completion; Guided and Practice checks are for learning.")
                    }
                    InspectorSection("Optional · exact checks and limits", symbol: "list.bullet.rectangle", identifier: "lesson.affine.exactCriteria", initiallyExpanded: false) {
                    ForEach(task.criteria) { criterion in LessonRichText(text: criterion.description) }
                    Text("CPU execution · \(task.budget.maximumTensorElements) tensor elements maximum · \(task.budget.milliseconds / 1000, specifier: "%.1f") s compute budget")
                        .font(.system(size: 10)).foregroundStyle(MLTheme.muted)
                    ForEach(task.prompts, id: \.self) { LessonRichText(text: $0) }
                    }
                    }
                } else { Text("Preparing the verified runtime contract…").font(.caption) }
                InspectorSection(hasReviewedCopy ? reviewedTitle("reference.notation"):"Optional · mathematical notation", symbol: "function", identifier: "lesson.affine.conventions", initiallyExpanded: false) {
                if hasReviewedCopy {LessonRichText(text:reviewedText("reference.notation"))}
                else {
                LessonRichText(text: "An affine map multiplies input values by fixed weights and adds a fixed offset, called bias. A matrix is a rectangular table of weights. x and y are one input and output observation; W holds the weights and b holds the offsets. Capital X and Y contain a batch, meaning several observations in rows. T means transpose: exchange rows and columns. ‘in’ and ‘out’ count features, not observations.")
                Text("Column-vector mathematics\ny = Wx + b\n\nRow-batch execution\nY = XWᵀ + b\nX: [batch, in] · W: [out, in]\nY: [batch, out]")
                    .font(.system(size: 11, design: .monospaced)).textSelection(.enabled)
                }
                LessonRichText(text: "The weights and offsets you enter are checked directly. This exercise does not use training to find them for you.")
                }
                if lesson.mode != .assessment { InspectorSection("Optional hints", symbol: "lightbulb", identifier: "lesson.affine.hints") { hints } }
                InspectorSection("Prediction & reasoning", symbol: "pencil", identifier: "lesson.affine.reflection") {
                LessonRichText(text: "Predict what should change in the output when one input number changes. After checking, compare your prediction with the measured result and explain any difference.")
                TextEditor(text: Binding(get: { lesson.draft?.notes ?? "" }, set: { if lesson.draft?.projectID == notesProjectID { lesson.setNotes(String($0.prefix(8000))) } }))
                    .font(.system(size: 12)).frame(minHeight: 110).scrollContentBackground(.hidden)
                    .background(MLTheme.background).accessibilityLabel("Prediction and reasoning notes").disabled(lesson.isBusy)
                LessonRichText(text: "Notes are saved with your draft and submitted receipt; they are not automatically graded.")
                }
            }.padding(14)
    }
    private var practicalObjective: String {
        if hasReviewedCopy {return reviewedText("reference.goal")}
        return "**Goal:** replace two consecutive multiply-and-add transformations with a construction that gives the same result. Each transformation is an affine map: weights multiply the inputs, then a fixed offset (bias) is added. Use the public tables below as your specification."
    }
    private var modeExplanation: String {
        switch lesson.mode {
        case .guided: return "Build with prompts and optional conceptual hints. Checks provide feedback, not assessment completion."
        case .practice: return "Choose your own construction and check its behavior. Practice results do not count as assessment completion."
        case .assessment: return "Submit your connected blocks and entered numbers. The checker uses additional private examples, and a saved passing result counts toward completion."
        }
    }
    private var hints: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let task = lesson.lessonTask, task.identity.assessmentID == AffineLessonIdentity.recovery(taskSeed: 0).assessmentID {
                hint("recovery-bias", title: "Zero-input offset", text: "Imagine both input numbers are zero. Multiplying them by any weight still gives zero. Which kind of value can still appear in the output?")
                hint("recovery-columns", title: "Basis-input changes", text: "A basis input switches on just one input feature: one value is 1 and the other is 0. Compare its output with the all-zero example. What contribution comes from that one feature?")
            } else if isReLU {
                hint("relu-sign", title: "Negative and positive inputs", text: "ReLU keeps whichever is larger: the input number or zero. What happens to a negative input, zero and a positive input?")
                hint("relu-independent", title: "One value at a time", text: "Changing a companion value should not change this value’s result. Softmax compares several numbers; Tanh and Sigmoid also change positive values. ReLU must preserve positives.")
                hint("relu-scaffold", title: "Preserve the input", text: "Review which Linear weights carry each feature to itself, which mix columns, and whether fixed offsets alter the input before clipping. A second ReLU does not undo or change the first clipping operation.")
            } else if isTanh {
                hint("tanh-signed", title: "Sign and midpoint", text: "A negative input should still give a negative output. What should zero become? Compare this with ReLU, which clips negative values, and Sigmoid, which places zero at one half.")
                hint("tanh-independent", title: "Independent values", text: "Transform one number at a time. Should changing its neighbor change its result? Softmax compares several numbers; Tanh does not.")
                hint("tanh-scaffold", title: "Preserve the input", text: "Review which Linear weights carry each feature to itself, which mix columns, and whether fixed offsets alter the inputs before Tanh.")
            } else if isSigmoid {
                hint("sigmoid-range", title: "Range and midpoint", text: "Negative inputs give values below one half; positive inputs give values above one half. What should zero become? The result is a transformed value, does not automatically estimate how likely an outcome is.")
                hint("sigmoid-independent", title: "Separate columns", text: "Each number is transformed independently. If you change its neighbor, should its own result change? Softmax works differently because it compares several numbers.")
                hint("sigmoid-scaffold", title: "Preserve the input", text: "Review which Linear weights carry a feature to itself, which mix columns, and whether fixed offsets change the input before Sigmoid.")
            } else if geluMode != nil {
                hint("gelu-negative", title: "Negative inputs", text: "GELU multiplies each input by a number between 0 and 1. If the input is negative, can the product become positive? ReLU is a different gate that sets negative inputs to zero.")
                hint("gelu-scaffold", title: "Preserve the input", text: "A Linear block can pass values through unchanged. Think about the weight connecting each feature to itself, weights connecting it to other features, and whether adding an offset would change it.")
                hint("gelu-approximation", title: "Approximation choice", text: "Check the GELU block’s Tanh approximation switch against this task. That switch chooses a faster approximate calculation; a separate Tanh block does not perform GELU.")
            } else if isFrozen {
                hint("frozen-diagonal", title: "Independent feature scales", text: "Each prepared feature uses only its matching input column. Which weights would accidentally mix two columns? Remember that multiplying by a scale differs from dividing by it.")
                hint("frozen-offset", title: "Scale the offset too", text: "First subtract the saved average, then divide by the saved scale. What happens to the subtracted amount when you rewrite this as multiplying the input and adding one fixed number?")
                hint("frozen-batch", title: "A stable inference contract", text: "A batch is a group of observations. Keep one row fixed while changing the other rows. Its prepared result should stay the same because the averages and scales were saved earlier.")
            } else if isBackward {
                hint("backward-order", title: "Reverse the path", text: "Start at the end of the original forward chain. Follow the second transformation back before the first. Reversing a transformation’s sensitivity exchanges the roles of its input and output connections.")
                hint("backward-bias", title: "Constant offsets", text: "An offset is fixed. When an input changes slightly, does that offset change too? Decide whether adding the original offset belongs in a calculation of sensitivity.")
                hint("backward-check", title: "Directional check", text: "Make a small change to an input and think about the resulting output change. The sensitivity sent backward should describe the same effect seen from the input side.")
            } else {
                hint("matrix-order", title: "Matrix order", text: "Apply the first transformation before the second. Count the features after each step. Check that each block’s input size matches the previous block’s output size.")
                hint("bias-transform", title: "Bias composition", text: "The first offset becomes part of the values entering the second transformation. Track how the second set of weights changes that offset before the second offset is added.")
            }
        }
    }
    private func hint(_ id: String, title: String, text: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            let originatingProjectID = lesson.draft?.projectID
            if lesson.draft?.hintIDs.contains(id) == true { Text(title).font(.caption.bold()); LessonRichText(text: text) }
            else { Button("Reveal \(title.lowercased()) hint") { if lesson.draft?.projectID == originatingProjectID { lesson.revealHint(id) } }.buttonStyle(.borderless).disabled(lesson.isBusy) }
        }
    }
    private var parameterPanel: some View {
        VStack(alignment: .leading, spacing: 14) {
            if let node = workspace.selectedNode, let definition = workspace.registry.component(type: node.type) {
                HStack { ComponentIcon(type: node.type, size: 24); Text(definition.title).font(.headline) }
                if node.type == "linear" {
                LessonRichText(text: "For Linear blocks, the weight table works as follows: each row produces one output feature and each column receives one input feature. Bias adds one fixed offset per output. Choose input/output sizes first, then enter the table and offsets below.")
                } else {
                    LessonRichText(text: ComponentStyle.purpose(node.type))
                    if definition.fields.isEmpty {
                        LessonRichText(text: "This block has no adjustable computation settings. You can change its label and connections. Surrounding blocks can still change the values it receives.")
                    }
                }
                SectionLabel(title: "Your label")
                TextField("Block name", text: Binding(get: { workspace.graph.nodes.first { $0.id == node.id }?.name ?? "" }, set: { if workspace.selectedNodeID == node.id { workspace.renameNode($0) } }))
                    .flatField().disabled(lesson.isBusy).accessibilityLabel("Lesson block name")
                if ActivationConfiguration.Function(rawValue:node.type) != nil {
                    let project=workspace.activeProjectID
                    let allowed=ActivationPalette.functions(allowedTypes:lesson.lessonTask?.allowedNodeTypes)
                    ActivationConfigurationView(node:node,configurationVersion:workspace.graph.schemaVersion,registry:workspace.registry,
                        allowedFunctions:allowed,isBusy:lesson.isBusy,
                        onPending:{lesson.setNumericEditPending(id:"activation:"+node.id,pending:$0)}) {function,configuration,expectedType,expectedConfig in
                            guard ActivationPalette.functions(allowedTypes:lesson.lessonTask?.allowedNodeTypes)==allowed else {return false}
                            return workspace.setActivation(function,configuration:configuration,nodeID:node.id,expectedType:expectedType,
                                expectedSourceConfig:expectedConfig,expectedProjectID:project,allowedFunctions:allowed)
                        }
                } else {
                ForEach(definition.fields, id: \.key) { field in
                    RegistryFieldControl(field: field, value: node.config[field.key] ?? field.defaultValue,
                        origin: "\(workspace.activeProjectID ?? ""):\(node.id):\(field.key)", currentOrigin: {
                            guard !lesson.isBusy, workspace.selectedNodeID == node.id else { return nil }
                            return "\(workspace.activeProjectID ?? ""):\(node.id):\(field.key)"
                        }, onDraftChange: { lesson.setNumericEditPending(id: $0, pending: $1) }) { value in
                            workspace.setConfigValue(field.key, value: value, nodeID: node.id)
                        }.disabled(lesson.isBusy)
                }
                }
                if node.type == "linear", node.config["bias"]?.boolValue == false {
                    Text("Bias is the fixed offset added after multiplication. It is disabled for this block, so any saved offset values here are ignored. Another block with bias enabled may still add the overall offset.")
                        .font(.caption).foregroundStyle(MLTheme.rose)
                }
                if node.type == "linear" { MatrixParameterEditor(lesson: lesson, workspace: workspace, node: node).id(node.id) }
                else { LessonRichText(text: "Select a Linear block to enter its explicit weight matrix and bias.") }
                Button("Delete selected block") { workspace.deleteSelection() }.buttonStyle(FlatButtonStyle()).disabled(lesson.isBusy)
            } else {
                Text("Select a block to edit its label and configuration. Add components and connect their ports to construct the requested function.").font(.system(size: 12)).foregroundStyle(MLTheme.muted)
            }
            if !lesson.linearNodes.isEmpty {
                FlatChoiceControl(title: "Select Linear block", selection: workspace.selectedNodeID ?? "", options: lesson.linearNodes.map { FlatChoiceOption(id: $0.id, title: $0.name.isEmpty ? "Linear" : $0.name) }) { workspace.selectNode($0) }.disabled(lesson.isBusy)
            }
            Divider(); SectionLabel(title: "Submission readiness")
            if let issue = lesson.lessonRuntimeIssue { Text(issue).font(.caption).foregroundStyle(MLTheme.rose) }
            if !lesson.pendingNumericEdits.isEmpty { Text("Apply or correct pending numeric edits before submitting.").font(.caption).foregroundStyle(MLTheme.rose) }
            if workspace.validation.isValid {
                ForEach(Array(lesson.parameterIssues.enumerated()), id: \.offset) { _, issue in Text(issue).font(.caption).foregroundStyle(MLTheme.rose) }
            }
            ForEach(workspace.validation.issues) { issue in
                Button { if let id = issue.nodeID { workspace.selectNode(id) } } label: { Text(issue.message).font(.caption).foregroundStyle(MLTheme.rose).multilineTextAlignment(.leading) }.buttonStyle(.plain)
            }
        }
    }
    private var resultNextStep: String {
        if isReLU { return "Review Linear weights and fixed offsets in Parameters. Preserve each input before ReLU and keep feature order. Negative values should become zero while positive values should stay unchanged. Recheck in Guided or Practice to compare the measured public table. Exact feedback below retains the checker’s details." }
        if isTanh { return "Review Linear weights and fixed offsets in Parameters. Preserve each input before Tanh and keep feature order. Negative values should remain negative and zero should remain zero. Recheck in Guided or Practice to compare the measured public table. Exact feedback below retains the checker’s details." }
        if isSigmoid { return "Review the Linear weights and fixed offsets in Parameters. Preserve each input before Sigmoid and keep the feature order. Sigmoid transforms each number separately, unlike Softmax. Recheck in Guided or Practice to compare the measured public table. Exact feedback below retains the checker’s details." }
        if geluMode != nil { return "Select each GELU block and check the requested approximation switch. In Parameters, review whether surrounding Linear blocks change the values before or after GELU. Recheck in Guided or Practice to compare measured outputs. Exact feedback below retains the checker’s details." }
        if isFrozen { return "Review the saved averages, scales and feature order in the task. In Parameters, check each Linear weight and offset. Confirm that changing another row would not change this row’s prepared value. Exact feedback below retains the checker’s details." }
        if isBackward { return "Follow the original chain backward from its end. In Parameters, review which input sensitivity contributes to each output sensitivity and whether fixed offsets belong in that calculation. Exact feedback below retains the checker’s details." }
        return "Open Parameters and review the weights and fixed offsets of each Linear block. Compare the public examples with measured outputs in Guided or Practice. Check the order of connected blocks, then submit again. Exact feedback below retains the checker’s details."
    }
    private var resultPanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            InspectorSection("Latest result", symbol: "checkmark.seal", identifier: "lesson.affine.latestResult") {
            if let grade = lesson.latestGrade {
                Text(lesson.latestAttempt == nil ? "Unrecorded attempt · no verified receipt" : (lesson.latestResultIsCurrent ? "Matches current construction" : "Previous construction · assess again after edits"))
                    .font(.caption).foregroundStyle(lesson.latestAttempt != nil && lesson.latestResultIsCurrent ? MLTheme.teal : MLTheme.rose)
                if let attempt = lesson.latestAttempt {
                    LessonRichText(text: "\(attempt.mode.rawValue.capitalized) · assessed \(attempt.timestamp.formatted(date: .abbreviated, time: .shortened))")
                    Text("Content \(attempt.contentVersion.major).\(attempt.contentVersion.minor).\(attempt.contentVersion.patch) · assessment \(attempt.assessmentVersion.major).\(attempt.assessmentVersion.minor).\(attempt.assessmentVersion.patch)").font(.system(size: 10)).foregroundStyle(MLTheme.muted)
                }
                if lesson.latestAttempt != nil && !lesson.latestAssessmentIsCurrent {
                    LessonRichText(text: "Historical assessment version. This receipt remains valid for its recorded contract; submit again to assess the current version.")
                }
                let state=LessonCheckState(grade:grade,running:lesson.isBusy,canSubmit:lesson.canSubmit,unavailable:false,current:lesson.latestResultIsCurrent)
                Label(state.label,systemImage:state.symbol).font(.system(size:12,weight:.semibold)).foregroundStyle(state.tint)
                LessonRichText(text: "The checker compares your connected blocks with the requested behavior. **Passed** means this attempt met its checks; completion also requires an Assessment attempt and a saved receipt. After changing blocks or numbers, check again.")
                InspectorSection("Checker summary", symbol: "text.alignleft", identifier: "lesson.affine.resultSummary") {
                    LessonRichText(text: grade.summary)
                }
                LessonCriteriaRowsView(items:grade.outcomes.enumerated().map {index,outcome in
                    .project(outcome,title:lesson.publicTask?.criteria.first {$0.id==outcome.criterionID}?.description ?? "Construction check \(index+1)",publicNextStep:outcome.passed ? nil:resultNextStep,assessment:false)
                },showsTechnicalFeedback:true)
                Text(lesson.resultSaved ? "Attempt receipt saved" : "Receipt not saved; this result is not a saved completion")
                    .font(.caption).foregroundStyle(lesson.resultSaved ? MLTheme.teal : MLTheme.rose)
            } else { LessonRichText(text: "No numerical assessment result yet.") }
            }
            if lesson.mode != .assessment {
                Divider()
                if let projection = lesson.formativeProjection {
                    LessonTextPanel(title: "Inspect your measured output", text: "Each row below is a checked observation. Compare its output values with the public task and your prediction. A feature is one column; a batch is a group of rows. These Guided/Practice measurements help you reason about this construction. Private Assessment examples are never displayed here.")
                    InspectorSection("Measured values", symbol: "tablecells", identifier: "lesson.affine.measured") {
                        AffineFormativeResultsView(projection: projection, backward: isBackward).id(projection.submissionDigest)
                    }
                } else {
                    LessonRichText(text: "Check the current construction in Guided or Practice mode to inspect measured outputs. Assessment attempts never expose their fixtures.")
                }
            }
            Divider()
            Text(lesson.hasCurrentCompletion ? "A passing assessment is saved for this lesson version." : "No saved passing assessment for this lesson version.").font(.caption)
            Text("Version completion records a past assessment; it does not certify an edited graph or general mastery.").font(.system(size: 10)).foregroundStyle(MLTheme.muted)
            InspectorSection("Saved attempts", symbol: "clock", identifier: "lesson.affine.attempts") {
            ForEach(Array(lesson.attempts.filter { $0.objectiveID == lesson.publicTask?.objectiveID }.suffix(12).reversed())) { attempt in
                VStack(alignment: .leading, spacing: 3) {
                    Text("\(attempt.mode.rawValue.capitalized) · \(attempt.grade.status.rawValue)").font(.caption.bold())
                    Text(attempt.timestamp.formatted(date: .abbreviated, time: .shortened)).font(.system(size: 10)).foregroundStyle(MLTheme.muted)
                    LessonRichText(text: attempt.grade.summary)
                }.padding(10).frame(maxWidth: .infinity, alignment: .leading)
                    .background(MLTheme.background).overlay(Rectangle().stroke(MLTheme.rule))
            }
            if lesson.attempts.filter({ $0.objectiveID == lesson.publicTask?.objectiveID }).count > 12 { LessonRichText(text: "Showing the 12 most recent saved attempts.") }
            }
        }
    }
}
