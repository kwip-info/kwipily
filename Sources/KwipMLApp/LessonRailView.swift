import SwiftUI
import KwipMLCore

private struct LessonRailPresentedKey: EnvironmentKey { static let defaultValue=false }
extension EnvironmentValues { var lessonRailPresented: Bool { get {self[LessonRailPresentedKey.self]} set {self[LessonRailPresentedKey.self]=newValue} } }

struct LessonRailView: View {
    @ObservedObject var workspace:WorkspaceStore
    @ObservedObject var affine:AffineLearningStore
    @ObservedObject var normalization:NormalizationLearningStore
    @ObservedObject var tensor:TensorLearningStore
    @ObservedObject var reading:LessonReadingStore
    let entryID:String
    @Environment(\.colorScheme) private var colorScheme
    @AppStorage(LessonTypography.preferenceKey) private var textSize = LessonTypography.defaultSize
    private var typography:LessonTypography {.init(bodySize:textSize)}
    @State private var showFreshCopy=false
    @State private var showSteps=false
    @State private var saveMessage:String?
    @State private var selectedStepID:String?
    private var content:LessonContent? {
        guard var result=authoredContent else {return nil}
        if affine.isActive {result.readingScopeID=affine.lessonTask?.identity.assessmentID}
        else if tensor.isActive,let operation=tensor.draft?.operation {result.readingScopeID=entryID+":"+operation}
        return result
    }
    private var authoredContent:LessonContent? {
        if let host=workspace.workflowSession,let progress=host.owner.document.guidedBuild,
           let plan=try? WorkflowGuidedBuildPlan.parallelClassifier(projectID:host.owner.projectID,version:progress.version),progress.courseID==plan.id {
            return .init(entryID:entryID,title:"Guided canvas build",kind:"Guided",steps:plan.steps.enumerated().map {index,step in
                var result=LessonContent.step("placement.\(index)",ClassifierLessonGuide.title(for:step.address.localID,fallback:step.title,version:progress.version),ClassifierLessonGuide.explanation(for:step.address.localID,fallback:step.explanation,version:progress.version),kind:.canvasAction)
                result.nodeFocus = .init(addresses:[step.address]+plan.completeDocument.edges.filter {$0.to.address==step.address}.map { $0.from.address })
                return result
            })
        }
        if affine.isActive,let task=affine.lessonTask {
            if task.identity == .relu {return .init(entryID:entryID,title:task.title,kind:"Construction",steps:AffineReLULessonCopy.steps)}
            if let copy=AffineRemainingActivationCopy.packet(task.identity) {return .init(entryID:entryID,title:task.title,kind:"Construction",steps:copy.steps)}
            if task.identity == .composition {
                return .init(entryID:entryID,title:task.title,kind:"Construction",steps:AffineCompositionLessonCopy.steps)
            }
            if task.identity == .frozenStandardization {
                return .init(entryID:entryID,title:task.title,kind:"Construction",steps:AffineFrozenLessonCopy.steps)
            }
            if task.identity == .backwardChain {
                return .init(entryID:entryID,title:task.title,kind:"Construction",steps:AffineBackwardLessonCopy.steps)
            }
            if task.identity.assessmentID == AffineLessonIdentity.recovery(taskSeed:0).assessmentID {
                return .init(entryID:entryID,title:task.title,kind:"Construction",steps:AffineRecoveryLessonCopy.steps)
            }
            return .init(entryID:entryID,title:task.title,kind:"Construction",steps:LessonContentCatalog.authored(objective:task.instructions,steps:["Build a construction using the allowed components. Select each component to configure its values, then compare your prediction with Check construction."],boundary:"A practice check and an assessment are separate. Inspect the actual measured feedback; reading or saving does not award correctness."))
        }
        var result=LessonContentCatalog.content(entryID:entryID,operation:tensor.isActive ? tensor.draft?.operation:nil)
        if tensor.isActive,TensorTopicLessonCopy.task(tensor.draft?.operation ?? "") == nil,LayerTopicLessonCopy.task(tensor.draft?.operation ?? "") == nil,TrainingTopicLessonCopy.steps(entryID,operation:tensor.draft?.operation) == nil,let task=tensor.publicTask,result?.steps.isEmpty == false {
            result?.steps[0].blocks.append(.init(id:"task.requirements",kind:.detail,title:"Task & supplied shapes",text:LessonAudienceGuide.requirements(operation:tensor.draft?.operation ?? "",fallback:task.instructions)))
        }
        return result
    }
    private var topicAccent:Color {
        let topic=LearningLibraryPresentation.items.first {$0.id==entryID}?.category ?? "Foundations"
        let family=LearningLibraryPresentation.family(topic)
        return colorScheme == .light ? family.secondary:family.primary
    }
    private func index(_ content:LessonContent)->Int {selectedStepID.flatMap {id in content.steps.firstIndex {$0.id==id}} ?? reading.position(content)}
    var body: some View {
        let content=self.content
        return VStack(spacing:0) {
            HStack(alignment:.top,spacing:8) {
                Image(systemName:LessonWindowPresentation.symbol(kind:content?.kind ?? "Practice")).foregroundStyle(topicAccent)
                VStack(alignment:.leading,spacing:4) {
                    Text(content?.title ?? "Lesson").font(.system(size:typography.size(13),weight:.semibold)).fixedSize(horizontal:false,vertical:true)
                    Text(content?.kind ?? "Practice").font(.caption).foregroundStyle(MLTheme.muted)
                }
                Spacer(minLength:0)
                Button {workspace.showLessonRail=false} label: {Image(systemName:"sidebar.right")}.help("Hide lesson").accessibilityLabel("Hide lesson")
            }.padding(14)
            HStack {
                Button("Save lesson",action:save).accessibilityIdentifier("lesson.save").disabled(workspace.lessonContext?.restoreState == .failed)
                if let projectID=workspace.activeProjectID,!reading.isUnsaved(projectID:projectID),
                   reading.record(entryID).savedProjectID==projectID || workspace.isLearningLessonActive {
                    Button("Unsave") {
                        saveMessage=reading.unsave(projectID:projectID) ? "Bookmark removed":reading.error ?? "Unsave needs attention"
                    }.help("Remove bookmark; keep this canvas and assessment history")
                }
                Button {showFreshCopy=true} label: {Image(systemName:"arrow.counterclockwise")}
                    .accessibilityLabel("Start fresh lesson canvas").help("Start fresh · Open the supplied canvas as a new copy")
                    .disabled(workspace.isBusy)
                    .confirmationDialog("Start fresh lesson canvas?",isPresented:$showFreshCopy,titleVisibility:.visible) {
                        Button("Open fresh copy") {
                            do {_ = try ProjectWindowRegistry.shared.startFreshLesson(entryID:entryID,operation:tensor.isActive ? tensor.draft?.operation:nil,workspace:workspace)}
                            catch {saveMessage=error.localizedDescription}
                        }
                        Button("Cancel",role:.cancel) {}
                    } message: {Text("Your current canvas and its saved history stay available.")}
                Spacer()
                Text(saveMessage ?? (workspace.hasUnsavedChanges ? "Unsaved changes":"On this Mac")).font(.caption2).foregroundStyle(MLTheme.muted).lineLimit(2)
            }.padding(.horizontal,14).padding(.bottom,10)
            HStack(spacing:10) {
                Text("Text size").font(.caption).foregroundStyle(MLTheme.muted)
                Spacer(minLength:0)
                Button {textSize=LessonTypography.normalized(typography.bodySize-2)} label:{Text("A−")}
                    .accessibilityLabel("Decrease lesson text size").disabled(typography.bodySize<=LessonTypography.minimumSize)
                Text("\(typography.bodySize) pt").monospacedDigit().font(.caption).frame(minWidth:36)
                    .accessibilityLabel("Lesson text size").accessibilityValue("\(typography.bodySize) points")
                Button {textSize=LessonTypography.normalized(typography.bodySize+2)} label:{Text("A+")}
                    .accessibilityLabel("Increase lesson text size").disabled(typography.bodySize>=LessonTypography.maximumSize)
            }.padding(.horizontal,14).padding(.bottom,10)
            Divider()
            if let content,!content.steps.isEmpty {
                let position=index(content)
                HStack {Text("Step \(position+1) of \(content.steps.count)");Spacer();Button("All steps") {showSteps.toggle()}}
                    .font(.caption).padding(14)
                if showSteps {
                    ThemedScrollView {
                        VStack(alignment:.leading,spacing:0) {
                            ForEach(Array(content.steps.enumerated()),id:\.element.id) {offset,step in
                                Button {navigate(offset,content:content);showSteps=false} label: {
                                    HStack(alignment:.top) {Text(String(offset+1)).frame(width:18);Text(step.title);Spacer(minLength:0)}
                                        .font(.caption).padding(9).frame(maxWidth:.infinity,alignment:.leading)
                                        .background(offset==position ? MLTheme.rose.opacity(0.10):.clear)
                                }.buttonStyle(.plain)
                            }
                        }
                    }.frame(maxHeight:200)
                    Divider()
                }
                ThemedScrollView {
                    VStack(alignment:.leading,spacing:12) {
                        Text(content.steps[position].title).font(.system(size:typography.size(14),weight:.semibold))
                        ForEach(content.steps[position].blocks) {block in
                            LessonContentBlockView(block:block,selectTarget:selectTarget)
                        }
                        if workspace.workflowSession?.owner.document.guidedBuild != nil,let host=workspace.workflowSession {
                            LessonBlockSection(title:"Build this step",symbol:"list.number",identifier:"lesson.guided.action") {WorkflowCourseCoachView(owner:host.owner,host:host)}
                        } else if workspace.tutorialStep != nil {
                            LessonBlockSection(title:"Build this step",symbol:"list.number",identifier:"lesson.tutorial.action") {CourseCoachView(store:workspace)}
                        }
                        if affine.isActive {
                            LessonBlockSection(title:"Task, supplied values & hints",symbol:"text.alignleft",collapsible:true,identifier:"lesson.affine.reference") {AffineLearningView(lesson:affine,workspace:workspace).taskContents}
                        }
                        if canCheck {checkSection}
                        if let error=reading.error {Text(error).font(.caption).foregroundStyle(.orange)}
                        if entryID != "model-output" && !affine.isActive && !tensor.isActive && workspace.workflowSession?.owner.document.guidedBuild == nil {
                            LessonBlockSection(title:"Exercise choices & reference",symbol:"text.alignleft",collapsible:true,identifier:"lesson.reference") {
                                LearnCatalogContent(store:workspace,affineLearning:affine,normalizationLearning:normalization,tensorLearning:tensor,section:"Library",search:"").entryContent(entryID)
                                    .environment(\.lessonRailPresented,true)
                                    .font(.caption).padding(.top,10)
                            }
                        }
                    }.padding(14).frame(maxWidth:.infinity,alignment:.leading)
                }.id(content.steps[position].id)
                Divider()
                HStack {
                    Button("Back") {navigate(position-1,content:content)}.disabled(position==0)
                    Spacer()
                    Text("\(position+1) / \(content.steps.count)").font(.system(size:10)).foregroundStyle(MLTheme.muted)
                    Spacer()
                    if position==content.steps.count-1 {
                        Button("Return to library") {ApplicationNavigationWindows.shared.returnToLearningLibrary()}
                            .accessibilityIdentifier("lesson.return-library")
                    } else {
                        Button("Next") {navigate(position+1,content:content)}
                    }
                }.padding(14)
                .onAppear {workspace.lessonNodeFocus=content.steps[position].canvasFocus;_=reading.visit(position,content:content)}
                .onChange(of:(content.readingScopeID ?? "")+content.revision) {_,_ in
                    selectedStepID=nil;workspace.lessonNodeFocus=content.steps[reading.position(content)].canvasFocus;_=reading.visit(reading.position(content),content:content)
                }
            } else {Text("This lesson is unavailable; your canvas is preserved.").padding(14);Spacer()}
        }.background(MLTheme.background).overlay(alignment:.leading) {Rectangle().fill(MLTheme.rule).frame(width:1)}
            .buttonStyle(FlatButtonStyle()).environment(\.lessonContentAccent,topicAccent).environment(\.lessonTypography,typography).tint(topicAccent)
            .environment(\.openURL,OpenURLAction {url in
                guard url.scheme == "kwip-node" else {return .systemAction}
                guard let address=LessonNodeLink.address(url),let host=workspace.workflowSession,
                      host.canvas.revealLessonNode(address) else {
                    workspace.statusMessage="This lesson component is not on the current canvas."
                    return .discarded
                }
                if let id=host.canvas.projection.displayIDByAddress[address] {workspace.lessonNodeFocus = .init(nodeIDs:[id])}
                return .handled
            }).accessibilityIdentifier("lesson.rail")
            .onChange(of:workspace.workflowSession?.owner.document.guidedBuild?.completedSteps) {_,count in
                guard let count,let content else {return}
                navigate(min(max(0,count-1),content.steps.count-1),content:content)
            }
            .onChange(of:workspace.hasUnsavedChanges) {_,dirty in if dirty {saveMessage=nil}}
    }
    private var currentGrade:LearningGrade? {tensor.isActive ? tensor.grade:affine.isActive ? affine.latestGrade:normalization.latestGrade}
    private var resultCurrent:Bool {tensor.isActive ? tensor.resultIsCurrent:affine.isActive ? affine.latestResultIsCurrent:normalization.latestResultIsCurrent}
    private var receiptSaved:Bool {tensor.isActive ? tensor.resultSaved:affine.isActive ? affine.resultSaved:normalization.resultSaved}
    private var checkState:LessonCheckState {.init(grade:currentGrade,running:workspace.isAssessingLearning,canSubmit:canSubmit,unavailable:tensor.isActive && tensor.runtimeReadinessError != nil,current:resultCurrent)}
    private var feedbackItems:[LessonFeedbackItem] {
        (currentGrade?.outcomes ?? []).enumerated().map {index,outcome in
            let guide=tensor.isActive ? LessonAudienceGuide.criterionFeedback(operation:tensor.draft?.operation ?? "",criterionID:outcome.criterionID):nil
            let publicTitle=guide?.title ?? (affine.isActive ? affine.publicTask?.criteria.first {$0.id==outcome.criterionID}?.description:nil) ?? "Construction check \(index+1)"
            let nextStep=guide.flatMap { guidance in
                if let copy=TensorTopicLessonCopy.task(tensor.draft?.operation ?? "") {return copy.controls+" Apply pending fields, then choose Submit assessment again."}
                if let copy=LayerTopicLessonCopy.task(tensor.draft?.operation ?? ""),checkMode == .assessment {return copy.controls+" Apply pending fields, then choose Submit assessment again."}
                return guidance.nextStep
            }
            return .project(outcome,title:publicTitle,publicNextStep:nextStep,assessment:checkMode == .assessment)
        }
    }
    private var checkSection:some View {
        VStack(spacing:12) {
            LessonBlockSection(title:"Check your construction",symbol:"checkmark.circle",identifier:"lesson.check") {
                HStack {Text("Submission");Spacer();Label(checkState.label,systemImage:checkState.symbol).font(.system(size:10,weight:.medium)).foregroundStyle(checkState.tint)}
                Button(workspace.isAssessingLearning ? "Cancel check":checkTitle) {
                    workspace.isAssessingLearning ? workspace.learningProjects.cancelAssessment():check()
                }.frame(maxWidth:.infinity).buttonStyle(FlatButtonStyle(prominent:canSubmit && !workspace.isAssessingLearning))
                    .disabled(!canSubmit && !workspace.isAssessingLearning)
                if !canSubmit && !workspace.isAssessingLearning {
                    LessonRichText(text:tensor.isActive ? tensor.runtimeReadinessError ?? "Complete the construction and apply any pending fields.":"Complete the construction and apply any pending fields.")
                }
                LessonBlockSection(title:"Check details",symbol:"text.alignleft",collapsible:true,identifier:"lesson.check.detail") {LessonRichText(text:checkStatus)}
            }
            if let grade=currentGrade,!workspace.isAssessingLearning {
                LessonBlockSection(title:checkMode == .assessment ? "Assessment feedback":"Example check feedback",symbol:"text.bubble",identifier:"lesson.feedback") {
                    if !resultCurrent {Text(checkMode == .assessment ? "Previous construction · assess again after edits":"Previous example · check again after edits").font(.system(size:10)).foregroundStyle(MLTheme.muted)}
                    LessonCriteriaRowsView(items:feedbackItems)
                    LessonBlockSection(title:"Checker summary",symbol:"text.alignleft",collapsible:true,identifier:"lesson.feedback.summary") {LessonRichText(text:grade.summary)}
                    Text(receiptSaved ? (checkMode == .assessment ? "Receipt saved on this Mac":"Example check saved on this Mac"):(checkMode == .assessment ? "Receipt not saved":"Example check not saved")).font(.system(size:10)).foregroundStyle(MLTheme.muted)
                }
            }
        }
    }
    private var checkMode:LearningMode {affine.isActive ? affine.mode:tensor.draft?.mode ?? normalization.mode}
    private var checkTitle:String {checkMode == .assessment ? "Submit assessment":"Check construction"}
    private var canCheck:Bool {affine.isActive || tensor.isActive || normalization.isActive}
    private var canSubmit:Bool {affine.isActive ? affine.canSubmit : tensor.isActive ? tensor.canSubmit:normalization.canSubmit}
    private var checkStatus:String {affine.isActive ? affine.statusMessage : tensor.isActive ? tensor.statusMessage:normalization.statusMessage}
    private func check() {if tensor.isActive {tensor.submit()} else if affine.isActive {affine.submit()} else if normalization.isActive {normalization.submit()}}
    private func navigate(_ next:Int,content:LessonContent) {
        guard content.steps.indices.contains(next) else {return}
        selectedStepID=content.steps[next].id;workspace.lessonNodeFocus=content.steps[next].canvasFocus;_=reading.visit(next,content:content)
    }
    private func save() {
        let saved=LessonSaveAction.save(workspace:workspace,entryID:entryID) {
            if tensor.isActive {return tensor.save()}
            if affine.isActive {return affine.save()}
            if normalization.isActive {return normalization.save()}
            return workspace.saveManagedProject()
        }
        saveMessage=saved ? "Saved just now":workspace.saveError ?? "Save needs attention"
    }
    private func selectTarget(_ type:String) {
        if let host=workspace.workflowSession,
           let node=host.canvas.graph.nodes.first(where:{host.canvas.canvasIconType(nodeID:$0.id)==type}) {
            host.canvas.selectNode(node.id,extending:false)
        } else if let node=workspace.graph.nodes.first(where:{$0.type==type}) {workspace.selectedNodeID=node.id}
        else {workspace.statusMessage="This lesson component is not on the current canvas."}
    }
}

struct LessonReadingBadge:View {
    let entryID:String
    @ObservedObject var reading:LessonReadingStore
    var body:some View {
        if let content=reading.record(entryID).content ?? LessonContentCatalog.content(entryID:entryID) {
            let status=reading.status(content)
            Label(status.rawValue,systemImage:status == .viewed ? "eye":status == .inProgress ? "circle.lefthalf.filled":"circle")
                .font(.system(size:10)).foregroundStyle(MLTheme.muted)
                .accessibilityLabel("Reading: "+status.rawValue)
        }
    }
}

enum LessonWindowPresentation {
    static func symbol(kind:String)->String {
        switch kind {case "Guided":return "point.topleft.down.to.point.bottomright.curvepath";case "Example":return "doc.text.magnifyingglass";case "Construction":return "point.3.connected.trianglepath.dotted";default:return "pencil.and.outline"}
    }
}
